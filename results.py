"""Analyze simulation runs from runs.jsonl.

Model-agnostic: reads 'reported' metrics and 'reported_series' from JSONL
records, prints summary statistics, runs sensitivity analysis, and saves
time-series plots to reported/{name}.png.

Usage: uv run python results.py [runs.jsonl]
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr


def load_runs(path="runs.jsonl"):
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def quantile_summary(vals, fmt=".2f"):
    a = np.array(vals, dtype=float)
    p10, p50, p90 = np.percentile(a, [10, 50, 90])
    mean = np.mean(a)
    return f"mean={mean:{fmt}}  p10={p10:{fmt}}  p50={p50:{fmt}}  p90={p90:{fmt}}"


def auto_fmt(vals):
    a = np.array(vals, dtype=float)
    med = np.median(a)
    # Boolean-like (0/1 only)
    if set(float(v) for v in vals) <= {0.0, 1.0}:
        return ".0%"
    if abs(med) < 0.01 and med != 0:
        return ".4f"
    if abs(med) < 1:
        return ".2f"
    if abs(med) < 100:
        return ".1f"
    return ".0f"


def sensitivity(runs, metric_fn, min_samples=20):
    n = len(runs)
    if n < min_samples:
        return []

    metric_values = np.array([metric_fn(r) for r in runs])
    if np.std(metric_values) == 0:
        return []

    all_names = set()
    for r in runs:
        all_names.update(r["prior"].keys())

    output = []
    for name in sorted(all_names):
        values = [r["prior"][name] for r in runs if name in r["prior"]]
        if len(values) < min_samples:
            continue
        median_val = float(np.median(values))
        full = np.array([r["prior"].get(name, median_val) for r in runs])
        if np.std(full) == 0:
            continue
        rho, p = spearmanr(full, metric_values)
        output.append((name, float(rho), float(p), len(values)))

    output.sort(key=lambda x: abs(x[1]), reverse=True)
    return output


def plot_time_series(runs, out_dir="reported"):
    """Save one PNG per reported metric with all trajectories overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Collect all reported keys from series data
    reported_keys = set()
    for r in runs:
        for _, d in r.get("reported_series", []):
            reported_keys.update(d.keys())

    if not reported_keys:
        return

    for key in sorted(reported_keys):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Collect all time series for this key
        all_times = []
        all_values = []
        for r in runs:
            series = r.get("reported_series", [])
            if not series:
                continue
            ts = [pt[0] for pt in series if key in pt[1]]
            vs = [pt[1][key] for pt in series if key in pt[1]]
            if not ts:
                continue
            all_times.append(ts)
            all_values.append(vs)
            ax.plot(ts, vs, color="C0", alpha=0.1, lw=0.5)

        if not all_values:
            plt.close(fig)
            continue

        # Compute mean trajectory (interpolate to common time grid)
        max_len = max(len(ts) for ts in all_times)
        ref_times = all_times[np.argmax([len(ts) for ts in all_times])]
        t_grid = np.array(ref_times)

        # Stack all values onto the common grid (assumes same time points)
        matrix = []
        for ts, vs in zip(all_times, all_values):
            if len(ts) == len(t_grid):
                matrix.append(vs)
            else:
                # Interpolate to common grid
                matrix.append(np.interp(t_grid, ts, vs))

        matrix = np.array(matrix)
        mean_vals = np.mean(matrix, axis=0)
        ax.plot(t_grid, mean_vals, color="C0", alpha=1.0, lw=2, label="mean")

        ax.set_xlabel("Time (days)")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.legend(loc="best")

        safe_name = key.replace(".", "_").replace("/", "_")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{safe_name}.png"), dpi=150)
        plt.close(fig)

    print(f"\n  Plots saved to {out_dir}/")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs.jsonl"
    runs = load_runs(path)
    n = len(runs)

    if n == 0:
        print("No runs found.")
        return

    # --- Dataset info ---
    times = sorted(r["time"] for r in runs)
    print(f"=== Dataset: {n} runs ===")
    if n >= 2:
        span = times[-1] - times[0]
        if span > 0:
            print(f"  Collected over {span:.0f}s ({n / span:.1f} runs/s)")

    # --- Reported metrics (final values) ---
    reported_keys = set()
    for r in runs:
        if "reported" in r:
            reported_keys.update(r["reported"].keys())

    if reported_keys:
        print(f"\n=== Reported Metrics (final) ===")
        max_key_len = max(len(k) for k in reported_keys)
        for key in sorted(reported_keys):
            vals = [r["reported"][key] for r in runs if key in r.get("reported", {})]
            if not vals:
                continue
            fmt = auto_fmt(vals)
            print(f"  {key:{max_key_len}s}: {quantile_summary(vals, fmt)}")

    # --- Event counts ---
    all_events = set()
    for r in runs:
        all_events.update(r.get("event_counts", {}).keys())

    if all_events:
        print(f"\n=== Event Counts ===")
        max_ev_len = max(len(e) for e in all_events)
        for ev in sorted(all_events):
            counts = [r.get("event_counts", {}).get(ev, 0) for r in runs]
            print(f"  {ev:{max_ev_len}s}: {quantile_summary(counts, '.0f')}")

    # --- Sensitivity for each reported metric ---
    if reported_keys and n >= 20:
        for key in sorted(reported_keys):
            vals = [r.get("reported", {}).get(key) for r in runs]
            if any(v is None for v in vals):
                continue
            # Skip constant metrics
            if np.std(vals) == 0:
                continue

            sens = sensitivity(runs, lambda r, k=key: r["reported"][k])
            sig_results = [(name, rho, p, ns) for name, rho, p, ns in sens if p < 0.05]
            if not sig_results:
                continue

            print(f"\n=== Sensitivity: {key} (p<0.05, top 10) ===")
            for name, rho, p, ns in sig_results[:10]:
                print(f"  {name:40s} rho={rho:+.3f}  p={p:.4f}")

    # --- Time-series plots ---
    has_series = any(r.get("reported_series") for r in runs)
    if has_series:
        plot_time_series(runs)


if __name__ == "__main__":
    main()
