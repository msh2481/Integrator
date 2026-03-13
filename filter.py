"""Particle filter: select simulations matching observed conditions, then
visualize the posterior over prior parameters.

Usage: uv run python filter.py [runs.jsonl]

Edit the FILTERS list below to specify conditions.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import ks_2samp, spearmanr, ttest_ind

from results import load_runs

# ===================================================================
# Helpers to query a run
# ===================================================================

def reported_at(run, day, key):
    """Get a reported metric at the closest time <= day."""
    series = run.get("reported_series", [])
    best = None
    for t, d in series:
        if t <= day + 0.5 and key in d:
            best = d[key]
    return best


def final(run, *path):
    """Walk into final_values by dotted path, e.g. final(r, "gulf", "pac3")."""
    node = run.get("final_values", {})
    for key in path:
        if isinstance(node, dict):
            node = node.get(key)
        else:
            return None
    return node


def prior(run, name):
    """Get a prior parameter value."""
    return run["prior"].get(name)


# ===================================================================
# FILTERS — edit these
#
# Each filter is a function: run -> bool.
# A run passes if ALL filters return True.
# ===================================================================

FILTERS = [
    # # By day 10, Iran has 20-40% of TELs remaining
    # lambda r: (
    #     (init := reported_at(r, 0, "iran.n_tels")) is not None
    #     and init > 0
    #     and (tel := reported_at(r, 10, "iran.n_tels")) is not None
    #     and 0.20 <= tel / init <= 0.40
    # ),

    # By day 10, Gulf interceptors are 30-50% depleted
    lambda r: (
        (init := reported_at(r, 0, "gulf.total_interceptors")) is not None
        and init > 0
        and (cur := reported_at(r, 10, "gulf.total_interceptors")) is not None
        and 0.30 <= 1 - cur / init <= 0.50
    ),
]


# ===================================================================
# Core
# ===================================================================

def apply_filters(runs, filters):
    return [r for r in runs if all(f(r) for f in filters)]


def _fmt_p(p):
    if p < 0.001:
        return f"[bold red]{p:.1e}[/]"
    if p < 0.01:
        return f"[red]{p:.4f}[/]"
    if p < 0.05:
        return f"[yellow]{p:.4f}[/]"
    return f"{p:.4f}"


def print_summary(runs, filtered):
    console = Console()
    n_total = len(runs)
    n_pass = len(filtered)
    console.print(f"\n[bold]Filter: {n_pass}/{n_total} runs pass ({100*n_pass/n_total:.1f}%)[/]\n")

    if n_pass == 0:
        console.print("No runs match filters.")
        return

    all_names = sorted({k for r in filtered for k in r["prior"]})

    # Compute tests for each prior parameter
    rows = []
    for name in all_names:
        prior_vals = np.array([r["prior"][name] for r in runs if name in r["prior"]])
        post_vals = np.array([r["prior"][name] for r in filtered if name in r["prior"]])
        if len(post_vals) < 2 or np.std(prior_vals) == 0:
            continue

        t_stat, t_p = ttest_ind(post_vals, prior_vals, equal_var=False)
        ks_stat, ks_p = ks_2samp(post_vals, prior_vals)
        rows.append((name, t_stat, t_p, ks_stat, ks_p))

    # Sort by KS p-value (most significant first)
    rows.sort(key=lambda r: r[4])

    table = Table(title="Prior vs Posterior", show_lines=False)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("t-stat (p)", justify="right")
    table.add_column("KS stat (p)", justify="right")

    for name, t_stat, t_p, ks_stat, ks_p in rows:
        table.add_row(
            name,
            f"{t_stat:+.2f} ({_fmt_p(t_p)})",
            f"{ks_stat:.3f} ({_fmt_p(ks_p)})",
        )

    console.print(table)


def plot_marginals(runs, filtered, out_dir="filter_report"):
    """One histogram per prior parameter: prior (grey) vs posterior (blue)."""
    os.makedirs(out_dir, exist_ok=True)

    all_names = sorted({k for r in filtered for k in r["prior"]})
    if not all_names:
        return

    for name in all_names:
        prior_vals = [r["prior"][name] for r in runs if name in r["prior"]]
        post_vals = [r["prior"][name] for r in filtered if name in r["prior"]]
        if len(post_vals) < 2:
            continue

        fig, ax = plt.subplots(figsize=(6, 3.5))

        # Common bins from prior range
        lo = min(prior_vals)
        hi = max(prior_vals)
        bins = np.linspace(lo, hi, 40)

        ax.hist(prior_vals, bins=bins, density=True, color="0.75", label="prior")
        ax.hist(post_vals, bins=bins, density=True, color="C0", alpha=0.7, label="posterior")
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(name, fontsize=10)

        safe = name.replace(".", "_").replace("/", "_")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{safe}.png"), dpi=120)
        plt.close(fig)

    print(f"\n  Marginal histograms saved to {out_dir}/")


def plot_correlation(filtered, out_dir="filter_report"):
    """Spearman correlation heatmap of prior parameters."""
    os.makedirs(out_dir, exist_ok=True)

    all_names = sorted({k for r in filtered for k in r["prior"]})
    if len(all_names) < 2 or len(filtered) < 5:
        return

    # Build matrix
    matrix = np.array([
        [r["prior"].get(name, np.nan) for name in all_names]
        for r in filtered
    ])

    # Drop columns with no variance
    stds = np.nanstd(matrix, axis=0)
    keep = stds > 0
    matrix = matrix[:, keep]
    names = [n for n, k in zip(all_names, keep) if k]

    if len(names) < 2:
        return

    n = len(names)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
            elif i < j:
                rho, _ = spearmanr(matrix[:, i], matrix[:, j])
                corr[i, j] = rho
                corr[j, i] = rho

    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(6, n * 0.35)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("Prior parameter correlations (filtered runs)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "corr.png"), dpi=120)
    plt.close(fig)

    print(f"  Correlation heatmap saved to {out_dir}/corr.png")


# ===================================================================
# Main
# ===================================================================

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs.jsonl"
    runs = load_runs(path)
    print(f"Loaded {len(runs)} runs from {path}")

    filtered = apply_filters(runs, FILTERS)
    print_summary(runs, filtered)

    if filtered:
        plot_marginals(runs, filtered)
        plot_correlation(filtered)


if __name__ == "__main__":
    main()
