"""Sensitivity analysis via Spearman rank correlation on prior draws."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import spearmanr

from simulate import Results


def sensitivity(
    results: Results,
    metric: Callable,
    min_samples: int = 5,
) -> list[tuple[str, float, float, int]]:
    """Compute Spearman rank correlations between prior draws and an output metric.

    Args:
        results: Ensemble results from simulate().
        metric: Callable taking a TrajectoryResult, returning a float.
        min_samples: Minimum number of non-imputed samples to report.

    Returns:
        List of (prior_name, rho, p_value, n_sampled) sorted by |rho| descending.
    """
    n = len(results)
    if n < min_samples:
        return []

    # Evaluate metric for each trajectory
    metric_values = np.array([metric(traj) for traj in results])

    # Collect union of all prior names
    all_names: set[str] = set()
    for traj in results:
        all_names.update(traj.prior_values.keys())

    output = []

    for name in sorted(all_names):
        # Build vector of sampled values, track which trajectories have it
        values = []
        present = []
        for i, traj in enumerate(results):
            if name in traj.prior_values:
                values.append(traj.prior_values[name])
                present.append(i)

        n_sampled = len(values)
        if n_sampled < min_samples:
            continue

        # Impute missing with median
        median_val = float(np.median(values))
        full_vector = np.array([
            traj.prior_values.get(name, median_val) for traj in results
        ])

        # Filter out constant vectors
        if np.std(full_vector) == 0 or np.std(metric_values) == 0:
            continue

        rho, p_value = spearmanr(full_vector, metric_values)
        output.append((name, float(rho), float(p_value), n_sampled))

    # Sort by |rho| descending
    output.sort(key=lambda x: abs(x[1]), reverse=True)
    return output
