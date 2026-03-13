"""Simulation loop: Euler stepping with PDMP transition handling."""

from __future__ import annotations

import copy
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from .core import Prior, StateGroup


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class TrajectoryResult:
    """Results from a single trajectory."""

    def __init__(self, snapshots, event_log, prior_values, seed):
        self.snapshots: list[tuple[float, dict]] = snapshots
        self.event_log: dict[str, list[float]] = event_log
        self.prior_values: dict[str, float] = prior_values
        self.seed: int | None = seed


class Results:
    """Collection of trajectory results from an ensemble run."""

    def __init__(self, trajectories: list[TrajectoryResult]):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def __iter__(self):
        return iter(self.trajectories)


# ---------------------------------------------------------------------------
# Single-trajectory runner (top-level for pickling)
# ---------------------------------------------------------------------------

def _run_trajectory(world_template: StateGroup, t_end: float, dt: float,
                    record_every: float, seed: int | None) -> TrajectoryResult:
    """Run one trajectory. Pure top-level function for process dispatch."""
    # Deep copy and wire up
    world = copy.deepcopy(world_template)
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    prior = Prior(rng)
    world._wire_root(world, prior)

    # Discover ODE and transition methods
    odes = world._collect_odes()
    transitions = world._collect_transitions()

    # Initialize event log
    event_log: dict[str, list[float]] = {name: [] for name, _, _ in transitions}

    # Recording
    snapshots: list[tuple[float, dict]] = []
    next_record = 0.0

    t = 0.0

    # Record initial state
    snapshots.append((t, world._snapshot()))
    next_record = record_every

    while t_end - t > 1e-12:
        remaining = min(dt, t_end - t)

        while remaining > 1e-12:
            # Evaluate transition rates
            rates = []
            trans_info = []
            for name, group, method in transitions:
                rate, effect = method()
                if not math.isfinite(rate) or rate < 0:
                    raise ValueError(
                        f"Invalid rate {rate} from transition '{name}'"
                    )
                rates.append(rate)
                trans_info.append((name, effect))

            total_rate = sum(rates)

            if total_rate == 0:
                # No transitions possible — advance full remaining
                _euler_step(world, odes, remaining)
                t += remaining
                remaining = 0.0
            else:
                # Sample waiting time
                tau = rng.exponential(1.0 / total_rate)

                if tau >= remaining:
                    # No event this substep
                    _euler_step(world, odes, remaining)
                    t += remaining
                    remaining = 0.0
                else:
                    # Event occurs at t + tau
                    _euler_step(world, odes, tau)
                    t += tau
                    remaining -= tau

                    # Choose transition proportional to rate
                    probs = np.array(rates) / total_rate
                    idx = rng.choice(len(rates), p=probs)
                    chosen_name, chosen_effect = trans_info[idx]
                    chosen_effect()
                    event_log[chosen_name].append(t)

        # Check if we should record
        while next_record <= t + 1e-12:
            snapshots.append((t, world._snapshot()))
            next_record += record_every

    # Final snapshot if not already recorded
    if len(snapshots) == 0 or abs(snapshots[-1][0] - t) > 1e-12:
        snapshots.append((t, world._snapshot()))

    return TrajectoryResult(
        snapshots=snapshots,
        event_log=event_log,
        prior_values=prior.values(),
        seed=seed,
    )


def _euler_step(world: StateGroup, odes: list, dt: float):
    """Apply one Euler step for all ODE methods."""
    for group, method in odes:
        derivs = method()
        if derivs:
            world._apply_derivatives(group, derivs, dt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    world: StateGroup,
    t_end: float,
    n_runs: int = 1,
    dt: float = 0.1,
    record_every: float = 1.0,
    rng=None,
    workers: int = 1,
) -> Results:
    """Run an ensemble of PDMP trajectories.

    Args:
        world: Template StateGroup (will be deepcopied per trajectory).
        t_end: Simulation end time.
        n_runs: Number of trajectories.
        dt: Outer Euler step size.
        record_every: Snapshot recording interval.
        rng: None, int seed, or np.random.Generator.
        workers: Number of parallel workers (1 = sequential).
    """
    # Generate per-trajectory seeds
    if rng is None:
        seeds = [None] * n_runs
    elif isinstance(rng, int):
        ss = np.random.SeedSequence(rng)
        child_seeds = ss.spawn(n_runs)
        seeds = [int(cs.generate_state(1)[0]) for cs in child_seeds]
    elif isinstance(rng, np.random.Generator):
        # Extract seed sequence from generator's bit_generator
        ss = rng.bit_generator.seed_seq
        if ss is None:
            seeds = [None] * n_runs
        else:
            child_seeds = ss.spawn(n_runs)
            seeds = [int(cs.generate_state(1)[0]) for cs in child_seeds]
    else:
        raise TypeError(f"Unsupported rng type: {type(rng)}")

    if workers <= 1:
        trajectories = [
            _run_trajectory(world, t_end, dt, record_every, seed)
            for seed in seeds
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_trajectory, world, t_end, dt, record_every, seed)
                for seed in seeds
            ]
            trajectories = [f.result() for f in futures]

    return Results(trajectories)
