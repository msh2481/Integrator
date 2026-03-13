"""Verification tests for the PDMP simulation engine."""

import math

import numpy as np
from scipy import stats

from core import StateGroup, ode, transition, Prior
from simulate import simulate
from sensitivity import sensitivity


# ===========================================================================
# 1. Exponential decay ODE -> check analytical solution
# ===========================================================================

class DecayModel(StateGroup):
    x: float = 100.0

    @ode
    def decay(self):
        return {"x": -0.1 * self.x}

    @transition
    def dummy(self):
        return 0.0, lambda: None


def test_exponential_decay():
    model = DecayModel()
    results = simulate(model, t_end=10.0, n_runs=1, dt=0.01, record_every=10.0, rng=42)
    traj = results[0]
    final = traj.snapshots[-1][1]["x"]
    expected = 100.0 * math.exp(-0.1 * 10.0)
    # Euler with dt=0.01 over 10s should be close
    assert abs(final - expected) / expected < 0.01, f"Decay: {final} vs {expected}"


# ===========================================================================
# 2. Fixed-rate transition -> Poisson jump count
# ===========================================================================

class PoissonModel(StateGroup):
    count: int = 0

    @ode
    def noop(self):
        return {}

    @transition
    def jump(self):
        rate = 2.0  # 2 events per unit time
        def effect():
            self.count += 1
        return rate, effect


def test_poisson_jumps():
    model = PoissonModel()
    n_runs = 500
    t_end = 10.0
    results = simulate(model, t_end=t_end, n_runs=n_runs, dt=0.1, record_every=t_end, rng=123)
    counts = [len(traj.event_log["jump"]) for traj in results]
    mean_count = np.mean(counts)
    expected = 2.0 * t_end  # Poisson mean = rate * time
    assert abs(mean_count - expected) / expected < 0.1, f"Poisson: mean={mean_count} vs expected={expected}"


# ===========================================================================
# 3. Multiple jumps in one outer dt
# ===========================================================================

class HighRateModel(StateGroup):
    n: int = 0

    @ode
    def noop(self):
        return {}

    @transition
    def fast_jump(self):
        rate = 100.0
        def effect():
            self.n += 1
        return rate, effect


def test_multiple_jumps_per_dt():
    model = HighRateModel()
    results = simulate(model, t_end=1.0, n_runs=1, dt=1.0, record_every=1.0, rng=42)
    # With rate=100, expect ~100 jumps in 1 time unit, all within one outer dt
    n_jumps = len(results[0].event_log["fast_jump"])
    assert n_jumps > 10, f"Expected many jumps in one dt, got {n_jumps}"


# ===========================================================================
# 4. ODE-driven increasing rate -> events cluster later
# ===========================================================================

class IncreasingRateModel(StateGroup):
    intensity: float = 0.0

    @ode
    def ramp(self):
        return {"intensity": 1.0}  # linear ramp

    @transition
    def event(self):
        rate = max(0.0, self.intensity)
        def effect():
            pass
        return rate, effect


def test_events_cluster_later():
    model = IncreasingRateModel()
    results = simulate(model, t_end=10.0, n_runs=200, dt=0.05, record_every=10.0, rng=99)
    all_times = []
    for traj in results:
        all_times.extend(traj.event_log["event"])
    if len(all_times) > 10:
        median_time = np.median(all_times)
        # With linearly increasing rate, events should cluster in the latter half
        assert median_time > 5.0, f"Events should cluster later, median={median_time}"


# ===========================================================================
# 5. Prior caching
# ===========================================================================

def test_prior_caching():
    rng = np.random.default_rng(0)
    prior = Prior(rng)
    v1 = prior.sample("x", stats.norm(0, 1))
    v2 = prior.sample("x", stats.norm(0, 1))
    assert v1 == v2, "Prior should cache"

    try:
        prior.sample("x", stats.uniform(0, 1))
        assert False, "Should have raised ValueError for different distribution"
    except ValueError:
        pass

    vals = prior.values()
    assert "x" in vals and vals["x"] == v1


# ===========================================================================
# 6. RNG reproducibility
# ===========================================================================

def test_rng_reproducibility():
    model = PoissonModel()
    r1 = simulate(model, t_end=5.0, n_runs=3, dt=0.1, record_every=5.0, rng=77)
    r2 = simulate(model, t_end=5.0, n_runs=3, dt=0.1, record_every=5.0, rng=77)
    for i in range(3):
        assert r1[i].event_log == r2[i].event_log, f"Trajectory {i} event logs differ"
        s1 = r1[i].snapshots[-1][1]
        s2 = r2[i].snapshots[-1][1]
        assert s1 == s2, f"Trajectory {i} final states differ"


# ===========================================================================
# 7. Parallel vs sequential ensemble
# ===========================================================================

def test_parallel_vs_sequential():
    model = PoissonModel()
    seq = simulate(model, t_end=5.0, n_runs=4, dt=0.1, record_every=5.0, rng=55, workers=1)
    par = simulate(model, t_end=5.0, n_runs=4, dt=0.1, record_every=5.0, rng=55, workers=2)
    for i in range(4):
        assert seq[i].event_log == par[i].event_log, f"Traj {i}: parallel differs from sequential"


# ===========================================================================
# 8. Sensitivity ranking on a known linear model
# ===========================================================================

class LinearModel(StateGroup):
    y: float = 0.0

    @ode
    def grow(self):
        a = self.prior.sample("slope", stats.uniform(0.5, 1.5))
        return {"y": a}

    @transition
    def dummy(self):
        return 0.0, lambda: None


def test_sensitivity():
    model = LinearModel()
    results = simulate(model, t_end=10.0, n_runs=100, dt=0.1, record_every=10.0, rng=42)

    def final_y(traj):
        return traj.snapshots[-1][1]["y"]

    sens = sensitivity(results, final_y)
    assert len(sens) > 0, "Should have sensitivity results"
    # The slope prior should have a strong positive correlation with final y
    name, rho, pval, n = sens[0]
    assert name == "slope", f"Expected 'slope' to be top, got '{name}'"
    assert rho > 0.5, f"Expected strong positive correlation, got rho={rho}"


# ===========================================================================
# 9. Cross-group effect
# ===========================================================================

class GroupA(StateGroup):
    value: float = 10.0

    @ode
    def noop(self):
        return {}


class GroupB(StateGroup):
    trigger: bool = False

    @ode
    def noop(self):
        return {}

    @transition
    def cross_effect(self):
        rate = 100.0  # fire quickly
        def effect():
            self._root.a.value = 0.0
            self.trigger = True
        return rate, effect


class CrossModel(StateGroup):
    a: GroupA
    b: GroupB


def test_cross_group_effect():
    model = CrossModel()
    results = simulate(model, t_end=1.0, n_runs=1, dt=0.1, record_every=1.0, rng=42)
    final = results[0].snapshots[-1][1]
    assert final["b"]["trigger"] is True
    assert final["a"]["value"] == 0.0


# ===========================================================================
# 10. dt sensitivity: convergence
# ===========================================================================

def test_dt_convergence():
    model = DecayModel()
    r_coarse = simulate(model, t_end=10.0, n_runs=1, dt=0.1, record_every=10.0, rng=42)
    r_fine = simulate(model, t_end=10.0, n_runs=1, dt=0.01, record_every=10.0, rng=42)
    coarse = r_coarse[0].snapshots[-1][1]["x"]
    fine = r_fine[0].snapshots[-1][1]["x"]
    exact = 100.0 * math.exp(-1.0)
    # Fine should be closer to exact than coarse
    assert abs(fine - exact) < abs(coarse - exact), "Finer dt should be more accurate"


# ===========================================================================
# Run all
# ===========================================================================

if __name__ == "__main__":
    tests = [
        test_exponential_decay,
        test_poisson_jumps,
        test_multiple_jumps_per_dt,
        test_events_cluster_later,
        test_prior_caching,
        test_rng_reproducibility,
        test_parallel_vs_sequential,
        test_sensitivity,
        test_cross_group_effect,
        test_dt_convergence,
    ]
    for t in tests:
        print(f"Running {t.__name__}...", end=" ")
        t()
        print("OK")
    print(f"\nAll {len(tests)} tests passed.")
