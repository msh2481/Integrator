# PDMP Simulation Engine

## Context
Lightweight Python framework for piecewise-deterministic Markov processes. Continuous ODEs plus stochastic transitions with state-dependent rates. Intended as a thinking tool for geopolitical and economic models: small codebase, readable model definitions, ensemble runs, and sensitivity analysis.

## Core Design

### Dependencies
- Use `uv` for environment and dependency management
- `numpy`
- `scipy` (`scipy.stats` for distributions, `scipy.stats.spearmanr` for sensitivity)
- No custom distribution wrappers; use `scipy.stats` directly

### `Prior` — trajectory-local sampler cache

```python
class Prior:
    def sample(self, name, dist):
        ...

prior = Prior(rng)
```

Usage inside model code goes through the current trajectory context, e.g. `self.prior.sample(...)` where `self.prior` resolves to `self._root._prior`.

- One `Prior` per trajectory
- First call per `name`: draw once from `dist` using the trajectory RNG, cache it
- Repeat calls: return cached value and assert distribution identity / parameters match
- `.values()`: `{name: sampled_value}` for that trajectory

No singleton or module-global `prior`; concurrent runs must not share sampler state.

### StateGroup

```python
class Military(StateGroup):
    interceptors: float = 100  # continuous, evolves via ODE

    @ode
    def dynamics(self):
        if not self._root.china.at_war:
            return {}
        rate = self.prior.sample("mil.attrition", stats.lognorm(s=0.5, scale=2))
        return {"interceptors": -rate}

    @property
    def depletion(self):
        return 1 - self.interceptors / 100

class China(StateGroup):
    readiness: float = 0.3    # continuous
    at_war: bool = False      # discrete, changed by transitions

    @ode
    def buildup(self):
        return {"readiness": 0.05 * (1 - self.readiness)}

    @transition
    def invades_taiwan(self):
        base = self.prior.sample("china.base_rate", stats.uniform(0, 0.5))
        rate = base * self._root.military.depletion * self.readiness
        def effect():
            self.at_war = True
        return rate, effect

class World(StateGroup):
    military: Military
    china: China
```

`World.__init__` creates children and passes `self` as `_root` to each descendant.

Rules:
- Scalar numeric fields may be targeted by ODE updates
- Discrete fields may be mutated by transition effects
- ODE methods return `{field_name: derivative}`
- Unknown fields, non-numeric ODE targets, `nan`, or `inf` derivatives are errors
- Transition names default to method names and should be unique within the model tree for reporting and event logs

### RNG and reproducibility

`simulate(..., rng=None)` accepts:
- `None`: nondeterministic run
- `int`: base seed
- `np.random.Generator`: explicit generator

Each trajectory gets its own RNG stream. For ensemble runs, spawn per-trajectory seeds from a base `SeedSequence` so sequential and parallel execution can both be reproducible when desired.

All randomness goes through the trajectory RNG:
- transition event sampling
- transition selection among competing rates
- prior draws from `scipy.stats`

### Integration: fixed-step Euler

Use fixed-step Euler: `x += dx * dt`. This keeps the implementation small and transparent. If needed later, swap in RK4 behind the same interface.

### Transition semantics

Each `@transition` returns `(rate, effect)`.

Rules:
- `rate` must be finite and `>= 0`
- `effect` is a callable with no arguments
- effects may mutate any reachable part of the state tree
- invalid rates (`nan`, `inf`, negative`) raise immediately

### Simulation loop

```python
def simulate(
    world,
    t_end,
    n_runs,
    dt=0.1,
    record_every=1.0,
    rng=None,
    workers=1,
):
    ...
```

Per trajectory:
1. `deepcopy(world)`
2. wire `_root` references
3. create trajectory-local RNG and `Prior`
4. discover bound ODE and transition methods
5. run until `t_end`

Within each outer step of size `dt`, allow multiple jumps:
1. set `remaining = dt`
2. evaluate transition rates on the current state
3. if total rate is zero, Euler-advance by `remaining` and finish the step
4. otherwise sample `tau ~ Exp(total_rate)`
5. if `tau >= remaining`, Euler-advance by `remaining` and finish the step
6. if `tau < remaining`, Euler-advance by `tau`, choose one transition proportional to its rate, apply its effect, subtract `tau` from `remaining`, and continue

Assumptions:
- This is much better than a single Bernoulli "did any event happen this step?" check
- Multiple events may occur within one outer `dt`
- Rates are treated as frozen between jump checks within each substep; Euler still controls the continuous approximation error
- Transition rates should not change too rapidly on the `dt` scale
- This design does not implement thinning or exact continuous-time hazard integration for fully time-varying rates

No flattening / unflattening layer needed; Euler operates directly on state object fields.

### Concurrency

Ensemble runs are embarrassingly parallel, so `simulate(..., workers=N)` should support process-level parallelism.

Approach:
- `workers=1`: sequential path
- `workers>1`: `concurrent.futures.ProcessPoolExecutor`
- each worker runs complete independent trajectories and returns serialized results
- no shared mutable state across workers

Motivation:
- Python model code is CPU-bound
- process isolation avoids shared `Prior` / RNG state issues
- should scale well for posterior / ensemble sampling workloads

Implementation note: keep the single-trajectory runner as a pure top-level function so it is easy to dispatch in worker processes.

### Results

`Results` should retain, per trajectory:
- recorded snapshots
- transition event log: `{transition_name: [t0, t1, ...]}`
- sampled prior values
- RNG seed or seed material when reproducibility is requested

Transition names come from the transition method name. When a transition fires, append the event time to that transition's list; untouched transitions have empty lists.

This gives one place to support summaries, sensitivity analysis, and later export helpers.

### Sensitivity

Store one sampled-prior dict per trajectory, then compute sensitivity after the ensemble run.

Algorithm:
1. choose an output metric, e.g. final readiness, time of first war event, max depletion
2. evaluate that metric for every trajectory
3. take the union of all prior names sampled across trajectories
4. for each prior name, build the vector of sampled values across runs, imputing missing entries with the median over trajectories where that prior was sampled
5. compute Spearman rank correlation between the metric vector and the prior vector
6. report `(metric_name, prior_name, rho, p_value, n)`
7. sort by `abs(rho)` descending

Notes:
- a prior may be sampled in some trajectories but not others because model execution is path-dependent
- median imputation keeps the vector length aligned with the metric vector while minimizing distortion from extreme values
- impose a minimum sample count before reporting a correlation

## File structure

```
pdmp/
    __init__.py
    core.py          # StateGroup, @ode, @transition, Prior, field discovery
    simulate.py      # simulate(), trajectory runner, Euler stepping, event handling
    sensitivity.py   # metrics + rank-correlation analysis
    example.py       # geopolitical example
```

## Verification

1. Exponential decay ODE -> check analytical solution
2. Fixed-rate transition -> jump count and waiting times match Poisson / exponential theory
3. Multiple jumps in one outer `dt` are possible when rates are high
4. ODE-driven increasing rate -> events cluster later
5. Prior caching: same name returns same value; different distribution for same name -> error
6. RNG reproducibility: same seed -> same trajectories and prior draws
7. Parallel vs sequential ensemble with same spawned seeds -> same results
8. Sensitivity ranking on a known linear / monotone model
9. Cross-group effect: China transition mutates Military field
10. `dt` sensitivity: compare results at `dt=0.1` vs `dt=0.01`, should converge
