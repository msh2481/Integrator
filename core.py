"""StateGroup, @ode, @transition decorators, Prior, and field discovery."""

from __future__ import annotations

import copy
import math
from typing import Any


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def ode(fn):
    """Mark a method as an ODE contributor."""
    fn._is_ode = True
    return fn


def transition(fn):
    """Mark a method as a stochastic transition."""
    fn._is_transition = True
    return fn


def reported(fn):
    """Mark a property as a reported metric (included in reported snapshot).

    Works with either decorator order:
        @property @reported  — fn is a function, mark it, return it
        @reported @property  — fn is a property, mark its fget
    """
    if isinstance(fn, property):
        fn.fget._is_reported = True
        return fn
    fn._is_reported = True
    return fn


# ---------------------------------------------------------------------------
# Prior — trajectory-local sampler cache
# ---------------------------------------------------------------------------

class Prior:
    """Draw-once-per-trajectory parameter cache."""

    def __init__(self, rng):
        self._rng = rng
        self._cache: dict[str, Any] = {}
        self._dists: dict[str, Any] = {}

    def sample(self, name: str, dist) -> float:
        if name in self._cache:
            # Verify same distribution type and parameters
            prev = self._dists[name]
            if prev.dist.name != dist.dist.name or dist.args != prev.args or dist.kwds != prev.kwds:
                raise ValueError(
                    f"Prior '{name}' already sampled with different distribution: "
                    f"{prev} vs {dist}"
                )
            return self._cache[name]
        value = dist.rvs(random_state=self._rng)
        self._cache[name] = float(value)
        self._dists[name] = dist
        return self._cache[name]

    def values(self) -> dict[str, float]:
        return dict(self._cache)


# ---------------------------------------------------------------------------
# StateGroup
# ---------------------------------------------------------------------------

class StateGroup:
    """Base class for state groups in a PDMP model.

    Subclasses declare:
    - float/int/bool fields as class-level annotations with defaults
    - child StateGroups as annotations with type hints
    - @ode methods returning {field: derivative}
    - @transition methods returning (rate, effect_callable)
    """

    def __init__(self, _root=None, _prior=None):
        self._root: StateGroup = _root if _root is not None else self
        self._prior_ref: Prior | None = _prior

        # Collect declared fields and children from MRO
        self._fields: dict[str, type] = {}
        self._children: dict[str, type] = {}

        for cls in reversed(type(self).__mro__):
            for name, ann in getattr(cls, "__annotations__", {}).items():
                if name.startswith("_"):
                    continue
                if isinstance(ann, type) and issubclass(ann, StateGroup):
                    self._children[name] = ann
                else:
                    self._fields[name] = ann

        # Initialize scalar fields from class defaults
        for name in self._fields:
            for cls in type(self).__mro__:
                if name in cls.__dict__ and not isinstance(cls.__dict__[name], property):
                    setattr(self, name, cls.__dict__[name])
                    break

        # Initialize child groups
        for name, child_cls in self._children.items():
            child = child_cls(_root=self._root, _prior=_prior)
            setattr(self, name, child)

    @property
    def prior(self) -> Prior:
        return self._root._prior_ref

    def _wire_root(self, root: StateGroup, prior: Prior | None = None):
        """Recursively set _root and _prior_ref after deepcopy."""
        self._root = root
        if prior is not None:
            self._prior_ref = prior
        for name in self._children:
            child = getattr(self, name)
            child._wire_root(root, prior)

    def _init_state(self):
        """Called once per trajectory after prior is wired. Override to sample initial state."""
        for name in self._children:
            getattr(self, name)._init_state()

    def _collect_odes(self) -> list[tuple[StateGroup, callable]]:
        """Find all @ode methods in this group and descendants."""
        results = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            attr = getattr(type(self), name, None)
            if callable(attr) and getattr(attr, "_is_ode", False):
                results.append((self, getattr(self, name)))
        for child_name in self._children:
            child = getattr(self, child_name)
            results.extend(child._collect_odes())
        return results

    def _collect_transitions(self) -> list[tuple[str, StateGroup, callable]]:
        """Find all @transition methods in this group and descendants."""
        results = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            attr = getattr(type(self), name, None)
            if callable(attr) and getattr(attr, "_is_transition", False):
                results.append((name, self, getattr(self, name)))
        for child_name in self._children:
            child = getattr(self, child_name)
            results.extend(child._collect_transitions())
        return results

    def _snapshot(self) -> dict[str, Any]:
        """Recursively snapshot all scalar fields."""
        snap = {}
        for name in self._fields:
            snap[name] = getattr(self, name)
        for child_name in self._children:
            child = getattr(self, child_name)
            snap[child_name] = child._snapshot()
        return snap

    def _reported_snapshot(self) -> dict[str, Any]:
        """Collect values of all @reported properties in the tree.

        Returns a flat dict with dotted keys, e.g. 'iran.mrbm_depletion'.
        """
        snap = {}
        for name in dir(self):
            if name.startswith("_"):
                continue
            attr = getattr(type(self), name, None)
            if isinstance(attr, property) and getattr(attr.fget, "_is_reported", False):
                snap[name] = getattr(self, name)
        for child_name in self._children:
            child = getattr(self, child_name)
            for k, v in child._reported_snapshot().items():
                snap[f"{child_name}.{k}"] = v
        return snap

    def _apply_derivatives(self, group: 'StateGroup', derivs: dict[str, float], dt: float):
        """Apply Euler step: field += deriv * dt, with validation."""
        for field, deriv in derivs.items():
            if field not in group._fields:
                raise ValueError(
                    f"ODE targets unknown field '{field}' in {type(group).__name__}"
                )
            val = getattr(group, field)
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"ODE targets non-numeric field '{field}' in {type(group).__name__}"
                )
            if not math.isfinite(deriv):
                raise ValueError(
                    f"Non-finite derivative for '{field}' in {type(group).__name__}: {deriv}"
                )
            setattr(group, field, val + deriv * dt)
