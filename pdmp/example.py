"""Geopolitical example model from the PLAN."""

from scipy import stats

from .core import StateGroup, ode, transition


class Military(StateGroup):
    interceptors: float = 100.0

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
    readiness: float = 0.3
    at_war: bool = False

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
