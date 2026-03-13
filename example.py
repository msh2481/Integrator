"""Iran conflict simulation — March 2026 scenario.

Models a multi-week US-Iran conflict with:
- Iranian ballistic missile, cruise missile, and drone salvos
- US/coalition strikes on TELs and production facilities
- Multi-layered missile defense (Arrow-3, Arrow-2, THAAD, PAC-3, David's Sling, Iron Dome)
- Production/replenishment on both sides

Time unit: 1 = 1 day.
"""

import json
import sys
import time

from scipy import stats

from core import StateGroup, ode, transition, reported
from simulate import simulate


# ===================================================================
# Assumptions
#
# Each range [low, high] is in natural units. All sampled log-uniform.
#
# Sources: IRAN_ARSENAL.md, IRAN_DOCTRINE.md, DEFENSE_SYSTEMS.md,
#          US_OFFENSE.md
# ===================================================================

# --- Iran: Initial Arsenal (count) ---
# High end includes possibility of undisclosed stockpiles / surge production.
# Total BM high end ~3k (MRBM+SRBM+cruise), drones ~50k.

IRAN_MRBM = (1000, 2000)             # low: conservative OSINT; high: undisclosed + surge
IRAN_SRBM = (1000, 6000)             # Fateh family is high-volume
IRAN_CRUISE = (100, 500)            # Paveh/Hoveyzeh/Quds
IRAN_DRONE = (2000, 50000)          # on-hand; high end = full wartime stockpile
IRAN_TEL = (80, 200)                # all types

# --- Iran: Production Rates (per year) ---

IRAN_MRBM_PROD = (30, 150)          # CSIS baseline + wartime surge
IRAN_SRBM_PROD = (80, 400)          # high-volume solid-fuel
IRAN_CRUISE_PROD = (30, 150)        # constrained by turbojet supply
IRAN_DRONE_PROD = (2000, 10000)     # IISS baseline + expanded facilities

# --- Iran: Salvo Parameters (natural units) ---

IRAN_MRBM_SALVO_INTERVAL = (1, 7)   # days between MRBM salvos
IRAN_MRBM_SALVO_FRAC = (0.01, 0.10) # fraction of available per salvo; Iran conserves stock
IRAN_MRBM_MIN_SALVO = 20            # minimum salvo size (fixed)
IRAN_MISSILES_PER_TEL = 2           # per salvo window (fixed)

IRAN_SRBM_SALVO_INTERVAL = (0.5, 3) # days between SRBM salvos
IRAN_SRBM_PER_SALVO = (10, 100)     # missiles per salvo

IRAN_DRONES_PER_DAY = (10, 300)     # drones launched per day (continuous ops)
IRAN_CRUISE_PER_DAY = (2, 40)       # cruise missiles launched per day

# --- Defense: Drone/Cruise Intercept Allocation ---

FRAC_DRONE_IRON_DOME = (0.30, 0.70)  # fraction of drone kills by Iron Dome (rest by fighters)
FRAC_CRUISE_DS = (0.40, 0.80)        # fraction of cruise kills by David's Sling
FRAC_CRUISE_PAC3 = (0.10, 0.40)      # fraction of cruise kills by PAC-3

# --- Defense: Interceptor Inventories (count) ---

DEFENSE_ARROW3 = (30, 120)
DEFENSE_ARROW2 = (80, 200)
DEFENSE_THAAD = (48, 96)            # 1-2 batteries
DEFENSE_PAC3 = (150, 600)
DEFENSE_DS = (50, 300)              # David's Sling Stunner
DEFENSE_IRON_DOME = (500, 5000)     # Tamir; huge range due to 2023-2025 drawdown

# --- Defense: Intercept Probabilities (Pk) ---

PK_ARROW3 = (0.60, 0.95)            # wide: MaRV degrades it heavily
PK_ARROW2 = (0.70, 0.92)
PK_THAAD = (0.80, 0.96)
PK_PAC3_VS_BM = (0.60, 0.92)        # mixed combat data
PK_PAC3_VS_SRBM = (0.80, 0.96)
PK_VS_DRONE = (0.60, 0.98)          # huge range: depends on fighter availability
PK_VS_CRUISE = (0.65, 0.95)

# --- Defense: Engagement Fractions per Layer ---

ENGAGE_FRAC_ARROW3 = (0.20, 0.50)
ENGAGE_FRAC_ARROW2 = (0.25, 0.60)
ENGAGE_FRAC_THAAD = (0.40, 0.75)

# --- Defense: Gulf-Based Interceptors (for SRBM defense) ---

GULF_PAC3 = (96, 384)               # 2-4 Patriot batteries in Gulf region
GULF_THAAD = (48, 96)               # 1-2 THAAD batteries in Gulf

# --- Defense: Resupply Rates (per month) ---

DEFENSE_TAMIR_RESUPPLY = (50, 500)   # peacetime to max wartime surge
DEFENSE_PAC3_RESUPPLY = (20, 80)
DEFENSE_ARROW_RESUPPLY = (0.5, 8)    # Arrow-2/3 combined

# --- US Offensive (count) ---

US_TOMAHAWK = (200, 600)
US_JASSM = (300, 1500)

# --- US: Resupply (per month) ---

US_TOMAHAWK_RESUPPLY = (50, 300)    # VLS reload requires port calls; slow
US_JASSM_RESUPPLY = (50, 300)       # airlift from CONUS, limited by production

# --- US: TEL Hunt ---

US_TEL_STRIKE_INTERVAL = (0.2, 1.0)  # days between strike packages
US_TEL_KILL_FRAC = (0.005, 0.08)     # 1991 was ~0; modern maybe 1-8%
US_TOMAHAWK_PER_TEL_STRIKE = (10, 40)
US_JASSM_PER_TEL_STRIKE = (3, 20)

# --- US: Production Facility Strikes ---

US_PROD_STRIKE_INTERVAL = (0.5, 1)   # days between strikes
US_PROD_STRIKE_EFFECT = (0.05, 0.30) # fraction of remaining capacity per strike
US_TOMAHAWK_PER_PROD_STRIKE = (20, 60)
US_JASSM_PER_PROD_STRIKE = (10, 40)


# ===================================================================
# Helper
# ===================================================================

def lu(low, high):
    """Log-uniform distribution over [low, high]."""
    return stats.loguniform(low, high)


# ===================================================================
# Iran's Arsenal
# ===================================================================

class IranMissiles(StateGroup):
    mrbm: float = 0.0
    srbm: float = 0.0
    cruise: float = 0.0
    drones: float = 0.0
    tels: float = 0.0
    production_capacity: float = 1.0

    @property
    @reported
    def prod_cap(self):
        return self.production_capacity
    
    @property
    @reported
    def n_tels(self):
        return self.tels

    @property
    @reported
    def n_mrbm(self):
        return self.mrbm

    def _init_state(self):
        self.mrbm = self.prior.sample("iran.mrbm_init", lu(*IRAN_MRBM))
        self.srbm = self.prior.sample("iran.srbm_init", lu(*IRAN_SRBM))
        self.cruise = self.prior.sample("iran.cruise_init", lu(*IRAN_CRUISE))
        self.drones = self.prior.sample("iran.drone_init", lu(*IRAN_DRONE))
        self.tels = self.prior.sample("iran.tel_init", lu(*IRAN_TEL))

    @ode
    def production(self):
        mrbm_yr = self.prior.sample("iran.mrbm_prod_yr", lu(*IRAN_MRBM_PROD))
        srbm_yr = self.prior.sample("iran.srbm_prod_yr", lu(*IRAN_SRBM_PROD))
        cruise_yr = self.prior.sample("iran.cruise_prod_yr", lu(*IRAN_CRUISE_PROD))
        drone_yr = self.prior.sample("iran.drone_prod_yr", lu(*IRAN_DRONE_PROD))

        cap = self.production_capacity
        return {
            "mrbm": mrbm_yr / 365 * cap,
            "srbm": srbm_yr / 365 * cap,
            "cruise": cruise_yr / 365 * cap,
            "drones": drone_yr / 365 * cap,
        }


# ===================================================================
# US/Coalition Offensive Capability
# ===================================================================

class USOffense(StateGroup):
    tomahawks: float = 0.0
    jassm: float = 0.0

    @property
    @reported
    def total_munitions(self):
        return self.tomahawks + self.jassm

    def _init_state(self):
        self.tomahawks = self.prior.sample("us.tomahawk_init", lu(*US_TOMAHAWK))
        self.jassm = self.prior.sample("us.jassm_init", lu(*US_JASSM))

    @ode
    def resupply(self):
        tomahawk_mo = self.prior.sample("us.tomahawk_resupply_mo", lu(*US_TOMAHAWK_RESUPPLY))
        jassm_mo = self.prior.sample("us.jassm_resupply_mo", lu(*US_JASSM_RESUPPLY))
        return {
            "tomahawks": tomahawk_mo / 30,
            "jassm": jassm_mo / 30,
        }


# ===================================================================
# Missile Defense Interceptors
# ===================================================================

class Interceptors(StateGroup):
    arrow3: float = 0.0
    arrow2: float = 0.0
    thaad: float = 0.0
    pac3: float = 0.0
    davids_sling: float = 0.0
    iron_dome: float = 0.0

    @property
    @reported
    def total_interceptors(self):
        return self.arrow3 + self.arrow2 + self.thaad + self.pac3 + self.davids_sling + self.iron_dome

    @property
    @reported
    def bm_interceptors(self):
        """Interceptors that can engage ballistic missiles."""
        return self.arrow3 + self.arrow2 + self.thaad + self.pac3

    def _init_state(self):
        self.arrow3 = self.prior.sample("def.arrow3_init", lu(*DEFENSE_ARROW3))
        self.arrow2 = self.prior.sample("def.arrow2_init", lu(*DEFENSE_ARROW2))
        self.thaad = self.prior.sample("def.thaad_init", lu(*DEFENSE_THAAD))
        self.pac3 = self.prior.sample("def.pac3_init", lu(*DEFENSE_PAC3))
        self.davids_sling = self.prior.sample("def.ds_init", lu(*DEFENSE_DS))
        self.iron_dome = self.prior.sample("def.iron_dome_init", lu(*DEFENSE_IRON_DOME))

        tamir_mo = self.prior.sample("def.tamir_resupply_mo", lu(*DEFENSE_TAMIR_RESUPPLY))
        pac3_mo = self.prior.sample("def.pac3_resupply_mo", lu(*DEFENSE_PAC3_RESUPPLY))
        arrow_mo = self.prior.sample("def.arrow_resupply_mo", lu(*DEFENSE_ARROW_RESUPPLY))
        return {
            "iron_dome": tamir_mo / 30,
            "pac3": pac3_mo / 30,
            "arrow2": arrow_mo / 30 * 0.6,
            "arrow3": arrow_mo / 30 * 0.4,
        }


# ===================================================================
# Gulf-Based Missile Defense (US bases in Gulf)
# ===================================================================

class GulfDefense(StateGroup):
    pac3: float = 0.0
    thaad: float = 0.0

    def _init_state(self):
        self.pac3 = self.prior.sample("gulf.pac3_init", lu(*GULF_PAC3))
        self.thaad = self.prior.sample("gulf.thaad_init", lu(*GULF_THAAD))

    @ode
    def resupply(self):
        pac3_mo = self.prior.sample("gulf.pac3_resupply_mo", lu(*DEFENSE_PAC3_RESUPPLY))
        return {"pac3": pac3_mo / 30}


# ===================================================================
# Damage Tracking
# ===================================================================

class Damage(StateGroup):
    missiles_intercepted: float = 0.0
    missiles_leaked: float = 0.0

    @property
    @reported
    def intercept_rate(self):
        total = self.missiles_intercepted + self.missiles_leaked
        return self.missiles_intercepted / total if total > 0 else 0.0



# ===================================================================
# Conflict (root model)
# ===================================================================

class Conflict(StateGroup):
    iran: IranMissiles
    us: USOffense
    defense: Interceptors
    gulf: GulfDefense
    damage: Damage

    outcome: float = 0.0  # +1 = US wins, -1 = Iran wins, 0 = undecided

    @property
    @reported
    def winner(self):
        return self.outcome

    @transition
    def us_victory(self):
        """US wins when Iran's production is crippled and MRBM+SRBM stock near zero."""
        if self.outcome != 0:
            return 0.0, lambda: None
        if (self.iran.production_capacity < 0.01 and self.iran.mrbm < 10 and self.iran.srbm < 10) or self.iran.n_tels < 5:
            return 1000.0, lambda: setattr(self, 'outcome', 1.0)
        return 0.0, lambda: None

    @transition
    def iran_victory(self):
        """Iran wins when coalition BM interceptors are essentially exhausted."""
        if self.outcome != 0:
            return 0.0, lambda: None
        if self.defense.bm_interceptors < 10:
            return 1000.0, lambda: setattr(self, 'outcome', -1.0)
        return 0.0, lambda: None

    # --- Iran launches MRBM salvo at Israel ---
    @transition
    def iran_ballistic_salvo(self):
        available = min(self.iran.mrbm, self.iran.tels * IRAN_MISSILES_PER_TEL)
        if available < IRAN_MRBM_MIN_SALVO:
            return 0.0, lambda: None

        interval = self.prior.sample("iran.mrbm_salvo_interval", lu(*IRAN_MRBM_SALVO_INTERVAL))
        rate = 1.0 / interval

        def effect():
            frac = self.prior.sample("iran.mrbm_salvo_frac", lu(*IRAN_MRBM_SALVO_FRAC))
            salvo = max(IRAN_MRBM_MIN_SALVO, self.iran.mrbm * frac)
            salvo = min(salvo, self.iran.mrbm, self.iran.tels * IRAN_MISSILES_PER_TEL)
            self.iran.mrbm = max(0, self.iran.mrbm - salvo)

            remaining = salvo

            # Arrow-3 (exo-atmospheric, first shot)
            if self.defense.arrow3 > 0:
                f = self.prior.sample("def.engage_frac_arrow3", lu(*ENGAGE_FRAC_ARROW3))
                engaged = min(remaining * f, self.defense.arrow3)
                pk = self.prior.sample("def.pk_arrow3", lu(*PK_ARROW3))
                self.defense.arrow3 = max(0, self.defense.arrow3 - engaged)
                remaining -= engaged * pk

            # Arrow-2 (endo-atmospheric)
            if self.defense.arrow2 > 0:
                f = self.prior.sample("def.engage_frac_arrow2", lu(*ENGAGE_FRAC_ARROW2))
                engaged = min(remaining * f, self.defense.arrow2)
                pk = self.prior.sample("def.pk_arrow2", lu(*PK_ARROW2))
                self.defense.arrow2 = max(0, self.defense.arrow2 - engaged)
                remaining -= engaged * pk

            # THAAD
            if self.defense.thaad > 0:
                f = self.prior.sample("def.engage_frac_thaad", lu(*ENGAGE_FRAC_THAAD))
                engaged = min(remaining * f, self.defense.thaad)
                pk = self.prior.sample("def.pk_thaad", lu(*PK_THAAD))
                self.defense.thaad = max(0, self.defense.thaad - engaged)
                remaining -= engaged * pk

            # PAC-3 (terminal, shoot-shoot = 2 interceptors per target)
            if self.defense.pac3 > 0:
                engaged_targets = min(remaining, self.defense.pac3 / 2)
                pk = self.prior.sample("def.pk_pac3_bm", lu(*PK_PAC3_VS_BM))
                self.defense.pac3 = max(0, self.defense.pac3 - engaged_targets * 2)
                remaining -= engaged_targets * pk

            leaked = max(0, remaining)
            self.damage.missiles_intercepted += salvo - leaked
            self.damage.missiles_leaked += leaked

        return rate, effect

    # --- Iran launches drones + cruise missiles (daily) ---
    @transition
    def iran_drone_cruise(self):
        if self.iran.drones < 10 and self.iran.cruise < 2:
            return 0.0, lambda: None

        rate = 1.0  # once per day

        def effect():
            n_drones = min(self.iran.drones,
                self.prior.sample("iran.drones_per_day", lu(*IRAN_DRONES_PER_DAY)))
            n_cruise = min(self.iran.cruise,
                self.prior.sample("iran.cruise_per_day", lu(*IRAN_CRUISE_PER_DAY)))

            self.iran.drones = max(0, self.iran.drones - n_drones)
            self.iran.cruise = max(0, self.iran.cruise - n_cruise)

            # Drones: Iron Dome + fighter aircraft
            pk_drone = self.prior.sample("def.pk_vs_drone", lu(*PK_VS_DRONE))
            drone_killed = n_drones * pk_drone
            drone_leaked = n_drones - drone_killed
            f_iron_dome = self.prior.sample("def.frac_drone_iron_dome", lu(*FRAC_DRONE_IRON_DOME))
            iron_dome_used = min(drone_killed * f_iron_dome, self.defense.iron_dome)
            self.defense.iron_dome = max(0, self.defense.iron_dome - iron_dome_used)

            # Cruise missiles: David's Sling + PAC-3
            pk_cruise = self.prior.sample("def.pk_vs_cruise", lu(*PK_VS_CRUISE))
            cruise_killed = n_cruise * pk_cruise
            cruise_leaked = n_cruise - cruise_killed
            f_ds = self.prior.sample("def.frac_cruise_ds", lu(*FRAC_CRUISE_DS))
            f_pac3 = self.prior.sample("def.frac_cruise_pac3", lu(*FRAC_CRUISE_PAC3))
            ds_used = min(cruise_killed * f_ds, self.defense.davids_sling)
            pac3_used = min(cruise_killed * f_pac3, self.defense.pac3)
            self.defense.davids_sling = max(0, self.defense.davids_sling - ds_used)
            self.defense.pac3 = max(0, self.defense.pac3 - pac3_used)

            total_leaked = max(0, drone_leaked) + max(0, cruise_leaked)
            self.damage.missiles_intercepted += (n_drones + n_cruise) - total_leaked
            self.damage.missiles_leaked += total_leaked

        return rate, effect

    # --- Iran launches SRBMs at US Gulf bases ---
    @transition
    def iran_srbm_salvo(self):
        if self.iran.srbm < 10:
            return 0.0, lambda: None

        interval = self.prior.sample("iran.srbm_salvo_interval", lu(*IRAN_SRBM_SALVO_INTERVAL))
        rate = 1.0 / interval

        def effect():
            salvo = min(self.iran.srbm,
                self.prior.sample("iran.srbm_per_salvo", lu(*IRAN_SRBM_PER_SALVO)))
            self.iran.srbm = max(0, self.iran.srbm - salvo)

            remaining = salvo

            # Gulf THAAD (upper tier)
            if self.gulf.thaad > 0:
                f = self.prior.sample("def.engage_frac_thaad", lu(*ENGAGE_FRAC_THAAD))
                engaged = min(remaining * f, self.gulf.thaad)
                pk = self.prior.sample("def.pk_thaad", lu(*PK_THAAD))
                self.gulf.thaad = max(0, self.gulf.thaad - engaged)
                remaining -= engaged * pk

            # Gulf PAC-3 (terminal, shoot-shoot)
            if self.gulf.pac3 > 0:
                engaged_targets = min(remaining, self.gulf.pac3 / 2)
                pk = self.prior.sample("def.pk_pac3_srbm", lu(*PK_PAC3_VS_SRBM))
                self.gulf.pac3 = max(0, self.gulf.pac3 - engaged_targets * 2)
                remaining -= engaged_targets * pk

            leaked = max(0, remaining)
            self.damage.missiles_intercepted += salvo - leaked
            self.damage.missiles_leaked += leaked

        return rate, effect

    # --- US strikes Iranian TELs ---
    @transition
    def us_strikes_tels(self):
        if self.iran.tels < 1:
            return 0.0, lambda: None

        interval = self.prior.sample("us.tel_strike_interval", lu(*US_TEL_STRIKE_INTERVAL))
        rate = 1.0 / interval

        def effect():
            if self.us.tomahawks < 1 or self.us.jassm < 1:
                return

            kill_frac = self.prior.sample("us.tel_kill_frac", lu(*US_TEL_KILL_FRAC))
            destroyed = self.iran.tels * kill_frac
            self.iran.tels = max(0, self.iran.tels - destroyed)

            t_used = min(self.prior.sample("us.tomahawk_per_tel_strike",
                lu(*US_TOMAHAWK_PER_TEL_STRIKE)), self.us.tomahawks)
            j_used = min(self.prior.sample("us.jassm_per_tel_strike",
                lu(*US_JASSM_PER_TEL_STRIKE)), self.us.jassm)
            self.us.tomahawks = max(0, self.us.tomahawks - t_used)
            self.us.jassm = max(0, self.us.jassm - j_used)

        return rate, effect

    # --- US strikes missile production facilities ---
    @transition
    def us_strikes_production(self):
        if self.iran.production_capacity < 0.001:
            return 0.0, lambda: None

        interval = self.prior.sample("us.prod_strike_interval", lu(*US_PROD_STRIKE_INTERVAL))
        rate = 1.0 / interval

        def effect():
            # Multiplicative: destroy a fraction of remaining capacity
            frac_destroyed = self.prior.sample("us.prod_strike_effect", lu(*US_PROD_STRIKE_EFFECT))
            self.iran.production_capacity *= (1 - frac_destroyed)

            # Munitions scale with remaining capacity — less to target = fewer munitions
            cap = self.iran.production_capacity
            t_base = self.prior.sample("us.tomahawk_per_prod_strike",
                lu(*US_TOMAHAWK_PER_PROD_STRIKE))
            j_base = self.prior.sample("us.jassm_per_prod_strike",
                lu(*US_JASSM_PER_PROD_STRIKE))
            t_used = min(t_base * cap, self.us.tomahawks)
            j_used = min(j_base * cap, self.us.jassm)
            self.us.tomahawks = max(0, self.us.tomahawks - t_used)
            self.us.jassm = max(0, self.us.jassm - j_used)

        return rate, effect


# ===================================================================
# Main — simulate and append to runs.jsonl
# ===================================================================

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t_end = 300.0  # 300 days
    output_file = "runs.jsonl"
    total = 0

    while True:
        t0 = time.perf_counter()
        results = simulate(Conflict(), t_end=t_end, n_runs=workers, dt=1.0,
                           record_every=1.0, rng=None, workers=workers)
        elapsed = time.perf_counter() - t0
        total += workers

        now = time.time()
        with open(output_file, "a") as f:
            for traj in results:
                record = {
                    "name": "example.py",
                    "time": now,
                    "seed": traj.seed,
                    "prior": traj.prior_values,
                    "reported": traj.reported,
                    "reported_series": [(t, d) for t, d in traj.reported_series],
                    "final_values": traj.snapshots[-1][1],
                    "event_counts": {k: len(v) for k, v in traj.event_log.items()},
                    "event_log": traj.event_log,
                }
                f.write(json.dumps(record) + "\n")

        print(f"[{total} total] +{workers} in {elapsed:.1f}s ({workers / elapsed:.1f} runs/s)")
