"""Iran conflict simulation — March 2026 scenario.

Models a multi-month US-Iran conflict with:
- Iranian ballistic missile, cruise missile, and drone salvos
- US/coalition strikes on TELs and production facilities
- Multi-layered missile defense (Arrow-3, Arrow-2, THAAD, PAC-3, David's Sling, Iron Dome)
- Production/replenishment on both sides
- Proxy fronts: Hezbollah (Lebanon), Houthis (Yemen), Iraqi militias
- Proxy activation/degradation dynamics

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
# Calibrated against: github.com/danielrosehill/Iran-Israel-War-2026-OSINT-Data
# ===================================================================

# --- Iran: Initial Arsenal (count) ---
# High end includes possibility of undisclosed stockpiles / surge production.

# OSINT: ~2,410 BMs fired in 13 days (TP4), conflict ongoing — must start above that
IRAN_MRBM = (2000, 4000)
# OSINT: SRBMs rarely identified in TP4 wave data; MRBMs dominate all fronts
IRAN_SRBM = (500, 3000)
# OSINT: only 4/42 TP4 waves used cruise missiles — clearly secondary weapon
IRAN_CRUISE = (50, 300)
# OSINT: ~3,560 drones in 13 days; 50k was speculative upper bound
IRAN_DRONE = (3000, 20000)
IRAN_TEL = (80, 200)                # all types

# --- Iran: Production Rates (per year) ---

IRAN_MRBM_PROD = (30, 150)          # CSIS baseline + wartime surge
IRAN_SRBM_PROD = (80, 400)          # high-volume solid-fuel
IRAN_CRUISE_PROD = (30, 150)        # constrained by turbojet supply
IRAN_DRONE_PROD = (2000, 10000)     # IISS baseline + expanded facilities

# --- Iran: Salvo Parameters (natural units) ---

# OSINT: 40 BM waves in 13 days = 3/day → 0.33 day interval (fixed)
IRAN_MRBM_SALVO_INTERVAL = 0.33
# OSINT: ~60 BMs/wave from ~3k stock = 2% per salvo (fixed)
IRAN_MRBM_SALVO_FRAC = 0.02
# OSINT: Wave 12 had only 5 BMs; many small salvos in data
IRAN_MRBM_MIN_SALVO = 5
IRAN_MISSILES_PER_TEL = 2           # per salvo window (fixed)

# OSINT: Gulf states hit almost daily with SRBMs
IRAN_SRBM_SALVO_INTERVAL = (0.5, 2)
# OSINT: Qatar Day 12: 9 BMs, UAE Day 11: 8 BMs, Saudi Day 12: 7 BMs
IRAN_SRBM_PER_SALVO = (5, 15)

# OSINT: ~274 drones/day avg; Wave 16 alone: 230 drones; low end of 10 impossible
IRAN_DRONES_PER_DAY = (100, 500)
# OSINT: only 4/42 waves used cruise missiles; negligible vs BMs + drones
IRAN_CRUISE_PER_DAY = (0.5, 10)

# --- Defense: Drone/Cruise Intercept Allocation ---

FRAC_DRONE_IRON_DOME = (0.30, 0.70)  # fraction of drone kills by Iron Dome (rest by fighters)
FRAC_CRUISE_DS = (0.40, 0.80)        # fraction of cruise kills by David's Sling
FRAC_CRUISE_PAC3 = (0.10, 0.40)      # fraction of cruise kills by PAC-3

# --- Defense: Interceptor Inventories (count) ---

# Sources: CSIS depleting inventory report, FPRI "Shallow Ramparts" analysis,
# CNN/WSJ reporting on TP3 interceptor usage, battery counts × interceptors/launcher.
# TP3 (Jun 2025) used ~131 Arrow-3, 100-150 THAAD, ~80 SM-3; Israel replenishing since.
DEFENSE_ARROW3 = (80, 200)          # replenished post-TP3 but not fully; accelerated IAI production
DEFENSE_ARROW2 = (50, 150)          # old stock, no longer in production; depleting
DEFENSE_THAAD = (96, 200)           # 2 batteries (96 on launchers) + theater reload stock
DEFENSE_PAC3 = (200, 500)           # US-provided; 600+ MSE delivered in 2025; multiple batteries
DEFENSE_DS = (50, 150)              # only 2 batteries; 6-12 Stunners per TEL; $1M each
DEFENSE_IRON_DOME = (2000, 5000)    # 10 batteries + massive US-funded resupply ($5.2B for AD)
# OSINT: Aegis SM-3 used in 5/42 TP4 waves; 414 in MDA arsenal Dec 2025 minus ~80 used in TP3
DEFENSE_SM3 = (30, 80)              # 2-3 Aegis destroyers in theater × 10-15 SM-3 each

# --- Defense: Intercept Probabilities (Pk) ---

# OSINT: still intercepting MaRV missiles Day 13; Qatar "all intercepted" for BMs
PK_ARROW3 = (0.85, 0.95)
PK_ARROW2 = (0.70, 0.92)
PK_THAAD = (0.80, 0.96)
PK_PAC3_VS_BM = (0.60, 0.92)        # mixed combat data
PK_PAC3_VS_SRBM = (0.80, 0.96)
# OSINT: SM-3 exoatmospheric intercepts confirmed TP4 Wave 1; similar Pk to Arrow-3
PK_SM3 = (0.70, 0.95)
ENGAGE_FRAC_SM3 = (0.10, 0.30)      # limited ships, can't engage everything
# OSINT: UAE "mostly intercepted" ~1,200 drones; Day 14 reports 95% drone attrition
PK_VS_DRONE = (0.80, 0.96)
PK_VS_CRUISE = (0.65, 0.95)
PK_CRAM = (0.30, 0.70)              # C-RAM vs rockets/drones; overwhelmed by swarms

# --- Defense: Targeting Split & Threat Filtering ---
# OSINT: Israel targeted in 37/42 waves, but Gulf states also heavily targeted.
# Many MRBMs go to Gulf/other countries, not just Israel.
MRBM_ISRAEL_FRAC = (0.40, 0.70)     # fraction of each MRBM salvo aimed at Israel (rest at Gulf etc.)
# Israel's BMD tracks each warhead's predicted impact point and only engages
# those heading toward populated/critical zones. Missiles heading for open ground
# (Negev desert, missed targets due to CEP/GPS jamming) are ignored.
# Core Iron Dome design principle, applied across all layers.
BM_THREAT_FRAC = (0.30, 0.60)       # of Israel-bound BMs, fraction warranting interception
SRBM_THREAT_FRAC = (0.40, 0.70)     # SRBMs at Gulf — bases are denser targets, higher frac

# --- Defense: Engagement Fractions per Layer ---

ENGAGE_FRAC_ARROW3 = (0.20, 0.50)
ENGAGE_FRAC_ARROW2 = (0.25, 0.60)
ENGAGE_FRAC_THAAD = (0.40, 0.75)

# --- Defense: Gulf-Based Interceptors (for SRBM defense) ---

GULF_PAC3 = (96, 384)               # 2-4 Patriot batteries in Gulf region
GULF_THAAD = (48, 96)               # 1-2 THAAD batteries in Gulf

# --- Defense: Global Production (per year) ---
# These are total annual production, NOT what's available to theater.
# Theater allocation is a fraction of this.

GLOBAL_TAMIR_PROD = (2000, 3000)     # Rafael Israel + RTX Camden (2026 ramp)
GLOBAL_PAC3_PROD = (500, 650)        # Lockheed MSE line; 620 in 2025, 650 target
GLOBAL_THAAD_PROD = (12, 96)         # 12/yr actual delivery (2024); 96/yr funded capacity
GLOBAL_ARROW_PROD = (40, 100)        # Arrow-2+3 combined; IAI tripled post-war
GLOBAL_DS_PROD = (100, 400)          # David's Sling Stunner; Rafael scaling up

# --- Defense: Theater Allocation Fractions ---
# What fraction of global production goes to this theater (Middle East).

THEATER_TAMIR_SHARE = (0.85, 0.95)   # nearly all Tamir is for Israel
THEATER_PAC3_SHARE = (0.15, 0.35)    # rest goes to NATO, Japan, Korea, etc.
THEATER_THAAD_SHARE = (0.15, 0.40)   # 2 of 8 batteries deployed to Israel
THEATER_DS_SHARE = (0.80, 0.95)      # primarily Israeli system
GULF_PAC3_SHARE = (0.20, 0.50)       # Gulf's share of theater PAC-3 allocation

# --- US Offensive (count) ---

# OSINT: ~2,000 targets struck with stand-off weapons by Day 5; deep stocks evident
US_TOMAHAWK = (400, 800)
US_JASSM = (500, 1500)

# --- US: Resupply (per month) ---

US_TOMAHAWK_RESUPPLY = (50, 300)    # VLS reload requires port calls; slow
US_JASSM_RESUPPLY = (50, 300)       # airlift from CONUS, limited by production

# --- US: TEL Hunt ---

# OSINT: "surface launchers ~60% destroyed by Day 7" → (1-f)^7=0.4 → f≈0.12
US_TEL_KILL_FRAC = (0.10, 0.15)
US_TOMAHAWK_PER_TEL_STRIKE = (10, 40)
US_JASSM_PER_TEL_STRIKE = (3, 20)

# --- US: Production Facility Strikes ---

US_PROD_STRIKE_INTERVAL = (0.5, 1)   # days between strikes
# OSINT: 86% degradation by Day 5; (1-f)^7=0.14 → f≈0.25; low end 0.05 too weak
US_PROD_STRIKE_EFFECT = (0.15, 0.30)
US_TOMAHAWK_PER_PROD_STRIKE = (20, 60)
US_JASSM_PER_PROD_STRIKE = (10, 40)

# --- US: Air Superiority (SEAD completion) ---
# OSINT: Day 5-7 US shifts from Tomahawk/JASSM to gravity bombs (JDAM/BLU-109)
# after Iranian air defenses destroyed. Once prod_capacity below this threshold,
# strikes are "free" (gravity bombs from aircraft, no standoff munition cost).
SEAD_THRESHOLD = (0.20, 0.40)       # prod_capacity below this → air superiority

# --- Casualties ---
# OSINT: ~16 Israeli civilians killed by Day 11 from BM impacts + cluster warheads
CASUALTIES_PER_LEAKED_BM = (0.5, 3.0)       # Israeli casualties per BM that gets through
CASUALTIES_PER_CLUSTER_INTERCEPT = (0.1, 0.5) # cluster submunitions even on "intercept"
CLUSTER_WARHEAD_FRAC = (0.10, 0.30)          # fraction of BMs with cluster warheads (Day 7+)
CLUSTER_START_DAY = 7                        # OSINT: cluster warheads first deployed Day 7
DRONE_LETHALITY_VS_BM = 0.2                  # drone casualty rate relative to BM
CRUISE_LETHALITY_VS_BM = 0.5                 # cruise missile casualty rate relative to BM
ARROW3_PRODUCTION_SHARE = 0.4                # ~40% of Arrow production is Arrow-3, 60% Arrow-2
# OSINT: Gulf damage per leaked missile, in $M USD (best-effort estimate)
# Anchors: BAPCO refinery $200-500M, ADNOC Ruwais $500M+, 5x KC-135 $200M,
# Dubai/Kuwait airports $50-100M/day, residential $5-50M.
# Most leaked BMs/drones hit a mix of military, energy, civilian.
GULF_DAMAGE_PER_LEAKED_BM = (50, 500)     # $M per leaked BM — high variance, energy infra skews up
GULF_DAMAGE_PER_LEAKED_DRONE = (5, 50)    # $M per leaked drone — lower warhead, less structural


# ===================================================================
# Hezbollah (post-2024 war residual stockpile)
#
# Pre-war: ~150k total. IDF destroyed 70-80%. Syria resupply corridor
# severed after Assad fell Dec 2024. Resupply near zero.
# ===================================================================

HEZB_SR_ROCKETS = (5000, 15000)       # short-range (Katyusha/Grad/Fajr-3); <10k est. remaining
HEZB_MR_ROCKETS = (300, 1500)         # medium-range (Fajr-5/Khaibar/Zelzal); <1k est. remaining
HEZB_PGM = (20, 200)                  # precision-guided (Fateh w/ guidance kits); <100 est.
HEZB_DRONES = (200, 1000)             # Ababil/Shahed remnants

# OSINT: Day 13 barrage of 150 rockets was described as major; 500 exceeds depleted capacity
HEZB_SR_PER_DAY = (50, 200)
HEZB_MR_SALVO_INTERVAL = (1, 5)       # days between MR salvos
HEZB_MR_PER_SALVO = (5, 30)
# OSINT: two PGM strikes in 13 days (Dado Base Day 5, Ramle Day 10) ≈ every 5 days
HEZB_PGM_INTERVAL = (3, 7)
HEZB_PGM_PER_STRIKE = (1, 5)

HEZB_LAUNCHERS = (100, 500)           # MRL vehicles + concealed launch sites
HEZB_ROCKETS_PER_LAUNCHER = 10        # effective rockets per launcher per day

# OSINT: Hezbollah active Day 3 (March 2) = exactly 2 days post-conflict start
HEZB_ACTIVATION_DELAY = 2.0
# OSINT: Day 5 "80 Hezb targets in 24hrs", ground invasion, drone factory destroyed;
# Day 14: 773 killed, 759k displaced — IDF hit extremely hard
HEZB_LAUNCHER_KILL_FRAC = (0.05, 0.12)

# ===================================================================
# Houthis (Ansar Allah, Yemen)
#
# Active since Oct 2023 Red Sea campaign. Iranian-supplied.
# Can reach Israel (~2000km BMs), Gulf bases, Red Sea shipping.
# Preliminary estimates — will refine with research data.
# ===================================================================

HOUTHI_BM = (100, 400)                # ~400 total incl ~120 Palestine-2 (UN 2025)
HOUTHI_CRUISE = (100, 200)            # Quds variants; limited data
HOUTHI_DRONES = (1000, 5000)          # Samad/Waid; large stockpile + domestic production

HOUTHI_BM_INTERVAL = (3, 14)          # days between BM salvos at Israel/Gulf
HOUTHI_BM_PER_SALVO = (1, 5)
HOUTHI_DRONE_PER_DAY = (1, 10)        # demonstrated ~40/month ≈ 1.3/day sustained

HOUTHI_LAUNCH_SITES = (10, 30)        # 14 mobile TELs confirmed; dispersed across 200km²
# OSINT: zero confirmed Houthi attacks through Day 15; deterred or conserving
HOUTHI_ACTIVATION_DELAY = (10, 30)
HOUTHI_SITE_KILL_FRAC = (0.005, 0.03) # hard to degrade (dispersed, mountainous terrain)

HOUTHI_DRONE_PROD_PER_DAY = (1, 5)    # domestic assembly with Iranian components

# ===================================================================
# Iraqi Militias (PMF/Kata'ib Hezbollah)
#
# Short-range rockets and one-way attack drones targeting US bases
# in Iraq/Syria/Kuwait. Direct overland resupply from Iran.
# ===================================================================

IRAQ_ROCKETS = (500, 2000)            # 107mm/122mm/Fajr + OWA drones
IRAQ_CELLS = (20, 80)                # active militia launch cells
# OSINT: 6 attacks in 13 days (Day 2,7,10,12,13,15) ≈ 0.46/day; not daily
IRAQ_PER_DAY = (0.3, 2)
# OSINT: Erbil hotel drone strike on Day 2 — clear and specific
IRAQ_ACTIVATION_DELAY = 2.0
IRAQ_CELL_KILL_FRAC = (0.01, 0.05)   # US strikes on militia positions
IRAQ_RESUPPLY_PER_DAY = (2, 20)       # overland from Iran; hard to interdict


# ===================================================================
# Helper
# ===================================================================

_NOOP = lambda: None
_lu_cache = {}

def lu(low, high):
    """Log-uniform distribution over [low, high]."""
    key = (low, high)
    if key not in _lu_cache:
        _lu_cache[key] = stats.loguniform(low, high)
    return _lu_cache[key]


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
    sm3: float = 0.0                # Aegis BMD / SM-3 (sea-based)

    @property
    @reported
    def total_interceptors(self):
        return self.arrow3 + self.arrow2 + self.thaad + self.pac3 + self.davids_sling + self.iron_dome + self.sm3

    @property
    @reported
    def bm_interceptors(self):
        """Interceptors that can engage ballistic missiles."""
        return self.arrow3 + self.arrow2 + self.thaad + self.pac3 + self.sm3

    def _init_state(self):
        self.arrow3 = self.prior.sample("def.arrow3_init", lu(*DEFENSE_ARROW3))
        self.arrow2 = self.prior.sample("def.arrow2_init", lu(*DEFENSE_ARROW2))
        self.thaad = self.prior.sample("def.thaad_init", lu(*DEFENSE_THAAD))
        self.pac3 = self.prior.sample("def.pac3_init", lu(*DEFENSE_PAC3))
        self.davids_sling = self.prior.sample("def.ds_init", lu(*DEFENSE_DS))
        self.iron_dome = self.prior.sample("def.iron_dome_init", lu(*DEFENSE_IRON_DOME))
        self.sm3 = self.prior.sample("def.sm3_init", lu(*DEFENSE_SM3))

    @ode
    def resupply(self):
        # Each system: global_prod * theater_share / 365 = daily resupply
        tamir_yr = self.prior.sample("global.tamir_prod_yr", lu(*GLOBAL_TAMIR_PROD))
        tamir_share = self.prior.sample("global.tamir_theater_share", lu(*THEATER_TAMIR_SHARE))

        pac3_yr = self.prior.sample("global.pac3_prod_yr", lu(*GLOBAL_PAC3_PROD))
        pac3_share = self.prior.sample("global.pac3_theater_share", lu(*THEATER_PAC3_SHARE))
        gulf_share = self.prior.sample("global.gulf_pac3_share", lu(*GULF_PAC3_SHARE))

        thaad_yr = self.prior.sample("global.thaad_prod_yr", lu(*GLOBAL_THAAD_PROD))
        thaad_share = self.prior.sample("global.thaad_theater_share", lu(*THEATER_THAAD_SHARE))

        arrow_yr = self.prior.sample("global.arrow_prod_yr", lu(*GLOBAL_ARROW_PROD))

        ds_yr = self.prior.sample("global.ds_prod_yr", lu(*GLOBAL_DS_PROD))
        ds_share = self.prior.sample("global.ds_theater_share", lu(*THEATER_DS_SHARE))

        return {
            "iron_dome": tamir_yr * tamir_share / 365,
            "pac3": pac3_yr * pac3_share * (1 - gulf_share) / 365,
            "thaad": thaad_yr * thaad_share / 365,
            "davids_sling": ds_yr * ds_share / 365,
            "arrow2": arrow_yr / 365 * (1 - ARROW3_PRODUCTION_SHARE),
            "arrow3": arrow_yr / 365 * ARROW3_PRODUCTION_SHARE,
        }


# ===================================================================
# Gulf-Based Missile Defense (US bases in Gulf)
# ===================================================================

class GulfDefense(StateGroup):
    pac3: float = 0.0
    thaad: float = 0.0

    @property
    @reported
    def total_interceptors(self):
        return self.pac3 + self.thaad

    def _init_state(self):
        self.pac3 = self.prior.sample("gulf.pac3_init", lu(*GULF_PAC3))
        self.thaad = self.prior.sample("gulf.thaad_init", lu(*GULF_THAAD))

    @ode
    def resupply(self):
        # Gulf gets gulf_share of the theater PAC-3 allocation
        pac3_yr = self.prior.sample("global.pac3_prod_yr", lu(*GLOBAL_PAC3_PROD))
        pac3_share = self.prior.sample("global.pac3_theater_share", lu(*THEATER_PAC3_SHARE))
        gulf_share = self.prior.sample("global.gulf_pac3_share", lu(*GULF_PAC3_SHARE))
        return {"pac3": pac3_yr * pac3_share * gulf_share / 365}


# ===================================================================
# Hezbollah (northern front)
# ===================================================================

class Hezbollah(StateGroup):
    active: float = 0.0           # 0 = inactive, 1 = active
    launchers: float = 0.0        # MRL vehicles + concealed launch sites
    rockets_sr: float = 0.0
    rockets_mr: float = 0.0
    pgm: float = 0.0
    drones: float = 0.0

    @property
    @reported
    def hezb_launchers(self):
        return self.launchers

    @property
    @reported
    def hezb_sr(self):
        return self.rockets_sr

    def _init_state(self):
        self.launchers = self.prior.sample("hezb.launchers_init", lu(*HEZB_LAUNCHERS))
        self.rockets_sr = self.prior.sample("hezb.sr_init", lu(*HEZB_SR_ROCKETS))
        self.rockets_mr = self.prior.sample("hezb.mr_init", lu(*HEZB_MR_ROCKETS))
        self.pgm = self.prior.sample("hezb.pgm_init", lu(*HEZB_PGM))
        self.drones = self.prior.sample("hezb.drone_init", lu(*HEZB_DRONES))


# ===================================================================
# Houthis (southern front)
# ===================================================================

class Houthis(StateGroup):
    active: float = 0.0
    launch_sites: float = 0.0    # dispersed TELs in mountains
    bm: float = 0.0
    cruise: float = 0.0
    drones: float = 0.0

    @property
    @reported
    def houthi_sites(self):
        return self.launch_sites

    @property
    @reported
    def houthi_drones(self):
        return self.drones

    def _init_state(self):
        self.launch_sites = self.prior.sample("houthi.sites_init", lu(*HOUTHI_LAUNCH_SITES))
        self.bm = self.prior.sample("houthi.bm_init", lu(*HOUTHI_BM))
        self.cruise = self.prior.sample("houthi.cruise_init", lu(*HOUTHI_CRUISE))
        self.drones = self.prior.sample("houthi.drone_init", lu(*HOUTHI_DRONES))

    @ode
    def drone_production(self):
        if self.active < 0.5 or self.launch_sites < 1:
            return {}
        prod = self.prior.sample("houthi.drone_prod_day", lu(*HOUTHI_DRONE_PROD_PER_DAY))
        return {"drones": prod}


# ===================================================================
# Iraqi Militias
# ===================================================================

class IraqiMilitias(StateGroup):
    active: float = 0.0
    cells: float = 0.0           # active militia launch cells
    rockets: float = 0.0

    @property
    @reported
    def iraq_cells(self):
        return self.cells

    def _init_state(self):
        self.cells = self.prior.sample("iraq.cells_init", lu(*IRAQ_CELLS))
        self.rockets = self.prior.sample("iraq.rockets_init", lu(*IRAQ_ROCKETS))

    @ode
    def resupply(self):
        """Overland from Iran — hard to interdict."""
        if self.active < 0.5:
            return {}
        rate = self.prior.sample("iraq.resupply_day", lu(*IRAQ_RESUPPLY_PER_DAY))
        return {"rockets": rate}


# ===================================================================
# Damage Tracking
# ===================================================================

class Damage(StateGroup):
    missiles_intercepted: float = 0.0
    missiles_leaked: float = 0.0
    israeli_casualties: float = 0.0   # civilian casualties from BM impacts + cluster
    gulf_damage: float = 0.0          # $M USD cumulative Gulf infrastructure/economic damage

    @property
    @reported
    def intercept_rate(self):
        total = self.missiles_intercepted + self.missiles_leaked
        return self.missiles_intercepted / total if total > 0 else 0.0

    @property
    @reported
    def il_casualties(self):
        return self.israeli_casualties

    @property
    @reported
    def gulf_dmg(self):
        return self.gulf_damage



# ===================================================================
# Conflict (root model)
# ===================================================================

class Conflict(StateGroup):
    iran: IranMissiles
    us: USOffense
    defense: Interceptors
    gulf: GulfDefense
    hezb: Hezbollah
    houthi: Houthis
    iraq: IraqiMilitias
    damage: Damage

    t: float = 0.0        # simulation clock (days)
    outcome: float = 0.0  # +1 = US wins, -1 = Iran wins, 0 = undecided

    @property
    @reported
    def winner(self):
        return self.outcome

    @transition
    def us_victory(self):
        """US wins when Iran's production is crippled and MRBM+SRBM stock near zero."""
        if self.outcome != 0:
            return 0.0, _NOOP
        if (self.iran.production_capacity < 0.01 and self.iran.mrbm < 10 and self.iran.srbm < 10) or self.iran.n_tels < 5:
            return 1000.0, lambda: setattr(self, 'outcome', 1.0)
        return 0.0, _NOOP

    @transition
    def iran_victory(self):
        """Iran wins when coalition BM interceptors are essentially exhausted."""
        if self.outcome != 0:
            return 0.0, _NOOP
        if self.defense.bm_interceptors < 10:
            return 1000.0, lambda: setattr(self, 'outcome', -1.0)
        return 0.0, _NOOP

    # ---------------------------------------------------------------
    # Clock + deterministic proxy activation
    # OSINT: Hezbollah Day 3, Iraq Day 2 — fixed times, not stochastic
    # ---------------------------------------------------------------

    @ode
    def clock(self):
        if self.t >= HEZB_ACTIVATION_DELAY and self.hezb.active < 0.5:
            self.hezb.active = 1.0
        if self.t >= IRAQ_ACTIVATION_DELAY and self.iraq.active < 0.5:
            self.iraq.active = 1.0
        return {"t": 1.0}

    @transition
    def activate_houthis(self):
        if self.houthi.active > 0.5:
            return 0.0, _NOOP
        delay = self.prior.sample("houthi.activation_delay", lu(*HOUTHI_ACTIVATION_DELAY))
        return 1.0 / delay, lambda: setattr(self.houthi, 'active', 1.0)

    # ---------------------------------------------------------------
    # Proxy degradation — Israel/US strikes reduce capacity
    # ---------------------------------------------------------------

    @transition
    def iaf_strikes_hezbollah(self):
        """IAF destroys Hezbollah launchers (no munition cost — aircraft + PGMs)."""
        if self.hezb.active < 0.5 or self.hezb.launchers < 1:
            return 0.0, _NOOP
        rate = 1.0  # daily strikes
        def effect():
            kill_frac = self.prior.sample("hezb.launcher_kill_frac", lu(*HEZB_LAUNCHER_KILL_FRAC))
            destroyed = self.hezb.launchers * kill_frac
            self.hezb.launchers = max(0, self.hezb.launchers - destroyed)
        return rate, effect

    @transition
    def us_strikes_houthis(self):
        """CENTCOM strikes on Houthi launch sites — costs Tomahawks."""
        if self.houthi.active < 0.5 or self.houthi.launch_sites < 1:
            return 0.0, _NOOP
        rate = 0.5  # every ~2 days
        def effect():
            if self.us.tomahawks < 5:
                return
            t_used = min(10, self.us.tomahawks)
            self.us.tomahawks = max(0, self.us.tomahawks - t_used)
            kill_frac = self.prior.sample("houthi.site_kill_frac", lu(*HOUTHI_SITE_KILL_FRAC))
            destroyed = self.houthi.launch_sites * kill_frac
            self.houthi.launch_sites = max(0, self.houthi.launch_sites - destroyed)
        return rate, effect

    @transition
    def us_strikes_iraq_militias(self):
        """US strikes on Iraqi militia cells."""
        if self.iraq.active < 0.5 or self.iraq.cells < 1:
            return 0.0, _NOOP
        rate = 0.5
        def effect():
            kill_frac = self.prior.sample("iraq.cell_kill_frac", lu(*IRAQ_CELL_KILL_FRAC))
            destroyed = self.iraq.cells * kill_frac
            self.iraq.cells = max(0, self.iraq.cells - destroyed)
        return rate, effect

    # ---------------------------------------------------------------
    # Hezbollah launches — draws Iron Dome, David's Sling
    # ---------------------------------------------------------------

    @transition
    def hezb_sr_barrage(self):
        """Hezbollah daily short-range rocket barrage → Iron Dome."""
        if self.hezb.active < 0.5 or self.hezb.rockets_sr < 10 or self.hezb.launchers < 1:
            return 0.0, _NOOP
        rate = 1.0  # daily
        def effect():
            max_day = self.prior.sample("hezb.sr_per_day", lu(*HEZB_SR_PER_DAY))
            launcher_cap = self.hezb.launchers * HEZB_ROCKETS_PER_LAUNCHER
            n = min(self.hezb.rockets_sr, max_day, launcher_cap)
            self.hezb.rockets_sr = max(0, self.hezb.rockets_sr - n)
            # Iron Dome intercepts
            pk = self.prior.sample("def.pk_vs_drone", lu(*PK_VS_DRONE))
            killed = n * pk
            iron_dome_used = min(killed, self.defense.iron_dome)
            self.defense.iron_dome = max(0, self.defense.iron_dome - iron_dome_used)
            leaked = n - killed
            self.damage.missiles_intercepted += killed
            self.damage.missiles_leaked += max(0, leaked)
        return rate, effect

    @transition
    def hezb_mr_salvo(self):
        """Hezbollah medium-range salvo → David's Sling."""
        if self.hezb.active < 0.5 or self.hezb.rockets_mr < 5 or self.hezb.launchers < 5:
            return 0.0, _NOOP
        interval = self.prior.sample("hezb.mr_salvo_interval", lu(*HEZB_MR_SALVO_INTERVAL))
        rate = 1.0 / interval
        def effect():
            n = min(self.hezb.rockets_mr,
                    self.prior.sample("hezb.mr_per_salvo", lu(*HEZB_MR_PER_SALVO)))
            self.hezb.rockets_mr = max(0, self.hezb.rockets_mr - n)
            # David's Sling intercepts
            pk = self.prior.sample("def.pk_vs_cruise", lu(*PK_VS_CRUISE))
            killed = n * pk
            ds_used = min(killed, self.defense.davids_sling)
            self.defense.davids_sling = max(0, self.defense.davids_sling - ds_used)
            leaked = n - killed
            self.damage.missiles_intercepted += killed
            self.damage.missiles_leaked += max(0, leaked)
        return rate, effect

    @transition
    def hezb_pgm_strike(self):
        """Hezbollah precision-guided missile strikes → David's Sling / Arrow-2."""
        if self.hezb.active < 0.5 or self.hezb.pgm < 1 or self.hezb.launchers < 1:
            return 0.0, _NOOP
        interval = self.prior.sample("hezb.pgm_interval", lu(*HEZB_PGM_INTERVAL))
        rate = 1.0 / interval
        def effect():
            n = min(self.hezb.pgm,
                    self.prior.sample("hezb.pgm_per_strike", lu(*HEZB_PGM_PER_STRIKE)))
            self.hezb.pgm = max(0, self.hezb.pgm - n)
            remaining = n
            # Arrow-2 engages first (these are ballistic-trajectory)
            if self.defense.arrow2 > 0:
                f = self.prior.sample("def.engage_frac_arrow2", lu(*ENGAGE_FRAC_ARROW2))
                engaged = min(remaining * f, self.defense.arrow2)
                pk = self.prior.sample("def.pk_arrow2", lu(*PK_ARROW2))
                self.defense.arrow2 = max(0, self.defense.arrow2 - engaged)
                remaining -= engaged * pk
            # David's Sling
            if self.defense.davids_sling > 0:
                engaged = min(remaining, self.defense.davids_sling)
                pk = self.prior.sample("def.pk_vs_cruise", lu(*PK_VS_CRUISE))
                self.defense.davids_sling = max(0, self.defense.davids_sling - engaged)
                remaining -= engaged * pk
            leaked = max(0, remaining)
            self.damage.missiles_intercepted += n - leaked
            self.damage.missiles_leaked += leaked
        return rate, effect

    # ---------------------------------------------------------------
    # Houthi launches — BMs at Israel (Arrow/THAAD), drones at Gulf
    # ---------------------------------------------------------------

    @transition
    def houthi_bm_salvo(self):
        """Houthi BM salvo at Israel → Arrow-3 / THAAD."""
        if self.houthi.active < 0.5 or self.houthi.bm < 1 or self.houthi.launch_sites < 1:
            return 0.0, _NOOP
        interval = self.prior.sample("houthi.bm_interval", lu(*HOUTHI_BM_INTERVAL))
        rate = 1.0 / interval
        def effect():
            n = min(self.houthi.bm,
                    self.prior.sample("houthi.bm_per_salvo", lu(*HOUTHI_BM_PER_SALVO)),
                    self.houthi.launch_sites)  # can't fire more than sites available
            self.houthi.bm = max(0, self.houthi.bm - n)
            remaining = n
            # Arrow-3
            if self.defense.arrow3 > 0:
                engaged = min(remaining, self.defense.arrow3)
                pk = self.prior.sample("def.pk_arrow3", lu(*PK_ARROW3))
                self.defense.arrow3 = max(0, self.defense.arrow3 - engaged)
                remaining -= engaged * pk
            # THAAD
            if self.defense.thaad > 0:
                engaged = min(remaining, self.defense.thaad)
                pk = self.prior.sample("def.pk_thaad", lu(*PK_THAAD))
                self.defense.thaad = max(0, self.defense.thaad - engaged)
                remaining -= engaged * pk
            leaked = max(0, remaining)
            self.damage.missiles_intercepted += n - leaked
            self.damage.missiles_leaked += leaked
        return rate, effect

    @transition
    def houthi_drone_wave(self):
        """Houthi daily drone/cruise wave → Iron Dome. Targets Israel."""
        if self.houthi.active < 0.5 or self.houthi.launch_sites < 1:
            return 0.0, _NOOP
        if self.houthi.drones < 2 and self.houthi.cruise < 1:
            return 0.0, _NOOP
        rate = 1.0  # daily
        def effect():
            n_drone = min(self.houthi.drones,
                          self.prior.sample("houthi.drone_per_day", lu(*HOUTHI_DRONE_PER_DAY)))
            self.houthi.drones = max(0, self.houthi.drones - n_drone)
            pk = self.prior.sample("def.pk_vs_drone", lu(*PK_VS_DRONE))
            killed = n_drone * pk
            iron_dome_used = min(killed, self.defense.iron_dome)
            self.defense.iron_dome = max(0, self.defense.iron_dome - iron_dome_used)
            leaked = max(0, n_drone - killed)
            self.damage.missiles_intercepted += killed
            self.damage.missiles_leaked += leaked
            # Drone leakers cause Israeli casualties (lower per-drone than per-BM)
            cas_per = self.prior.sample("cas.per_leaked_bm", lu(*CASUALTIES_PER_LEAKED_BM))
            self.damage.israeli_casualties += leaked * cas_per * DRONE_LETHALITY_VS_BM
        return rate, effect

    # ---------------------------------------------------------------
    # Iraqi militia attacks → Gulf C-RAM / PAC-3
    # ---------------------------------------------------------------

    @transition
    def iraq_attacks(self):
        """Iraqi militia rockets/drones on US Gulf bases → C-RAM (no interceptor draw)."""
        if self.iraq.active < 0.5 or self.iraq.rockets < 1 or self.iraq.cells < 1:
            return 0.0, _NOOP
        attacks_day = self.prior.sample("iraq.attacks_per_day", lu(*IRAQ_PER_DAY))
        rate = attacks_day
        def effect():
            # Each event is one attack from one cell (~1-3 rockets)
            n = min(self.iraq.rockets, 3)
            self.iraq.rockets = max(0, self.iraq.rockets - n)
            # C-RAM intercepts — no expensive interceptor draw
            pk = self.prior.sample("def.pk_cram", lu(*PK_CRAM))
            killed = n * pk
            leaked = max(0, n - killed)
            self.damage.missiles_intercepted += killed
            self.damage.missiles_leaked += leaked
            # Gulf damage from militia attacks ($M USD) — smaller warheads than SRBMs
            dmg_per = self.prior.sample("cas.gulf_dmg_per_drone", lu(*GULF_DAMAGE_PER_LEAKED_DRONE))
            self.damage.gulf_damage += leaked * dmg_per
        return rate, effect

    # --- Iran launches MRBM salvo at Israel ---
    @transition
    def iran_ballistic_salvo(self):
        available = min(self.iran.mrbm, self.iran.tels * IRAN_MISSILES_PER_TEL)
        if available < IRAN_MRBM_MIN_SALVO:
            return 0.0, _NOOP

        rate = 1.0 / IRAN_MRBM_SALVO_INTERVAL

        def effect():
            frac = IRAN_MRBM_SALVO_FRAC
            salvo = max(IRAN_MRBM_MIN_SALVO, self.iran.mrbm * frac)
            salvo = min(salvo, self.iran.mrbm, self.iran.tels * IRAN_MISSILES_PER_TEL)
            self.iran.mrbm = max(0, self.iran.mrbm - salvo)

            # Split: fraction aimed at Israel vs Gulf/other targets
            il_frac = self.prior.sample("iran.mrbm_israel_frac", lu(*MRBM_ISRAEL_FRAC))
            at_israel = salvo * il_frac
            at_gulf = salvo - at_israel

            # Gulf-bound MRBMs: route through actual GulfDefense interceptors
            gulf_remaining = at_gulf
            if self.gulf.thaad > 0:
                f = self.prior.sample("def.engage_frac_thaad", lu(*ENGAGE_FRAC_THAAD))
                engaged = min(gulf_remaining * f, self.gulf.thaad)
                pk = self.prior.sample("def.pk_thaad", lu(*PK_THAAD))
                self.gulf.thaad = max(0, self.gulf.thaad - engaged)
                gulf_remaining -= engaged * pk
            if self.gulf.pac3 > 0:
                engaged_targets = min(gulf_remaining, self.gulf.pac3 / 2)
                pk = self.prior.sample("def.pk_pac3_bm", lu(*PK_PAC3_VS_BM))
                self.gulf.pac3 = max(0, self.gulf.pac3 - engaged_targets * 2)
                gulf_remaining -= engaged_targets * pk
            gulf_leaked = max(0, gulf_remaining)
            dmg_per = self.prior.sample("cas.gulf_dmg_per_bm", lu(*GULF_DAMAGE_PER_LEAKED_BM))
            self.damage.gulf_damage += gulf_leaked * dmg_per

            # Israel-bound: threat filtering — only engage BMs heading for valuable targets
            threat_frac = self.prior.sample("def.bm_threat_frac", lu(*BM_THREAT_FRAC))
            engaged_total = at_israel * threat_frac  # BMs that defense tries to intercept

            remaining = engaged_total

            # SM-3 (sea-based exoatmospheric, fires alongside Arrow-3)
            if self.defense.sm3 > 0:
                f = self.prior.sample("def.engage_frac_sm3", lu(*ENGAGE_FRAC_SM3))
                engaged = min(remaining * f, self.defense.sm3)
                pk = self.prior.sample("def.pk_sm3", lu(*PK_SM3))
                self.defense.sm3 = max(0, self.defense.sm3 - engaged)
                remaining -= engaged * pk

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

            leaked_engaged = max(0, remaining)  # engaged BMs that got through defense
            intercepted = engaged_total - leaked_engaged
            self.damage.missiles_intercepted += intercepted
            self.damage.missiles_leaked += leaked_engaged

            # Israeli casualties: only from BMs that leaked through defense toward targets
            cas_per = self.prior.sample("cas.per_leaked_bm", lu(*CASUALTIES_PER_LEAKED_BM))
            self.damage.israeli_casualties += leaked_engaged * cas_per

            # Cluster warhead casualties even on intercept (submunitions scatter)
            if self.t >= CLUSTER_START_DAY:
                cluster_frac = self.prior.sample("cas.cluster_frac", lu(*CLUSTER_WARHEAD_FRAC))
                cas_cluster = self.prior.sample("cas.per_cluster_intercept",
                    lu(*CASUALTIES_PER_CLUSTER_INTERCEPT))
                self.damage.israeli_casualties += intercepted * cluster_frac * cas_cluster

        return rate, effect

    # --- Iran launches drones + cruise missiles (daily) ---
    @transition
    def iran_drone_cruise(self):
        if self.iran.drones < 10 and self.iran.cruise < 2:
            return 0.0, _NOOP

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

            # Israeli casualties from leaked drones/cruise (lower lethality than BMs)
            cas_per = self.prior.sample("cas.per_leaked_bm", lu(*CASUALTIES_PER_LEAKED_BM))
            self.damage.israeli_casualties += max(0, drone_leaked) * cas_per * DRONE_LETHALITY_VS_BM
            self.damage.israeli_casualties += max(0, cruise_leaked) * cas_per * CRUISE_LETHALITY_VS_BM

        return rate, effect

    # --- Iran launches SRBMs at US Gulf bases ---
    @transition
    def iran_srbm_salvo(self):
        if self.iran.srbm < 10:
            return 0.0, _NOOP

        interval = self.prior.sample("iran.srbm_salvo_interval", lu(*IRAN_SRBM_SALVO_INTERVAL))
        rate = 1.0 / interval

        def effect():
            salvo = min(self.iran.srbm,
                self.prior.sample("iran.srbm_per_salvo", lu(*IRAN_SRBM_PER_SALVO)))
            self.iran.srbm = max(0, self.iran.srbm - salvo)

            # Threat filtering — Gulf bases are denser targets than Israel open areas
            threat_frac = self.prior.sample("def.srbm_threat_frac", lu(*SRBM_THREAT_FRAC))
            remaining = salvo * threat_frac

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

            # Gulf economic/infrastructure damage from leaked SRBMs ($M USD)
            dmg_per = self.prior.sample("cas.gulf_dmg_per_bm", lu(*GULF_DAMAGE_PER_LEAKED_BM))
            self.damage.gulf_damage += leaked * dmg_per

        return rate, effect

    # --- US strikes Iranian TELs ---
    @transition
    def us_strikes_tels(self):
        if self.iran.tels < 1:
            return 0.0, _NOOP

        rate = 1.0

        def effect():
            # OSINT: after SEAD complete (~Day 5-7), US uses gravity bombs — no standoff cost
            sead_thresh = self.prior.sample("us.sead_threshold", lu(*SEAD_THRESHOLD))
            air_superiority = self.iran.production_capacity < sead_thresh

            if not air_superiority:
                if self.us.tomahawks < 1 or self.us.jassm < 1:
                    return

            kill_frac = self.prior.sample("us.tel_kill_frac", lu(*US_TEL_KILL_FRAC))
            destroyed = self.iran.tels * kill_frac
            self.iran.tels = max(0, self.iran.tels - destroyed)

            if not air_superiority:
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
            return 0.0, _NOOP

        interval = self.prior.sample("us.prod_strike_interval", lu(*US_PROD_STRIKE_INTERVAL))
        rate = 1.0 / interval

        def effect():
            # Check SEAD before the strike (bug fix: was using post-strike capacity)
            sead_thresh = self.prior.sample("us.sead_threshold", lu(*SEAD_THRESHOLD))
            needs_standoff = self.iran.production_capacity >= sead_thresh

            # Multiplicative: destroy a fraction of remaining capacity
            frac_destroyed = self.prior.sample("us.prod_strike_effect", lu(*US_PROD_STRIKE_EFFECT))
            self.iran.production_capacity *= (1 - frac_destroyed)

            # OSINT: after SEAD complete, gravity bombs — no standoff cost
            if needs_standoff:
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
    t_end = 100.0
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
