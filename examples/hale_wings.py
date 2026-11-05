"""Beam models of two solar-electric HALE wings, for aeroelastic checks with gebtaero-jax.

1. **EcoHALE** -- the CO2-minimal design of ``EcoHALE_AeroSandbox_1.ipynb`` (re-implementation of
   mid2SUPAERO/ecoHALE; Duriez, Guadano Martin & Morlier, Sci. Rep. 13, 11994, 2023): Sandwich-1
   (CFRP faces / PS-foam core) skins and spars, span 55.6 m, root chord 1.4 m, taper 0.3,
   gross mass 233.7 kg (structure 82.4 kg, PV + batteries 116.7 kg), cruise 34.5 m/s at
   rho = 0.055 kg/m^3, motors at the wing tips, wingbox between 10 % and 60 % chord.

2. **Dawn** -- the point design of the DawnDesignTool (P. Sharpe, A. Dewald et al., AIAA 2021-2451,
   Table 2): span 39.9 m, AR 23.4, S = 67.9 m^2, TOGW 375 kg, 30.4 m/s at about 18.7 km, wing CL ~ 1.1;
   from ``design_opt.py``: taper break at 67 % semi-span, outboard taper 0.5, HALE_03 airfoil
   (t/c = 13.2 %), outboard booms at 80 % semi-span, carbon-tube spar sized for a 3 g ultimate load.

What is published and what is reconstructed
-------------------------------------------
Planforms, masses, speeds and densities above are taken from the sources. The *distribution* of the
structure is not published in a form a beam model can use, so it is reconstructed here and
documented as ASSUMPTION below: EcoHALE spar webs sit at the 1 mm lower bound and the skins taper
root-to-tip, scaled to the reported 82.4 kg, with t/c calibrated on the reported 1.7 m gust deflection;
the Dawn spar is a CFRP tube sized by bending stress at 3 g; secondary-structure,
solar-cell, boom and battery masses are split so that the totals match. Treat the results as a
representative aeroelastic check of these designs, not as their certified properties.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import gebtaero_jax as gx

G0 = 9.81


def isa_density(h):
    """ISA density (troposphere + lower stratosphere, h <= 20 km)."""
    if h <= 11000:
        T = 288.15 - 0.0065 * h
        p = 101325 * (T / 288.15) ** 5.2559
    else:
        T = 216.65
        p = 22632.1 * np.exp(-(h - 11000) / 6341.62)
    return p / (287.05 * T)


# ======================================================================================
# EcoHALE
# ======================================================================================
class EcoHALEWing:
    """Half-wing of the EcoHALE CO2 optimum. Design vector ``x = [t_skin (n), t_spar (n)]`` (m)."""

    SPAN, C_ROOT, TAPER = 55.6, 1.40, 0.30
    V_CRUISE, RHO = 34.5, 0.055
    V_DIVE = 1.4 * 34.5  # EcoHALE gust point: V_D = 1.4 V with a 3.4 m/s gust
    GUST = 3.4
    MASS_GROSS, MASS_STRUCT, MASS_PV_BATT = 233.7, 82.4, 116.7
    SANDWICH_1 = dict(E=42.5e9, G=16.3e9, rho=504.5)  # homogenised CFRP / PS-foam (EcoHALE catalogue)
    BOX_WIDTH, BOX_H_MEAN, X_EA = 0.50, 0.1093, 0.35  # NACA 63412 box (10-60 % c) at t/c = 0.12
    # ASSUMPTIONS -------------------------------------------------------------------------
    # t/c is an EcoHALE design variable whose optimum is not printed. A 0.13 -> 0.11 shape is scaled by 1.727 so that the
    # gust-point tip deflection (V_D, n = 1.1, +atan(3.4/V_D)) equals the notebook's 1.7 m: CALIBRATED, see notebook 09.
    TC_SCALE = 1.727
    TC_ROOT, TC_TIP = 0.13 * TC_SCALE, 0.11 * TC_SCALE
    MOTOR_MASS = 6.0  # kg per motor at each tip (estimated from the mass closure: ~14 kg unassigned)
    PV_CG = 0.50  # chordwise CG of PV + batteries (spread over the whole chord)
    THICK_TAPER = 3.0  # root / tip skin-thickness ratio of the reconstructed baseline
    T_SPAR = 1.0e-3  # spar webs at EcoHALE's 1 mm lower bound; skins carry the rest of the 82.4 kg
    ALPHA0 = np.radians(-10.0)  # zero-lift angle standing in for EcoHALE's jig twist (up to 15 deg)

    def __init__(self, n: int = 12, aero=gx.Aero.PETERS, motor_mass: float | None = None):
        self.n = n
        self.aero = aero
        self.motor_mass = self.MOTOR_MASS if motor_mass is None else motor_mass
        b2 = self.SPAN / 2
        self.dy = b2 / n
        eta = (np.arange(n) + 0.5) / n
        self.eta = eta
        self.chord = self.C_ROOT * (1 - (1 - self.TAPER) * eta)
        self.tc = self.TC_ROOT + (self.TC_TIP - self.TC_ROOT) * eta
        self.w = self.BOX_WIDTH * self.chord
        self.h = self.BOX_H_MEAN * self.chord * self.tc / 0.12
        self.a = (self.X_EA - 0.5) / 0.5
        self.area_half = 0.5 * self.SPAN * self.C_ROOT * (1 + self.TAPER) / 2
        self.mu_pv = self.MASS_PV_BATT / 2 / self.area_half * self.chord  # kg/m

    def baseline(self) -> jnp.ndarray:
        """Reconstructed walls reproducing the reported 82.4 kg structure.

        Spar webs at the 1 mm lower bound, skins tapered linearly (root/tip = 3) and scaled to the mass:
        bending/buckling-driven optimisers put the material in the skins.
        """
        shape = self.THICK_TAPER + (1 - self.THICK_TAPER) * self.eta
        rho = self.SANDWICH_1["rho"]
        m_spar = np.sum(rho * 2 * self.h * self.T_SPAR * self.dy)
        t0 = (self.MASS_STRUCT / 2 - m_spar) / np.sum(rho * 2 * self.w * shape * self.dy)
        return jnp.asarray(np.concatenate([t0 * shape, np.full(self.n, self.T_SPAR)]))

    def section_properties(self, x):
        n, mat = self.n, self.SANDWICH_1
        ts, tw = x[:n], x[n:]
        w, h, c = jnp.asarray(self.w), jnp.asarray(self.h), jnp.asarray(self.chord)
        EI_flap = mat["E"] * (2 * w * ts * (h / 2) ** 2 + 2 * tw * h**3 / 12)
        EI_lag = mat["E"] * (2 * ts * w**3 / 12 + 2 * tw * h * (w / 2) ** 2)
        GJ = mat["G"] * 4 * (w * h) ** 2 / (2 * w / ts + 2 * h / tw)
        mu_s = mat["rho"] * (2 * w * ts + 2 * h * tw)
        mu_pv = jnp.asarray(self.mu_pv)
        mu = mu_s + mu_pv
        e_pv = (self.PV_CG - self.X_EA) * c  # + aft of the elastic axis
        nu = mu_pv * e_pv / mu
        i33 = mu_s * w**2 / 12 + mu_pv * (c**2 / 12 + e_pv**2)  # about the elastic axis
        i22 = mu_s * h**2 / 12 + mu_pv * (0.02 * c) ** 2
        return dict(EI_flap=EI_flap, EI_lag=EI_lag, GJ=GJ, mu=mu, mu_s=mu_s, nu=nu, i22=i22, i33=i33)

    def structural_mass(self, x):
        """Wingbox mass of the whole aircraft (both half-wings), kg."""
        return 2 * jnp.sum(self.section_properties(x)["mu_s"] * self.dy)

    def build(self, x, V: float | None = None, alpha: float = 0.0, gravity: float = G0, aero=None, **flight_kw):
        p = self.section_properties(x)
        frame = gx.Frame.from_axis([0, 1, 0])
        secs = [
            gx.WingSection(float(self.chord[i]), float(self.a), self.dy,
                           gx.CrossSection.from_isotropic(p["EI_flap"][i], p["EI_lag"][i], p["GJ"][i]).with_mass(
                               p["mu"][i], p["i22"][i], p["i33"][i], 0.0, p["nu"][i]),
                           frame, 1, alpha0=self.ALPHA0)
            for i in range(self.n)
        ]
        model = gx.Wing.from_sections(secs).discretize(self.aero if aero is None else aero)
        model = model.with_point_mass(self.SPAN / 2 - 1e-6, self.motor_mass)
        V = self.V_CRUISE if V is None else V
        return model, gx.FlightCondition(velocity=V, rho=self.RHO, alpha=alpha, gravity=gravity, **flight_kw)

    @property
    def root_carried_mass(self):
        """Mass not on the wing (fuselage, payload, avionics, margin), per half-wing: carried at the root."""
        return self.MASS_GROSS / 2 - self.MASS_STRUCT / 2 - self.MASS_PV_BATT / 2 - self.motor_mass


# ======================================================================================
# Dawn
# ======================================================================================
class DawnWing:
    """Half-wing of the Dawn point design. Design vector ``x = t_tube (n)`` (m), spar wall thickness."""

    SPAN, AREA, AR = 39.9, 67.9, 23.4
    BREAK, TAPER = 0.67, 0.5
    TOGW, V_CRUISE, ALTITUDE = 375.0, 30.4, 18_700.0
    TC = 0.1324  # HALE_03 maximum thickness (computed from studies/airfoil_optimizer/HALE_03.dat)
    BOOM_ETA = 0.80
    MASS_PAYLOAD, MASS_BATTERY = 30.0, 167.6  # payload (Table/mission); 75.4 kWh / 450 Wh/kg
    N_ULT = 3.0  # structural_load_factor in design_opt.py
    # ASSUMPTIONS -------------------------------------------------------------------------
    X_SPAR = 0.30  # tube spar at 30 % chord, which is also the elastic axis
    SPAR_FILL = 0.85  # tube diameter / local airfoil thickness at the spar
    TUBE = dict(E=120e9, G=20e9, rho=1600.0, sigma=350e6)  # UD-dominated CFRP tube, ultimate allowable
    T_MIN = 0.6e-3
    SECONDARY_AREAL = 0.55  # kg/m^2: ribs, leading/trailing edge, covering (CG at 40 % c)
    SOLAR_AREAL, SOLAR_FRACTION = 0.40, 0.80  # kg/m^2 of cell area (incl. encapsulation, wiring), 80 % of S
    BOOM_POD_MASS = 18.0  # kg per outboard boom: tail boom, tails, motor + propeller, CG 1.0 m ahead of EA
    ALPHA0 = np.radians(-6.0)  # cambered HALE_03 airfoil

    def __init__(self, n: int = 12, aero=gx.Aero.PETERS, batteries: str = "fuselage"):
        assert batteries in ("fuselage", "wing")
        self.n, self.aero, self.batteries = n, aero, batteries
        b2 = self.SPAN / 2
        self.c_root = self.AREA / (self.SPAN * (self.BREAK + (1 - self.BREAK) * (1 + self.TAPER) / 2))
        self.dy = b2 / n
        eta = (np.arange(n) + 0.5) / n
        self.eta = eta
        self.chord = np.where(eta < self.BREAK, self.c_root,
                              self.c_root * (1 - (1 - self.TAPER) * (eta - self.BREAK) / (1 - self.BREAK)))
        self.r = 0.5 * self.SPAR_FILL * self.TC * self.chord  # tube radius (thickness ~ max t/c near 30 % c)
        self.rho = isa_density(self.ALTITUDE)
        self.mu_sec = self.SECONDARY_AREAL * self.chord
        self.mu_sol = self.SOLAR_AREAL * self.SOLAR_FRACTION * self.chord
        self.mu_bat = (self.MASS_BATTERY / 2) / (self.AREA / 2) * self.chord if batteries == "wing" else 0 * self.chord

    def sized_spar(self) -> jnp.ndarray:
        """Wall thickness from the 3 g ultimate bending moment of a chord-proportional lift (no inertia relief)."""
        y = np.linspace(0, self.SPAN / 2, 401)
        c = np.interp(y / (self.SPAN / 2), self.eta, self.chord, left=self.chord[0], right=self.chord[-1])
        lift = self.N_ULT * G0 * self.TOGW / 2 * c / np.trapezoid(c, y)  # N/m
        M = np.array([np.trapezoid(lift[i:] * (y[i:] - y[i]), y[i:]) for i in range(len(y))])
        Me = np.interp(self.eta * self.SPAN / 2, y, M)
        t = Me / (np.pi * self.r**2 * self.TUBE["sigma"])
        return jnp.asarray(np.maximum(t, self.T_MIN))

    def section_properties(self, t):
        r = jnp.asarray(self.r)
        E, G, rho = self.TUBE["E"], self.TUBE["G"], self.TUBE["rho"]
        EI = E * jnp.pi * r**3 * t
        GJ = G * 2 * jnp.pi * r**3 * t
        mu_s = rho * 2 * jnp.pi * r * t
        c = jnp.asarray(self.chord)
        parts = [(mu_s, 0.0), (jnp.asarray(self.mu_sec), 0.40 - self.X_SPAR), (jnp.asarray(self.mu_sol), 0.50 - self.X_SPAR),
                 (jnp.asarray(self.mu_bat), 0.30 - self.X_SPAR)]
        mu = sum(m for m, _ in parts)
        nu = sum(m * e * c for m, e in parts) / mu
        i33 = sum(m * ((e * c) ** 2 + (c**2 / 12 if k else r**2 / 2)) for k, (m, e) in enumerate(parts))  # about the elastic axis
        i22 = mu_s * r**2 / 2 + (mu - mu_s) * (0.03 * c) ** 2
        return dict(EI_flap=EI, EI_lag=EI * 25.0, GJ=GJ, mu=mu, mu_s=mu_s, nu=nu, i22=i22, i33=i33)

    def spar_mass(self, t):
        return 2 * jnp.sum(self.section_properties(t)["mu_s"] * self.dy)

    def mass_budget(self, t):
        sp = float(self.spar_mass(t))
        sec = 2 * float(np.sum(self.mu_sec * self.dy))
        sol = 2 * float(np.sum(self.mu_sol * self.dy))
        bat = 2 * float(np.sum(self.mu_bat * self.dy))
        booms = 2 * self.BOOM_POD_MASS
        on_wing = sp + sec + sol + bat + booms
        return dict(spar=sp, secondary=sec, solar=sol, batteries_in_wing=bat, booms=booms, on_wing=on_wing,
                    fuselage_and_rest=self.TOGW - on_wing)

    def build(self, t, V: float | None = None, alpha: float = 0.0, gravity: float = G0, aero=None, **flight_kw):
        p = self.section_properties(t)
        frame = gx.Frame.from_axis([0, 1, 0])
        a = (self.X_SPAR - 0.5) / 0.5
        secs = [
            gx.WingSection(float(self.chord[i]), a, self.dy,
                           gx.CrossSection.from_isotropic(p["EI_flap"][i], p["EI_lag"][i], p["GJ"][i]).with_mass(
                               p["mu"][i], p["i22"][i], p["i33"][i], 0.0, p["nu"][i]),
                           frame, 1, alpha0=self.ALPHA0)
            for i in range(self.n)
        ]
        model = gx.Wing.from_sections(secs).discretize(self.aero if aero is None else aero)
        model = model.with_point_mass(self.BOOM_ETA * self.SPAN / 2, self.BOOM_POD_MASS, offset=(1.0, 0.0),
                                      inertia=(1.0, 6.0, 6.0))
        V = self.V_CRUISE if V is None else V
        return model, gx.FlightCondition(velocity=V, rho=self.rho, alpha=alpha, gravity=gravity, **flight_kw)

    def root_carried_mass(self, t):
        return self.mass_budget(t)["fuselage_and_rest"] / 2


def trim_alpha(build_fn, root_mass_half, alpha0=0.05, n_load_steps=3, tol=1e-6, n=1.0):
    """Root angle of attack such that the half-wing lift balances ``n`` x (its weight + half of the root-carried mass).

    ``build_fn(alpha)`` must build the model with gravity ``n * g`` (inertia relief of the wing masses).

    Newton iteration on alpha; d(lift)/d(alpha) comes from AD through the nonlinear static equilibrium.
    """
    import jax

    def net(alpha):
        model, flight = build_fn(alpha)
        x = gx.solve_static(model, flight, n_load_steps=n_load_steps)
        return gx.lift_force(model, x) - n * G0 * root_mass_half, x

    alpha = alpha0
    for _ in range(20):
        f, g = jax.value_and_grad(lambda a: net(a)[0])(alpha)
        step = float(np.clip(f / g, -0.05, 0.05))  # damped Newton: at most ~3 deg per step (stay on the physical branch)
        alpha = alpha - step
        if abs(step) < tol:
            break
    model, flight = build_fn(alpha)
    return float(alpha), gx.solve_static(model, flight, n_load_steps=n_load_steps)
