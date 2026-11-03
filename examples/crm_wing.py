"""CRM-like wing: planform, aluminium wingbox sizing variables and a beam model builder.

The planform uses approximate public values of the NASA Common Research Model (CRM) wing:
semi-span 29.38 m, quarter-chord sweep 35 deg, Yehudi break at 37 % semi-span, root / break / tip
chords of about 13.6 / 7.3 / 2.7 m (streamwise). The structure is a *simplified* single-cell
aluminium wingbox (front spar 15 %, rear spar 65 % of the trapezoidal reference chord) with
non-structural mass representing fuel, ribs, stringers, leading/trailing edges and systems.
It is **not** the uCRM finite-element model: absolute numbers are illustrative only.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import gebtaero_jax as gx

SEMI_SPAN = 29.38
ETA_BREAK = 0.37
C_ROOT, C_BREAK, C_TIP = 13.6, 7.3, 2.7  # streamwise chords (m)
SWEEP_LE = np.radians(37.2)
TC_ROOT, TC_TIP = 0.14, 0.10  # thickness-to-chord ratio
FRONT_SPAR, REAR_SPAR = 0.15, 0.65  # fractions of the trapezoidal reference chord
ALU = dict(E=72e9, G=27e9, rho=2800.0)
STIFFENED = 1.3  # stringers: effective skin thickness factor (stiffness and mass)


def planform(n_sections: int = 20):
    """Section mid-span stations and streamwise geometry."""
    y_edges = np.linspace(0.0, SEMI_SPAN, n_sections + 1)
    y = 0.5 * (y_edges[1:] + y_edges[:-1])
    eta = y / SEMI_SPAN
    chord = np.where(eta < ETA_BREAK,
                     C_ROOT + (C_BREAK - C_ROOT) * eta / ETA_BREAK,
                     C_BREAK + (C_TIP - C_BREAK) * (eta - ETA_BREAK) / (1 - ETA_BREAK))
    # trapezoidal reference chord (extends the outboard panel to the root)
    c_trap = C_TIP + (C_BREAK - C_TIP) * (1 - eta) / (1 - ETA_BREAK)
    tc = TC_ROOT + (TC_TIP - TC_ROOT) * eta
    return dict(y_edges=y_edges, y=y, eta=eta, chord=chord, c_trap=c_trap, tc=tc)


def box_axis_sweep():
    """Sweep of the wingbox centre line (40 % of the trapezoidal chord)."""
    xc = 0.5 * (FRONT_SPAR + REAR_SPAR)
    tan_le = np.tan(SWEEP_LE)
    # x_ref(y) = y tan(LE) + xc c_trap(y);  d c_trap/dy = -(C_BREAK - C_TIP)/((1-eta_b) b)
    dc = -(C_BREAK - C_TIP) / ((1 - ETA_BREAK) * SEMI_SPAN)
    return np.arctan(tan_le + xc * dc)


def baseline_thickness(n_sections: int = 20):
    eta = planform(n_sections)["eta"]
    t_skin = 0.022 - 0.016 * eta  # 22 mm -> 6 mm
    t_spar = 0.016 - 0.010 * eta  # 16 mm -> 6 mm
    return jnp.asarray(np.concatenate([t_skin, t_spar]))


class CRMWing:
    """Builder ``x -> (BeamModel, FlightCondition)`` with ``x = [t_skin (n), t_spar (n)]`` in metres."""

    def __init__(self, n_sections: int = 20, aero: gx.Aero = gx.Aero.PETERS, rho: float = 0.55,
                 ns_mass_per_area: float = 60.0):
        self.n = n_sections
        self.aero = aero
        self.rho = rho
        g = planform(n_sections)
        self.geo = g
        self.sweep = box_axis_sweep()
        cosL = np.cos(self.sweep)
        self.seg_len = (g["y_edges"][1] - g["y_edges"][0]) / cosL  # along the swept axis
        # section geometry normal to the swept beam axis
        self.c_n = g["chord"] * cosL
        self.w_box = (REAR_SPAR - FRONT_SPAR) * g["c_trap"] * cosL
        self.h_box = 0.9 * g["tc"] * g["chord"]
        # reference axis = box centre; position from the leading edge, normal to the axis
        x_ref = 0.5 * (FRONT_SPAR + REAR_SPAR) * g["c_trap"] * cosL
        self.a = x_ref / (0.5 * self.c_n) - 1.0  # Theodorsen 'a' (semi-chords, + aft)
        # non-structural mass: per unit planform area, CG at 45 % of the local chord
        self.mu_ns = ns_mass_per_area * self.c_n
        self.nu_ns = 0.45 * self.c_n - x_ref  # + aft of the reference axis
        self.frame = gx.Frame.from_angles(sweep=self.sweep)

    def section_properties(self, x):
        """Stiffness and mass per unit length of every section (jax-traceable in ``x``)."""
        n = self.n
        ts = STIFFENED * x[:n]
        tw = x[n:]
        w, h = jnp.asarray(self.w_box), jnp.asarray(self.h_box)
        E, G, rho = ALU["E"], ALU["G"], ALU["rho"]
        EI_flap = E * (2 * w * ts * (h / 2) ** 2 + 2 * tw * h**3 / 12)
        EI_lag = E * (2 * ts * w**3 / 12 + 2 * tw * h * (w / 2) ** 2)
        GJ = G * 4 * (w * h) ** 2 / (2 * w / ts + 2 * h / tw)
        mu_s = rho * (2 * w * ts + 2 * h * tw)
        mu_ns = jnp.asarray(self.mu_ns)
        nu_ns = jnp.asarray(self.nu_ns)
        mu = mu_s + mu_ns
        nu = mu_ns * nu_ns / mu
        cn = jnp.asarray(self.c_n)
        i22 = mu_s * h**2 / 12 + mu_ns * (0.05 * cn) ** 2
        i33 = mu_s * w**2 / 12 + mu_ns * ((0.25 * cn) ** 2 + nu_ns**2) - mu * nu**2
        return dict(EI_flap=EI_flap, EI_lag=EI_lag, GJ=GJ, mu=mu, mu_s=mu_s, nu=nu, i22=i22, i33=i33)

    def structural_mass(self, x):
        """Wingbox mass of the semi-span (kg)."""
        return jnp.sum(self.section_properties(x)["mu_s"] * self.seg_len)

    def build(self, x, V: float = 0.0, aero=None):
        p = self.section_properties(x)
        sections = []
        for i in range(self.n):
            cs = gx.CrossSection.from_isotropic(p["EI_flap"][i], p["EI_lag"][i], p["GJ"][i]).with_mass(
                p["mu"][i], p["i22"][i], p["i33"][i], 0.0, p["nu"][i])
            sections.append(gx.WingSection(float(self.c_n[i]), float(self.a[i]), float(self.seg_len), cs,
                                           self.frame, 1))
        model = gx.Wing.from_sections(sections).discretize(self.aero if aero is None else aero)
        return model, gx.FlightCondition(velocity=V, rho=self.rho)
