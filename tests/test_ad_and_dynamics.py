import jax
import jax.numpy as jnp
import numpy as np
from conftest import GOLAND

import gebtaero_jax as gx


def test_static_tip_deflection_gradient_matches_fd():
    """Implicit differentiation of the nonlinear equilibrium (custom_root)."""

    def tip_w(params):
        cs = gx.CrossSection.from_isotropic(params["EI"], 0.0, 1e4).with_mass(0.75, 1e-4, 0.1)
        m = gx.straight_wing(cs, 16.0, 1.0, 0.0, 8, aero=gx.Aero.PETERS)
        fl = gx.FlightCondition(velocity=12.0, rho=0.0889, alpha=params["alpha"], gravity=9.81)
        x = gx.solve_static(m, fl, n_load_steps=3)
        return gx.tip_displacement(m, x)[2]

    p = {"EI": 2e4, "alpha": 0.05}
    g = jax.grad(tip_w)(p)
    for k, h in (("EI", 10.0), ("alpha", 1e-5)):
        pp, pm = dict(p), dict(p)
        pp[k] += h
        pm[k] -= h
        fd = (tip_w(pp) - tip_w(pm)) / (2 * h)
        np.testing.assert_allclose(g[k], fd, rtol=1e-5)


def test_flutter_speed_gradient_matches_fd(goland_cs):
    def build(p, V):
        cs = gx.CrossSection.from_isotropic(9.77e6, 0.0, p["GJ"]).with_mass(35.71, 8.64e-4, 8.64, 0.0, p["nu"])
        return gx.straight_wing(cs, aero=gx.Aero.PETERS, **GOLAND), gx.FlightCondition(velocity=V, rho=1.225)

    p = {"GJ": 0.99e6, "nu": 0.18288}
    r = gx.flutter_speed_and_grad(build, p, 120, 150, n_grid=4, tol=1e-8)
    for k, h in (("GJ", 1e3), ("nu", 1e-4)):
        pp, pm = dict(p), dict(p)
        pp[k] += h
        pm[k] -= h
        vp = gx.flutter_speed(*build(pp, 0.0), 120, 150, n_grid=4, tol=1e-9).velocity
        vm = gx.flutter_speed(*build(pm, 0.0), 120, 150, n_grid=4, tol=1e-9).velocity
        np.testing.assert_allclose(r.dV_dparams[k], (vp - vm) / (2 * h), rtol=1e-4)


def test_time_domain_growth_rate_matches_modal(goland_cs):
    m = gx.straight_wing(goland_cs, aero=gx.Aero.PETERS, **GOLAND)
    fl = gx.FlightCondition(velocity=140.0, rho=1.225)
    s = gx.modal_analysis(m, fl)
    idx = s.oscillatory()
    sigma = s.eigvals.real[idx].max()

    def pulse(t, f):
        return f.replace(tip_force=jnp.array([0.0, 0.0, -jnp.where(t < 0.02, 2000.0, 0.0)]))

    h = gx.simulate(m, fl, 3.0, 0.002, flight_fn=pulse)
    tw = h.tip_rotation_angle()
    # fit the growth of the successive peaks of the tip twist (after the transient)
    pk = [i for i in range(1, len(tw) - 1) if tw[i] > tw[i - 1] and tw[i] >= tw[i + 1] and h.t[i] > 1.0]
    rate = np.polyfit(h.t[pk], np.log(tw[pk]), 1)[0]
    np.testing.assert_allclose(rate, sigma, rtol=0.05)


def test_structural_damping_adds_damping(patil_cs):
    cs = patil_cs
    m0 = gx.straight_wing(cs, 16.0, 1.0, 0.0, 8, aero=gx.Aero.NONE)
    m1 = gx.straight_wing(cs, 16.0, 1.0, 0.0, 8, aero=gx.Aero.NONE, damping=jnp.full(6, 1e-3))
    z0 = gx.modal_analysis(m0, gx.FlightCondition()).table(3)[:, 0]
    z1 = gx.modal_analysis(m1, gx.FlightCondition()).table(3)[:, 0]
    assert np.all(np.abs(z0) < 1e-8) and np.all(z1 < -1e-6)
