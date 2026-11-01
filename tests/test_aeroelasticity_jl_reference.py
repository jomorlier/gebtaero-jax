"""Verification cases from Aeroelasticity.jl's documentation."""

import numpy as np
import pytest

import gebtaero_jax as gx
from gebtaero_jax import coupled
from gebtaero_jax import section as sec


@pytest.fixture(scope="module")
def hp_params():
    # Hodges & Pierce typical section used by Aeroelasticity.jl
    return sec.SectionParameters.from_nondimensional(a=-0.2, e=-0.1, mu=20, r2=0.24, sigma=0.4)


@pytest.mark.parametrize("model", [sec.Steady(), sec.QuasiSteady(), sec.Wagner(), sec.Peters(6)])
def test_section_divergence_is_analytic(hp_params, model):
    Vd = sec.divergence_reduced_velocity(sec.section_model(model), hp_params, 0.5, 3.5, 15)
    np.testing.assert_allclose(Vd, np.sqrt(0.24 * 20 / (2 * 0.3)), rtol=1e-6)


def test_section_unsteady_flutter_close_to_theodorsen(hp_params):
    V_exact, _ = sec.theodorsen_flutter(hp_params, (2.17, 0.65))
    for model in (sec.Wagner(), sec.Peters(6)):
        Vf, _ = sec.flutter_reduced_velocity(sec.section_model(model), hp_params, 1.5, 3.0, 16)
        assert abs(Vf - 2.2) < 0.05  # "flutter reduced velocity around 2.2"
        assert abs(Vf / V_exact - 1) < 0.012


def test_section_flutter_gradient_matches_fd():
    model = sec.section_model(sec.Peters(6))

    def params_fn(d, V):
        return sec.SectionParameters.from_nondimensional(a=-0.2, e=-0.1, mu=d, r2=0.24, sigma=0.4, V=V)

    vf = lambda mu: coupled.critical_value(model, lambda V: params_fn(mu, V), 1.5, 3.0, 16)
    V, s = vf(20.0)
    g, _ = coupled.critical_value_gradient(model, params_fn, 20.0, V, s)
    fd = (vf(20.0 + 1e-4)[0] - vf(20.0 - 1e-4)[0]) / 2e-4
    np.testing.assert_allclose(float(g), fd, rtol=1e-5)


def test_goland_aeroelasticity_jl_setup():
    """Aeroelasticity.jl goland-stability: 69.0 rad/s, flutter between their 135 and 140 m/s grid points."""
    c, i11 = 1.8288, 8.64
    cs = gx.CrossSection.from_isotropic(9.77e6, 0, 0.99e6).with_mass(35.71, 0.1 * i11, 0.9 * i11, 0.0, 0.1 * c)
    m = gx.straight_wing(cs, 6.096, c, 0.33 - 0.5, 8, aero=gx.Aero.PETERS, a0=0.85 * 2 * np.pi)
    p = gx.flutter_speed(m, gx.FlightCondition(rho=1.02, speed_of_sound=343.0), 100, 200, n_grid=11)
    assert 135.0 <= p.velocity <= 140.0
    assert abs(2 * np.pi * p.frequency_hz - 69.0) < 0.5


def test_wagner_and_peters_agree_on_goland(goland_cs):
    from conftest import GOLAND

    vs = [
        gx.flutter_speed(gx.straight_wing(goland_cs, aero=a, **GOLAND), gx.FlightCondition(rho=1.225), 100, 180, n_grid=9).velocity
        for a in (gx.Aero.PETERS, gx.Aero.WAGNER)
    ]
    assert abs(vs[0] / vs[1] - 1) < 0.02
