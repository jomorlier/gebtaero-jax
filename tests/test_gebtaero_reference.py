"""Regression values from GEBTAero's ``cas_test`` suite (Goland and Patil wings)."""

import numpy as np
import pytest
from conftest import DEG, GOLAND, PATIL

import gebtaero_jax as gx


def test_patil_structural_modes():
    i11 = 0.1
    cs = gx.CrossSection.from_isotropic(2e4, 4e6, 1e4).with_mass(0.75, 1e-3 * i11, i11, 0.0, 0.0)
    m = gx.straight_wing(cs, 16.0, 1.0, 0.0, 20, aero=gx.Aero.NONE)
    f = gx.modal_analysis(m, gx.FlightCondition()).table(5)[:, 1]
    np.testing.assert_allclose(f, [0.357, 2.257, 4.941, 5.048, 6.428], atol=1e-2)


@pytest.mark.parametrize("aero, ref", [(gx.Aero.PETERS, 136.5), (gx.Aero.QUASI_STEADY_ADDED_MASS, 64.34), (gx.Aero.QUASI_STEADY, 35.2)])
def test_goland_flutter(goland_cs, aero, ref):
    m = gx.straight_wing(goland_cs, aero=aero, **GOLAND)
    p = gx.flutter_speed(m, gx.FlightCondition(rho=1.225), 10, 200, n_grid=20)
    assert p.found and abs(p.velocity - ref) < 0.15


def test_goland_divergence(goland_cs):
    m = gx.straight_wing(goland_cs, aero=gx.Aero.QUASI_STEADY, **GOLAND)
    assert abs(gx.divergence_speed(m, gx.FlightCondition(rho=1.225), 5, 300).velocity - 253) < 2


@pytest.mark.parametrize(
    "geom, flight",
    [
        (dict(sweep=15 * DEG), dict(beta=15 * DEG, gravity=9.81)),
        (dict(dihedral=15 * DEG), dict(gravity=9.81)),
        (dict(twist=np.pi / 4), dict(alpha=-np.pi / 4)),
    ],
)
def test_goland_geometric_invariance(goland_cs, geom, flight):
    m = gx.straight_wing(goland_cs, aero=gx.Aero.PETERS, **GOLAND, **geom)
    p = gx.flutter_speed(m, gx.FlightCondition(rho=1.225, **flight), 100, 160, n_grid=7)
    assert abs(p.velocity - 136.5) < 0.1


def test_patil_flutter_and_divergence(patil_cs):
    m = gx.straight_wing(patil_cs, aero=gx.Aero.PETERS, **PATIL)
    fl = gx.FlightCondition(rho=0.0889)
    assert abs(gx.flutter_speed(m, fl, 1, 50, n_grid=10).velocity - 32.12) < 0.1
    assert abs(gx.divergence_speed(m, fl, 1, 100).velocity - 37.25) < 0.2


def test_patil_deformed_by_gravity(patil_cs):
    """GEBTAero: 23.35 m/s with its approximate aero Jacobian; exact AD linearisation gives 23.13."""
    m = gx.straight_wing(patil_cs, aero=gx.Aero.PETERS, **PATIL)
    p = gx.flutter_speed(m, gx.FlightCondition(rho=0.0889, gravity=9.81), 1, 50, n_grid=10, n_load_steps=4)
    assert abs(p.velocity - 23.35) < 0.35


def test_legacy_dat_reader():
    case = gx.read_dat("examples/data/goland.dat")
    m = case.wing.discretize(case.aero_flag)
    assert abs(gx.flutter_speed(m, case.flight, 100, 160, n_grid=7).velocity - 136.5) < 0.1
