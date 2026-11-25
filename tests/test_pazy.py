"""Pazy wing (AePW3 LDWG) regression against UM/NAST (Riso & Cesnik) on the same equivalent-beam data."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import gebtaero_jax as gx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from pazy_wing import PazyWing, load_reference  # noqa: E402


@pytest.fixture(scope="module")
def wing():
    return PazyWing()


def test_pazy_undeformed_frequencies(wing):
    m, f = wing.build(aero=gx.Aero.NONE)
    np.testing.assert_allclose(gx.modal_analysis(m, f).table(5)[:, 1], load_reference("umnast_frequencies_undeformed.txt"), rtol=5e-3)


def test_pazy_static_deflection_aoa7(wing):
    ref = load_reference("umnast_static_aoa7.txt")
    x = None
    for V in (20.0, 35.0, 50.0):
        m, f = wing.build(V=V, alpha_deg=7.0)
        x = gx.solve_static(m, f, x0=x, n_load_steps=1 if x is not None else 3)
        z_ref = np.interp(V, ref[:, 0], ref[:, 1])
        assert abs(wing.tip_z_percent(m, x) - z_ref) < 0.04 * z_ref + 0.3


def test_pazy_hump_onset_aoa5(wing):
    p = gx.flutter_speed(*wing.build(V=0.0, alpha_deg=5.0), 38.0, 46.0, n_grid=5, fmax_hz=60.0, deformed=True, n_load_steps=3)
    assert abs(p.velocity - 43.14) < 1.0  # UM/NAST 43.14 m/s, experiment 43 m/s
    assert 28.0 < p.frequency_hz < 32.0


def test_rigid_body_mass_matrix_and_convert_state():
    from gebtaero_jax.cross_section import rigid_body_mass_matrix

    M = np.asarray(rigid_body_mass_matrix(2.0, [0.1, -0.2, 0.3], np.diag([1.0, 2.0, 3.0])))
    np.testing.assert_allclose(M, M.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(M) > 0)
    cs = gx.CrossSection.from_isotropic(2e4, 4e6, 1e4).with_mass(0.75, 1e-4, 0.1)
    mp = gx.straight_wing(cs, 16.0, 1.0, 0.0, 4, aero=gx.Aero.PETERS)
    ms = gx.straight_wing(cs, 16.0, 1.0, 0.0, 4, aero=gx.Aero.NONE)
    x = jnp.arange(mp.n_dof, dtype=float)
    y = gx.convert_state(x, mp, ms)
    assert y.shape == (ms.n_dof,)
    np.testing.assert_array_equal(gx.convert_state(y, ms, mp)[: 6 + 18], x[: 6 + 18])
