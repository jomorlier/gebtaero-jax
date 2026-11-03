"""Bordered eigen-adjoint (S. He's mode-tracking adjoint, complex non-symmetric version) and MAC tracking."""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import GOLAND

import gebtaero_jax as gx
from gebtaero_jax import tracking as tk
from gebtaero_jax.stability import eigenvalue_gradient


def _patil_builder(p):
    cs = gx.CrossSection.from_isotropic(p[0], 4e6 * p[2], p[1]).with_mass(0.75, 1e-3 * 0.1, 0.1, 0.0, p[3])
    return gx.straight_wing(cs, 16.0, 1.0, 0.0, 6, aero=gx.Aero.NONE), gx.FlightCondition()


@pytest.fixture(scope="module")
def patil_setup():
    p0 = jnp.array([2e4, 1e4, 1.0, 0.0])
    m0, f0 = _patil_builder(p0)
    ref = tk.reference_modes(m0, gx.modal_analysis(m0, f0), 4)
    p1 = jnp.array([2.3e4, 0.8e4, 0.7, 0.05])  # perturbed design: shapes differ from the reference
    return ref, p1


def test_tracked_frequency_adjoint_matches_left_right_formula(patil_setup):
    ref, p1 = patil_setup
    ts = tk.tracked_sensitivities(_patil_builder, p1, ref)
    m, f = _patil_builder(p1)
    r = gx.modal_analysis(m, f)
    for i in range(ref.n):
        j = ts.tracked.index[i]
        _, _, im_p, _ = eigenvalue_gradient(lambda p, V: _patil_builder(p), p1, 0.0, r.eigvals[j], r.right[:, j], r.left[:, j])
        ref_grad = np.asarray(im_p) / (2 * np.pi)
        np.testing.assert_allclose(ts.dfreq[i], ref_grad, rtol=1e-6, atol=1e-8 * np.abs(ref_grad).max())


def test_mac_loss_adjoint_matches_fd(patil_setup):
    ref, p1 = patil_setup
    ts = tk.tracked_sensitivities(_patil_builder, p1, ref)
    assert 1e-4 < ts.mac_loss < 0.5

    def loss(p):
        m, f = _patil_builder(jnp.asarray(p))
        return 1 - tk.track(m, gx.modal_analysis(m, f), ref).mac.mean()

    rel = np.array([1e-4, 1e-4, 1e-4, 1e-3])
    for k in range(4):
        h = rel[k] * max(abs(float(p1[k])), 1.0)
        e = np.zeros(4)
        e[k] = h
        fd = (loss(np.asarray(p1) + e) - loss(np.asarray(p1) - e)) / (2 * h)
        np.testing.assert_allclose(ts.dmac_loss[k], fd, rtol=2e-4, atol=1e-12)


def test_flutter_adjoint_matches_eigenvector_method(goland_cs):
    def build(p, V):
        cs = gx.CrossSection.from_isotropic(9.77e6, 0.0, p[0]).with_mass(35.71, 8.64e-4, 8.64, 0.0, p[1])
        return gx.straight_wing(cs, aero=gx.Aero.PETERS, **GOLAND), gx.FlightCondition(velocity=V, rho=1.225)

    p = jnp.array([0.99e6, 0.18288])
    ref = gx.flutter_speed_and_grad(build, p, 120, 150, n_grid=4, tol=1e-8)
    dV, _, _ = tk.flutter_gradient_adjoint(lambda pv: build(pv[0], pv[1]), p, ref.point.velocity, ref.point.eigval)
    np.testing.assert_allclose(dV, np.asarray(ref.dV_dparams), rtol=1e-6)


def test_tracking_follows_modes_through_a_crossing():
    """Lag stiffness swept through the bending frequencies: sorted order changes, tracked labels do not."""

    def build(eil):
        cs = gx.CrossSection.from_isotropic(2e4, eil, 1e4).with_mass(0.75, 1e-4, 0.1)
        return gx.straight_wing(cs, 16.0, 1.0, 0.0, 8, aero=gx.Aero.NONE), gx.FlightCondition()

    m0, f0 = build(4e6)
    ref = tk.reference_modes(m0, gx.modal_analysis(m0, f0), 4)
    ranks, macs = [], []
    for eil in np.geomspace(4e6, 2e4, 12):
        m, f = build(eil)
        tr = tk.track(m, gx.modal_analysis(m, f), ref)
        ranks.append(tr.sorted_rank)
        macs.append(tr.mac)
    assert any(not np.array_equal(r, np.arange(4)) for r in ranks)  # frequencies did cross
    assert np.min(macs) > 0.99  # every tracked mode kept its shape (uncoupled modes)
