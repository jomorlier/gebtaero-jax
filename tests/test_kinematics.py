import jax
import jax.numpy as jnp
import numpy as np

import gebtaero_jax as gx
from gebtaero_jax.rotations import rodrigues_from_axis_angle, rodrigues_to_angle, rotation_rodrigues


def test_rotation_is_orthonormal_and_matches_axis_angle():
    th = rodrigues_from_axis_angle(jnp.array([0.3, -1.0, 0.5]), 1.2)
    C = rotation_rodrigues(th)
    np.testing.assert_allclose(C @ C.T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(C), 1.0, atol=1e-14)
    np.testing.assert_allclose(rodrigues_to_angle(th), 1.2, atol=1e-14)


def test_frame_from_axis_matches_gebtaero_convention():
    C = np.asarray(gx.Frame.from_axis([0, 1, 0]).Cab)
    np.testing.assert_allclose(C, [[0, 1, 0], [1, 0, 0], [0, 0, -1]], atol=1e-15)


def test_undeformed_state_is_an_equilibrium(patil_cs):
    m = gx.straight_wing(patil_cs, 16.0, 1.0, n_elements=5)
    x = jnp.zeros(m.n_dof)
    r = gx.residual(x, x, m, gx.FlightCondition(velocity=10.0, rho=0.0889))
    assert float(jnp.abs(r).max()) < 1e-14


def test_peters_constants_match_aeroelasticity_jl_definition():
    A, b, c = gx.aerodynamics.peters_constants(4)
    np.testing.assert_allclose(c, [2, 1, 2 / 3, 0.5])
    np.testing.assert_allclose(b[-1], -1.0)
    assert A.shape == (4, 4)


def test_clt_isotropic_plate():
    alu = gx.IsoMaterial(73e9, 0.3, 2700.0)
    cs = gx.CrossSection.composite_plate(0.2, gx.Laminate.from_plies([gx.Ply(alu, 2e-3, 0.0)]))
    k = cs.stiffness_summary
    np.testing.assert_allclose(k["GJ"], alu.G * 0.2 * 2e-3**3 / 3, rtol=1e-12)
    np.testing.assert_allclose(k["EI2"], 73e9 * 0.2 * 2e-3**3 / 12, rtol=1e-12)
    np.testing.assert_allclose(cs.mass[0, 0], 2700 * 0.2 * 2e-3)


def test_bend_twist_coupling_symmetries():
    """Cross-ply has no bend-twist coupling; mirroring all ply angles flips its sign."""
    t300 = gx.OrthoMaterial(181e9, 10.3e9, 0.28, 7.17e9, 1600.0)

    def K(angles):
        lam = gx.Laminate.from_plies([gx.Ply(t300, 1e-4, a) for a in angles])
        S = jnp.linalg.pinv(gx.CrossSection.composite_plate(0.1, lam).flexibility)
        return float(S[3, 4]), float(S[3, 3])

    k0, gj = K((0.0, 90.0, 90.0, 0.0))
    assert abs(k0) < 1e-9 * gj
    kp, _ = K((30.0, 30.0, 30.0))
    km, _ = K((-30.0, -30.0, -30.0))
    assert abs(kp) > 1e-3 * gj
    np.testing.assert_allclose(kp, -km, rtol=1e-10)


def test_clt_gradient_wrt_ply_angle_matches_fd():
    t300 = gx.OrthoMaterial(181e9, 10.3e9, 0.28, 7.17e9, 1600.0)

    def coupling(theta):
        lam = gx.Laminate.from_plies([gx.Ply(t300, 1e-4, theta), gx.Ply(t300, 1e-4, theta)])
        return jnp.linalg.pinv(gx.CrossSection.composite_plate(0.1, lam).flexibility)[3, 4]

    g = jax.grad(coupling)(20.0)
    fd = (coupling(20.0 + 1e-4) - coupling(20.0 - 1e-4)) / 2e-4
    np.testing.assert_allclose(g, fd, rtol=1e-5)
