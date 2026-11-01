"""Beam cross-section: 6x6 flexibility and 6x6 mass matrices (GEBT conventions).

Generalised strains are ordered ``(gamma_11, 2 gamma_12, 2 gamma_13, kappa_1,
kappa_2, kappa_3)`` in the cross-section frame ``b`` where ``x_b`` is the beam
axis, ``y_b`` points towards the leading edge and ``z_b`` completes the
right-handed triad. ``kappa_1`` is torsion, ``kappa_2`` flap (span-wise)
bending and ``kappa_3`` lag (chord-wise) bending.

A zero diagonal term in the flexibility matrix means the corresponding strain
is suppressed (rigid). This is natural in GEBT's mixed formulation.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._pytree import pytree_dataclass
from .materials import IsoMaterial, Laminate


def _safe_inv(x):
    x = jnp.asarray(x, dtype=float)
    return jnp.where(x > 0, 1.0 / jnp.where(x > 0, x, 1.0), 0.0)


def mass_matrix(mu, i22, i33, i23=0.0, nu=0.0, zcg=0.0) -> jnp.ndarray:
    """Cross-section mass matrix, identical to GEBTAero's ``SetMassMatrix``.

    Parameters
    ----------
    mu : mass per unit length (kg/m)
    i22 : mass moment of inertia about ``y_b`` (flap bending), kg.m
    i33 : mass moment of inertia about ``z_b`` (lag bending), kg.m
    i23 : product of inertia
    nu : distance between the reference axis and the centre of gravity,
        positive when the CG is *behind* the reference axis (``y_cg = -nu``)
    zcg : ``z_b`` coordinate of the CG
    """
    ycg = -jnp.asarray(nu, dtype=float)
    M = jnp.zeros((6, 6))
    M = M.at[0, 0].set(mu).at[1, 1].set(mu).at[2, 2].set(mu)
    M = M.at[3, 3].set(i22 + i33).at[4, 4].set(i22).at[5, 5].set(i33)
    M = M.at[4, 5].set(-i23).at[5, 4].set(-i23)
    M = M.at[0, 5].set(-mu * ycg).at[5, 0].set(-mu * ycg)
    M = M.at[2, 3].set(mu * ycg).at[3, 2].set(mu * ycg)
    M = M.at[0, 4].set(mu * zcg).at[4, 0].set(mu * zcg)
    M = M.at[1, 3].set(-mu * zcg).at[3, 1].set(-mu * zcg)
    return M


@pytree_dataclass
class CrossSection:
    """Sectional flexibility (compliance) and mass matrices."""

    flexibility: jnp.ndarray
    mass: jnp.ndarray

    # ------------------------------------------------------------------ builders
    @staticmethod
    def from_isotropic(EI2, EI3=0.0, GJ=0.0) -> CrossSection:
        """Flexibility from classical stiffnesses (``SetFlexibilityMatrixByIsotropicValues``).

        A zero stiffness suppresses the corresponding degree of freedom.
        """
        F = jnp.zeros((6, 6))
        F = F.at[3, 3].set(_safe_inv(GJ)).at[4, 4].set(_safe_inv(EI2)).at[5, 5].set(_safe_inv(EI3))
        return CrossSection(F, jnp.eye(6))

    @staticmethod
    def from_stiffness(stiffness: jnp.ndarray, active=None) -> CrossSection:
        """Build from a 6x6 stiffness matrix; ``active`` lists the non-rigid strain indices."""
        stiffness = jnp.asarray(stiffness, dtype=float)
        if active is None:
            active = [i for i in range(6)]
        idx = jnp.array(active)
        Fsub = jnp.linalg.inv(stiffness[jnp.ix_(idx, idx)])
        F = jnp.zeros((6, 6)).at[jnp.ix_(idx, idx)].set(Fsub)
        return CrossSection(F, jnp.eye(6))

    @staticmethod
    def rectangular(h, w, material: IsoMaterial, rigid_lag=False, neglect_i22=False) -> CrossSection:
        """Solid isotropic rectangle ``h`` (thickness, along z_b) x ``w`` (width, along y_b).

        Torsion constant from Roark (as in GEBTAero).
        """
        E, nu, rho = material.E, material.nu, material.rho
        G = E / (2.0 * (1.0 + nu))
        Ig2 = w * h**3 / 12.0
        Ig3 = 0.0 if rigid_lag else w**3 * h / 12.0
        a, b = 0.5 * w, 0.5 * h
        J = a * b**3 * (16.0 / 3.0 - 3.36 * b / a * (1.0 - b**4 / (12.0 * a**4)))
        cs = CrossSection.from_isotropic(E * Ig2, E * Ig3, G * J)
        i33 = rho * w**3 * h / 12.0
        i22 = 1e-4 * i33 if neglect_i22 else rho * w * h**3 / 12.0
        return cs.with_mass(rho * w * h, i22, i33)

    @staticmethod
    def composite_plate(
        chord,
        laminate: Laminate,
        rigid_extension: bool = True,
        rigid_lag: bool = True,
    ) -> CrossSection:
        """Thin laminated flat-plate wing, condensed from Classical Laminate Theory.

        Differentiable replacement of GEBTAero's periodic 3D-FEM (CalculiX)
        homogenisation for plate-like wings. The free edges are modelled with
        ``N_y = N_xy = M_y = 0`` and the plate kinematics are mapped onto the
        beam strains with ``eps_x = gamma_11``, ``kappa_x = kappa_2`` and
        ``kappa_xy = -2 kappa_1``. This yields the classical (Weisshaar)
        bending/torsion/bend-twist coupling stiffnesses::

            EI = c (D11 - D12^2/D22),  GJ = 4 c (D66 - D26^2/D22),
            K  = -2 c (D16 - D12 D26/D22)

        The reference axis is at mid-chord, on the laminate mid-plane.
        """
        ABD = laminate.abd()
        keep = jnp.array([0, 3, 5])  # eps_x, kappa_x, kappa_xy
        cond = jnp.array([1, 2, 4])  # eps_y, gamma_xy, kappa_y  (free: N = 0)
        Skk = ABD[jnp.ix_(keep, keep)]
        Skc = ABD[jnp.ix_(keep, cond)]
        Scc = ABD[jnp.ix_(cond, cond)]
        Dc = Skk - Skc @ jnp.linalg.solve(Scc, Skc.T)
        # plate strains = T @ (gamma_11, kappa_1, kappa_2)
        T = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -2.0, 0.0]])
        S3 = chord * T.T @ Dc @ T
        S = jnp.zeros((6, 6))
        idx3 = jnp.array([0, 3, 4])
        S = S.at[jnp.ix_(idx3, idx3)].set(S3)
        active = [0, 3, 4]
        if rigid_extension:
            active = [3, 4]
        if not rigid_lag:
            # in-plane (lag) bending of the strip: EI3 = c^3/12 * reduced A11
            S = S.at[5, 5].set(chord**3 / 12.0 * Dc[0, 0])
            active = active + [5]
        cs = CrossSection.from_stiffness(S, active)
        m0, m1, m2 = laminate.mass_properties()
        mu = chord * m0
        zcg = m1 / m0
        i22 = chord * m2
        i33 = mu * chord**2 / 12.0
        return cs.with_mass(mu, i22, i33, 0.0, 0.0, zcg)

    # ------------------------------------------------------------------ mass
    def with_mass(self, mu, i22, i33, i23=0.0, nu=0.0, zcg=0.0) -> CrossSection:
        """Return a copy with the GEBTAero-style mass matrix (see :func:`mass_matrix`)."""
        return CrossSection(self.flexibility, mass_matrix(mu, i22, i33, i23, nu, zcg))

    def with_mass_matrix(self, M) -> CrossSection:
        return CrossSection(self.flexibility, jnp.asarray(M, dtype=float))

    # ------------------------------------------------------------------ helpers
    @property
    def mass_per_length(self):
        return self.mass[0, 0]

    @property
    def stiffness_summary(self) -> dict:
        """Equivalent GJ, EI2, EI3 (inverse of diagonal flexibility terms)."""
        f = jnp.diag(self.flexibility)
        return {"GJ": _safe_inv(f[3]), "EI2": _safe_inv(f[4]), "EI3": _safe_inv(f[5])}
