"""Materials, plies and Classical Laminate Theory (CLT).

Everything is written with ``jax.numpy`` so laminate stiffnesses are
differentiable with respect to ply angles, thicknesses and material constants.
This is what enables gradient-based aeroelastic tailoring.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from ._pytree import pytree_dataclass


@pytree_dataclass
class IsoMaterial:
    """Isotropic material.

    Parameters
    ----------
    E : Young's modulus (Pa)
    nu : Poisson ratio
    rho : density (kg/m^3)
    """

    E: float
    nu: float
    rho: float = 0.0

    @property
    def G(self):
        return self.E / (2.0 * (1.0 + self.nu))

    def q_matrix(self) -> jnp.ndarray:
        return _q_matrix(self.E, self.E, self.nu, self.G)


@pytree_dataclass
class OrthoMaterial:
    """Orthotropic (transversely isotropic) ply material.

    Parameters
    ----------
    E1 : modulus in the fibre direction (Pa)
    E2 : transverse modulus (Pa)
    nu12 : major Poisson ratio
    G12 : in-plane shear modulus (Pa)
    rho : density (kg/m^3)
    """

    E1: float
    E2: float
    nu12: float
    G12: float
    rho: float = 0.0

    def q_matrix(self) -> jnp.ndarray:
        return _q_matrix(self.E1, self.E2, self.nu12, self.G12)


def _q_matrix(E1, E2, nu12, G12) -> jnp.ndarray:
    """Plane-stress reduced stiffness of a ply in its material axes."""
    nu21 = nu12 * E2 / E1
    den = 1.0 - nu12 * nu21
    return jnp.array(
        [
            [E1 / den, nu12 * E2 / den, 0.0],
            [nu12 * E2 / den, E2 / den, 0.0],
            [0.0, 0.0, G12],
        ]
    )


def q_bar(Q: jnp.ndarray, angle_rad) -> jnp.ndarray:
    """Rotate the reduced stiffness ``Q`` by ``angle_rad`` (fibre angle from the x axis)."""
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    c2, s2 = c * c, s * s
    Qb11 = Q11 * c2 * c2 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * s2 * s2
    Qb22 = Q11 * s2 * s2 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * c2 * c2
    Qb12 = (Q11 + Q22 - 4 * Q66) * s2 * c2 + Q12 * (s2 * s2 + c2 * c2)
    Qb66 = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s2 * c2 + Q66 * (s2 * s2 + c2 * c2)
    Qb16 = (Q11 - Q12 - 2 * Q66) * s * c2 * c + (Q12 - Q22 + 2 * Q66) * s2 * s * c
    Qb26 = (Q11 - Q12 - 2 * Q66) * s2 * s * c + (Q12 - Q22 + 2 * Q66) * s * c2 * c
    return jnp.array(
        [
            [Qb11, Qb12, Qb16],
            [Qb12, Qb22, Qb26],
            [Qb16, Qb26, Qb66],
        ]
    )


@pytree_dataclass
class Ply:
    """A ply: material, thickness (m) and fibre orientation (degrees, from the span axis)."""

    material: IsoMaterial | OrthoMaterial
    thickness: float
    angle_deg: float = 0.0


@pytree_dataclass
class Laminate:
    """Stack of plies, listed from the bottom (``-z``) to the top (``+z``) face."""

    plies: tuple

    @staticmethod
    def from_plies(plies: Sequence[Ply]) -> Laminate:
        return Laminate(tuple(plies))

    def with_angles(self, angles_deg) -> Laminate:
        """Return a copy with new ply angles (handy for optimisation)."""
        return Laminate(tuple(p.replace(angle_deg=a) for p, a in zip(self.plies, angles_deg)))

    def z_interfaces(self) -> jnp.ndarray:
        t = jnp.stack([jnp.asarray(p.thickness, dtype=float) for p in self.plies])
        h = jnp.sum(t)
        return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(t)]) - 0.5 * h

    @property
    def thickness(self):
        return sum(p.thickness for p in self.plies)

    def abd(self) -> jnp.ndarray:
        """6x6 ABD matrix ordered (eps_x, eps_y, gamma_xy, kappa_x, kappa_y, kappa_xy)."""
        z = self.z_interfaces()
        A = jnp.zeros((3, 3))
        B = jnp.zeros((3, 3))
        D = jnp.zeros((3, 3))
        for k, ply in enumerate(self.plies):
            Qb = q_bar(ply.material.q_matrix(), jnp.deg2rad(ply.angle_deg))
            z0, z1 = z[k], z[k + 1]
            A = A + Qb * (z1 - z0)
            B = B + Qb * (z1**2 - z0**2) / 2.0
            D = D + Qb * (z1**3 - z0**3) / 3.0
        return jnp.block([[A, B], [B, D]])

    def mass_properties(self):
        """Return (areal mass, first moment, second moment) through the thickness."""
        z = self.z_interfaces()
        m0 = m1 = m2 = 0.0
        for k, ply in enumerate(self.plies):
            rho = ply.material.rho
            z0, z1 = z[k], z[k + 1]
            m0 = m0 + rho * (z1 - z0)
            m1 = m1 + rho * (z1**2 - z0**2) / 2.0
            m2 = m2 + rho * (z1**3 - z0**3) / 3.0
        return m0, m1, m2
