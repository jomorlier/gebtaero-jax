r"""Two-degree-of-freedom typical section with interchangeable 2D aerodynamics.

Port of Aeroelasticity.jl's ``Section`` structural model, its ``Steady``,
``QuasiSteady``, ``Wagner`` and ``Peters{N}`` aerodynamic models and the
corresponding couplings, built on :mod:`gebtaero_jax.coupled`.

Equations of motion (``h`` positive down, ``theta`` nose up)::

    m h'' + S_theta theta'' + k_h h     = -L
    S_theta h'' + I_theta theta'' + k_theta theta = M

Example
-------
>>> from gebtaero_jax import section as sec
>>> model = sec.section_model(sec.Peters(6))
>>> p = sec.SectionParameters.from_nondimensional(a=-0.2, e=-0.1, mu=20, r2=0.24, sigma=0.4, V=1.0)
>>> V_f, s_f = sec.flutter_reduced_velocity(model, p)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.special

from . import aerodynamics as aero
from ._pytree import pytree_dataclass
from .coupled import CoupledModel, critical_value


# ------------------------------------------------------------------------------ parameters
@pytree_dataclass
class SectionParameters:
    """Same parameters (and order) as Aeroelasticity.jl's section examples."""

    a: float  # reference point from mid-chord (semi-chords)
    b: float  # semi-chord
    a0: float  # lift slope
    alpha0: float  # zero-lift angle
    cd0: float
    cm0: float
    kh: float
    ktheta: float
    m: float
    Stheta: float
    Itheta: float
    U: float
    rho: float
    c: float = 0.0  # speed of sound (0: incompressible)

    @staticmethod
    def from_nondimensional(a=-0.2, e=-0.1, mu=20.0, r2=0.24, sigma=0.4, V=1.0, b=1.0, rho=1.0, omega_theta=1.0,
                            a0=2 * np.pi, alpha0=0.0, cd0=0.0, cm0=0.0, c=0.0) -> SectionParameters:
        """Hodges & Pierce non-dimensional parameters (``V = U / (b omega_theta)``)."""
        m = mu * rho * np.pi * b**2
        Stheta = m * (e - a) * b
        Itheta = r2 * m * b**2
        kh = m * (sigma * omega_theta) ** 2
        ktheta = Itheta * omega_theta**2
        return SectionParameters(a, b, a0, alpha0, cd0, cm0, kh, ktheta, m, Stheta, Itheta, V * b * omega_theta, rho, c)

    @property
    def beta(self):
        return aero.prandtl_glauert(self.U, self.c)


# ------------------------------------------------------------------------------ submodels
@dataclass(frozen=True)
class TypicalSection:
    """States ``(h, theta, hdot, thetadot)``, inputs ``(L, M)``."""

    n_states: int = 4

    def residual(self, dx, x, y, p, t):
        dh, dth, dhd, dthd = dx
        h, th, hd, thd = x
        L, M = y
        return jnp.stack(
            [
                dh - hd,
                dth - thd,
                p.m * dhd + p.Stheta * dthd + p.kh * h + L,
                p.Stheta * dhd + p.Itheta * dthd + p.ktheta * th - M,
            ]
        )


@dataclass(frozen=True)
class Steady:
    n_states: int = 0

    def residual(self, dx, x, y, p, t):
        return jnp.zeros(0)


@dataclass(frozen=True)
class QuasiSteady:
    n_states: int = 0

    def residual(self, dx, x, y, p, t):
        return jnp.zeros(0)


@dataclass(frozen=True)
class Wagner:
    n_states: int = 2

    def residual(self, dx, x, y, p, t):
        u, v, omega = y
        return aero.wagner_residual(p.a, p.b, p.alpha0, u, v, omega, x, dx)


@dataclass(frozen=True)
class Peters:
    n: int = 6

    @property
    def n_states(self) -> int:
        return self.n

    def residual(self, dx, x, y, p, t):
        A, _, c = aero.peters_constants(self.n)
        u, omega, vdot, omegadot = y
        return aero.peters_residual(A, c, p.a, p.b, u, omega, vdot, omegadot, x, dx)


# ------------------------------------------------------------------------------ coupling
def _kinematics(p, x_s, dx_s):
    h, th, hd, thd = x_s
    u, v, omega = p.U, p.U * th + hd, thd
    vdot, omegadot = dx_s[2], dx_s[3]
    return u, v, omega, vdot, omegadot


def section_model(aero_model) -> CoupledModel:
    """Couple an aerodynamic submodel with :class:`TypicalSection` (aero states first)."""
    structure = TypicalSection()

    def coupling(dxs, xs, p, t):
        dx_a, dx_s = dxs
        x_a, x_s = xs
        u, v, omega, vdot, omegadot = _kinematics(p, x_s, dx_s)
        beta = p.beta
        if isinstance(aero_model, Steady):
            N, M = aero.steady_loads(p.a, p.b, p.a0, p.alpha0, p.cm0, p.rho, beta, u, v)
            y_a = ()
        elif isinstance(aero_model, QuasiSteady):
            N, M = aero.quasisteady_loads(p.a, p.b, p.a0, p.alpha0, p.cm0, p.rho, beta, u, v, omega, vdot, omegadot)
            y_a = ()
        elif isinstance(aero_model, Wagner):
            N, M = aero.wagner_loads(p.a, p.b, p.a0, p.alpha0, p.cm0, p.rho, beta, u, v, omega, vdot, omegadot, x_a)
            y_a = (u, v, omega)
        elif isinstance(aero_model, Peters):
            _, bbar, _ = aero.peters_constants(aero_model.n)
            N, M = aero.peters_loads(p.a, p.b, p.a0, p.alpha0, p.cm0, p.rho, beta, bbar, u, v, omega, vdot, omegadot, x_a)
            y_a = (u, omega, vdot, omegadot)
        else:
            raise TypeError(f"unknown aerodynamic model {aero_model!r}")
        return y_a, (N, M)

    return CoupledModel((aero_model, structure), coupling, lambda p: (p, p))


def flutter_reduced_velocity(model: CoupledModel, p: SectionParameters, v_min=0.05, v_max=4.0, n_grid=40):
    """Flutter reduced velocity ``U_f / (b omega_theta)`` (``omega_theta = sqrt(k_theta / I_theta)``)."""
    wt = float(np.sqrt(p.ktheta / p.Itheta))
    scale = float(p.b) * wt
    V, s = critical_value(model, lambda V: p.replace(U=V * scale), v_min, v_max, n_grid, "flutter")
    return V, (s / wt if s is not None and np.all(np.isfinite(s)) else s)


def divergence_reduced_velocity(model: CoupledModel, p: SectionParameters, v_min=0.05, v_max=4.0, n_grid=40):
    wt = float(np.sqrt(p.ktheta / p.Itheta))
    scale = float(p.b) * wt
    V, _ = critical_value(model, lambda V: p.replace(U=V * scale), v_min, v_max, n_grid, "divergence")
    return V


# ------------------------------------------------------------------------------ reference
def theodorsen_C(k):
    """Theodorsen's function ``C(k)``."""
    H1 = scipy.special.hankel2(1, k)
    H0 = scipy.special.hankel2(0, k)
    return H1 / (H1 + 1j * H0)


def theodorsen_flutter(p: SectionParameters, guess=(2.0, 0.5)):
    """Exact (Theodorsen) flutter point ``(V, omega/omega_theta)`` of the typical section.

    Independent frequency-domain reference used to verify the time-domain
    state-space models: solves ``det A(i omega, U) = 0`` for real ``omega, U``.
    """
    a, b, m, S, I, kh, kt, rho = (float(v) for v in (p.a, p.b, p.m, p.Stheta, p.Itheta, p.kh, p.ktheta, p.rho))
    wt = np.sqrt(kt / I)

    def det(z):
        V, w = z[0], z[1] * wt
        U = V * b * wt
        k = w * b / U
        C = theodorsen_C(k)
        s = 1j * w
        d = b / 2 - a * b
        # loads for unit h and unit theta (columns), v = U theta + h_dot, omega = theta_dot
        def loads(h, th):
            v, om = U * th + s * h, s * th
            vd, omd = s * s * h, s * s * th
            N = 2 * np.pi * rho * U * b * C * (v + d * om) + np.pi * rho * b**3 * (vd / b + U / b * om - a * omd)
            M = -np.pi * rho * b**3 * (vd / 2 + U * om + b * (1 / 8 - a / 2) * omd) + (b / 2 + a * b) * N
            return N, M

        Nh, Mh = loads(1.0, 0.0)
        Nt, Mt = loads(0.0, 1.0)
        A = np.array([[s * s * m + kh + Nh, s * s * S + Nt], [s * s * S - Mh, s * s * I + kt - Mt]])
        D = np.linalg.det(A) / (m * I * wt**4)
        return [D.real, D.imag]

    sol = scipy.optimize.fsolve(det, guess, full_output=True)
    return float(sol[0][0]), float(sol[0][1])
