r"""Two-dimensional (strip) aerodynamic models, shared by the beam and the typical section.

The formulation follows Aeroelasticity.jl (T. McDonnell, BYU FLOW lab). An
airfoil section is described by its kinematics at the reference point

* ``u`` -- free-stream (chordwise) velocity,
* ``v`` -- normal velocity (``U theta + h_dot``, positive for upward lift),
* ``omega`` -- pitch rate, and the accelerations ``vdot``, ``omegadot``,

and by the section parameters ``a`` (reference point from mid-chord, in
semi-chords), ``b`` (semi-chord), ``a0`` (lift slope), ``alpha0`` (zero-lift
angle), ``cd0`` and ``cm0``. All loads are *per unit span*; ``N`` is the normal
force (~ lift) and ``M`` the moment about the reference point, both divided by
the Prandtl-Glauert factor ``beta``.

Unsteady models carry their own states and residuals:

* Wagner (Jones' two-term approximation of Wagner's function), 2 states,
* Peters' finite-state induced inflow, ``N`` states.

With ``a0 = 2 pi`` and ``beta = 1`` the quasi-steady and Peters loads are
identical to GEBTAero's ``aero_flag = 2, 3``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

# Jones' approximation of Wagner's function
WAGNER_C1, WAGNER_C2, WAGNER_EPS1, WAGNER_EPS2 = 0.165, 0.335, 0.0455, 0.3


def peters_constants(n: int = 6):
    """``(A, b, c)`` matrices of Peters' finite-state inflow theory."""
    D = np.zeros((n, n))
    for m in range(1, n):
        D[m, m - 1] = 1.0 / (2.0 * (m + 1))
    for m in range(2, n + 1):
        D[m - 2, m - 1] = -1.0 / (2.0 * (m - 1))
    b = np.zeros(n)
    for k in range(1, n):
        b[k - 1] = (-1.0) ** (k - 1) * math.factorial(n + k - 1) / math.factorial(n - k - 1) / math.factorial(k) ** 2
    b[n - 1] = (-1.0) ** (n - 1)
    c = 2.0 / np.arange(1, n + 1)
    d = np.zeros(n)
    d[0] = 0.5
    A = D + np.outer(d, b) + np.outer(c, d) + 0.5 * np.outer(c, b)
    return jnp.asarray(A), jnp.asarray(b), jnp.asarray(c)


def _noncirculatory(a, b, rho, u, omega, vdot, omegadot):
    tmp2 = jnp.pi * rho * b**3
    N_nc = tmp2 * (vdot / b + u / b * omega - a * omegadot)
    M_nc = -tmp2 * (vdot / 2 + u * omega + b * (1.0 / 8 - a / 2) * omegadot)
    return N_nc, M_nc


def _finish(N, M, a, b, rho, u, cm0, beta):
    M = M + 2 * rho * b**2 * u**2 * cm0 + (b / 2 + a * b) * N
    return N / beta, M / beta


def steady_loads(a, b, a0, alpha0, cm0, rho, beta, u, v):
    """Steady thin-airfoil theory."""
    N = a0 * rho * b * u * (v - u * alpha0)
    return _finish(N, 0.0, a, b, rho, u, cm0, beta)


def quasisteady_loads(a, b, a0, alpha0, cm0, rho, beta, u, v, omega, vdot, omegadot):
    """Quasi-steady thin-airfoil theory with apparent-mass terms."""
    d = b / 2 - a * b
    N_nc, M_nc = _noncirculatory(a, b, rho, u, omega, vdot, omegadot)
    N = a0 * rho * u * b * (v + d * omega - u * alpha0) + N_nc
    return _finish(N, M_nc, a, b, rho, u, cm0, beta)


def legacy_quasisteady_loads(a, b, rho, u, v, omega):
    """GEBTAero ``aero_flag = 1``: quasi-steady theory without apparent mass."""
    N = 2 * jnp.pi * rho * b * u * (v + b * (0.5 - a) * omega)
    M = b * (0.5 + a) * N - 0.5 * jnp.pi * rho * u * b**3 * omega
    return N, M


def wagner_loads(a, b, a0, alpha0, cm0, rho, beta, u, v, omega, vdot, omegadot, lam):
    d = b / 2 - a * b
    phi0 = 1.0 - WAGNER_C1 - WAGNER_C2
    N_nc, M_nc = _noncirculatory(a, b, rho, u, omega, vdot, omegadot)
    N = a0 * rho * u * b * ((v + d * omega - u * alpha0) * phi0 + lam[0] + lam[1]) + N_nc
    return _finish(N, M_nc, a, b, rho, u, cm0, beta)


def wagner_residual(a, b, alpha0, u, v, omega, lam, lamdot):
    d = b / 2 - a * b
    w = v + d * omega - u * alpha0
    r1 = lamdot[0] + WAGNER_EPS1 * u / b * lam[0] - WAGNER_C1 * WAGNER_EPS1 * u / b * w
    r2 = lamdot[1] + WAGNER_EPS2 * u / b * lam[1] - WAGNER_C2 * WAGNER_EPS2 * u / b * w
    return jnp.stack([r1, r2])


def peters_loads(a, b, a0, alpha0, cm0, rho, beta, bbar, u, v, omega, vdot, omegadot, lam):
    d = b / 2 - a * b
    lam0 = 0.5 * jnp.dot(bbar, lam)
    N_nc, M_nc = _noncirculatory(a, b, rho, u, omega, vdot, omegadot)
    N = a0 * rho * u * b * (v + d * omega - lam0 - u * alpha0) + N_nc
    return _finish(N, M_nc, a, b, rho, u, cm0, beta)


def peters_residual(A, cbar, a, b, u, omega, vdot, omegadot, lam, lamdot):
    return A @ lamdot - cbar * (vdot + u * omega + (b / 2 - a * b) * omegadot) + u / b * lam


def drag_per_span(b, rho, u, cd0):
    """Skin-friction drag per unit span (not affected by Prandtl-Glauert)."""
    return rho * b * u**2 * cd0


def prandtl_glauert(u, speed_of_sound):
    """``sqrt(1 - M^2)``; ``speed_of_sound <= 0`` disables compressibility."""
    mach2 = jnp.where(speed_of_sound > 0, (u / jnp.where(speed_of_sound > 0, speed_of_sound, 1.0)) ** 2, 0.0)
    return jnp.sqrt(1.0 - mach2)
