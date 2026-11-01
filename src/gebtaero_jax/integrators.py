"""Implicit time integration shared by the beam and the coupled-model framework."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def generalized_alpha_coefficients(rho_inf: float):
    """First-order generalised-alpha parameters (Jansen, Whiting & Hulbert 2000).

    ``rho_inf = 1`` recovers the (non-dissipative) mid-point/trapezoidal family;
    smaller values damp spurious high-frequency content while keeping
    second-order accuracy.
    """
    alpha_m = 0.5 * (3.0 - rho_inf) / (1.0 + rho_inf)
    alpha_f = 1.0 / (1.0 + rho_inf)
    gamma = 0.5 + alpha_m - alpha_f
    return alpha_m, alpha_f, gamma


@partial(jax.jit, static_argnames=("fun", "n_steps", "rho_inf", "newton_iter"))
def generalized_alpha(fun, x0, xd0, params, dt, n_steps: int, rho_inf: float = 0.9, newton_iter: int = 4):
    """Integrate ``fun(x, xdot, t, params) = 0`` and return the states at steps ``1..n_steps``.

    The loop is a ``lax.scan`` with a fixed number of Newton iterations per
    step (exact AD Jacobian), so the whole trajectory is differentiable with
    respect to ``params``, ``x0`` and ``dt``.
    """
    am, af, gam = generalized_alpha_coefficients(rho_inf)

    def step(carry, k):
        x, xd = carry
        t = (k + af) * dt

        def xd_new(xn):
            return xd + (xn - x - dt * xd) / (gam * dt)

        def f(xn):
            return fun(x + af * (xn - x), xd + am * (xd_new(xn) - xd), t, params)

        def body(_, xn):
            return xn + jnp.linalg.solve(jax.jacfwd(f)(xn), -f(xn))

        xn = jax.lax.fori_loop(0, newton_iter, body, x + dt * xd)
        return (xn, xd_new(xn)), xn

    _, xs = jax.lax.scan(step, (x0, xd0), jnp.arange(n_steps))
    return xs
