r"""Modular monolithic coupling of implicit models (Aeroelasticity.jl's ``CoupledModel`` in JAX).

Each submodel ``i`` is an implicit first-order system

.. math:: 0 = f_i(\dot x_i, x_i, y_i, p_i, t)

with inputs ``y_i``. A coupling function defines the inputs from the rates,
states and parameters of all submodels,

.. math:: (y_1, \dots, y_n) = G(\dot x, x, p, t),

giving the monolithic residual ``F(xdot, x, p, t) = 0``. JAX provides the
state and rate Jacobians (``K = dF/dx``, ``M = dF/dxdot``) exactly, and every
analysis below -- steady state, eigenvalues, flutter search, time marching --
is differentiable with respect to the parameters ``p`` (any pytree).

Compared with Aeroelasticity.jl, sparsity detection/colouring is unnecessary
for the small dense systems targeted here and ``DifferentialEquations.jl`` is
replaced by a compiled generalised-alpha integrator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.optimize

from .integrators import generalized_alpha


class Submodel(Protocol):
    n_states: int

    def residual(self, dx, x, y, p, t): ...


@dataclass(frozen=True)
class CoupledModel:
    """Monolithic coupled model.

    Parameters
    ----------
    submodels : sequence of objects with ``n_states`` and ``residual(dx, x, y, p, t)``
    coupling : ``coupling(dxs, xs, p, t) -> ys`` where ``dxs``/``xs`` are tuples
        of per-submodel rate/state vectors and ``ys`` a tuple of inputs
    params : ``params(p) -> (p_1, ..., p_n)`` extracts each submodel's
        parameters from the global parameter pytree ``p``
    """

    submodels: tuple
    coupling: Callable
    params: Callable

    def __hash__(self):
        return id(self)

    @property
    def n_states(self) -> int:
        return sum(m.n_states for m in self.submodels)

    @property
    def indices(self) -> list[slice]:
        out, k = [], 0
        for m in self.submodels:
            out.append(slice(k, k + m.n_states))
            k += m.n_states
        return out

    def split(self, x):
        return tuple(x[s] for s in self.indices)

    def residual(self, dx, x, p, t=0.0):
        dxs, xs = self.split(dx), self.split(x)
        ys = self.coupling(dxs, xs, p, t)
        ps = self.params(p)
        res = [m.residual(dxi, xi, yi, pi, t) for m, dxi, xi, yi, pi in zip(self.submodels, dxs, xs, ys, ps)]
        return jnp.concatenate([jnp.atleast_1d(r) for r in res if jnp.size(r) > 0])

    def __call__(self, dx, x, p, t=0.0):
        return self.residual(dx, x, p, t)


# ------------------------------------------------------------------------------ analyses
@partial(jax.jit, static_argnums=0)
def linearize(model: CoupledModel, x, p, dx=None, t=0.0):
    """State and rate Jacobians ``K = dF/dx``, ``M = dF/dxdot`` (Aeroelasticity.jl ``linearize``)."""
    dx = jnp.zeros_like(x) if dx is None else dx
    K = jax.jacfwd(model.residual, argnums=1)(dx, x, p, t)
    M = jax.jacfwd(model.residual, argnums=0)(dx, x, p, t)
    return K, M


@partial(jax.jit, static_argnums=(0, 3))
def steady_state(model: CoupledModel, p, x0=None, maxiter: int = 50):
    """Steady state ``F(0, x, p) = 0`` (Newton, implicitly differentiable)."""
    x0 = jnp.zeros(model.n_states) if x0 is None else x0
    f = lambda x: model.residual(jnp.zeros_like(x), x, p, 0.0)

    def solve(g, y0):
        def body(_, y):
            return y + jnp.linalg.solve(jax.jacfwd(g)(y), -g(y))

        return jax.lax.fori_loop(0, maxiter, body, y0)

    tangent_solve = lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(y), y)
    x = jax.lax.custom_root(f, x0, solve, tangent_solve)
    # singular Jacobian (e.g. inflow states at U = 0): keep the initial guess
    return jnp.where(jnp.all(jnp.isfinite(x)), x, x0)


def dense_eigen(K, M, max_abs: float = 1e8):
    """Eigenvalues ``s`` of ``K v + s M v = 0`` with left (rows of ``U``) and right eigenvectors.

    Left eigenvectors are normalised so that ``U M V = I`` (as in
    Aeroelasticity.jl). Infinite eigenvalues (singular ``M``) are removed and
    the modes are sorted by ascending frequency.
    """
    from .solvers import equilibrated_eig

    K, M = np.asarray(K), np.asarray(M)
    w, vl, vr = equilibrated_eig(K, M)
    keep = np.isfinite(w) & (np.abs(w) < max_abs)
    w, vl, vr = w[keep], vl[:, keep], vr[:, keep]
    order = np.lexsort((w.real, np.abs(w.imag)))
    w, vl, vr = w[order], vl[:, order], vr[:, order]
    U = vl.conj().T
    scale = np.einsum("ij,jk,ki->i", U, M, vr)
    U = U / scale[:, None]
    return w, U, vr


def correlate_eigenmodes(U_prev, M, V, tracked=None):
    """Mode correlation through ``C = U_prev M V`` (Aeroelasticity.jl ``correlate_eigenmodes``).

    Returns the permutation ``perm`` such that ``V[:, perm]`` follows the
    previous modes, and a corruption index (second best / best correlation;
    close to 1 means an ambiguous match). Uses an optimal assignment instead
    of the greedy pass.
    """
    C = np.abs(U_prev @ np.asarray(M) @ V)
    n_prev, n_new = C.shape
    rows = np.arange(n_prev) if tracked is None else np.asarray(tracked)
    r, c = scipy.optimize.linear_sum_assignment(-C[rows])
    perm = np.empty(len(rows), dtype=int)
    perm[r] = c
    srt = np.sort(C[rows], axis=1)
    corruption = srt[:, -2] / np.maximum(srt[:, -1], 1e-300) if n_new > 1 else np.zeros(len(rows))
    return perm, corruption


@dataclass
class EigenSweep:
    parameter: np.ndarray
    eigvals: np.ndarray  # (n_points, n_modes)

    @property
    def freq(self):
        return self.eigvals.imag

    @property
    def damping(self):
        return self.eigvals.real


def eigen_sweep(model: CoupledModel, params_at: Callable[[float], Any], values, x0=None, steady=True) -> EigenSweep:
    """Eigenvalues along a parameter sweep with left-eigenvector mode correlation.

    ``params_at(v)`` returns the parameter pytree for the sweep value ``v``.
    """
    values = np.asarray(values, dtype=float)
    out = []
    U_prev = None
    x = jnp.zeros(model.n_states) if x0 is None else x0
    for v in values:
        p = params_at(v)
        if steady:
            x = steady_state(model, p, x)
        K, M = linearize(model, x, p)
        w, U, V = dense_eigen(K, M)
        if U_prev is not None and len(w) == U_prev.shape[0]:
            perm, _ = correlate_eigenmodes(U_prev, M, V)
            w, U, V = w[perm], U[perm], V[:, perm]
        out.append(w)
        U_prev = U
    n = min(len(w) for w in out)
    return EigenSweep(values, np.array([w[:n] for w in out]))


def critical_value(
    model: CoupledModel,
    params_at: Callable[[float], Any],
    v_min: float,
    v_max: float,
    n_grid: int = 40,
    kind: str = "flutter",
    imag_tol: float = 1e-6,
    tol: float = 1e-8,
    steady: bool = True,
    rel_noise: float = 1e-8,
):
    """Smallest sweep value where an oscillatory (``kind='flutter'``) or real
    (``kind='divergence'``) eigenvalue crosses into the right half-plane.

    Real parts below ``rel_noise * |s|`` count as neutral (undamped roots of
    e.g. steady aerodynamics are otherwise flagged by round-off).
    Returns ``(value, eigenvalue)`` or ``(nan, nan)``.
    """

    def growth(v):
        p = params_at(v)
        x = steady_state(model, p) if steady else jnp.zeros(model.n_states)
        w, _, _ = dense_eigen(*linearize(model, x, p))
        sel = w.imag > imag_tol if kind == "flutter" else np.abs(w.imag) <= imag_tol
        if not np.any(sel):
            return -1.0, None
        idx = np.where(sel)[0]
        # subtract a tiny relative threshold so undamped (Re = 0) roots are not flagged
        g = w.real[idx] - rel_noise * np.maximum(np.abs(w[idx]), 1.0)
        j = idx[np.argmax(g)]
        return g.max(), w[j]

    vals = np.linspace(v_min, v_max, n_grid)
    g_prev = growth(vals[0])[0]
    for v0, v1 in zip(vals[:-1], vals[1:]):
        g1 = growth(v1)[0]
        if g_prev <= 0 < g1:
            vc = scipy.optimize.brentq(lambda v: growth(v)[0], v0, v1, xtol=tol)
            return vc, growth(vc)[1]
        g_prev = g1
    return np.nan, np.nan


def simulate(
    model: CoupledModel,
    p,
    x0,
    t_end: float,
    dt: float,
    dx0=None,
    rho_inf: float = 0.9,
    newton_iter: int = 4,
):
    """Time marching with the generalised-alpha method (replaces ``DifferentialEquations.jl``)."""
    fun = lambda xn, xdn, t, prm: model.residual(xdn, xn, prm, t)
    n_steps = int(round(t_end / dt))
    x0 = jnp.asarray(x0, dtype=float)
    dx0 = jnp.zeros_like(x0) if dx0 is None else dx0
    xs = generalized_alpha(fun, x0, dx0, p, dt, n_steps, rho_inf, newton_iter)
    t = np.arange(n_steps + 1) * dt
    return t, np.concatenate([np.asarray(x0)[None], np.asarray(xs)])


def concatenate_submodels(submodels: Sequence) -> int:
    return sum(m.n_states for m in submodels)


def critical_value_gradient(model: CoupledModel, params_fn: Callable, design, value: float, s_c: complex, steady=True):
    """Gradient of a critical value (flutter/divergence) w.r.t. ``design``.

    ``params_fn(design, v) -> p``. With ``K v + s M v = 0`` and left vectors
    normalised ``u M v = 1``: ``ds = -u (dK + s dM) v``; at the critical point
    ``Re s = 0`` hence ``dv*/d design = -Re(ds/d design) / Re(ds/dv)``.
    Returns ``(dvalue_ddesign, dimag_ddesign)``.
    """
    v0 = jnp.asarray(value, dtype=float)
    p = params_fn(design, v0)
    x = steady_state(model, p) if steady else jnp.zeros(model.n_states)
    w, U, V = dense_eigen(*linearize(model, x, p))
    j = int(np.argmin(np.abs(w - s_c)))
    s, u, vr = w[j], jnp.asarray(U[j]), jnp.asarray(V[:, j])

    def ds(d, v):
        prm = params_fn(d, v)
        xx = steady_state(model, prm, x) if steady else jnp.zeros(model.n_states)
        K = jax.jacfwd(model.residual, argnums=1)(jnp.zeros_like(xx), xx, prm, 0.0)
        M = jax.jacfwd(model.residual, argnums=0)(jnp.zeros_like(xx), xx, prm, 0.0)
        val = -(u @ ((K + s * M) @ vr))
        return jnp.stack([val.real, val.imag])

    J_d, J_v = jax.jacrev(ds, argnums=(0, 1))(design, v0)
    dvalue = jax.tree_util.tree_map(lambda g: -g[0] / J_v[0], J_d)
    dimag = jax.tree_util.tree_map(lambda g, dv: g[1] + J_v[1] * dv, J_d, dvalue)
    return dvalue, dimag
