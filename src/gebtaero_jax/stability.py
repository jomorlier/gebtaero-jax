r"""Aeroelastic stability: flutter/divergence speeds, V-g sweeps and their gradients.

Critical speeds are found with a coarse velocity scan followed by Brent's
method (flutter, on the largest real part of the oscillatory roots) or
bisection on ``sign(det K)`` (divergence, a real root crossing zero).

Gradients use AD and the implicit function theorem. For a simple eigenvalue
``s`` of the pencil ``(-K, M)`` with right/left eigenvectors ``phi``/``psi``

.. math:: ds = \frac{\psi^H(-dK - s\, dM)\phi}{\psi^H M \phi}

and at the flutter point ``Re s(V_f, p) = 0`` hence

.. math:: \frac{dV_f}{dp} = -\frac{\partial\, \mathrm{Re}\, s / \partial p}{\partial\, \mathrm{Re}\, s / \partial V}.

``dK``/``dM`` come from :func:`jax.grad` through ``jacfwd`` of the residual --
and through the (implicitly differentiated) nonlinear equilibrium when the
wing is pre-deformed by gravity or angle of attack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from scipy.optimize import linear_sum_assignment

from .beam import FlightCondition, tangent_matrices
from .solvers import ModalResult, _tangent_jit, modal_analysis, solve_static, static_residual_norm
from .wing import BeamModel

_REL_NOISE = 1e-8

Builder = Callable[[Any, float], "tuple[BeamModel, FlightCondition]"]


def _needs_equilibrium(model: BeamModel, flight: FlightCondition) -> bool:
    vals = [flight.gravity, flight.alpha, *np.asarray(flight.tip_force), *np.asarray(flight.tip_moment)]
    vals += [*np.asarray(flight.dist_force), *np.asarray(flight.dist_moment)]
    return any(abs(float(v)) > 0 for v in vals)


class _Evaluator:
    """Caches equilibria along a velocity sweep (continuation on the initial guess)."""

    def __init__(self, model, flight, deformed, n_load_steps):
        self.model, self.flight = model, flight
        self.deformed = _needs_equilibrium(model, flight) if deformed is None else deformed
        self.n_load_steps = n_load_steps
        self._x = None

    def equilibrium(self, V):
        """Static equilibrium at speed ``V`` (``None`` if no equilibrium could be found)."""
        fl = self.flight.replace(velocity=V)
        if not self.deformed:
            return jnp.zeros(self.model.n_dof), fl
        guesses = [] if self._x is None else [(self._x, 1)]
        guesses.append((None, max(self.n_load_steps, 4)))
        for x0, steps in guesses:
            x = solve_static(self.model, fl, x0=x0, n_load_steps=steps if x0 is None else 1)
            if bool(jnp.all(jnp.isfinite(x))) and static_residual_norm(self.model, fl, x) < 1e-6:
                self._x = x
                return x, fl
        return None, fl  # no static equilibrium (beyond static divergence / limit point)

    def modes(self, V) -> ModalResult | None:
        x, fl = self.equilibrium(V)
        if x is None:
            return None
        return modal_analysis(self.model, fl, x)


def _flutter_growth(res: ModalResult | None, imag_tol, fmax_hz):
    if res is None:  # no equilibrium: treat as unstable
        return np.inf, None
    idx = res.oscillatory(imag_tol)
    if fmax_hz is not None:
        idx = idx[res.freq_hz[idx] <= fmax_hz]
    if len(idx) == 0:
        return -np.inf, None
    # real parts below 1e-8 |s| are round-off on neutrally stable (spurious) roots
    g = res.eigvals.real[idx] - _REL_NOISE * np.maximum(np.abs(res.eigvals[idx]), 1.0)
    k = int(np.argmax(g))
    return g[k], idx[k]


@dataclass
class CriticalPoint:
    """Result of a critical speed search."""

    kind: str  # "flutter" or "divergence"
    velocity: float
    frequency_hz: float
    eigval: complex | None = None
    found: bool = True

    def __repr__(self) -> str:
        if not self.found:
            return f"CriticalPoint({self.kind}: not found in range)"
        return f"CriticalPoint({self.kind}: V={self.velocity:.4f} m/s, f={self.frequency_hz:.4f} Hz)"


def flutter_speed(
    model: BeamModel,
    flight: FlightCondition,
    v_min: float = 1.0,
    v_max: float = 200.0,
    n_grid: int = 40,
    tol: float = 1e-4,
    imag_tol: float = 1e-4,
    fmax_hz: float | None = 1000.0,
    deformed: bool | None = None,
    n_load_steps: int = 1,
) -> CriticalPoint:
    """First flutter speed in ``[v_min, v_max]`` (modal method).

    ``deformed`` (auto-detected) linearises about the nonlinear static
    equilibrium at each speed (gravity, angle of attack, loads).
    Modes above ``fmax_hz`` (numerically spurious high-frequency roots of the
    mixed formulation) are ignored, like GEBTAero's ``FreqLim``.
    """
    ev = _Evaluator(model, flight, deformed, n_load_steps)
    g = lambda V: _flutter_growth(ev.modes(V), imag_tol, fmax_hz)[0]
    Vs = np.linspace(v_min, v_max, n_grid)
    prev_V, prev_g = Vs[0], g(Vs[0])
    if prev_g > 0:
        return CriticalPoint("flutter", v_min, np.nan, None, found=False)
    for V in Vs[1:]:
        gv = g(V)
        if gv > 0:
            if not np.isfinite(prev_g):
                prev_g = -1.0
            if not np.isfinite(gv):
                gv = 1.0
            Vf = scipy.optimize.brentq(lambda v: np.clip(g(v), -1e6, 1e6), prev_V, V, xtol=tol)
            # identify the critical mode slightly above V_f, where it is clearly growing
            res_hi = ev.modes(min(V, Vf + max(100 * tol, 1e-6 * Vf)))
            res = ev.modes(Vf)
            if res is None or res_hi is None:
                return CriticalPoint("static-limit", float(Vf), 0.0, None)
            _, j_hi = _flutter_growth(res_hi, imag_tol, fmax_hz)
            j = int(np.argmin(np.abs(res.eigvals - res_hi.eigvals[j_hi])))
            return CriticalPoint("flutter", float(Vf), float(res.freq_hz[j]), complex(res.eigvals[j]))
        prev_V, prev_g = V, gv
    return CriticalPoint("flutter", np.nan, np.nan, None, found=False)


def _det_sign(ev: _Evaluator, V):
    x, fl = ev.equilibrium(V)
    if x is None:
        return 0.0
    K, _ = _tangent_jit(x, ev.model, fl)
    sign, _ = np.linalg.slogdet(np.asarray(K))
    return sign


def divergence_speed(
    model: BeamModel,
    flight: FlightCondition,
    v_min: float = 1.0,
    v_max: float = 300.0,
    n_grid: int = 40,
    tol: float = 1e-4,
    deformed: bool | None = None,
) -> CriticalPoint:
    """First static divergence speed: a real eigenvalue crosses zero <=> ``det K`` changes sign."""
    ev = _Evaluator(model, flight, deformed, 1)
    Vs = np.linspace(v_min, v_max, n_grid)
    s0 = _det_sign(ev, Vs[0])
    lo = Vs[0]
    for V in Vs[1:]:
        if _det_sign(ev, V) != s0:
            hi = V
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                if _det_sign(ev, mid) == s0:
                    lo = mid
                else:
                    hi = mid
            return CriticalPoint("divergence", float(0.5 * (lo + hi)), 0.0, 0.0)
        lo = V
    return CriticalPoint("divergence", np.nan, 0.0, None, found=False)


def critical_speeds(model, flight, v_min=1.0, v_max=200.0, **kw) -> dict:
    """Both flutter and divergence speeds (GEBTAero ``ModalCriticalSpeed(mode=3)``)."""
    return {
        "flutter": flutter_speed(model, flight, v_min, v_max, **kw),
        "divergence": divergence_speed(model, flight, v_min, v_max, **{k: v for k, v in kw.items() if k in ("n_grid", "tol", "deformed")}),
    }


# --------------------------------------------------------------------------------------
# V-g sweep with mode tracking
# --------------------------------------------------------------------------------------
@dataclass
class ModeSweep:
    velocity: np.ndarray  # (nV,)
    eigvals: np.ndarray  # (nV, n_modes) complex

    @property
    def freq_hz(self):
        return self.eigvals.imag / (2 * np.pi)

    @property
    def damping(self):
        return self.eigvals.real / (2 * np.pi)

    @property
    def zeta(self):
        return -self.eigvals.real / np.maximum(np.abs(self.eigvals), 1e-300)


def _mac(a, b):
    num = np.abs(a.conj().T @ b) ** 2
    den = np.outer(np.sum(np.abs(a) ** 2, 0), np.sum(np.abs(b) ** 2, 0))
    return num / den


def mode_sweep(
    model: BeamModel,
    flight: FlightCondition,
    velocities,
    n_modes: int = 4,
    imag_tol: float = 1e-4,
    fmax_hz: float = 1000.0,
    deformed: bool | None = None,
    n_load_steps: int = 1,
) -> ModeSweep:
    """Track the ``n_modes`` lowest structural modes over a velocity sweep (V-g / V-f diagram).

    Mode correlation uses the MAC between successive eigenvectors (Hungarian
    assignment), like GEBTAero's ``EigenTabSorted``. A tracked mode that
    becomes real (e.g. divergence) keeps its continuous branch.
    """
    ev = _Evaluator(model, flight, deformed, n_load_steps)
    velocities = np.asarray(velocities, dtype=float)
    out = np.zeros((len(velocities), n_modes), dtype=complex)
    prev_vec = None
    prev_s = None
    for i, V in enumerate(velocities):
        res = ev.modes(V)
        if res is None:
            out[i] = np.nan
            continue
        cand = np.where((res.eigvals.imag >= -imag_tol) & (np.abs(res.eigvals) < 2 * np.pi * fmax_hz))[0]
        if prev_vec is None:
            osc = res.oscillatory(imag_tol)[:n_modes]
            sel = osc
        else:
            mac = _mac(prev_vec, res.right[:, cand])
            ds = np.abs(prev_s[:, None] - res.eigvals[cand][None, :]) / (1.0 + np.abs(prev_s[:, None]))
            cost = (1.0 - mac) + 0.2 * np.minimum(ds, 5.0)
            r, c = linear_sum_assignment(cost)
            sel = cand[c[np.argsort(r)]]
        out[i] = res.eigvals[sel]
        prev_vec = res.right[:, sel]
        prev_s = res.eigvals[sel]
    return ModeSweep(velocities, out)


# --------------------------------------------------------------------------------------
# sensitivities
# --------------------------------------------------------------------------------------
def _matrices(build: Builder, params, V, deformed, x_guess):
    model, flight = build(params, V)
    if deformed:
        x0 = solve_static(model, flight, x0=x_guess)
    else:
        x0 = jnp.zeros(model.n_dof)
    return tangent_matrices(x0, model, flight)


@partial(jax.jit, static_argnames=("build", "deformed"))
def _eig_jacobian(build, params, V, s, phi, psi, deformed, x_guess):
    K0, M0 = _matrices(build, params, V, deformed, x_guess)
    c = jnp.vdot(psi, M0 @ phi)

    def h(p, v):
        K, M = _matrices(build, p, v, deformed, x_guess)
        val = jnp.vdot(psi, (-K - s * M) @ phi) / c
        return jnp.stack([val.real, val.imag])

    return jax.jacrev(h, argnums=(0, 1))(params, V)


def eigenvalue_gradient(
    build: Builder,
    params,
    V: float,
    s: complex,
    phi: np.ndarray,
    psi: np.ndarray,
    deformed: bool = False,
    x_guess=None,
):
    """Gradient of an eigenvalue ``s`` (real and imaginary parts) w.r.t. ``params`` and ``V``.

    ``phi``/``psi`` are the right/left eigenvectors of ``-K phi = s M phi``.
    Returns ``(dRe/dparams, dRe/dV, dIm/dparams, dIm/dV)``.
    """
    dp, dv = _eig_jacobian(
        build, params, jnp.asarray(V, dtype=float), jnp.asarray(s), jnp.asarray(phi), jnp.asarray(psi),
        bool(deformed), x_guess,
    )
    re_p = jax.tree_util.tree_map(lambda a: a[0], dp)
    im_p = jax.tree_util.tree_map(lambda a: a[1], dp)
    return re_p, dv[0], im_p, dv[1]


@dataclass
class CriticalSensitivity:
    point: CriticalPoint
    dV_dparams: Any
    dfreq_dparams: Any | None = None


def flutter_speed_and_grad(
    build: Builder,
    params,
    v_min: float = 1.0,
    v_max: float = 200.0,
    deformed: bool | None = None,
    **kw,
) -> CriticalSensitivity:
    """Flutter speed and its exact gradient w.r.t. an arbitrary pytree of design parameters.

    ``build(params, V) -> (model, flight)`` must be written with ``jax.numpy``.
    """
    model, flight = build(params, v_min)
    if deformed is None:
        deformed = _needs_equilibrium(model, flight)
    pt = flutter_speed(model, flight, v_min, v_max, deformed=deformed, **kw)
    if not pt.found:
        return CriticalSensitivity(pt, None, None)
    model, flight = build(params, pt.velocity)
    x_guess = solve_static(model, flight) if deformed else None
    res = modal_analysis(model, flight, x_guess)
    j = int(np.argmin(np.abs(res.eigvals - pt.eigval)))
    s, phi, psi = res.eigvals[j], res.right[:, j], res.left[:, j]
    re_p, re_v, im_p, im_v = eigenvalue_gradient(build, params, pt.velocity, s, phi, psi, deformed, x_guess)
    dV = jax.tree_util.tree_map(lambda g: -g / re_v, re_p)
    dfreq = jax.tree_util.tree_map(lambda gi, dv: (gi + im_v * dv) / (2 * np.pi), im_p, dV)
    return CriticalSensitivity(pt, dV, dfreq)


def divergence_speed_and_grad(
    build: Builder, params, v_min: float = 1.0, v_max: float = 300.0, **kw
) -> CriticalSensitivity:
    """Divergence speed and its gradient (real eigenvalue crossing zero)."""
    model, flight = build(params, v_min)
    pt = divergence_speed(model, flight, v_min, v_max, deformed=False, **kw)
    if not pt.found:
        return CriticalSensitivity(pt, None)
    model, flight = build(params, pt.velocity)
    res = modal_analysis(model, flight)
    j = int(np.argmin(np.abs(res.eigvals)))
    s, phi, psi = res.eigvals[j], res.right[:, j], res.left[:, j]
    re_p, re_v, _, _ = eigenvalue_gradient(build, params, pt.velocity, s, phi, psi, False)
    dV = jax.tree_util.tree_map(lambda g: -g / re_v, re_p)
    return CriticalSensitivity(pt, dV)
