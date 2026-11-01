"""Static (nonlinear), modal and transient solvers.

* :func:`solve_static` -- Newton-Raphson on ``R(x, 0) = 0`` with an exact
  AD Jacobian. Wrapped in :func:`jax.lax.custom_root`, so the equilibrium is
  *implicitly differentiable*: ``jax.grad`` of any output w.r.t. any model or
  flight parameter costs one extra linear solve, not a re-solve.
* :func:`modal_analysis` -- linearisation about an equilibrium and generalised
  eigenvalue problem ``-K phi = s M phi``.
* :func:`simulate` -- implicit generalised-alpha time marching (GEBT uses the
  trapezoidal rule, recovered with ``rho_inf=1``), compiled with
  :func:`jax.lax.scan`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from .beam import FlightCondition, node_displacements, residual, split_state, tangent_matrices
from .integrators import generalized_alpha, generalized_alpha_coefficients  # noqa: F401
from .rotations import rodrigues_to_angle
from .wing import BeamModel


# --------------------------------------------------------------------------------------
# Newton
# --------------------------------------------------------------------------------------
def _newton(fun, x0, tol=1e-10, maxiter=30):
    def cond(c):
        i, _, err = c
        return (i < maxiter) & (err > tol)

    def body(c):
        i, x, _ = c
        f, J = fun(x), jax.jacfwd(fun)(x)
        dx = jnp.linalg.solve(J, -f)
        x = x + dx
        err = jnp.linalg.norm(dx) / (1.0 + jnp.linalg.norm(x))
        return i + 1, x, err

    _, x, _ = jax.lax.while_loop(cond, body, (0, x0, jnp.inf))
    return x


@partial(jax.jit, static_argnames=("tol", "maxiter"))
def _static_root(model: BeamModel, flight: FlightCondition, x0, tol=1e-10, maxiter=30):
    xd = jnp.zeros_like(x0)
    f = lambda x: residual(x, xd, model, flight)
    solve = lambda g, y0: _newton(g, y0, tol, maxiter)
    tangent_solve = lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(y), y)
    return jax.lax.custom_root(f, x0, solve, tangent_solve)


def solve_static(
    model: BeamModel,
    flight: FlightCondition,
    x0: jnp.ndarray | None = None,
    n_load_steps: int = 1,
    tol: float = 1e-10,
    maxiter: int = 30,
) -> jnp.ndarray:
    """Nonlinear steady (static aeroelastic) equilibrium.

    ``n_load_steps > 1`` ramps ``flight.load_factor`` (gravity, density and
    applied loads) for robustness with large deflections. The returned state
    is differentiable (implicit function theorem) w.r.t. ``model`` and
    ``flight``.
    """
    x = jnp.zeros(model.n_dof) if x0 is None else x0
    for k in range(1, n_load_steps):
        fl = flight.replace(load_factor=flight.load_factor * k / n_load_steps)
        x = _static_root(model, fl, jax.lax.stop_gradient(x), tol=tol, maxiter=maxiter)
    x = _static_root(model, flight, jax.lax.stop_gradient(x), tol=tol, maxiter=maxiter)
    return x


def static_residual_norm(model: BeamModel, flight: FlightCondition, x) -> float:
    return float(jnp.linalg.norm(residual(x, jnp.zeros_like(x), model, flight)))


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------
@dataclass
class StaticOutputs:
    """Root loads and tip deflection, GEBTAero ``StaticLoads`` style (frame a)."""

    root_force: np.ndarray
    root_moment: np.ndarray
    tip_displacement: np.ndarray
    tip_rotation: np.ndarray

    def as_array(self) -> np.ndarray:
        return np.stack([self.root_force, self.root_moment, self.tip_displacement, self.tip_rotation])


def static_outputs(model: BeamModel, x) -> StaticOutputs:
    root, _, tip = split_state(model, x)
    return StaticOutputs(*(np.asarray(v) for v in (root[:3], root[3:], tip[:3], tip[3:])))


def root_loads(model: BeamModel, x):
    """Differentiable root force and moment (frame a, GEBT sign convention)."""
    root, _, _ = split_state(model, x)
    return root[:3], root[3:]


def tip_displacement(model: BeamModel, x):
    """Differentiable tip displacement (frame a)."""
    _, _, tip = split_state(model, x)
    return tip[:3]


def tip_twist(model: BeamModel, x):
    """Differentiable elastic tip rotation about the local beam axis (rad)."""
    _, _, tip = split_state(model, x)
    return 2.0 * jnp.arctan(0.5 * jnp.dot(tip[3:], model.Cab[-1][:, 0]))


def lift_force(model: BeamModel, x):
    """Total aerodynamic + gravity vertical load carried at the root (positive up)."""
    F, _ = root_loads(model, x)
    return -F[2]


# --------------------------------------------------------------------------------------
# modal analysis
# --------------------------------------------------------------------------------------
_tangent_jit = jax.jit(tangent_matrices)


@dataclass
class ModalResult:
    """Eigen-solution of the linearised aeroelastic system ``-K phi = s M phi``."""

    eigvals: np.ndarray  # complex continuous-time eigenvalues s (rad/s)
    right: np.ndarray  # (n_dof, n_modes) right eigenvectors
    left: np.ndarray  # (n_dof, n_modes) left eigenvectors
    K: np.ndarray
    M: np.ndarray

    @property
    def freq_hz(self) -> np.ndarray:
        return self.eigvals.imag / (2 * np.pi)

    @property
    def damping(self) -> np.ndarray:
        """Real part / 2 pi (GEBTAero's output convention; > 0 means unstable)."""
        return self.eigvals.real / (2 * np.pi)

    @property
    def zeta(self) -> np.ndarray:
        """Damping ratio ``-Re(s)/|s|``."""
        return -self.eigvals.real / np.maximum(np.abs(self.eigvals), 1e-300)

    def oscillatory(self, imag_tol: float = 1e-4) -> np.ndarray:
        """Indices of oscillatory modes (one per conjugate pair), sorted by frequency."""
        idx = np.where(self.eigvals.imag > imag_tol)[0]
        return idx[np.argsort(self.eigvals.imag[idx])]

    def real_modes(self, imag_tol: float = 1e-4) -> np.ndarray:
        idx = np.where(np.abs(self.eigvals.imag) <= imag_tol)[0]
        return idx[np.argsort(-self.eigvals.real[idx])]

    def table(self, n: int = 10, imag_tol: float = 1e-4) -> np.ndarray:
        """``[damping, frequency]`` of the ``n`` lowest oscillatory modes (GEBTAero ``Eigenvalues``)."""
        idx = self.oscillatory(imag_tol)[:n]
        return np.stack([self.damping[idx], self.freq_hz[idx]], axis=1)


def equilibrated_eig(K, M, n_sweeps: int = 3):
    """Solve ``-K v = s M v`` after diagonal row/column equilibration of the pencil.

    The mixed formulation mixes forces, moments, lengths and momenta, which
    makes the raw pencil badly scaled; equilibration leaves the eigenvalues
    unchanged but makes spurious high-frequency roots far less sensitive to
    round-off. Returns ``(s, left, right)`` in the original coordinates.
    """
    dr = np.ones(K.shape[0])
    dc = np.ones(K.shape[1])
    for _ in range(n_sweeps):
        A = np.abs(K) * dr[:, None] * dc[None, :] + np.abs(M) * dr[:, None] * dc[None, :]
        dr = dr / np.sqrt(np.maximum(A.max(axis=1), 1e-300))
        A = np.abs(K) * dr[:, None] * dc[None, :] + np.abs(M) * dr[:, None] * dc[None, :]
        dc = dc / np.sqrt(np.maximum(A.max(axis=0), 1e-300))
    Ks = K * dr[:, None] * dc[None, :]
    Ms = M * dr[:, None] * dc[None, :]
    w, vl, vr = scipy.linalg.eig(-Ks, Ms, left=True, right=True)
    return w, vl * dr[:, None], vr * dc[:, None]


def modal_analysis(
    model: BeamModel,
    flight: FlightCondition,
    x0: jnp.ndarray | None = None,
    max_abs: float = 1e7,
) -> ModalResult:
    """Linearise about ``x0`` (default: undeformed state) and solve the eigenproblem."""
    if x0 is None:
        x0 = jnp.zeros(model.n_dof)
    K, M = _tangent_jit(x0, model, flight)
    K, M = np.asarray(K), np.asarray(M)
    w, vl, vr = equilibrated_eig(K, M)
    keep = np.isfinite(w) & (np.abs(w) < max_abs)
    w, vl, vr = w[keep], vl[:, keep], vr[:, keep]
    order = np.argsort(np.abs(w.imag) + 1e-9 * np.abs(w.real))
    return ModalResult(w[order], vr[:, order], vl[:, order], K, M)


def mode_shape(model: BeamModel, phi: np.ndarray):
    """Nodal displacement/rotation shapes (real part after phase normalisation)."""
    phi = np.asarray(phi)
    k = np.argmax(np.abs(phi))
    phi = phi * np.exp(-1j * np.angle(phi[k]))
    phi = phi / np.max(np.abs(phi))
    u, th = node_displacements(model, jnp.asarray(phi.real))
    return np.asarray(u), np.asarray(th)


# --------------------------------------------------------------------------------------
# transient simulation
# --------------------------------------------------------------------------------------
@dataclass
class TimeHistory:
    t: np.ndarray
    x: np.ndarray  # (n_steps + 1, n_dof)
    model: BeamModel

    def tip(self) -> np.ndarray:
        return self.x[:, -6:]

    def tip_rotation_angle(self) -> np.ndarray:
        return np.asarray(rodrigues_to_angle(jnp.asarray(self.x[:, -3:])))

    def max_rotation(self) -> np.ndarray:
        """Largest element rotation angle at every time step."""
        _, E, _ = jax.vmap(lambda x: split_state(self.model, x))(jnp.asarray(self.x))
        return np.asarray(jnp.max(rodrigues_to_angle(E[..., 3:6]), axis=-1))


@cache
def _beam_fun(flight_fn):
    def fun(x, xd, t, prm):
        model, flight = prm
        fl = flight if flight_fn is None else flight_fn(t, flight)
        return residual(x, xd, model, fl)

    return fun


def simulate(
    model: BeamModel,
    flight: FlightCondition,
    t_end: float,
    dt: float,
    x0: jnp.ndarray | None = None,
    xd0: jnp.ndarray | None = None,
    flight_fn: Callable[[float, FlightCondition], FlightCondition] | None = None,
    newton_iter: int = 4,
    rho_inf: float = 0.9,
) -> TimeHistory:
    """Nonlinear transient response (generalised-alpha time integration).

    GEBT uses the trapezoidal rule; here the first-order generalised-alpha
    method is used instead (``rho_inf=1`` is trapezoid-like). A little
    high-frequency dissipation (``rho_inf < 1``) is needed when some strains
    are rigid (zero flexibility), which makes the beam a higher-index DAE.

    ``flight_fn(t, flight) -> flight`` (jax-traceable, hashable) lets you
    prescribe time-dependent loads, gusts (``v_a``) or manoeuvres.
    The whole time loop is compiled with ``lax.scan`` and is differentiable.
    """
    n_steps = int(round(t_end / dt))
    x0 = jnp.zeros(model.n_dof) if x0 is None else x0
    xd0 = jnp.zeros_like(x0) if xd0 is None else xd0
    xs = generalized_alpha(_beam_fun(flight_fn), x0, xd0, (model, flight), dt, n_steps, float(rho_inf), newton_iter)
    t = np.arange(n_steps + 1) * dt
    return TimeHistory(t, np.concatenate([np.asarray(x0)[None], np.asarray(xs)]), model)
