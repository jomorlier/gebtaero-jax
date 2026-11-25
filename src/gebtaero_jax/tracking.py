r"""Mode tracking and adjoint sensitivities of tracked eigenpairs.

Generalises the bordered eigen-adjoint of S. He (note to J. Morlier, *mode tracking with the eigenvalue
problem adjoint*) from a real symmetric problem to the complex, non-symmetric pencil of the
geometrically exact beam

.. math:: (\mathbf K + s\,\mathbf M)\,\boldsymbol\phi = \mathbf 0 ,
   \qquad \mathbf K = \partial\mathbf R/\partial\mathbf x,\ \mathbf M = \partial\mathbf R/\partial\dot{\mathbf x}.

For a tracked mode the eigenpair ``q = (phi, s)`` is defined by the holomorphic residual

.. math:: \mathbf r(\mathbf q, \mathbf p) =
   \begin{bmatrix}(\mathbf K(\mathbf p) + s\,\mathbf M(\mathbf p))\boldsymbol\phi \\
   \mathbf e^T\boldsymbol\phi - 1\end{bmatrix} = \mathbf 0 ,
   \qquad
   \frac{\partial\mathbf r}{\partial\mathbf q} =
   \begin{bmatrix}\mathbf K + s\mathbf M & \mathbf M\boldsymbol\phi\\ \mathbf e^T & 0\end{bmatrix},

with an *analytic* normalisation ``e^T phi = 1`` (``e`` built from the reference shape). He's choice
``phi^T phi = 1`` is the real-symmetric special case. Any real function ``g(phi, s)`` -- the MAC with a
reference shape, a frequency, a damping -- then has the adjoint gradient

.. math:: \frac{dg}{d\mathbf p} = -\boldsymbol\psi^T\frac{\partial\mathbf r}{\partial\mathbf p},\qquad
   \Big(\frac{\partial\mathbf r}{\partial\mathbf q}\Big)^T\boldsymbol\psi = \frac{\partial g}{\partial\mathbf q},

i.e. **one linear solve per tracked function**, whatever the number of design variables. Everything is
real-split (``q = [Re phi, Re s, Im phi, Im s]``). The partial ``psi^T dr/dp`` is evaluated
*matrix-free*: ``K phi`` and ``M phi`` are Jacobian-vector products of the beam residual, and one
reverse-mode pass through them gives the whole gradient (``dK/dp`` is never formed).

Tracking: modes are paired with reference shapes by the Modal Assurance Criterion and an optimal
(Hungarian) assignment, so a constraint on "the first torsion mode" stays attached to that mode when
frequencies cross during an optimisation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

from .beam import FlightCondition, residual
from .solvers import ModalResult, modal_analysis
from .wing import BeamModel

Builder = Callable[[Any], "tuple[BeamModel, FlightCondition]"]


# --------------------------------------------------------------------------------------
# shapes and MAC
# --------------------------------------------------------------------------------------
def shape_dofs(model: BeamModel):
    """Indices and weights of the displacement-like DOFs used for mode shapes.

    Uses the element displacements ``u`` (weight 1) and Rodrigues rotations ``theta``
    weighted by the local semi-chord squared, so that rotations count like the
    displacement of the leading/trailing edges (all in metres).
    """
    ne = model.n_elem_dof
    idx, w = [], []
    b2 = (0.5 * np.asarray(model.chord)) ** 2
    for e in range(model.n_el):
        base = 6 + e * ne
        idx += list(range(base, base + 6))
        w += [1.0, 1.0, 1.0, b2[e], b2[e], b2[e]]
    return np.asarray(idx), np.sqrt(np.asarray(w))


def shapes(model: BeamModel, Phi: np.ndarray) -> np.ndarray:
    """Weighted mode shapes (rows: shape DOFs, columns: modes)."""
    idx, sw = shape_dofs(model)
    Phi = np.asarray(Phi)
    return Phi[idx] * (sw[:, None] if Phi.ndim == 2 else sw)


def mac_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``MAC_ij = |a_i^H b_j|^2 / (|a_i|^2 |b_j|^2)`` for complex shapes in columns."""
    A = np.atleast_2d(A.T).T
    B = np.atleast_2d(B.T).T
    num = np.abs(A.conj().T @ B) ** 2
    return num / np.outer(np.sum(np.abs(A) ** 2, 0), np.sum(np.abs(B) ** 2, 0))


def normalise_shape(v: np.ndarray) -> np.ndarray:
    """Unit norm, phase chosen so that the largest component is real positive."""
    v = np.asarray(v, dtype=complex)
    v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
    return v / np.linalg.norm(v)


@dataclass
class ReferenceModes:
    """Reference shapes ``phi_r,i`` (weighted, unit norm) and their labels/frequencies."""

    shapes: np.ndarray  # (n_shape_dofs, n_r) complex
    freq_hz: np.ndarray
    labels: list

    @property
    def n(self) -> int:
        return self.shapes.shape[1]


def reference_modes(model: BeamModel, res: ModalResult, n: int, labels: Sequence[str] | None = None,
                    imag_tol: float = 1e-4) -> ReferenceModes:
    """Take the ``n`` lowest oscillatory modes of ``res`` as references."""
    osc = res.oscillatory(imag_tol)[:n]
    S = np.stack([normalise_shape(v) for v in shapes(model, res.right[:, osc]).T], axis=1)
    labels = list(labels) if labels is not None else [f"mode {k + 1}" for k in range(n)]
    return ReferenceModes(S, res.freq_hz[osc], labels)


def reference_from_tracked(model: BeamModel, tracked: TrackedModes, labels: Sequence[str]) -> ReferenceModes:
    """New references from the modes just tracked (step-to-step tracking along a continuation path)."""
    S = np.stack([normalise_shape(v) for v in shapes(model, tracked.vectors).T], axis=1)
    return ReferenceModes(S, tracked.freq_hz, list(labels))


@dataclass
class TrackedModes:
    """Modes of the current design paired with the reference shapes."""

    index: np.ndarray  # index into ModalResult arrays, one per reference mode
    eigvals: np.ndarray  # tracked eigenvalues s
    vectors: np.ndarray  # full eigenvectors (n_dof, n_r)
    mac: np.ndarray  # MAC with own reference (n_r,)
    mac_full: np.ndarray  # MAC(reference, candidates)
    sorted_rank: np.ndarray  # position of each tracked mode in the frequency-sorted list

    @property
    def freq_hz(self):
        return self.eigvals.imag / (2 * np.pi)


def track(model: BeamModel, res: ModalResult, ref: ReferenceModes, fmax_hz: float = 1000.0,
          imag_tol: float = 1e-4) -> TrackedModes:
    """Pair the oscillatory modes of ``res`` with the reference shapes (MAC + Hungarian assignment)."""
    cand = res.oscillatory(imag_tol)
    cand = cand[res.freq_hz[cand] <= fmax_hz]
    C = shapes(model, res.right[:, cand])
    MAC = mac_matrix(ref.shapes, C)
    r, c = linear_sum_assignment(-MAC)
    order = np.argsort(r)
    sel = cand[c[order]]
    rank = np.array([int(np.where(cand == j)[0][0]) for j in sel])
    return TrackedModes(sel, res.eigvals[sel], res.right[:, sel], MAC[r[order], c[order]], MAC, rank)


# --------------------------------------------------------------------------------------
# bordered eigen-adjoint (He), complex non-symmetric, matrix-free partials
# --------------------------------------------------------------------------------------
def _real_split(z):
    return jnp.concatenate([jnp.real(z), jnp.imag(z)])


def _complex_join(q):
    n = q.shape[0] // 2
    return q[:n] + 1j * q[n:]


def _K_M_matvec(model: BeamModel, flight: FlightCondition, x0, v):
    """``K v`` and ``M v`` for complex ``v`` via two JVPs each (no matrix assembly)."""
    xd0 = jnp.zeros_like(x0)
    fx = lambda x: residual(x, xd0, model, flight)
    fxd = lambda xd: residual(x0, xd, model, flight)
    Kr = jax.jvp(fx, (x0,), (jnp.real(v),))[1]
    Ki = jax.jvp(fx, (x0,), (jnp.imag(v),))[1]
    Mr = jax.jvp(fxd, (xd0,), (jnp.real(v),))[1]
    Mi = jax.jvp(fxd, (xd0,), (jnp.imag(v),))[1]
    return Kr + 1j * Ki, Mr + 1j * Mi


def bordered_residual(build: Builder, params, q, e, x0_fn=None):
    """Real-split residual ``r(q, p)`` of one eigenpair (see module docstring)."""
    model, flight = build(params)
    x0 = jnp.zeros(model.n_dof) if x0_fn is None else x0_fn(model, flight)
    z = _complex_join(q)
    phi, s = z[:-1], z[-1]
    Kphi, Mphi = _K_M_matvec(model, flight, x0, phi)
    r = jnp.concatenate([Kphi + s * Mphi, jnp.atleast_1d(jnp.dot(e, phi) - 1.0)])
    return _real_split(r)


def bordered_jacobian(K, M, phi, s, e):
    """Real form of ``[[K + sM, M phi], [e^T, 0]]`` (assembled from the modal-analysis matrices)."""
    n = K.shape[0]
    Jc = np.zeros((n + 1, n + 1), dtype=complex)
    Jc[:n, :n] = K + s * M
    Jc[:n, n] = M @ phi
    Jc[n, :n] = e
    return np.block([[Jc.real, -Jc.imag], [Jc.imag, Jc.real]])


@dataclass
class EigenpairAdjoint:
    """Values and adjoint gradients of functions of one tracked eigenpair."""

    values: list
    grads: list  # pytrees like params
    psi: np.ndarray  # adjoint vectors (columns)


def normalisation_vector(model: BeamModel, ref_shape: np.ndarray) -> np.ndarray:
    """``e`` such that ``e^T phi = phi_r^H W phi / |phi_r|^2`` (analytic in ``phi``)."""
    idx, sw = shape_dofs(model)
    e = np.zeros(model.n_dof, dtype=complex)
    e[idx] = sw * np.conj(ref_shape) / np.vdot(ref_shape, ref_shape).real
    return e


@partial(jax.jit, static_argnames=("build", "x0_fn"))
def _vjp_params(build, params, q, e, cotangents, x0_fn=None):
    """``-psi_k^T dr/dp`` for every cotangent (column of ``cotangents``) with one linearisation."""
    f = lambda p: bordered_residual(build, p, q, e, x0_fn)
    _, vjp_fn = jax.vjp(f, params)
    return jax.vmap(lambda ct: jax.tree_util.tree_map(lambda g: -g, vjp_fn(ct)[0]), in_axes=1)(cotangents)


def eigenpair_adjoint(
    build: Builder,
    params,
    model: BeamModel,
    res: ModalResult,
    j: int,
    e: np.ndarray,
    funcs: Sequence[Callable],
    x0_fn=None,
) -> EigenpairAdjoint:
    """Adjoint gradients of ``funcs`` (each ``g(phi, s) -> real``, jax-traceable) for mode ``j`` of ``res``.

    ``res`` must be the modal analysis of ``build(params)``. Cost: one ``(2n+2)`` LU factorisation,
    ``len(funcs)`` back-substitutions and one batched reverse pass, independent of ``len(params)``.
    """
    phi = res.right[:, j] / (e @ res.right[:, j])  # enforce e^T phi = 1
    s = res.eigvals[j]
    q = np.concatenate([np.r_[phi.real, s.real], np.r_[phi.imag, s.imag]])
    J = bordered_jacobian(res.K, res.M, phi, s, e)
    lu = scipy.linalg.lu_factor(J)

    def g_of_q(g):
        def h(qq):
            z = _complex_join(qq)
            return g(z[:-1], z[-1])

        return h

    values, rhs = [], []
    for g in funcs:
        h = g_of_q(g)
        values.append(float(h(jnp.asarray(q))))
        rhs.append(np.asarray(jax.grad(h)(jnp.asarray(q))))
    Psi = scipy.linalg.lu_solve(lu, np.stack(rhs, axis=1), trans=1)
    G = _vjp_params(build, params, jnp.asarray(q), jnp.asarray(e), jnp.asarray(Psi), x0_fn)
    grads = [jax.tree_util.tree_map(lambda a, k=k: np.asarray(a[k]), G) for k in range(len(funcs))]
    return EigenpairAdjoint(values, grads, Psi)


# ready-made functions of an eigenpair --------------------------------------------------
def mac_function(model: BeamModel, ref_shape: np.ndarray) -> Callable:
    """``g(phi, s) = MAC(phi_r, phi)`` on the weighted shape DOFs (real, not holomorphic)."""
    idx, sw = shape_dofs(model)
    idx_j, sw_j = jnp.asarray(idx), jnp.asarray(sw)
    r = jnp.asarray(ref_shape)
    rr = jnp.real(jnp.vdot(r, r))

    def g(phi, s):
        v = phi[idx_j] * sw_j
        return jnp.abs(jnp.vdot(r, v)) ** 2 / (rr * jnp.real(jnp.vdot(v, v)))

    return g


def frequency_function(phi, s):
    """``g = Im(s) / 2 pi`` (Hz)."""
    return jnp.imag(s) / (2 * jnp.pi)


def growth_function(phi, s):
    """``g = Re(s)`` (1/s): positive means unstable."""
    return jnp.real(s)


# --------------------------------------------------------------------------------------
# convenience: tracked structural modes, MAC loss and frequencies with gradients
# --------------------------------------------------------------------------------------
@dataclass
class TrackedSensitivities:
    tracked: TrackedModes
    mac_loss: float  # f = 1 - mean(MAC_ii)   (He's loss with MAC instead of phi_r^T phi)
    dmac_loss: Any
    freq_hz: np.ndarray
    dfreq: list  # gradient pytree per tracked mode


def tracked_sensitivities(build: Builder, params, ref: ReferenceModes, fmax_hz: float = 1000.0,
                          x0_fn=None) -> TrackedSensitivities:
    """MAC loss ``f = 1 - (1/n_r) sum_i MAC(phi_r,i, phi_i)`` and tracked frequencies, with adjoint gradients.

    One bordered adjoint system per tracked mode (two right-hand sides: MAC and frequency).
    """
    model, flight = build(params)
    x0 = None if x0_fn is None else x0_fn(model, flight)
    res = modal_analysis(model, flight, x0)
    tr = track(model, res, ref, fmax_hz)
    loss, dloss, freqs, dfreqs = 1.0, None, [], []
    for i in range(ref.n):
        e = normalisation_vector(model, ref.shapes[:, i])
        adj = eigenpair_adjoint(build, params, model, res, int(tr.index[i]), e,
                                [mac_function(model, ref.shapes[:, i]), frequency_function], x0_fn)
        loss -= adj.values[0] / ref.n
        term = jax.tree_util.tree_map(lambda g: -g / ref.n, adj.grads[0])
        dloss = term if dloss is None else jax.tree_util.tree_map(np.add, dloss, term)
        freqs.append(adj.values[1])
        dfreqs.append(adj.grads[1])
    return TrackedSensitivities(tr, loss, dloss, np.asarray(freqs), dfreqs)


def flutter_gradient_adjoint(build_V: Callable, params, V: float, eigval: complex, fmax_hz: float = 1000.0):
    """Flutter-speed gradient with the bordered adjoint (one solve, matrix-free partials).

    ``build_V((params, V)) -> (model, flight)``. Uses ``dV_f/dp = -(dRe s/dp)/(dRe s/dV)``,
    with both partials obtained from the *same* adjoint vector (``V`` is appended to the parameters).
    """
    model, flight = build_V((params, V))
    res = modal_analysis(model, flight)
    j = int(np.argmin(np.abs(res.eigvals - eigval)))
    idx, sw = shape_dofs(model)
    e = normalisation_vector(model, normalise_shape(shapes(model, res.right[:, j])))
    adj = eigenpair_adjoint(build_V, (params, jnp.asarray(V, float)), model, res, j, e,
                            [growth_function, frequency_function])
    dRe_p, dRe_V = adj.grads[0]
    dF_p, dF_V = adj.grads[1]
    dV = jax.tree_util.tree_map(lambda g: -g / float(dRe_V), dRe_p)
    dfreq = jax.tree_util.tree_map(lambda gf, dv: gf + float(dF_V) * dv, dF_p, dV)
    return dV, dfreq, res.eigvals[j]
