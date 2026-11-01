"""Matplotlib helpers (V-g diagrams, 3D wing shapes, time histories)."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .beam import node_displacements, node_positions
from .rotations import rotation_rodrigues
from .stability import ModeSweep
from .wing import BeamModel


def plot_vg(sweep: ModeSweep, damping: str = "zeta", labels=None, axes=None, ylim_damping=None):
    """Frequency and damping versus velocity (GEBTAero ``EigenFreqDamping``).

    ``damping``: ``"zeta"`` (damping ratio, negative = unstable), ``"real"``
    (``Re(s)/2pi``, positive = unstable).
    """
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_f, ax_g = axes
    V = sweep.velocity
    d = -sweep.zeta if damping == "zeta" else sweep.damping
    for k in range(sweep.eigvals.shape[1]):
        lab = labels[k] if labels is not None else f"mode {k + 1}"
        ax_f.plot(V, sweep.freq_hz[:, k], "-o", ms=3, label=lab)
        ax_g.plot(V, d[:, k], "-o", ms=3, label=lab)
    ax_g.axhline(0.0, color="k", lw=0.8)
    ax_f.set_ylabel("frequency (Hz)")
    ax_g.set_ylabel("-ζ (damping ratio)" if damping == "zeta" else "Re(s)/2π (1/s)")
    ax_g.set_xlabel("velocity (m/s)")
    if ylim_damping is not None:
        ax_g.set_ylim(*ylim_damping)
    ax_f.legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3)
    return axes


def wing_outline(model: BeamModel, x=None, scale: float = 1.0):
    """Reference line, leading and trailing edges (frame a) of the (deformed) wing."""
    pts = np.asarray(node_positions(model))
    if x is None:
        u = np.zeros_like(pts)
        th = np.zeros_like(pts)
    else:
        u, th = node_displacements(model, jnp.asarray(x))
        u, th = scale * np.asarray(u), scale * np.asarray(th)
    ref = pts + u
    Cab_nodes = np.concatenate([np.asarray(model.Cab[:1]), np.asarray(model.Cab)])
    b = 0.5 * np.concatenate([np.asarray(model.chord[:1]), np.asarray(model.chord)])
    a = np.concatenate([np.asarray(model.a[:1]), np.asarray(model.a)])
    le, te = [], []
    for i in range(len(ref)):
        C = np.asarray(rotation_rodrigues(jnp.asarray(th[i]))) @ Cab_nodes[i]
        yb = C[:, 1]
        mid = ref[i] + a[i] * b[i] * yb
        le.append(mid + b[i] * yb)
        te.append(mid - b[i] * yb)
    return ref, np.array(le), np.array(te)


def plot_wing_3d(model: BeamModel, x=None, scale: float = 1.0, ax=None, color="C0", alpha=0.35, label=None):
    """3D surface of the wing planform (z_a is flipped so that 'up' is up)."""
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(projection="3d")
    ref, le, te = wing_outline(model, x, scale)
    flip = np.array([1.0, 1.0, -1.0])
    ref, le, te = ref * flip, le * flip, te * flip
    X = np.stack([le[:, 0], te[:, 0]])
    Y = np.stack([le[:, 1], te[:, 1]])
    Z = np.stack([le[:, 2], te[:, 2]])
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color=color, lw=2, label=label)
    ax.set_xlabel("x_a (fwd)")
    ax.set_ylabel("y_a (span)")
    ax.set_zlabel("-z_a (up)")
    return ax


def set_equal_3d(ax):
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    c = lims.mean(axis=1)
    r = 0.5 * np.max(lims[:, 1] - lims[:, 0])
    ax.set_xlim3d(c[0] - r, c[0] + r)
    ax.set_ylim3d(c[1] - r, c[1] + r)
    ax.set_zlim3d(c[2] - r, c[2] + r)
