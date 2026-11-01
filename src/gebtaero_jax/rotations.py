"""Rotation utilities based on Rodrigues parameters (as in GEBT).

GEBT parameterises finite rotations with the Rodrigues vector
``theta = 2 tan(phi / 2) n`` where ``phi`` is the rotation angle and ``n`` the
unit axis. All functions here are pure ``jax.numpy`` so they can be traced,
jit-compiled and differentiated.
"""

from __future__ import annotations

import jax.numpy as jnp

I3 = jnp.eye(3)
E1 = jnp.array([1.0, 0.0, 0.0])


def tilde(v: jnp.ndarray) -> jnp.ndarray:
    """Skew-symmetric matrix such that ``tilde(a) @ b == cross(a, b)``."""
    return jnp.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def rotation_rodrigues(theta: jnp.ndarray) -> jnp.ndarray:
    """Rotation matrix ``C^T`` (deformed frame -> undeformed frame) from Rodrigues parameters.

    This is GEBT's ``DirCosineTRodrigues``::

        C^T = ((1 - t.t/4) I + tilde(t) + t t^T / 2) / (1 + t.t/4)
    """
    t2_4 = jnp.dot(theta, theta) * 0.25
    return ((1.0 - t2_4) * I3 + tilde(theta) + 0.5 * jnp.outer(theta, theta)) / (1.0 + t2_4)


def rodrigues_from_axis_angle(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Rodrigues vector for a rotation of ``angle`` (rad) about ``axis``."""
    axis = jnp.asarray(axis, dtype=float)
    n = axis / jnp.linalg.norm(axis)
    return 2.0 * jnp.tan(0.5 * angle) * n


def rodrigues_to_angle(theta: jnp.ndarray) -> jnp.ndarray:
    """Rotation angle (rad) represented by a Rodrigues vector (last axis)."""
    return 2.0 * jnp.arctan(0.5 * jnp.linalg.norm(theta, axis=-1))
