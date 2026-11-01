"""Wing geometry (frames, sections) and its discretisation into beam elements.

Frame ``a`` is the aircraft frame attached to the wing root: ``x_a`` points
forward (flight direction), ``y_a`` along the right wing and ``z_a`` down.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import jax.numpy as jnp
import numpy as np

from ._pytree import pytree_dataclass, static_field
from .cross_section import CrossSection


class Aero(IntEnum):
    """Strip aerodynamic model.

    Values 0-3 are GEBTAero's ``aero_flag``; 4-5 come from Aeroelasticity.jl.
    """

    NONE = 0
    QUASI_STEADY = 1  # GEBTAero legacy quasi-steady (no apparent mass)
    QUASI_STEADY_ADDED_MASS = 2  # = Aeroelasticity.jl ``QuasiSteady``
    PETERS = 3  # Peters finite-state inflow (6 states) = Aeroelasticity.jl ``Peters{6}``
    STEADY = 4  # Aeroelasticity.jl ``Steady`` thin-airfoil theory
    WAGNER = 5  # Aeroelasticity.jl ``Wagner`` (Jones approximation, 2 states)


N_INFLOW_STATES = 6
N_AERO_STATES = {Aero.PETERS: N_INFLOW_STATES, Aero.WAGNER: 2}


@pytree_dataclass
class Frame:
    """Undeformed cross-section frame ``b`` expressed in frame ``a`` (``Cab``; columns = x_b, y_b, z_b)."""

    Cab: jnp.ndarray

    @staticmethod
    def from_axis(axis=(0.0, 1.0, 0.0), twist=0.0) -> Frame:
        """GEBTAero's ``Frame(Axis, Twist)``.

        ``x_b`` is the (normalised) wing axis oriented outboard (+y_a),
        ``y_b = x_b x z_a`` (towards the leading edge for an unswept wing),
        ``z_b = x_b x y_b``, then a rotation ``twist`` (rad) about ``x_b``.
        """
        axis = jnp.asarray(axis, dtype=float)
        xb = jnp.where(axis[1] > 0, axis, -axis)
        xb = xb / jnp.linalg.norm(xb)
        za = jnp.array([0.0, 0.0, 1.0])
        yb = jnp.cross(xb, za)
        yb = yb / jnp.linalg.norm(yb)
        zb = jnp.cross(xb, yb)
        a = jnp.cos(0.5 * twist)
        w = xb * jnp.sin(0.5 * twist)
        yb = yb + 2 * a * jnp.cross(w, yb) + 2 * jnp.cross(w, jnp.cross(w, yb))
        zb = zb + 2 * a * jnp.cross(w, zb) + 2 * jnp.cross(w, jnp.cross(w, zb))
        return Frame(jnp.stack([xb, yb, zb], axis=1))

    @staticmethod
    def from_angles(sweep=0.0, dihedral=0.0, twist=0.0) -> Frame:
        """Convenience: wing axis from sweep (positive aft) and dihedral (positive up), in rad."""
        axis = jnp.array(
            [-jnp.sin(sweep), jnp.cos(sweep) * jnp.cos(dihedral), -jnp.sin(dihedral)]
        )
        return Frame.from_axis(axis, twist)

    @property
    def axis(self):
        return self.Cab[:, 0]


@pytree_dataclass
class WingSection:
    """A straight wing segment between two key points (GEBTAero ``WingSection``).

    Parameters
    ----------
    chord : chord length (m)
    a : Theodorsen parameter, position of the reference axis from mid-chord in
        semi-chords (positive towards the trailing edge)
    length : segment length (m)
    cross_section : :class:`CrossSection`
    frame : :class:`Frame`
    n_elements : number of finite elements in the segment
    a0, alpha0, cm0, cd0 : section lift slope, zero-lift angle, moment and drag
        coefficients (Aeroelasticity.jl section parameters)
    damping : (6,) stiffness-proportional structural damping coefficients
        (GXBeam convention), zero by default
    """

    chord: float
    a: float
    length: float
    cross_section: CrossSection
    frame: Frame
    n_elements: int = static_field(default=10)
    a0: float = 2.0 * np.pi
    alpha0: float = 0.0
    cm0: float = 0.0
    cd0: float = 0.0
    damping: jnp.ndarray = None  # type: ignore[assignment]


@pytree_dataclass
class BeamModel:
    """Discretised wing: one entry per finite element (all arrays have leading dim ``n_el``)."""

    dL: jnp.ndarray  # (n,) element lengths
    Cab: jnp.ndarray  # (n,3,3) undeformed element frames
    coord: jnp.ndarray  # (n,3) element mid-points relative to the root
    flex: jnp.ndarray  # (n,6,6) flexibility matrices
    mass: jnp.ndarray  # (n,6,6) mass matrices
    chord: jnp.ndarray  # (n,)
    a: jnp.ndarray  # (n,)
    root: jnp.ndarray  # (3,) root position in frame a
    a0: jnp.ndarray = None  # type: ignore[assignment]  # (n,) lift slope
    alpha0: jnp.ndarray = None  # type: ignore[assignment]
    cm0: jnp.ndarray = None  # type: ignore[assignment]
    cd0: jnp.ndarray = None  # type: ignore[assignment]
    damping: jnp.ndarray = None  # type: ignore[assignment]  # (n,6)
    aero: int = static_field(default=int(Aero.PETERS))

    def __post_init__(self):
        n = self.dL.shape[0]
        defaults = {"a0": 2 * np.pi, "alpha0": 0.0, "cm0": 0.0, "cd0": 0.0}
        for k, v in defaults.items():
            if getattr(self, k) is None:
                object.__setattr__(self, k, jnp.full((n,), v))
        if self.damping is None:
            object.__setattr__(self, "damping", jnp.zeros((n, 6)))

    @property
    def n_el(self) -> int:
        return int(self.dL.shape[0])

    @property
    def n_states(self) -> int:
        return N_AERO_STATES.get(Aero(self.aero), 0)

    @property
    def n_elem_dof(self) -> int:
        return 18 + self.n_states

    @property
    def n_dof(self) -> int:
        return 12 + self.n_el * self.n_elem_dof

    @property
    def minv(self) -> jnp.ndarray:
        return jnp.linalg.inv(self.mass)

    def with_aero(self, aero: Aero | int) -> BeamModel:
        return self.replace(aero=int(aero))

    def node_arclength(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(np.asarray(self.dL))])


@pytree_dataclass
class Wing:
    """Chain of :class:`WingSection` clamped at the root (GEBTAero ``Wing``)."""

    sections: tuple
    root: jnp.ndarray = None  # type: ignore[assignment]
    name: str = static_field(default="wing")

    def __post_init__(self):
        if self.root is None:
            object.__setattr__(self, "root", jnp.zeros(3))
        object.__setattr__(self, "sections", tuple(self.sections))

    @staticmethod
    def from_sections(sections: Sequence[WingSection], root=None, name: str = "wing") -> Wing:
        return Wing(tuple(sections), jnp.zeros(3) if root is None else jnp.asarray(root, float), name)

    def key_points(self) -> jnp.ndarray:
        kps = [self.root]
        for s in self.sections:
            kps.append(kps[-1] + s.frame.axis * s.length)
        return jnp.stack(kps)

    @property
    def weight(self):
        return 9.81 * sum(s.length * s.cross_section.mass_per_length for s in self.sections)

    @property
    def surface(self):
        return sum(s.length * s.chord for s in self.sections)

    def discretize(self, aero: Aero | int = Aero.PETERS) -> BeamModel:
        """Mesh every section into ``n_elements`` equal elements."""
        dL, Cab, coord, flex, mass, chord, a = [], [], [], [], [], [], []
        a0, alpha0, cm0, cd0, damp = [], [], [], [], []
        start = jnp.zeros(3)
        for s in self.sections:
            n = s.n_elements
            seg = s.frame.axis * s.length
            frac = (jnp.arange(n) + 0.5) / n
            dL.append(jnp.full((n,), s.length / n))
            Cab.append(jnp.broadcast_to(s.frame.Cab, (n, 3, 3)))
            coord.append(start[None, :] + frac[:, None] * seg[None, :])
            flex.append(jnp.broadcast_to(s.cross_section.flexibility, (n, 6, 6)))
            mass.append(jnp.broadcast_to(s.cross_section.mass, (n, 6, 6)))
            chord.append(jnp.full((n,), s.chord))
            a.append(jnp.full((n,), s.a))
            a0.append(jnp.full((n,), s.a0))
            alpha0.append(jnp.full((n,), s.alpha0))
            cm0.append(jnp.full((n,), s.cm0))
            cd0.append(jnp.full((n,), s.cd0))
            d = jnp.zeros(6) if s.damping is None else jnp.broadcast_to(jnp.asarray(s.damping, float), (6,))
            damp.append(jnp.broadcast_to(d, (n, 6)))
            start = start + seg
        cat = lambda xs: jnp.concatenate([jnp.asarray(x, dtype=float) for x in xs], axis=0)
        return BeamModel(
            dL=cat(dL),
            Cab=cat(Cab),
            coord=cat(coord),
            flex=cat(flex),
            mass=cat(mass),
            chord=cat(chord),
            a=cat(a),
            root=jnp.asarray(self.root, dtype=float),
            a0=cat(a0),
            alpha0=cat(alpha0),
            cm0=cat(cm0),
            cd0=cat(cd0),
            damping=cat(damp),
            aero=int(aero),
        )


def straight_wing(
    cross_section: CrossSection,
    span: float,
    chord: float,
    a: float = 0.0,
    n_elements: int = 10,
    sweep: float = 0.0,
    dihedral: float = 0.0,
    twist: float = 0.0,
    aero: Aero | int = Aero.PETERS,
    **section_kw,
) -> BeamModel:
    """One-liner for the common uniform cantilever wing.

    Extra keywords (``a0``, ``alpha0``, ``cm0``, ``cd0``, ``damping``) are
    forwarded to :class:`WingSection`.
    """
    frame = Frame.from_angles(sweep, dihedral, twist)
    sec = WingSection(chord, a, span, cross_section, frame, n_elements, **section_kw)
    return Wing.from_sections([sec]).discretize(aero)
