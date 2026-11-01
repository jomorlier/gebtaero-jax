r"""Geometrically exact beam (GEBT mixed formulation) coupled with strip aerodynamics.

The whole aeroelastic system is written as a single *implicit* residual

.. math::  R(x, \dot x; p) = 0

where ``x`` gathers, for every element, the displacement ``u``, Rodrigues
rotation ``theta``, internal force ``F`` and moment ``M`` (frame ``a``/``B``),
linear and angular momenta ``P``/``H`` and, for Peters' model, the induced-flow
states ``lambda``; plus the root reactions and the tip displacement/rotation.

This mirrors ``Element.f90``/``System.f90`` of GEBT/GEBTAero, but none of the
hand-derived Jacobians are needed: the tangent stiffness ``dR/dx`` and the
mass matrix ``dR/dxdot`` are obtained with :func:`jax.jacfwd`, and every
quantity computed from them is differentiable with respect to the model
parameters ``p``.

State vector layout (``ne = 18 + n_states``)::

    x = [ root reactions (F0, M0)   : 6,
          element 1 : u, theta, F, M, P, H, lambda   : ne,
          ...
          element n,
          tip displacement / rotation (u_t, theta_t) : 6 ]
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import aerodynamics
from ._pytree import pytree_dataclass, static_field
from .rotations import E1, rotation_rodrigues
from .wing import Aero, BeamModel


@pytree_dataclass
class FlightCondition:
    """Flight / loading parameters (all differentiable).

    Parameters
    ----------
    velocity : free-stream velocity (m/s)
    rho : air density (kg/m^3)
    alpha : aircraft angle of attack (rad)
    beta : aircraft side-slip angle (rad)
    gravity : gravitational acceleration (m/s^2), 0 disables gravity
    omega_a : angular velocity of frame ``a`` (rad/s, frame a)
    v_a : linear velocity of the root/frame ``a`` (m/s, frame a)
    tip_force, tip_moment : dead loads applied at the tip (frame a)
    dist_force, dist_moment : uniform dead loads per unit span (frame a)
    load_factor : multiplies gravity, air density and applied loads (continuation)
    speed_of_sound : enables the Prandtl-Glauert correction when > 0
    tip_follower : if True the tip loads are follower loads given in the tip
        cross-section frame
    """

    velocity: float = 0.0
    rho: float = 1.225
    alpha: float = 0.0
    beta: float = 0.0
    gravity: float = 0.0
    omega_a: jnp.ndarray = None  # type: ignore[assignment]
    v_a: jnp.ndarray = None  # type: ignore[assignment]
    tip_force: jnp.ndarray = None  # type: ignore[assignment]
    tip_moment: jnp.ndarray = None  # type: ignore[assignment]
    dist_force: jnp.ndarray = None  # type: ignore[assignment]
    dist_moment: jnp.ndarray = None  # type: ignore[assignment]
    load_factor: float = 1.0
    speed_of_sound: float = 0.0
    tip_follower: bool = static_field(default=False)

    def __post_init__(self):
        for name in ("omega_a", "v_a", "tip_force", "tip_moment", "dist_force", "dist_moment"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, jnp.zeros(3))

    @property
    def flow_direction(self) -> jnp.ndarray:
        """Unit flight direction in frame ``a`` (``-wind``)."""
        a, b = self.alpha, self.beta
        return jnp.array([jnp.cos(a) * jnp.cos(b), jnp.sin(b), jnp.sin(a) * jnp.cos(b)])


def peters_matrices(n: int = 6):
    """Return ``(A, b, c)`` of Peters' finite-state inflow theory with ``n`` states."""
    return aerodynamics.peters_constants(n)


_PETERS = aerodynamics.peters_constants(6)


# --------------------------------------------------------------------------------------
# element
# --------------------------------------------------------------------------------------
def _unit(v):
    return v / jnp.linalg.norm(v)


def element_residual(xe, xed, el, fl: FlightCondition, aero: int, ns: int):
    """Residual of one element.

    Returns ``(left, right, internal)``: the 12 equations attached to the
    element's start node, the 12 attached to its end node, and the
    ``6 + ns`` element-internal equations (kinematics and inflow dynamics).
    """
    dL, Cab, coord, flex, minv, mass, chord, a, a0, alpha0, cm0, cd0, damp = el
    hL = 0.5 * dL
    u, th, F, M, P, H = (xe[3 * i : 3 * i + 3] for i in range(6))
    ud, thd = xed[0:3], xed[3:6]
    PHd = xed[12:18]
    Pd, Hd = PHd[:3], PHd[3:]
    FM = xe[6:12]
    PH = xe[12:18]

    CT, dCT = jax.jvp(rotation_rodrigues, (th,), (thd,))
    CTCab = CT @ Cab
    # strains, with optional stiffness-proportional damping (GXBeam): F = S (e + mu e_dot)
    strain = flex @ FM - damp * (flex @ xed[6:12])
    gam = strain[:3] + E1  # e1 + gamma
    kap = strain[3:]
    V = minv[:3] @ PH
    Om = minv[3:] @ PH

    om = fl.omega_a
    v_i = fl.v_a + jnp.cross(om, coord)

    # ---- equilibrium / compatibility at the two element ends ---------------------
    fu = CTCab @ F
    fm_c = hL * (CTCab @ jnp.cross(gam, F))
    fm = CTCab @ M
    fF_c = hL * (CTCab @ gam - Cab[:, 0])
    t = hL * (Cab @ kap)
    fT_c = t + 0.5 * jnp.cross(th, t) + 0.25 * th * jnp.dot(th, t)

    # ---- inertia --------------------------------------------------------------------
    CTCabP = CTCab @ P
    CTCabH = CTCab @ H
    dCTCabP = dCT @ (Cab @ P) + CTCab @ Pd
    dCTCabH = dCT @ (Cab @ H) + CTCab @ Hd
    iner_f = hL * (jnp.cross(om, CTCabP) + dCTCabP)
    iner_m = hL * (jnp.cross(om, CTCabH) + CTCab @ jnp.cross(V, P) + dCTCabH)

    # ---- external loads (per half element) ------------------------------------------
    lf = fl.load_factor
    f_ext = hL * lf * fl.dist_force
    m_ext = hL * lf * fl.dist_moment
    g = fl.gravity * lf
    mu = mass[0, 0]
    r_cg = CTCab @ jnp.array([0.0, mass[2, 3], mass[0, 4]]) / mu
    f_g = hL * mu * g * jnp.array([0.0, 0.0, 1.0])  # z_a points down
    f_ext = f_ext + f_g
    m_ext = m_ext + jnp.cross(r_cg, f_g)

    internal_aero = jnp.zeros((ns,))
    if aero != Aero.NONE:
        rho = fl.rho * lf
        U = fl.velocity
        b = 0.5 * chord
        xflow = fl.flow_direction
        xB, yB = CTCab[:, 0], CTCab[:, 1]
        d_lift = _unit(jnp.cross(xB, xflow))
        d_mom = _unit(jnp.cross(xflow, d_lift))
        alpha = jnp.arcsin(jnp.clip(jnp.dot(yB, d_lift), -1.0, 1.0))
        # section kinematics in Aeroelasticity.jl notation (u, v, omega, vdot, omegadot)
        omega = jnp.dot(CTCab @ Om, d_mom)
        hd = -jnp.dot(CTCab @ V, d_lift)
        Vd = minv[:3] @ PHd
        Omd = minv[3:] @ PHd
        omegad = jnp.dot(CTCab @ Omd, d_mom)
        vd = -jnp.dot(CTCab @ Vd, d_lift)
        v = U * alpha + hd
        beta = aerodynamics.prandtl_glauert(U, fl.speed_of_sound)
        if aero == Aero.QUASI_STEADY:
            N, Mo = aerodynamics.legacy_quasisteady_loads(a, b, rho, U, v, omega)
        elif aero == Aero.STEADY:
            N, Mo = aerodynamics.steady_loads(a, b, a0, alpha0, cm0, rho, beta, U, v)
        elif aero == Aero.QUASI_STEADY_ADDED_MASS:
            N, Mo = aerodynamics.quasisteady_loads(a, b, a0, alpha0, cm0, rho, beta, U, v, omega, vd, omegad)
        elif aero == Aero.WAGNER:
            lam, lamd = xe[18 : 18 + ns], xed[18 : 18 + ns]
            N, Mo = aerodynamics.wagner_loads(a, b, a0, alpha0, cm0, rho, beta, U, v, omega, vd, omegad, lam)
            internal_aero = aerodynamics.wagner_residual(a, b, alpha0, U, v, omega, lam, lamd)
        else:  # PETERS
            A, bv, cv = _PETERS
            lam, lamd = xe[18 : 18 + ns], xed[18 : 18 + ns]
            N, Mo = aerodynamics.peters_loads(a, b, a0, alpha0, cm0, rho, beta, bv, U, v, omega, vd, omegad, lam)
            internal_aero = aerodynamics.peters_residual(A, cv, a, b, U, omega, vd, omegad, lam, lamd)
        D = aerodynamics.drag_per_span(b, rho, U, cd0)
        f_ext = f_ext + hL * (N * d_lift - D * xflow)
        m_ext = m_ext + hL * Mo * d_mom

    fu_ext = -iner_f + f_ext
    fm_ext = -iner_m + m_ext
    left = jnp.concatenate([fu + fu_ext, fm + fm_c + fm_ext, -u + fF_c, -th + fT_c])
    right = jnp.concatenate([-fu + fu_ext, -fm + fm_c + fm_ext, u + fF_c, th + fT_c])

    # ---- element kinematics ---------------------------------------------------------
    fP = v_i + jnp.cross(om, u) - CTCab @ V + ud
    fH = CTCab.T @ om - Om + Cab.T @ ((thd - 0.5 * jnp.cross(th, thd)) / (1.0 + 0.25 * jnp.dot(th, th)))
    internal = jnp.concatenate([fP, fH, internal_aero])
    return left, right, internal


# --------------------------------------------------------------------------------------
# system
# --------------------------------------------------------------------------------------
def split_state(model: BeamModel, x: jnp.ndarray):
    """Return ``(root(6), elements(n_el, ne), tip(6))`` views of a state vector."""
    ne = model.n_elem_dof
    n = model.n_el
    return x[:6], x[6 : 6 + n * ne].reshape(n, ne), x[6 + n * ne :]


def residual(x: jnp.ndarray, xdot: jnp.ndarray, model: BeamModel, flight: FlightCondition) -> jnp.ndarray:
    """Full aeroelastic residual ``R(x, xdot)`` (root clamped, tip loaded)."""
    ns = model.n_states
    root, E, tip = split_state(model, x)
    _, Ed, _ = split_state(model, xdot)
    el = (
        model.dL, model.Cab, model.coord, model.flex, model.minv, model.mass, model.chord, model.a,
        model.a0, model.alpha0, model.cm0, model.cd0, model.damping,
    )
    elem = partial(element_residual, aero=model.aero, ns=ns)
    L, R, I = jax.vmap(elem, in_axes=(0, 0, 0, None))(E, Ed, el, flight)

    lf = flight.load_factor
    Ft, Mt = lf * flight.tip_force, lf * flight.tip_moment
    if flight.tip_follower:
        Ctip = rotation_rodrigues(tip[3:]) @ model.Cab[-1]
        Ft, Mt = Ctip @ Ft, Ctip @ Mt

    eq_root = L[0] + jnp.concatenate([-root, jnp.zeros(6)])  # clamped: u = theta = 0
    eq_mid = R[:-1] + L[1:]
    eq_tip = R[-1] + jnp.concatenate([Ft, Mt, -tip])
    return jnp.concatenate([eq_root, eq_mid.ravel(), eq_tip, I.ravel()])


def tangent_matrices(x: jnp.ndarray, model: BeamModel, flight: FlightCondition):
    """Linearisation about a steady state: ``K = dR/dx`` and ``M = dR/dxdot``."""
    xd = jnp.zeros_like(x)
    K = jax.jacfwd(residual, argnums=0)(x, xd, model, flight)
    M = jax.jacfwd(residual, argnums=1)(x, xd, model, flight)
    return K, M


def zero_state(model: BeamModel) -> jnp.ndarray:
    return jnp.zeros(model.n_dof)


# --------------------------------------------------------------------------------------
# post-processing helpers
# --------------------------------------------------------------------------------------
def node_displacements(model: BeamModel, x: jnp.ndarray):
    """Nodal displacements and Rodrigues rotations (``n_el + 1`` nodes, frame a)."""
    _, E, tip = split_state(model, x)

    def ends(xe, dL, Cab, flex):
        th = xe[3:6]
        CTCab = rotation_rodrigues(th) @ Cab
        gam = flex[:3] @ xe[6:12] + E1
        kap = flex[3:] @ xe[6:12]
        hL = 0.5 * dL
        du = hL * (CTCab @ gam - Cab[:, 0])
        t = hL * (Cab @ kap)
        dth = t + 0.5 * jnp.cross(th, t) + 0.25 * th * jnp.dot(th, t)
        return xe[0:3] - du, th - dth, xe[0:3] + du, th + dth

    u0, t0, u1, t1 = jax.vmap(ends)(E, model.dL, model.Cab, model.flex)
    u_nodes = jnp.concatenate([jnp.zeros((1, 3)), u1[:-1], tip[None, :3]])
    th_nodes = jnp.concatenate([jnp.zeros((1, 3)), t1[:-1], tip[None, 3:]])
    return u_nodes, th_nodes


def node_positions(model: BeamModel, x: jnp.ndarray | None = None):
    """Undeformed (``x=None``) or deformed nodal positions in frame ``a``."""
    pts = [model.root]
    for i in range(model.n_el):
        pts.append(pts[-1] + model.dL[i] * model.Cab[i][:, 0])
    pts = jnp.stack(pts)
    if x is None:
        return pts
    u, _ = node_displacements(model, x)
    return pts + u
