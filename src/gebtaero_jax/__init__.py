"""gebtaero-jax: differentiable nonlinear aeroelasticity of high-aspect-ratio wings.

A JAX re-implementation of GEBTAero (B. Kirsch, ISAE-Supaero / French Air
Force Academy), itself built on Prof. Wenbin Yu's Geometrically Exact Beam
Theory code (GEBT). Jacobians, mass matrices and design sensitivities are
obtained by automatic differentiation.

It also maps the modular design of Aeroelasticity.jl (T. McDonnell, BYU FLOW
lab): generic coupled implicit models (:mod:`gebtaero_jax.coupled`), the 2D
Steady/QuasiSteady/Wagner/Peters aerodynamics (:mod:`gebtaero_jax.aerodynamics`)
and the typical section (:mod:`gebtaero_jax.section`).
"""

import jax

jax.config.update("jax_enable_x64", True)

from . import aerodynamics, coupled, integrators, plotting, rotations, section, tracking  # noqa: E402
from .beam import (  # noqa: E402
    FlightCondition,
    convert_state,
    node_displacements,
    node_positions,
    peters_matrices,
    residual,
    split_state,
    tangent_matrices,
)
from .coupled import CoupledModel  # noqa: E402
from .cross_section import CrossSection, mass_matrix  # noqa: E402
from .legacy import read_dat  # noqa: E402
from .materials import IsoMaterial, Laminate, OrthoMaterial, Ply  # noqa: E402
from .solvers import (  # noqa: E402
    ModalResult,
    TimeHistory,
    lift_force,
    modal_analysis,
    mode_shape,
    root_loads,
    simulate,
    solve_static,
    static_outputs,
    tip_displacement,
    tip_twist,
)
from .stability import (  # noqa: E402
    CriticalPoint,
    ModeSweep,
    critical_speeds,
    divergence_speed,
    divergence_speed_and_grad,
    eigenvalue_gradient,
    flutter_speed,
    flutter_speed_and_grad,
    mode_sweep,
)
from .wing import Aero, BeamModel, Frame, Wing, WingSection, straight_wing  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "aerodynamics", "coupled", "integrators", "plotting", "rotations", "section", "tracking", "CoupledModel",
    "Aero", "BeamModel", "convert_state", "CriticalPoint", "CrossSection", "FlightCondition", "Frame",
    "IsoMaterial", "Laminate", "ModalResult", "ModeSweep", "OrthoMaterial", "Ply",
    "TimeHistory", "Wing", "WingSection", "critical_speeds", "divergence_speed",
    "divergence_speed_and_grad", "eigenvalue_gradient", "flutter_speed",
    "flutter_speed_and_grad", "lift_force", "mass_matrix", "modal_analysis", "mode_shape",
    "mode_sweep", "node_displacements", "node_positions", "peters_matrices", "read_dat",
    "residual", "root_loads", "simulate", "solve_static", "split_state", "static_outputs",
    "straight_wing", "tangent_matrices", "tip_displacement", "tip_twist",
]
