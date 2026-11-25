# Mapping: GEBTAero and Aeroelasticity.jl → gebtaero-jax

Status legend: ✅ ported · 🔁 replaced by a different (documented) approach · ➕ new in gebtaero-jax · ⏳ not ported yet

## 1. GEBTAero (Fortran core)

| GEBTAero | gebtaero-jax | Status / notes |
|---|---|---|
| `Element.f90::ElemEqn` | `beam.element_residual` | ✅ same equations (left/right half-element + kinematics) |
| `Element.f90::ElemJacobian`, `ElemMass` (~600 lines of hand derivatives) | `beam.tangent_matrices` = `jax.jacfwd` | 🔁 exact AD; also linearises the aero terms GEBTAero approximated (`Jdir_lift`, `Jalpha_THETA`) |
| `System.f90`, `Member.f90` (sparse assembly, point conditions) | `beam.residual` | 🔁 dense vectorised assembly (`vmap`), clamped root / loaded tip |
| `PrescribedCondition.f90` follower loads | `FlightCondition.tip_follower` | ✅ tip loads; ⏳ general prescribed DOFs at intermediate key points |
| `GlobalDataFun.f90::DirCosineTRodrigues`, `CT_THETA` | `rotations.rotation_rodrigues` (+ `jax.jvp`) | ✅ |
| `GlobalDataFun.f90::Peters, MatA, VecB, VecC` | `aerodynamics.peters_constants` | ✅ |
| `SolveMumps.f90` Newton–Raphson | `solvers.solve_static` | 🔁 Newton in `lax.while_loop`, wrapped in `lax.custom_root` (implicit differentiation) |
| `EigenSolveMumps.f90` (ARPACK shift-invert) | `solvers.modal_analysis` | 🔁 dense QZ (`scipy.linalg.eig`) with pencil equilibration + left eigenvectors |
| `Analysis.f90` time marching (trapezoidal) | `solvers.simulate`, `integrators.generalized_alpha` | 🔁 generalised-α (trapezoid unstable with rigid strains); `lax.scan` |
| `TimeFunction.f90` | `flight_fn(t, flight)` callables | 🔁 any jax-traceable function |
| `IOaero.f90` `.dat` input | `legacy.read_dat` | ✅ chain of straight members, clamped/free |
| `.vtk` output, Paraview scripts | `plotting.plot_wing_3d`, `plot_vg` | 🔁 matplotlib |
| curved members (`ncurv`) | — | ⏳ |
| aero gravity moment `-g μ X_cg e_z` (legacy) | physical gravity at the CG from the mass matrix | 🔁 |

## 2. GEBTAero (Python pre/post-processor)

| GEBTAero | gebtaero-jax | Status / notes |
|---|---|---|
| `IsoMaterial`, `OrthoMaterial`, `CompositePly` | `IsoMaterial`, `OrthoMaterial`, `Ply`, `Laminate` | ✅ pytrees (differentiable) |
| `CrossSection.SetFlexibilityMatrixByIsotropicValues` | `CrossSection.from_isotropic` | ✅ |
| `CrossSection.SetMassMatrix` | `CrossSection.with_mass` / `mass_matrix` | ✅ identical entries |
| `CrossSection.SetFlexibilityMatrixByRectBeamValues`, `SetMassMatrixByRectBeamValues` | `CrossSection.rectangular` | ✅ |
| `CompositePlate` + CalculiX periodic 3D FEM | `CrossSection.composite_plate` (CLT, free-edge condensation) | 🔁 analytic & differentiable; thin-plate approximation |
| `CompositeBox`, `ExternalMesh` (CalculiX) | — | ⏳ (thin-walled box theory planned) |
| `Frame(Axis, Twist)` | `Frame.from_axis`, `Frame.from_angles` | ✅ |
| `WingSection`, `Wing` | `WingSection`, `Wing`, `Wing.discretize` | ✅ + section aero coefficients and damping |
| `Simulation.Eigenvalues` | `modal_analysis(...).table()` | ✅ |
| `Simulation.StaticLoads` | `solve_static` + `static_outputs` | ✅ |
| `ModalFlutterSpeed`, `ModalCriticalSpeed`, `ModalDivergenceSpeed`, `*Sorted` | `flutter_speed`, `divergence_speed`, `critical_speeds` | 🔁 grid scan + Brent (flutter) / `sign det K` bisection (divergence) |
| `EigenTabSorted`, `EigenTab` | `mode_sweep` | ✅ MAC + optimal assignment |
| `DeformedModalCriticalSpeed` | `flutter_speed(..., deformed=True)` (auto) | ✅ |
| `EquilibriumAoA` | Newton on α with `jax.grad` through `solve_static` (notebook 03) | 🔁 |
| `TemporalFlutterSpeed`, `FlutterVtk`, `TemporalDynamic`, `TemporalDivergenceSpeed` | `simulate` (notebook 04) | 🔁 |
| `GebtPlot.EigenFreqDamping` | `plotting.plot_vg` | ✅ |
| `examples/optimisation` (OpenMDAO, genetic) | `flutter_speed_and_grad` + SLSQP (notebook 05) | 🔁 exact gradients |
| — | `eigenvalue_gradient`, `flutter_speed_and_grad`, `divergence_speed_and_grad` | ➕ |

## 3. Aeroelasticity.jl

| Aeroelasticity.jl | gebtaero-jax | Status / notes |
|---|---|---|
| `CoupledModel(submodels, parameters; finput, fparam)` | `coupled.CoupledModel(submodels, coupling, params)` | ✅ submodel residual `residual(dx, x, y, p, t)` |
| sparsity detection / colouring (`SparseDiffTools`, `Symbolics`) | — | 🔁 not needed for dense systems |
| `rate_jacobian!`, `state_jacobian!`, `linearize` | `coupled.linearize` (`jax.jacfwd`) | ✅ |
| `dense_eigen` | `coupled.dense_eigen` | ✅ left vectors normalised `U M V = I` |
| `sparse_eigen` (Arpack) | — | 🔁 dense only |
| `correlate_eigenmodes` | `coupled.correlate_eigenmodes` | ✅ (optimal assignment instead of greedy) |
| `ODEFunction`/`DAEFunction` + `DifferentialEquations.jl` | `coupled.simulate` | 🔁 generalised-α |
| `SteadyStateProblem` | `coupled.steady_state` | ✅ implicitly differentiable |
| `Steady`, `QuasiSteady`, `Wagner`, `Peters{N}` + `*_loads` | `aerodynamics.*`, `section.Steady/QuasiSteady/Wagner/Peters` | ✅ identical load expressions (a0, α0, cm0, cd0, β) |
| `Section` (typical section) | `section.TypicalSection` | ✅ |
| `SteadySection`, `QuasiSteadySection`, `WagnerSection`, `PetersSection` couplings | `section.section_model(aero)` | ✅ |
| `section_coordinates` | — | ⏳ (plotting only) |
| `LiftingLine` + `*LiftingLine` couplings | strip aerodynamics inside `beam.element_residual` (`Aero.STEADY/…/WAGNER/PETERS`) | 🔁 one strip per beam element, same section models |
| `GXBeamAssembly` (GXBeam.jl) | `beam` (GEBT) | 🔁 same theory family (both derive from Hodges' mixed formulation) |
| `LiftingLineGXBeam` coupling (`V`, `rho`, `beta`, gravity, prescribed conditions) | `FlightCondition` (velocity, rho, alpha, beta, gravity, speed_of_sound…) | ✅ |
| GXBeam stiffness-proportional damping | `WingSection(damping=...)` | ✅ |
| `RigidBody` (6-DOF), `LiftingLineRigidBody` | prescribed frame motion `omega_a`, `v_a` only | ⏳ free-flight coupling |
| `examples/section-stability`, `section-simulation` | notebook 06 | ✅ |
| `examples/cantilever-steady`, `cantilever-stability` | notebooks 01, 03 | ✅ |
| `examples/goland-stability` | notebook 02 | ✅ |
| — | `coupled.critical_value`, `critical_value_gradient`, `section.theodorsen_flutter` | ➕ |

## 4. Behavioural differences worth knowing

* **Exact linearisation.** GEBTAero's hand-written aerodynamic Jacobians neglect the dependence of the lift/moment
  directions on the deformed rotation. About the undeformed state this is exact; about a drooped wing it is not.
  This explains 23.13 m/s (exact) vs 23.35 m/s (GEBTAero) for the gravity-deformed Patil wing.
* **Legacy quasi-steady model** (`Aero.QUASI_STEADY`, GEBTAero flag 1) has no apparent mass and a different
  pitch-rate moment term than Aeroelasticity.jl's `QuasiSteady` (= `Aero.QUASI_STEADY_ADDED_MASS`).
* **Peters inflow at U → 0.** GEBTAero replaces the inflow stiffness `U` by `max(U, 1/U)` to decouple
  inflow modes at very low speed; gebtaero-jax keeps the physical `U` (searches start at `v_min > 0`).
* **Spurious roots.** The mixed formulation with rigid strains and tiny rotary inertias produces very
  high-frequency roots; they are ignored above `fmax_hz` (default 1000 Hz, like GEBTAero's `FreqLim`) and the
  eigen-pencil is equilibrated before solving.
* **Time integration.** Generalised-α with `rho_inf = 0.9` by default (`rho_inf = 1` ≈ GEBT's trapezoid).

## 4bis. Mode tracking and eigen-adjoint (S. He's note, `sensitivity_modetracking.ipynb`)

| Source | gebtaero-jax | Status / notes |
|---|---|---|
| He: residual `r_i = [A phi_i - lambda_i phi_i ; phi_i^T phi_i - 1]` | `tracking.bordered_residual`: `[(K + sM) phi ; e^T phi - 1]` | 🔁 complex, non-symmetric pencil of the beam; analytic normalisation `e^T phi = 1` |
| He: `dr/dq = [[A - lambda I, -phi], [2 phi^T, 0]]` | `tracking.bordered_jacobian` (real split) | ✅ |
| He: `g_i = phi_r^T phi_i`, `f = 1 - sum g_i / N` | `tracking.mac_function`, `tracked_sensitivities().mac_loss` | 🔁 MAC (phase/scale invariant, complex shapes) |
| He: adjoint `(dr/dq)^T psi = dg/dq`, `dg/dx = -psi^T dr/dx` | `tracking.eigenpair_adjoint` | ✅ one LU per mode, one back-substitution per function, `dr/dx` matrix-free by AD |
| notebook §3.1 MAC step-to-step / reference tracking | `tracking.track` (+ `stability.mode_sweep`) | ✅ Hungarian assignment |
| notebook §3.3 complex modes, left eigenvector | `stability.eigenvalue_gradient` (used to verify the adjoint) | ✅ |
| notebook §4.2–4.4 flutter-point bordered system and adjoint | `tracking.flutter_gradient_adjoint` | 🔁 modal flutter point (`Re s = 0`) instead of the p–k flutter equations |
| notebook §4.5 flutter-constrained optimisation | notebook 08 (CRM-like wingbox, 32 variables) | ✅ |

## 4ter. Solar HALE design tools (notebook 09)

| Source | gebtaero-jax | Status / notes |
|---|---|---|
| EcoHALE notebook: optimum planform, masses, speeds, Sandwich-1 properties, 10–60 % wingbox | `examples/hale_wings.EcoHALEWing` | ✅ published values; wall distribution reconstructed (spars at 1 mm bound), t/c calibrated on the published 1.7 m gust deflection |
| EcoHALE notebook: gust point (V_D = 1.4 V, n = 1.1, 3.4 m/s) | notebook 09 §3 (nonlinear, trimmed) | ✅ |
| EcoHALE notebook: 15 % deflection limit, linear beam | geometrically exact deflection, linear comparison | 🔁 |
| EcoHALE: buckling / stress / material catalogue / CO₂ objective | — | ⏳ (stays in the AeroSandbox notebook) |
| DawnDesignTool: point design (AIAA 2021-2451 Table 2), break 0.67, taper 0.5, booms 0.8, HALE_03, 3 g | `examples/hale_wings.DawnWing` | ✅ published values; CFRP tube spar sized by 3 g bending stress, secondary/solar/boom masses estimated |
| DawnDesignTool: `mass_wing_spar` (mass only) | tube spar stiffness + mass, nonlinear deflection, flutter | 🔁 |
| DawnDesignTool: trajectory / energy / solar / propulsion models | — | ⏳ (out of scope: these size the aircraft; we check its aeroelasticity) |
| motors, booms, pods as point masses | `BeamModel.with_point_mass` | ➕ |
| trim (lift = weight) | `hale_wings.trim_alpha` (Newton, AD through the nonlinear equilibrium) | ➕ |

## 4quater. AePW3 Large Deflection Working Group — Pazy wing (notebook 10)

| Source | gebtaero-jax | Status / notes |
|---|---|---|
| AePW3-LDWG `00_Models/01_Pazy_Technion/02_Models_Beam` (coordinates, inertia, stiffness, skin0/skin1) | `examples/pazy_wing.PazyWing` | ✅ data unchanged (xlsx → csv); 4×4 strain-based stiffness inverted into the GEBT flexibility, shear rigid; nodal CONM2 inertia → sectional mass via `cross_section.rigid_body_mass_matrix` |
| spanwise lift slope `aero_coeff.txt` | `BeamModel.a0` per element | ✅ moment-slope column not used |
| LDWG task 1: static aeroelastic sweeps (3/5/7°) | notebook 10 §2 | ✅ vs UM/NAST and Technion experiment |
| LDWG task 2: wind-off frequencies vs deflection | notebook 10 §3, `convert_state`, `tracking.reference_from_tracked` | ✅ step-to-step MAC tracking |
| LDWG task 3: hump-mode onset/offset vs AoA | notebook 10 §4 | ✅ vs UM/NAST, experiment and LDWG mean (Table 4) |
| UVLM / DLM / CFD entries, tip-loss strip variant, stall / LCO | — | ⏳ |
| — | onset-speed gradients through the deformed equilibrium, first-order UQ | ➕ |
