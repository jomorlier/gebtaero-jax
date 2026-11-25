# Pazy wing data (Technion model, skin, no tip mass)

Copied from https://github.com/UM-A2SRL/AePW3-LDWG (Riso & Cesnik, University of Michigan / Georgia Tech).
Equivalent beam (reference axis at 44 % chord, 16 nodes / 15 elements) extracted by Riso & Cesnik from the
Technion built-up FEM (Avin, Raveh et al.). Spreadsheets were converted to CSV, values unchanged.

| file | content |
|---|---|
| coordinates.csv | beam nodes x, y, z (m), MSC Nastran frame, origin at root |
| inertia_skin1.csv | per node: mass, CG offset (cgx, cgy, cgz), inertia about own CG (CONM2 convention) |
| stiffness_skin1.csv | per element: K11 axial, K22 torsion, K33 out-of-plane, K44 in-plane bending + couplings (Su & Cesnik strain-based) |
| inertia_skin0.csv, stiffness_skin0.csv | same, model without the foil skin |
| aero_coeff.txt | y (m), lift slope (1/rad), quarter-chord moment slope (1/rad) — spanwise strip coefficients |
| umnast_static_aoa{5,7}.txt | UM/NAST tip sweep: U (m/s), z, y, x (% semispan), twist (deg) — nonlinear, follower, no gravity |
| umnast_static_aoa7_linear.txt | same, linear kinematics |
| umnast_modal_deformed_aoa7.txt | U, z tip (%), wind-off frequencies of modes 1-5 (Hz), tracked step to step (its undeformed frequencies, 4.03 / 27.9 / 38.6 / 82.6 / 101.8 Hz, differ slightly from both beam variants; its tip deflections equal the skin1 static sweep) |
| umnast_flutter_undeformed.txt | U, then (freq Hz, Re s rad/s) for 3 modes |
| umnast_flutter_{onset,offset}_vs_aoa.txt | root AoA (deg), hump-mode onset/offset speed (m/s) |
| umnast_flutter_onset_vs_uztip.txt | tip deflection at onset (% semispan), onset speed |
| umnast_frequencies_undeformed{,_skin0}.txt | 5 undeformed frequencies (Hz), with / without skin |
| exp_static_aoa{5,7}_{disp,twist}.txt | Technion experiment: U (m/s), tip z (% semispan) / twist (deg) |

Experimental flutter (Ritter et al., J. Aircraft 2026, Table 4): onset 49 / 43 / 38 m/s and offset 55 / 48 / — m/s at 3 / 5 / 7 deg.

Please cite Riso & Cesnik (J. Aircraft 60(3) 2023; J. Fluids Struct. 120, 2023), Avin et al. (AIAA J. 60(3) 2022),
Drachinsky et al. (AIAA J. 60(9) 2022) and Ritter et al. (J. Aircraft, 2026, doi 10.2514/1.C038508).
