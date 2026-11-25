"""Pazy wing (AePW3 Large Deflection Working Group benchmark) as a gebtaero-jax beam model.

Data: the equivalent-beam model of Riso & Cesnik extracted from the Technion built-up FEM (with skin, no tip mass),
https://github.com/UM-A2SRL/AePW3-LDWG, copied in ``examples/data/pazy`` (see the README there).

* Geometry: semi-span 0.55 m, chord 0.1 m, NACA 0018, reference axis at 44 % chord, 16 nodes / 15 elements.
* Stiffness: per element 4x4 (axial, torsion, out-of-plane, in-plane) of the strain-based formulation; transverse
  shear is rigid (zero flexibility), which GEBT's mixed formulation handles directly.
* Inertia: lumped CONM2-like masses per node (mass, CG offset, inertia about the CG). Each node's rigid-body mass
  matrix is split between the two beam elements adjacent to it (each data element is subdivided ``refine`` times
  so the lumped masses stay close to their nodes).
* Aerodynamics: strip theory with the spanwise lift slope of ``aero_coeff.txt`` (it includes the 3D/tip effects and
  goes to zero at the tip) and Peters' unsteady inflow. The (small) moment-slope correction of the same file is not
  used: our strip model puts the aerodynamic centre at the quarter chord.
* Frames: MSC Nastran frame (x aft, y span, z up) -> beam frame b (x_b span, y_b forward, z_b up):
  ``x_b = y_n, y_b = -x_n, z_b = z_n``. The signs of the stiffness couplings (all small except axial/in-plane,
  whose sign does not change the frequencies) are taken as given.
* The wing is tested vertically in the tunnel: gravity is ignored, as in the LDWG reference computations.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import gebtaero_jax as gx
from gebtaero_jax.cross_section import rigid_body_mass_matrix

DATA = Path(__file__).parent / "data" / "pazy"
SEMISPAN, CHORD, X_REF = 0.55, 0.1, 0.44
RHO = 1.225
R_NB = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # b <- Nastran


def _csv(name):
    return np.genfromtxt(DATA / name, delimiter=",", names=True)


def load_reference(name):
    return np.loadtxt(DATA / name)


class PazyWing:
    """Builder of the Pazy beam model. ``params`` (all scale factors, default 1) make it differentiable:

    ``GJ``, ``EI_oop``, ``EI_ip`` scale the torsion / out-of-plane / in-plane stiffnesses, ``mass`` the inertia,
    ``cla`` the lift slope.
    """

    def __init__(self, refine: int = 1, aero=gx.Aero.PETERS, skin: bool = True):
        self.refine, self.aero, self.skin = refine, aero, skin
        tag = "skin1" if skin else "skin0"
        xyz = np.loadtxt(DATA / "coordinates.csv", delimiter=",", skiprows=1)
        self.y_nodes = xyz[:, 1]
        self.inertia = _csv(f"inertia_{tag}.csv")
        self.stiff = _csv(f"stiffness_{tag}.csv")
        ac = np.loadtxt(DATA / "aero_coeff.txt")
        self.n_data = len(self.y_nodes) - 1
        # sub-elements
        edges = []
        for e in range(self.n_data):
            edges += list(np.linspace(self.y_nodes[e], self.y_nodes[e + 1], refine + 1)[:-1])
        edges.append(self.y_nodes[-1])
        self.edges = np.asarray(edges)
        self.dL = np.diff(self.edges)
        self.mid = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.parent = np.repeat(np.arange(self.n_data), refine)
        self.cla = np.interp(self.mid, ac[:, 0], ac[:, 1])
        self._mass = self._lumped_mass()
        self.a = (X_REF - 0.5) / 0.5

    # ------------------------------------------------------------------ data -> beam
    def _lumped_mass(self):
        """Per-element (n_el, 6, 6) mass per unit length; node k shared by its two adjacent sub-elements.

        A node's matrix is about the node, while an element's mass acts about its mid-point: the offset (half a
        sub-element along x_b) is accounted for by shifting the CG accordingly.
        """
        n_el = len(self.dL)
        M = np.zeros((n_el, 6, 6))
        node_el = {}  # node index -> list of (element, signed offset of the node from the element mid-point)
        for k in range(len(self.y_nodes)):  # root node included (it barely moves; keeps every element massive)
            j = int(np.argmin(np.abs(self.edges - self.y_nodes[k])))
            adj = [e for e in (j - 1, j) if 0 <= e < n_el]
            node_el[k] = [(e, self.y_nodes[k] - self.mid[e]) for e in adj]
        for k, lst in node_el.items():
            r = self.inertia[k]
            I = np.array([[r["Ixx"], r["Ixy"], r["Ixz"]], [r["Ixy"], r["Iyy"], r["Iyz"]], [r["Ixz"], r["Iyz"], r["Izz"]]])
            cg = R_NB @ np.array([r["cgx"], r["cgy"], r["cgz"]])
            for e, dx in lst:
                Mk = np.asarray(rigid_body_mass_matrix(r["mass"] / len(lst), cg + np.array([dx, 0, 0]), R_NB @ I @ R_NB.T / len(lst)))
                M[e] += Mk / self.dL[e]
        return M

    def total_mass(self):
        return float(np.sum(self._mass[:, 0, 0] * self.dL))

    def flexibility(self, p=None):
        p = {} if p is None else p
        s = self.stiff
        rows = []
        for e in self.parent:
            r = s[e]
            K = jnp.array([[r["K11"], r["K12"], r["K13"], r["K14"]],
                           [r["K12"], r["K22"] * p.get("GJ", 1.0), r["K23"], r["K24"]],
                           [r["K13"], r["K23"], r["K33"] * p.get("EI_oop", 1.0), r["K34"]],
                           [r["K14"], r["K24"], r["K34"], r["K44"] * p.get("EI_ip", 1.0)]])
            F4 = jnp.linalg.inv(K)
            idx = jnp.array([0, 3, 4, 5])
            rows.append(jnp.zeros((6, 6)).at[jnp.ix_(idx, idx)].set(F4))
        return jnp.stack(rows)

    def build(self, params=None, V=0.0, alpha_deg=0.0, aero=None):
        """Return ``(model, flight)``; ``alpha_deg`` is the root angle of attack (positive: lift up)."""
        p = {} if params is None else params
        aero = self.aero if aero is None else aero
        n = len(self.dL)
        frame = gx.Frame.from_axis([0, 1, 0])
        model = gx.BeamModel(
            dL=jnp.asarray(self.dL), Cab=jnp.broadcast_to(frame.Cab, (n, 3, 3)),
            coord=jnp.stack([jnp.zeros(n), jnp.asarray(self.mid), jnp.zeros(n)], axis=1),
            flex=self.flexibility(p), mass=jnp.asarray(self._mass) * p.get("mass", 1.0),
            chord=jnp.full((n,), CHORD), a=jnp.full((n,), self.a), root=jnp.zeros(3),
            a0=jnp.asarray(self.cla) * p.get("cla", 1.0), aero=int(aero))
        return model, gx.FlightCondition(velocity=V, rho=RHO, alpha=jnp.deg2rad(alpha_deg))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def tip_z_percent(model, x):
        return -100.0 * float(gx.tip_displacement(model, x)[2]) / SEMISPAN

    @staticmethod
    def tip_twist_deg(model, x):
        return float(np.degrees(gx.tip_twist(model, x)))
