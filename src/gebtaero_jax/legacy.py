"""Reader for GEBTAero's Fortran ``.dat`` input files (migration helper).

Supported subset: a chain of straight members (no curvature, no distributed
load functions, no time functions), the first key point clamped and the last
key point free (optionally with applied loads), one aerodynamic line per
material. That covers every file written by GEBTAero's Python pre-processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .beam import FlightCondition
from .cross_section import CrossSection
from .wing import Frame, Wing, WingSection


@dataclass
class LegacyCase:
    wing: Wing
    flight: FlightCondition
    analysis_flag: int
    aero_flag: int
    grav_flag: int
    n_modes: int | None


class _Tokens:
    def __init__(self, text: str):
        toks = []
        for line in text.splitlines():
            line = line.split("#", 1)[0]
            toks.extend(line.split())
        self.toks = toks
        self.i = 0

    def take(self, n=1, cast=float):
        vals = [cast(float(t)) if cast is int else cast(t) for t in self.toks[self.i : self.i + n]]
        if len(vals) < n:
            raise ValueError("unexpected end of GEBTAero input file")
        self.i += n
        return vals if n > 1 else vals[0]


def read_dat(path: str | Path) -> LegacyCase:
    """Parse a GEBTAero ``.dat`` file into a :class:`Wing` and :class:`FlightCondition`."""
    tk = _Tokens(Path(path).read_text())
    analysis_flag, aero_flag, grav_flag, _niter, _nstep, _nvtk = tk.take(6, int)
    omega = np.zeros(3)
    v_a = np.zeros(3)
    n_modes = None
    if analysis_flag != 0:
        omega = np.array(tk.take(3))
        tk.take(3, int)
        v_a = np.array(tk.take(3))
        tk.take(3, int)
        if analysis_flag == 3:
            n_modes = tk.take(1, int)
    npoint, nmemb, ncond_pt, nmate, nframe, ncond_mb, ndistr, ntimefun, ncurv = tk.take(9, int)
    if ncond_mb or ndistr or ncurv:
        raise NotImplementedError("distributed loads / curvatures are not supported by the legacy reader")

    kps = {}
    for _ in range(npoint):
        idx = tk.take(1, int)
        kps[idx] = np.array(tk.take(3))
    members = []
    for _ in range(nmemb):
        _id, kp1, kp2, m1, m2, fr, ndiv, _curv = tk.take(8, int)
        if m1 != m2:
            raise NotImplementedError("linearly varying sections (mat1 != mat2) are not supported")
        members.append((kp1, kp2, m1, fr, ndiv))

    tip_force = np.zeros(3)
    tip_moment = np.zeros(3)
    last_kp = members[-1][1]
    for _ in range(ncond_pt):
        kp = tk.take(1, int)
        dofs = tk.take(6, int)
        vals = np.array(tk.take(6))
        tk.take(6, int)  # time functions
        follower = tk.take(6, int)
        if kp == last_kp and list(dofs) == [7, 8, 9, 10, 11, 12]:
            tip_force, tip_moment = vals[:3], vals[3:]
            if any(follower):
                raise NotImplementedError("follower tip loads are not supported by the legacy reader")

    mats = {}
    U = rho = alpha = beta = 0.0
    for _ in range(nmate):
        idx = tk.take(1, int)
        flex = np.array(tk.take(36)).reshape(6, 6)
        mass = np.array(tk.take(36)).reshape(6, 6)
        chord, a = 1.0, 0.0
        if aero_flag > 0:
            U, rho, chord, a, alpha, beta, _ = tk.take(7)
            if grav_flag:
                tk.take(1)  # Xcg (legacy, unused)
        mats[idx] = (CrossSection(jnp.asarray(flex), jnp.asarray(mass)), chord, a)

    frames = {}
    for _ in range(nframe):
        idx = tk.take(1, int)
        frames[idx] = Frame(jnp.asarray(np.array(tk.take(9)).reshape(3, 3)))

    sections = []
    for kp1, kp2, m, fr, ndiv in members:
        length = float(np.linalg.norm(kps[kp2] - kps[kp1]))
        cs, chord, a = mats[m]
        sections.append(WingSection(chord, a, length, cs, frames[fr], ndiv))
    wing = Wing.from_sections(sections, root=kps[members[0][0]], name=Path(path).stem)
    flight = FlightCondition(
        velocity=U,
        rho=rho,
        alpha=alpha,
        beta=beta,
        gravity=9.81 if grav_flag else 0.0,
        omega_a=jnp.asarray(omega),
        v_a=jnp.asarray(v_a),
        tip_force=jnp.asarray(tip_force),
        tip_moment=jnp.asarray(tip_moment),
    )
    return LegacyCase(wing, flight, analysis_flag, aero_flag, grav_flag, n_modes)
