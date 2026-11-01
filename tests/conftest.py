import numpy as np
import pytest

import gebtaero_jax as gx


@pytest.fixture(scope="session")
def goland_cs():
    """Goland wing cross-section as in GEBTAero's test suite."""
    i11 = 8.64
    return gx.CrossSection.from_isotropic(9.77e6, 0.0, 0.99e6).with_mass(35.71, 1e-4 * i11, i11, 0.0, 0.18288)


@pytest.fixture(scope="session")
def patil_cs():
    i11 = 0.1
    return gx.CrossSection.from_isotropic(2e4, 0.0, 1e4).with_mass(0.75, 1e-3 * i11, i11, 0.0, 0.0)


GOLAND = dict(span=6.096, chord=1.8288, a=-0.34, n_elements=10)
PATIL = dict(span=16.0, chord=1.0, a=0.0, n_elements=10)
DEG = np.pi / 180
