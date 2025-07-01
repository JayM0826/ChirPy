from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy

from utils_ext import cartesian_to_spherical


@dataclass
class Configuration:

    """
    cfg = Configuration()
    print(cfg.ORIGIN_ATOM_INDEX)

    if you want to override the default values, it is fine:
    cfg = Configuration(NUMBER_PER_UNIT_DISTANCE=64, SIGMA=1.)
    """

    # **********************CONSTANT**************************************
    ONE_OVER_TWO_PI_POWER_1DIV2: float = 1 / np.power(2 * np.pi, 0.5)
    ONE_OVER_TWO_PI_POWER_3DIV2: float = np.power(ONE_OVER_TWO_PI_POWER_1DIV2, 3)
    NUMBER_PER_UNIT_DISTANCE: int = 5
    # base 0
    ORIGIN_ATOM_INDEX: int = 0
    # CUT_OFF = 2. # 1nit angstrom: 10**(-10) m
    CUT_OFF: float = 1.8  # unit angstrom: 10**(-10) m
    # n = 0,1,2,3,4,5,6, base 0,
    # different from the quantum mechanics num, here it does not have
    # physical meaning.
    N_MAX: int = 5
    # l=0,1,2,3,4,5, here L can be larger than N_MAX, because N_MAX
    # is not the quantum principle num as before.
    # but here as a convention, L_MAX=N_MAX-1
    L_MAX: int = 8
    DVR_BASIS_NUM = N_MAX+1  # for numerical radial basis
    # Minimal value which can be evaluated with the spherical bessel function
    SPHERICAL_BESSEL_FUNCTION_FTOL = 1e-6
    BASE_PATH = Path(__file__).parent
    SIGMAS = None

    # numerical spherical grid
    # LEBEDEV_ORDER: int = 131
    # LEBEDEV_POINTS, LEBEDEV_WEIGHTS = scipy.integrate.lebedev_rule(LEBEDEV_ORDER)
    # LEBEDEV_THETA, LEBEDEV_PHI = cartesian_to_spherical(*LEBEDEV_POINTS)




