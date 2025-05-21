from dataclasses import dataclass
import numpy as np
import scipy

import utils_ext

from pathlib import Path



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
    # CUT_OFF = 2. # unit angstrom: 10**(-10) m
    CUT_OFF: float = 5.  # unit angstrom: 10**(-10) m
    N_MAX: int = 7 # n = 0,1,2,3,4,5,6, base 0, it is different from the quantum mechanics num, here it does not have physical meaning.
    L_MAX: int = N_MAX - 1 # l=0,1,2,3,4,5, here L can be larger than N_MAX, because N_MAX is not the quantum principle num as before. but here as a convention, L_MAX=N_MAX-1
    LEBEDEV_ORDER: int = 131
    LEBEDEV_POINTS, LEBEDEV_WEIGHTS = scipy.integrate.lebedev_rule(LEBEDEV_ORDER)
    LEBEDEV_THETA, LEBEDEV_PHI = utils_ext.cartesian_to_spherical(*LEBEDEV_POINTS)
    DVR_BASIS_NUM = N_MAX # for numerical radial basis
    # Minimal value which can be evaluated with the spherical bessel function
    SPHERICAL_BESSEL_FUNCTION_FTOL = 1e-5
    BASE_PATH = Path(__file__).parent


