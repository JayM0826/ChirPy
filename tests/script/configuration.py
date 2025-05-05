from dataclasses import dataclass
import numpy as np
import scipy

import utils_ext

from pathlib import Path



@dataclass
class Configration:

    """
    cfg = Configration()
    print(cfg.ORIGIN_ATOM_INDEX)

    if you want to override the default values, it is fine:
    cfg = Config(NUMBER_PER_UNIT_DISTANCE=64, SIGMA=1.)
    """

    # **********************CONSTANT**************************************
    ONE_OVER_TWO_PI_POWER_1DIV2: float = 1 / np.power(2 * np.pi, 0.5)
    ONE_OVER_TWO_PI_POWER_3DIV2: float = np.power(ONE_OVER_TWO_PI_POWER_1DIV2, 3)
    NUMBER_PER_UNIT_DISTANCE: int = 5
    # base 0
    ORIGIN_ATOM_INDEX: int = 0
    # CUT_OFF = 2. # unit angstrom: 10**(-10) m
    CUT_OFF: float = 5.  # unit angstrom: 10**(-10) m
    N_MAX: int = 7 # n = 1,2,3,4,5,6,7
    L_MAX: int = N_MAX - 1 # l=0,1,2,3,4,5,6
    LEBEDEV_ORDER: int = 131
    LEBEDEV_POINTS, LEBEDEV_WEIGHTS = scipy.integrate.lebedev_rule(LEBEDEV_ORDER)
    LEBEDEV_THETA, LEBEDEV_PHI = utils_ext.cartesian_to_spherical(*LEBEDEV_POINTS)
    SAMPLE_POINTS_NUM = N_MAX # for numerical radial basis
    BASE_PATH = Path(__file__).parent


