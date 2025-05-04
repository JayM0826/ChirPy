import numpy as np
from scipy.special import lpmv
from scipy.special import spherical_in
from sympy.core.numbers import pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos

from tests.script.configuration import Configration
from tests.script.utils_ext import cartesian_to_spherical, n_l_m_pairs

SQRT_2=np.sqrt(2)

def A_l_m_normalization_constant(l, m):
    return sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))


def compute_coefficients(atom_positions, sigmas, config:Configration):
    """
    compute the coefficients in sequence one by one
    """
    first_frame = atom_positions[0][:, 0:3]
    origin_atom_position = first_frame[config.ORIGIN_ATOM_INDEX]

    coefficients = np.zeros(shape=(config.N_MAX, config.L_MAX + 1, config.L_MAX * 2 +1))

    for idx, atom_position in enumerate(first_frame):
        if idx != config.ORIGIN_ATOM_INDEX: # exclude the origin atom
            for n, l, m in n_l_m_pairs(config.N_MAX):
                coeff = coefficient(origin_atom_position, atom_position - origin_atom_position, 0.5 * (sigmas[idx - 1] ** -2), n, l, m, config.L_MAX, config.SAMPLE_POINTS_NUM)
                coefficients[n - 1, l, m + config.L_MAX] += coeff

    return coefficients

def dvr_radial_I_nl_ij(n, l, r_cutoff, one_over_2_sigma_squared, r_ij, SAMPLE_POINTS_NUM = 20):
    # Gauss-Legendre quadrature points and weights
    x, w = np.polynomial.legendre.leggauss(SAMPLE_POINTS_NUM)

    # Scale quadrature points and weights
    x_n = (r_cutoff / 2.) * (x[n - 1] + 1)
    w_n = w[n - 1]

    # Compute the radial integral, spherical_in: modified spherical Bessel function of the first kind
    I_nl_ij = r_cutoff / 2. * np.sqrt(w_n) * x_n ** 2 * spherical_in(l, 2 * one_over_2_sigma_squared * x_n * np.linalg.norm(r_ij))
    return I_nl_ij

def coefficient(origin_atom, r_ij, one_over_2_sigma_squared, n, l, m, r_cutoff, SAMPLE_POINTS_NUM):
    theta, phi,  = cartesian_to_spherical(*r_ij, np.linalg.norm(r_ij))
    # TODO double check the order of the theta, phi when passing to Y_lm_real_scipy
    return (4 * np.pi * Y_bar_l_m(l, m, phi, theta)
            * np.exp(-one_over_2_sigma_squared * np.sum(r_ij ** 2))
            * dvr_radial_I_nl_ij(n, l, r_cutoff, one_over_2_sigma_squared, r_ij, SAMPLE_POINTS_NUM))

def P_bar_l_m(l, m, theta):
    """
        https://lab-cosmo.github.io/librascal/SOAP.html#spherical-harmonics

        lpmv: Associated Legendre function of integer order and real degree.
    """
    return np.sqrt(float((2 * l + 1) * factorial(l - m) / (2 * np.pi) / factorial(l + m))) * lpmv(m, l, np.cos(theta))


def Y_bar_l_m(l, m, theta, phi):

    """
    https://lab-cosmo.github.io/librascal/SOAP.html#spherical-harmonics
    """
    if m > 0:
        return cos(m * phi) * P_bar_l_m(l, m, theta)
    elif m < 0:
        return sin(-m * phi) * P_bar_l_m(l, -m, theta)
    else:
        return SQRT_2 * P_bar_l_m(l, 0, theta)


