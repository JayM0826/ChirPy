from math import sqrt

import numpy as np
from numpy import pi
from scipy.special import eval_legendre, roots_legendre, sph_harm_y
from scipy.special import lpmv, spherical_in
from sympy.functions.combinatorial.factorials import factorial

from tests.script.configuration import Configuration
from tests.script.utils_ext import cartesian_to_spherical, l_m_pairs

# *********CONSTANTS USED*************************
SQRT_2=sqrt(2)
PI_POW_3_DIV_2 = np.pi ** (1.5)
# ************************************************


# --- radial part
def phi_n(n, x):
    """
    Orthonormal Legendre polynomial function on [-1, 1]
    from numpy.polynomial.legendre import leggauss, Legendre
    It equals Legendre.basis(n)(x) / norm below, because Legendre.basis(n)(x)
    is not normalized
    namely eval_legendre(n, x) = Legendre.basis(n)(x)
    n:base 0

    """
    # common weight function:
    # https://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf,
    # page 17 Eq. 2.50 and 2.55, This weight is used for integration rather
    # than sum
    # Polynomial dependent, for definition, see Eq. 2.1. in Light, J. C., &
    # Carrington, T. (2007). Discrete-Variable Representations and their
    # Utilization (pp. 263–310). John Wiley & Sons, Ltd.
    # https://doi.org/10.1002/9780470141731.ch4
    weight_func = 1
    norm = np.sqrt((2 * n + 1) / 2)  # for normalization
    return norm * eval_legendre(n, x) * np.sqrt(weight_func)


# Construct DVR basis function ψ_j(x)
def dvr_basis_function(j, sample_points, LEGENDRE_ORDER_NUM):
    """
    here sample_points must be in [-1, 1]
    """
    root_x, weight_x = roots_legendre(LEGENDRE_ORDER_NUM)
    return sum(phi_n(n, sample_points) * phi_n(n, root_x[j])
               for n in range(LEGENDRE_ORDER_NUM)) * np.sqrt(weight_x[j])


def I_nl_ij_dvr(n, l, r_cutoff, one_over_2_sigma_squared, r_ij_norm,
                LEGENDRE_ORDER_NUM):
    # Gauss-Legendre quadrature points and weights
    x, w = roots_legendre(LEGENDRE_ORDER_NUM)

    # Scale quadrature points and weights from [-1,1] -> [0,r_c]
    x_n = (r_cutoff / 2.) * (x[n] + 1)
    # Here: required division by x_n has been omitted because it is absorbed by
    # the Jacobian x_n**2 below: x_n**2/x_n = x_n
    w_n = w[n] / (r_cutoff / 2.)
    # https://lab-cosmo.github.io/librascal/SOAP.html#eq:real-spherical-harmonics
    # Compute the radial integral, spherical_in: modified spherical Bessel
    # function of the first kind
    I_nl_ij = (
                r_cutoff / 2.  # dr_dx
                * np.sqrt(w_n)
                * x_n
                * np.exp(-one_over_2_sigma_squared * x_n ** 2)
                * spherical_in(
                    l,
                    2 * one_over_2_sigma_squared * x_n * r_ij_norm
                    )
                )
    return I_nl_ij



def compute_coefficients_in_sequence(atom_positions, sigmas, config:Configuration):
    """
    compute the coefficients in sequence one by one
    Due to the length of all ls belonging to different ns  is same,
    here we store the coefficients as shape=(n, (l+1)^2)
    """
    first_frame = atom_positions[0][:, 0:3]
    origin_atom_position = first_frame[config.ORIGIN_ATOM_INDEX]

    full_coeff = []

    l_m_list = list(l_m_pairs(config.L_MAX))
    num_lm = len(l_m_list)

    for n in range(config.N_MAX):
        coeff_n_lm = np.zeros(num_lm, dtype=np.float64)

        for idx, atom_position in enumerate(first_frame):
            if idx == config.ORIGIN_ATOM_INDEX:
                continue

            coeff_n_lm_i = np.array([
                coefficient(atom_position - origin_atom_position, 0.5 * (sigmas[idx] ** -2), n, l, m, config.CUT_OFF, config.DVR_BASIS_NUM) for (l, m) in l_m_list
            ], dtype=np.float64)

            coeff_n_lm += coeff_n_lm_i

        full_coeff.extend(coeff_n_lm.tolist())

    return np.array(full_coeff, dtype=np.float64)


def compute_coefficients_in_dict_old(atom_positions, sigmas, config: Configuration):
    """
    Compute the coefficients indexed by (n, l, m) quantum numbers.
    """
    first_frame = atom_positions[0][:, 0:3]
    origin_atom_position = first_frame[config.ORIGIN_ATOM_INDEX]

    # Dictionary to store coefficients with (n, l, m) keys
    coefficients = {}

    l_m_list = list(l_m_pairs(config.L_MAX))

    for n in range(config.N_MAX):
        for l, m in l_m_list:
            coefficients[(n, l, m)] = 0.0

        for idx, atom_position in enumerate(first_frame):
            if idx == config.ORIGIN_ATOM_INDEX:
                continue

            for l, m in l_m_list:
                coeff_n_lm_i = coefficient(
                    origin_atom_position,
                    atom_position - origin_atom_position,
                    0.5 * (sigmas[idx] ** -2),
                    n, l, m,
                    config.CUT_OFF,
                    config.DVR_BASIS_NUM
                )
                coefficients[(n, l, m)] += coeff_n_lm_i

    return coefficients

def compute_coefficients_in_dict(atom_positions, config: Configuration):
    """
    SJ CHANGED CODE: atom_positions are the relative positions within cutoff
    Compute the coefficients indexed by (n, l, m) quantum numbers.
    """
    _frame = 0

    # Dictionary to store coefficients with (n, l, m) keys
    # SJ: dicts are slow
    coefficients = {}

    l_m_list = list(l_m_pairs(config.L_MAX))

    for n in range(config.N_MAX+1):
        for l, m in l_m_list:
            coefficients[(n, l, m)] = 0.0

        for _pos, _s in zip(atom_positions[_frame][:, :3], config.SIGMAS):
            for l, m in l_m_list:
                coeff_n_lm_i = coefficient(
                    _pos,
                    0.5*(_s**-2),
                    n, l, m,
                    config.CUT_OFF,  # SJ removed BUG: not L_MAX
                    config.DVR_BASIS_NUM
                )
                coefficients[(n, l, m)] += coeff_n_lm_i

    return coefficients



if __name__ == '__main__1':

    import matplotlib.pyplot as plt
    # Number of DVR points / polynomial order
    N = 120

    # Get Gauss-Legendre quadrature points (nodes) and weights
    zero_point_grid, w = roots_legendre(N)

    # Grid for plotting
    sample_points_plot = np.linspace(-1, 1, 1000)

    # transform to R
    r_c = 15
    scaled_sample_points_plot = sample_points_plot * r_c / 2 + r_c / 2
    scaled_zero_point_grid = zero_point_grid * r_c / 2 + r_c / 2

    # -- test settings
    a = 4
    r_neighbours = [2., 4. , 3, 1]

    # --- summed density
    rho_tilde = np.zeros_like(scaled_sample_points_plot)
    # Plot DVR basis functions
    plt.figure(figsize=(10, 6))

    # here a, namely sigma,  controls the connection between the below density function and the superposition of the dvr radial basis
    for r_ij in r_neighbours:
        for j in range(N):
            # why does l only take the value of 0???
            cj = np.exp(-a*r_ij**2) * r_c / 2 * np.sqrt(w[j]) * np.exp(-a * scaled_zero_point_grid[j] ** 2) * spherical_in(0, 2 * a * scaled_zero_point_grid[j] * r_ij)
            # plt.plot(r_plot, dvr_function(j, x_plot)*np.sqrt(w[j]), color='k', alpha=(1.2-j/N)/1.2*0.2, lw=1, ls=':', zorder=-1000)
            plt.plot(scaled_sample_points_plot, cj * dvr_basis_function(j, sample_points_plot, N) * scaled_sample_points_plot ** 2, '--') # r_plot**2 jacobian
            rho_tilde += cj * dvr_basis_function(j, sample_points_plot, N) * scaled_sample_points_plot ** 2


    # --- compare with radial part of Gaussian function * R**2
    # this is the radial density function, not radial basis
    rho = np.zeros_like(rho_tilde)
    for r_ij in r_neighbours:
        # rho += r_c / 2. * np.exp(-a * (scaled_sample_points_plot ** 2 + r_ij ** 2)) * np.sinh(
        #     2 * a * scaled_sample_points_plot * r_ij) / (2 * a * r_ij) * scaled_sample_points_plot

        integral_gaussion = 2 * np.pi / (a * scaled_sample_points_plot *  r_ij) * np.exp(-a * (scaled_sample_points_plot ** 2 + r_ij ** 2)) * np.sinh(2 * a * scaled_sample_points_plot * r_ij)
        rho += r_c / 2. * (integral_gaussion / (2 * np.pi)  / 2) *  scaled_sample_points_plot ** 2

    plt.plot(scaled_sample_points_plot, rho, 'r-', lw=2, label=r'$\rho(r)$')
    plt.plot(scaled_sample_points_plot, rho_tilde, 'k-', lw=2, label=r'$\tilde\rho(r)$')

    plt.title("Legendre DVR Expansion")
    plt.xlabel('r')
    plt.ylabel('ψ_j(x)')
    plt.ylim(-0.5, 1.1)
    plt.grid(True)
    plt.legend()
    plt.show()

    assert np.allclose(rho, rho_tilde, rtol=0.01, atol=0.01), 'Backward check failed'



def coefficient(r_ij, one_over_2_sigma_squared, n, l, m, r_cutoff,
                DVR_BASIS_NUM):
    r_ij_norm = np.linalg.norm(r_ij)
    theta, phi = cartesian_to_spherical(*r_ij, r_ij_norm)

    return (4*np.pi * sph_harm_y(l, m, theta, phi)
            * np.exp(-one_over_2_sigma_squared*r_ij_norm**2)
            * I_nl_ij_dvr(
                n,
                l,
                r_cutoff,
                one_over_2_sigma_squared,
                r_ij_norm,
                DVR_BASIS_NUM
                )
            )


def P_bar_l_m(l, m, theta):
    m = abs(m)
    norm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return norm * lpmv(m, l, np.cos(theta)) * (-1) ** m

def Y_bar_l_m(l, m, theta, phi):
    if m > 0:
        return SQRT_2 * P_bar_l_m(l, m, theta) * np.cos(m * phi)
    elif m < 0:
        return SQRT_2 * P_bar_l_m(l, -m, theta) * np.sin(-m * phi)
    else:
        return P_bar_l_m(l, 0, theta)

