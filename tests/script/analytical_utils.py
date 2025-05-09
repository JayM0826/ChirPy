import numpy as np
from scipy.special import lpmv, gamma, hyp1f1, spherical_in, eval_legendre, roots_legendre
from sympy.core.numbers import pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos



from tests.script.configuration import Configration
from tests.script.utils_ext import cartesian_to_spherical, n_l_m_pairs


# *********CONSTANTS USED*************************
SQRT_2=np.sqrt(2)
PI_POW_3_DIV_2 = np.pi ** (1.5)
# ************************************************

def A_l_m_normalization_constant(l, m):
    return sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))

def Nn_squared(n, sigma_n):
    return 2 / (sigma_n ** (2 * n + 3) * gamma(n + 1.5))


# GTO radial basis
# when we got the radial coefficient of gto, we can reconstruct the original density function theoretically
def R_basis_gto(n, sigma, r):
    N_n = sqrt(Nn_squared(n, sigma))
    return N_n * r**n * np.exp(-0.5 * (sigma ** (-2)) * r**2)


def c_ij_nlm_gto(n, l, m, sigma, r_ij_vec):
    """
    Compute c^{ij GTO}_{nlm} coefficient for some ij pair. i: origin atom, j: neighbouring atom

    Parameters:
    - n, l, m: quantum numbers
    - a, b: Gaussian parameters: a?=b: 0.5 * (sigma_n ** (-2))
    - r_ij_vec: relative position vector between atoms i and j (3D array-like)


    Returns:
    - c_ij_gto: real coefficient
    """
    a = b =  0.5 * (sigma ** (-2))

    r_ij = np.linalg.norm(r_ij_vec)

    # Compute unit vector for spherical harmonics
    theta, phi = cartesian_to_spherical(*r_ij_vec, r_ij)

    # Gamma function parts
    numerator_gamma = gamma((n + l + 3) / 2)
    denominator_gamma = gamma(l + 1.5)

    # Hypergeometric function
    z = (a ** 2 * r_ij ** 2) / (a + b)
    hyp = hyp1f1((n + l + 3) / 2, l + 1.5, z)

    # Main expression
    prefactor = PI_POW_3_DIV_2 * sqrt(Nn_squared(n, sigma)) * numerator_gamma / denominator_gamma * (a + b) ** (-(n + l + 3) / 2)
    radial = np.exp(-a * r_ij ** 2) * (a * r_ij) ** l

    return prefactor * Y_bar_l_m(l, m, theta, phi) * radial * hyp


# I_nl^GTO
def I_nl_ij_gto(n, l, k, a, b, rij, sigma=1.0):
    N_n = sqrt(Nn_squared(n, sigma))
    prefactor = N_n * (np.sqrt(np.pi) / 4)

    gamma_num = gamma((n + l + k + 3) / 2)
    gamma_den = gamma(l + 1.5)

    a_l = a ** l
    rij_l = rij ** l
    ab_term = (a + b) ** (- (n + l + k + 3) / 2)
    hyper_arg = (a ** 2 * rij ** 2) / (a + b)

    confluent_hypergeom = hyp1f1((n + l + k + 3) / 2, l + 1.5, hyper_arg)

    return prefactor * (gamma_num / gamma_den) * a_l * rij_l * ab_term * confluent_hypergeom




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
                coeff = coefficient(origin_atom_position, atom_position - origin_atom_position, 0.5 * (sigmas[idx - 1] ** -2), n, l, m, config.L_MAX, config.DVR_BASIS_NUM)
                coefficients[n - 1, l, m + config.L_MAX] += coeff # shifting m by config.L_MAX to store them

    return coefficients


# Orthonormal Legendre polynomial function on [-1, 1]
def phi_n(n, x):
    norm = np.sqrt((2 * n + 1) / 2)
    return norm * eval_legendre(n, x)

# Construct DVR basis function ψ_j(x)
def dvr_basis_function(j, grid_x, LEGENDRE_ORDER_NUM):
    root_x, weight_x = roots_legendre(LEGENDRE_ORDER_NUM)
    return sum(phi_n(n, root_x[j]) * phi_n(n, grid_x) for n in range(LEGENDRE_ORDER_NUM))


def I_nl_ij_dvr(n, l, r_cutoff, one_over_2_sigma_squared, r_ij, LEGENDRE_ORDER_NUM):
    # Gauss-Legendre quadrature points and weights
    x, w = roots_legendre(LEGENDRE_ORDER_NUM)

    # Scale quadrature points and weights
    x_n = (r_cutoff / 2.) * (x[n - 1] + 1)
    w_n = w[n - 1]

    # Compute the radial integral, spherical_in: modified spherical Bessel function of the first kind
    I_nl_ij = r_cutoff / 2. * np.sqrt(w_n) * x_n ** 2 * spherical_in(l, 2 * one_over_2_sigma_squared * x_n * np.linalg.norm(r_ij))
    return I_nl_ij

def coefficient(origin_atom, r_ij, one_over_2_sigma_squared, n, l, m, r_cutoff, LEGENDRE_ORDER_NUM):
    theta, phi,  = cartesian_to_spherical(*r_ij, np.linalg.norm(r_ij))
    return (4 * np.pi * Y_bar_l_m(l, m, phi, theta)
            * np.exp(-one_over_2_sigma_squared * np.sum(r_ij ** 2))
            * I_nl_ij_dvr(n, l, r_cutoff, one_over_2_sigma_squared, r_ij, LEGENDRE_ORDER_NUM))

def P_bar_l_m(l, m, theta):
    """
        https://lab-cosmo.github.io/librascal/SOAP.html#spherical-harmonics

        lpmv: Associated Legendre function of integer order and real degree.
    """
    return np.sqrt(float((2 * l + 1) * factorial(l - m) / (2 * np.pi) / factorial(l + m))) * lpmv(m, l, np.cos(theta))


def Y_bar_l_m(l, m, theta, phi):
    """
    l: angular momentum quantum num, [0, n-1]
    m: magnetic quantum num, [-l, l]
    theta: polar angle, [0, pi]
    phi: azimuthal angle, [0, 2pi]
    """

    """
    https://lab-cosmo.github.io/librascal/SOAP.html#spherical-harmonics
    """
    if m > 0:
        return cos(m * phi) * P_bar_l_m(l, m, theta)
    elif m < 0:
        return sin(-m * phi) * P_bar_l_m(l, -m, theta)
    else:
        return SQRT_2 * P_bar_l_m(l, 0, theta)


import matplotlib.pyplot as plt
# Number of DVR points / polynomial order
N = 20

# Get Gauss-Legendre quadrature points (nodes) and weights
x_grid, w = roots_legendre(N)



# Grid for plotting
x_plot = np.linspace(-1, 1, 1000)

# transform to R
r_c = 6
r_plot = x_plot*r_c/2 + r_c/2
r_grid = x_grid*r_c/2 + r_c/2

# -- test settings
a = 4
r_neighbours = [2., 4.]

# --- summed density
rho_tilde = np.zeros_like(r_plot)
# Plot DVR basis functions
plt.figure(figsize=(10, 6))

# here a, namely sigma,  controls the connection between the below density function and the superposition of the dvr radial basis
for r_ij in r_neighbours:
    for j in range(N):
        cj =  np.exp(-a*r_ij**2) * r_c/2  * np.sqrt(w[j]) * np.exp(-a*r_grid[j]**2) * spherical_in(0, 2*a*r_grid[j]*r_ij)
        # plt.plot(r_plot, dvr_function(j, x_plot)*np.sqrt(w[j]), color='k', alpha=(1.2-j/N)/1.2*0.2, lw=1, ls=':', zorder=-1000)
        plt.plot(r_plot, cj * dvr_basis_function(j, x_plot, N) * r_plot**2, '--') # r_plot**2 jacobian
        rho_tilde += cj * dvr_basis_function(j, x_plot, N) * r_plot**2

# --- compare with radial part of Gaussian function * R**2
# this is the radial density function, not radial basis
rho = np.zeros_like(rho_tilde)
for r_ij in r_neighbours:
    rho += r_c/2. * np.exp(-a*(r_plot**2+r_ij**2))* np.sinh(2*a*r_plot*r_ij) / (2*a*r_ij) * r_plot

plt.plot(r_plot, rho, 'r-', lw=2, label=r'$\rho(r)$')
plt.plot(r_plot, rho_tilde, 'k-', lw=2, label=r'$\tilde\rho(r)$')

plt.title("Legendre DVR Expansion")
plt.xlabel('r')
plt.ylabel('ψ_j(x)')
plt.ylim(-0.5, 1.1)
plt.grid(True)
plt.legend()
plt.show()

assert np.allclose(rho, rho_tilde, rtol=0.01, atol=0.01), 'Backward check failed'