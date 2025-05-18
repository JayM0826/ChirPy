import numpy as np
from scipy.special import lpmv, gamma, hyp1f1, spherical_in, eval_legendre, roots_legendre
from sympy.core.numbers import pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos

from scipy.integrate import quad


from tests.script.configuration import Configration
from tests.script.utils_ext import cartesian_to_spherical, l_m_pairs

# *********CONSTANTS USED*************************
SQRT_2=np.sqrt(2)
PI_POW_3_DIV_2 = np.pi ** (1.5)
# ************************************************


def phi_n(n, x):
    """
    Orthonormal Legendre polynomial function on [-1, 1]
    n:base 0
    """
    weight_func = 1 # Polynomial dependent, for definition, see Eq. 2.1. in Light, J. C., & Carrington, T. (2007). Discrete-Variable Representations and their Utilization (pp. 263–310). John Wiley & Sons, Ltd. https://doi.org/10.1002/9780470141731.ch4
    norm = np.sqrt((2 * n + 1) / 2)
    return norm * eval_legendre(n, x) * np.sqrt(weight_func)

# Construct DVR basis function ψ_j(x)
def dvr_basis_function(j, grid_x, LEGENDRE_ORDER_NUM):
    root_x, weight_x = roots_legendre(LEGENDRE_ORDER_NUM)
    return sum(phi_n(n, root_x[j]) * phi_n(n, grid_x) for n in range(LEGENDRE_ORDER_NUM)) * np.sqrt(w[j])


def I_nl_ij_dvr(n, l, r_cutoff, one_over_2_sigma_squared, r_ij, LEGENDRE_ORDER_NUM):
    # Gauss-Legendre quadrature points and weights
    x, w = roots_legendre(LEGENDRE_ORDER_NUM)

    # Scale quadrature points and weights
    x_n = (r_cutoff / 2.) * (x[n - 1] + 1)
    w_n = w[n - 1]

    # Compute the radial integral, spherical_in: modified spherical Bessel function of the first kind
    I_nl_ij = r_cutoff / 2. * np.sqrt(w_n) * x_n ** 2 * spherical_in(l, 2 * one_over_2_sigma_squared * x_n * np.linalg.norm(r_ij))
    return I_nl_ij


def compute_coefficients(atom_positions, sigmas, config:Configration):
    """
    compute the coefficients in sequence one by one
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
                coefficient(origin_atom_position, atom_position - origin_atom_position, 0.5 * (sigmas[idx] ** -2), n, l, m, config.L_MAX, config.DVR_BASIS_NUM) for (l, m) in l_m_list
            ], dtype=np.float64)

            coeff_n_lm += coeff_n_lm_i

        full_coeff.extend(coeff_n_lm.tolist())

    return np.array(full_coeff, dtype=np.float64)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # Number of DVR points / polynomial order
    N = 100

    # Get Gauss-Legendre quadrature points (nodes) and weights
    x_grid, w = roots_legendre(N)

    # Grid for plotting
    x_plot = np.linspace(-1, 1, 1000)

    # transform to R
    r_c = 15
    r_plot = x_plot * r_c/2 + r_c/2
    r_grid = x_grid * r_c/2 + r_c/2

    # -- test settings
    a = 4
    r_neighbours = [2., 4. , 3, 1]

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


###############below is for gto


def A_l_m_normalization_constant(l, m):
    return np.sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))

def normalization_factor_n(n, sigma_n):
    # N_n=sqrt(2/\Gamma(n+3/2)) b_n^{3+2n}/4
    # #this->radial_norm_factors(radial_n) =
    #           0.25 * std::sqrt(2.0 / (std::tgamma(1.5 + radial_n) *
    #                                   pow(this->radial_sigmas[radial_n],
    #                                       3 + 2 * radial_n)));
    # TODO why there is a 0.25 in rascal, i think which is wrong
    # return 0.25 *  np.sqrt(2 / (gamma(1.5 + n) * sigma_n ** (3 + 2 * n)))
    return np.sqrt(2 / (gamma(1.5 + n) * sigma_n ** (3 + 2 * n)))

# GTO radial basis
# when we got the radial coefficient of gto, we can reconstruct the original density function theoretically
# the basis is normalization but not orthogonal to each other
def R_basis_gto(n, r_coordinate, r_cutoff, n_max):
    """
    n:[0, n_max)
    """
    sigma = get_gto_basis_sigma(n, n_max, r_cutoff)
    # 0.5 * pow(this->radial_sigmas[radial_n], -2)
    b = 0.5 * (sigma ** (-2))
    # normalization factor
    N_n = normalization_factor_n(n, sigma)
    return  N_n * (r_coordinate ** n) * np.exp(-b * r_coordinate **2)


def get_gto_basis_sigma(n, n_max, r_cutoff):
    """
        n:[0, n_max)
        # this->radial_sigmas[radial_n] =
    #               std::max(std::sqrt(static_cast<double>(radial_n)), 1.0) *
    #               this->interaction_cutoff / static_cast<double>(this->max_radial);

    """
    sigma = (r_cutoff) * max(np.sqrt(n), 1) / (n_max)
    return sigma


def orthogonality_integral(n, m, r_cutoff, n_max):
    def integrand(r):
        Rn = R_basis_gto(n, r, r_cutoff, n_max)
        Rm = R_basis_gto(m, r, r_cutoff, n_max)
        return Rn * Rm * r ** 2

    result, _ = quad(integrand, 0, 15 * r_cutoff, epsabs=1e-8, epsrel=1e-8)
    return result


def orthogonality_integral_test():
    r_cutoff = 5
    n_max = 5
    test_pairs =[ (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 0), (6,6), (2,4)]
    for n, m in test_pairs:
        integral = orthogonality_integral(n, m, r_cutoff, n_max)
        print(f"Integral <R_{n}|R_{m}> = {integral:.6f}")

# orthogonality_integral_test()


def safe_hyp1f1_scaled(a, b, x):
    """Scaled hyp1f1 = hyp1f1 * exp(-x), safe for large x"""
    if x > 50:  # heuristic threshold, adjust as needed
        # Use asymptotic approximation instead
        return (x**(a - b)) * np.exp(0)  # scaled version is constant
    else:
        return hyp1f1(a, b, x) * np.exp(-x)


# I_nl^GTO
def I_nl_ij_gto(n, l, r_ij, r_cutoff, n_max):
    """
    n:[0,n_max)
    l:[0, n_max)  not sure
    """
    sigma = get_gto_basis_sigma(n, n_max, r_cutoff)
    # b_n = 1 / (2*\sigma_n^2)
    # this->fac_b[radial_n] = 0.5 * pow(this->radial_sigmas[radial_n], -2);
    # b should be different for different atoms, but here sigmas are same
    b = 0.5 * (sigma ** (-2))
    # a = 0.5 * (radial + angular + 3)
    a = 4
    N_n = normalization_factor_n(n, sigma)
    prefactor = N_n * (np.sqrt(np.pi) / 4)

    numerator_gamma = gamma((n + l + 3) / 2)
    denominator_gamma = gamma(l + 1.5)

    a_l = a ** l
    rij_l = r_ij ** l
    ab_term = (a + b) ** (-(n + l + 3) / 2)
    hyper_arg = (a ** 2 * r_ij ** 2) / (a + b)
    print(f"hyper_arg={hyper_arg}")
    first_parameter = (n + l + 3) / 2.
    # b = angular + 1.5
    second_parameter = l + 1.5
    confluent_hypergeom = hyp1f1(first_parameter, second_parameter, float(hyper_arg))
    # confluent_hypergeom_scaled = safe_hyp1f1_scaled(first_parameter, second_parameter, float(hyper_arg))
    # confluent_hypergeom = confluent_hypergeom_scaled * np.exp(hyper_arg)
    return prefactor * (numerator_gamma / denominator_gamma) * a_l * rij_l * ab_term * confluent_hypergeom



def c_ij_nlm_gto(n, l, m, r_ij_vec, r_cutoff, n_max):
    a = 0.5 * (n + l + 3)
    r_ij = np.linalg.norm(r_ij_vec)
    I_nl_ij = I_nl_ij_gto(n, l, r_ij, r_cutoff, n_max)

    # Main expression
    prefactor = PI_POW_3_DIV_2 * I_nl_ij * np.exp(-a * r_ij ** 2)
    # Compute unit vector for spherical harmonics
    theta, phi = cartesian_to_spherical(*r_ij_vec, r_ij)
    return prefactor * Y_bar_l_m(l, m, theta, phi)








def coefficient(origin_atom, r_ij, one_over_2_sigma_squared, n, l, m, r_cutoff, DVR_BASIS_NUM):
    theta, phi,  = cartesian_to_spherical(*r_ij, np.linalg.norm(r_ij))
    return (4 * np.pi * Y_bar_l_m(l, m, theta, phi) # be careful the order of theta and phi
            * np.exp(-one_over_2_sigma_squared * np.sum(r_ij ** 2))
            * I_nl_ij_dvr(n, l, r_cutoff, one_over_2_sigma_squared, r_ij, DVR_BASIS_NUM))

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

    Note that it lacks (-1)^m compared to the wiki definition: https://en.wikipedia.org/wiki/Spherical_harmonics#Real%20form
    When times (-1)^m again, it is same as the implementation in tests.script.numerical_utils.Y_lm_real_scipy
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




if __name__ == '__main__':
    import numpy as np
    from scipy.special import sph_harm, lpmv, factorial

    SQRT_2 = 1.0 / np.sqrt(2)


    def P_bar_l_m(l, m, theta):
        return (-1) ** m * np.sqrt((2 * l + 1) * factorial(l - m) / (2 * np.pi) / factorial(l + m)) * lpmv(m, l, np.cos(
            theta))


    def Y_bar_l_m(l, m, theta, phi):
        if m > 0:
            return np.cos(m * phi) * P_bar_l_m(l, m, theta)
        elif m < 0:
            return np.sin(-m * phi) * P_bar_l_m(l, -m, theta)
        else:
            return SQRT_2 * P_bar_l_m(l, 0, theta)


    def sph_harm_y(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)  # Note: scipy order (m, l, phi, theta)


    def Y_lm_real_scipy(l, m, theta, phi):
        if m == 0:
            return sph_harm_y(l, 0, theta, phi).real
        elif m > 0:
            return np.sqrt(2) * (-1) ** m * sph_harm_y(l, m, theta, phi).real
        else:
            return np.sqrt(2) * (-1) ** m * sph_harm_y(l, -m, theta, phi).imag


    # Generate 100,000 random test values for theta in [0, pi] and phi in [0, 2pi]
    np.random.seed(42)
    theta_rand = np.random.uniform(0, np.pi, 100000)
    phi_rand = np.random.uniform(0, 2 * np.pi, 100000)

    # Test over a few (l, m) combinations
    results = {}
    for l, m in [(1, 0), (2, 1), (3, 2), (4, -1), (5, -3)]:
        y1 = Y_bar_l_m(l, m, theta_rand, phi_rand)
        y2 = Y_lm_real_scipy(l, m, theta_rand, phi_rand)
        max_diff = np.max(np.abs(y1 - y2))
        results[(l, m)] = max_diff

    print(results)
