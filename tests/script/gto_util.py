import numpy as np
from numpy import pi
from scipy.integrate import quad
from scipy.special import gamma, hyp1f1
from sympy.functions.combinatorial.factorials import factorial


###############below is for gto, be careful to use it, because it always blows up


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