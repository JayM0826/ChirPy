import numpy as np
from scipy.special import sph_harm, lpmv, factorial

from tests.script.analytical_utils import Y_bar_l_m, dvr_basis_function
from tests.script.numerical_utils import Y_lm_real_scipy, Y_lm_real_fun_sympy
from tests.script.utils_ext import l_m_pairs

def test_spherical_integral():
    import sympy as sp

    # Define symbols
    a, r, rij, theta, phi = sp.symbols('a r rij theta phi', real=True, positive=True)

    # Define the integrand
    k = 2 * a * r * rij
    integrand = sp.exp(-k * sp.cos(theta)) * sp.sin(theta)

    # First integrate over theta
    theta_integral = sp.integrate(integrand, (theta, 0, sp.pi))

    # Then integrate over phi (trivial)
    phi_integral = sp.integrate(1, (phi, 0, 2 * sp.pi))

    # Total result
    total_integral = sp.simplify(phi_integral * theta_integral)

    # Print result
    sp.pprint(total_integral, use_unicode=True)


def testYlm():
    SQRT_2 = 1.0 / np.sqrt(2)

    # Generate 100,000 random test values for theta in [0, pi] and phi in [0, 2pi]
    np.random.seed(42)
    theta_rand = np.random.uniform(0, np.pi, 100000)
    phi_rand = np.random.uniform(0, 2 * np.pi, 100000)
    for l, m in l_m_pairs(10):
        print(l, m)
        y1 = Y_bar_l_m(l, m, theta_rand, phi_rand)
        y2 = Y_lm_real_scipy(l, m, theta_rand, phi_rand)
        y3_fun = Y_lm_real_fun_sympy(l, m)
        y3 = y3_fun(theta_rand, phi_rand)
        assert np.allclose(y2, y1, rtol=0.000001, atol=0.00001), 'Backward check failed'


def test_sph_orthogonal_normality(fun):
    # Set up grid for integration
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    # Test orthonormality for l = 1, m = -1, 0, 1
    for l1, m1 in l_m_pairs(5):
        for l2, m2 in l_m_pairs(5):
            Y1 = fun(l1, m1, theta_grid, phi_grid)
            Y2 = fun(l2, m2, theta_grid, phi_grid)
            integrand = Y1 * Y2 * np.sin(theta_grid)
            result = np.sum(integrand) * dtheta * dphi
            print(f"⟨Y_{l1}^{m1}, Y_{l2}^{m2}⟩ = {result:.5f}")

# test_sph_orthogonal_normality(Y_bar_l_m)

def test_dvr_orthogonal_normality(fun):
    x_plot = np.linspace(-1, 1, 1000)
    delta_x = x_plot[1] - x_plot[0]
    LEGENDRE_ORDER_NUM = 7
    for n1 in range(LEGENDRE_ORDER_NUM):
        for n2 in range(LEGENDRE_ORDER_NUM):
            integrand = fun(n1, x_plot, LEGENDRE_ORDER_NUM) * fun(n2, x_plot, LEGENDRE_ORDER_NUM) * delta_x
            result = np.sum(integrand)
            print(f"⟨dvr_basis_function_{n1}, dvr_basis_function_{n2}⟩ = {result:.15f}")
test_dvr_orthogonal_normality(dvr_basis_function)
