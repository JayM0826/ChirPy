import random

import chirpy as cp
import numpy as np

import numerical_utils as num_utils
from tests.script import utils_ext
from tests.script.analytical_utils import Y_bar_l_m
from tests.script.configuration import Configuration
from tests.script.numerical_utils import Y_lm_real_scipy
from tests.script.utils_ext import l_m_pairs
from pathlib import Path

from scipy.spatial.transform import Rotation as R
import analytical_utils as ana_utils


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
        # y3_fun = Y_lm_real_fun_sympy(l, m)
        # y3 = y3_fun(theta_rand, phi_rand)
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
# test_dvr_orthogonal_normality(dvr_basis_function)





def compare_arrays(name, arr1, arr2):
    if np.allclose(arr1, arr2):
        print(f"{name}: ✅ Equal")
    else:
        print(f"{name}: ❌ Not Equal")


def test_translational_invariant_for_generate_grid_and_bounds():
    BASE_PATH = Path(__file__).parent
    filepath = utils_ext.get_relative_path(BASE_PATH, "tartrate.xyz").as_posix()
    system = cp.trajectory.XYZ(filepath).expand()
    frame_1 = system.data[0][:, 0:3]
    sigmas = utils_ext.get_sigmas(frame_1)
    config = Configuration()
    first_frame_within_cutoff, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(frame_1,
                                                                                          config.ORIGIN_ATOM_INDEX,
                                                                                          config.CUT_OFF)

    frame_trans = frame_1 + np.random.randint(1, 100 + 1)
    first_frame_within_cutoff_trans, smooth_coefficients_trans = utils_ext.filter_atoms_within_cutoff(frame_trans,
                                                                                                      config.ORIGIN_ATOM_INDEX,
                                                                                                      config.CUT_OFF)

    grid_xv_trans, grid_yv_trans, grid_zv_trans, xyz_bounds_trans, R_x_trans, R_y_trans, R_z_trans = utils_ext.generate_grid_and_bounds(
        first_frame_within_cutoff_trans, sigmas,
        config.NUMBER_PER_UNIT_DISTANCE,
        config.CUT_OFF, config.ORIGIN_ATOM_INDEX)

    grid_xv, grid_yv, grid_zv, xyz_bounds, R_x, R_y, R_z = utils_ext.generate_grid_and_bounds(
        first_frame_within_cutoff, sigmas,
        config.NUMBER_PER_UNIT_DISTANCE,
        config.CUT_OFF, config.ORIGIN_ATOM_INDEX)

    compare_arrays("grid_xv", grid_xv_trans, grid_xv)
    compare_arrays("grid_yv", grid_yv_trans, grid_yv)
    compare_arrays("grid_zv", grid_zv_trans, grid_zv)
    compare_arrays("xyz_bounds", xyz_bounds_trans, xyz_bounds)
    compare_arrays("R_x", R_x_trans, R_x)
    compare_arrays("R_y", R_y_trans, R_y)
    compare_arrays("R_z", R_z_trans, R_z)

# test_translational_invariant_for_generate_grid_and_bounds()


def test_translational_invariant_density_fun():
    # === Create example 3D coordinates ===
    np.random.seed(42)
    frame_1 = np.random.rand(5, 3)  # 5 random 3D points

    sigmas = utils_ext.get_sigmas(frame_1)
    config = Configuration()
    config.SIGMAS = sigmas
    first_frame_within_cutoff, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(frame_1,
                                                                                          config.ORIGIN_ATOM_INDEX,
                                                                                          config.CUT_OFF)

    density_fun = num_utils.trans_invariant_density_fun(first_frame_within_cutoff, config, smooth_coefficients)
    frame_trans = frame_1 + np.random.randint(1, 100 + 1)
    first_frame_within_cutoff_trans, smooth_coefficients_trans = utils_ext.filter_atoms_within_cutoff(frame_trans,
                                                                                                      config.ORIGIN_ATOM_INDEX,
                                                                                                      config.CUT_OFF)
    density_fun_trans = num_utils.trans_invariant_density_fun(first_frame_within_cutoff_trans, config, smooth_coefficients_trans)

    grid_xv, grid_yv, grid_zv, *_ = utils_ext.generate_grid_and_bounds(
        first_frame_within_cutoff, sigmas,
        config.NUMBER_PER_UNIT_DISTANCE,
        config.CUT_OFF, config.ORIGIN_ATOM_INDEX)

    compare_arrays("density_function_invariant", density_fun_trans(grid_xv, grid_yv, grid_zv), density_fun(grid_xv, grid_yv, grid_zv))
    return density_fun

# test_translational_invariant_density_fun()

def generate_random_rotation_matrix():
    # Generate random Euler angles in degrees
    angles = np.random.uniform(low=-180.0, high=180.0, size=3)
    # R.from_euler('xyz', [α, β, γ], degrees=True)
    rot = R.from_euler('xyz', angles, degrees=True)
    return rot.as_matrix(), angles


def test_rotational_invariant_density_fun():
    # === Create example 3D coordinates ===
    np.random.seed(42)
    frame_1 = np.random.rand(5, 3)  # 5 random 3D points
    R_mat, angles = generate_random_rotation_matrix()
    xyz_rot = frame_1 @ R_mat.T



    sigmas = utils_ext.get_sigmas(frame_1)
    config = Configuration()
    config.SIGMAS = sigmas
    first_frame_within_cutoff, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(frame_1,
                                                                                          config.ORIGIN_ATOM_INDEX,
                                                                                          config.CUT_OFF)

    density_fun = num_utils.trans_invariant_density_fun(first_frame_within_cutoff, config, smooth_coefficients)

    first_frame_within_cutoff_rot, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(xyz_rot,
                                                                                          config.ORIGIN_ATOM_INDEX,
                                                                                          config.CUT_OFF)

    density_fun_rot = num_utils.trans_invariant_density_fun(first_frame_within_cutoff_rot, config, smooth_coefficients)

    grid_xv, grid_yv, grid_zv, *_ = utils_ext.generate_grid_and_bounds(
        first_frame_within_cutoff, sigmas,
        config.NUMBER_PER_UNIT_DISTANCE,
        config.CUT_OFF, config.ORIGIN_ATOM_INDEX)

    compare_arrays("test_rotational_invariant_density_fun", density_fun(grid_xv, grid_yv, grid_zv), density_fun_rot(grid_xv, grid_yv, grid_zv),)


# test_rotational_invariant_density_fun() , this is wrong

def test_if_analytical_coefficients_are_equal():
    # === Create example 3D coordinates ===
    np.random.seed(42)
    frame_1 = np.random.rand(5, 3)  # 5 random 3D points

    sigmas = utils_ext.get_sigmas(frame_1)
    config = Configuration()
    config.SIGMAS = sigmas
    first_frame_within_cutoff, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(frame_1,
                                                                                          config.ORIGIN_ATOM_INDEX,
                                                                                          config.CUT_OFF)
    coefficients_in_dict = ana_utils.compute_coefficients_in_dict([first_frame_within_cutoff], sigmas, config)
    coefficients_in_sequence = ana_utils.compute_coefficients_in_sequence([first_frame_within_cutoff], sigmas, config)
    # Sort coefficients by (n, l, m) and convert to array
    sorted_keys = sorted(coefficients_in_dict.keys(), key=lambda x: (x[0], x[1], x[2]))
    sorted_coefficients = np.array([coefficients_in_dict[key] for key in sorted_keys], dtype=np.float64)

    assert np.allclose(sorted_coefficients, coefficients_in_sequence, 1e-5, 1e-5), "coefficients not equal"

# test_if_coefficients_are_equal()

def test_if_numerical_coefficients_are_equal():
    # === Create example 3D coordinates ===
    np.random.seed(42)
    frame_1 = np.random.rand(5, 3)  # 5 random 3D points

    sigmas = utils_ext.get_sigmas(frame_1)
    config = Configuration()
    config.SIGMAS = sigmas
    first_frame_within_cutoff, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(frame_1,
                                                                                          config.ORIGIN_ATOM_INDEX,
                                                                                          config.CUT_OFF)

    analytical_density_fun = num_utils.trans_invariant_density_fun(first_frame_within_cutoff, config,
                                                                   smooth_coefficients)
    coefficients_in_dict = num_utils.compute_coefficients_in_dict(analytical_density_fun, config)
    coefficients_in_sequence = num_utils.compute_coefficients_in_sequence(analytical_density_fun, config)
    # Sort coefficients by (n, l, m) and convert to array
    sorted_keys = sorted(coefficients_in_dict.keys(), key=lambda x: (x[0], x[1], x[2]))
    sorted_coefficients = np.array([coefficients_in_dict[key] for key in sorted_keys], dtype=np.float64)

    assert np.allclose(sorted_coefficients, coefficients_in_sequence, 1e-5, 1e-5), "coefficients not equal"

# test_if_numerical_coefficients_are_equal()
#
def test_density_fun_with_numerical_coefficients():

    coordinates = [
        [0.0, 0.0, 0.0],
        [0.804, -0.310, 0.107],
        [0.085, 0.804, 0.607],
        [0.889, 0.494, 0.714]
    ]

    frame_1 = np.array(coordinates, dtype=np.float64)


    sigmas = utils_ext.get_sigmas(frame_1)
    config = Configuration()
    config.SIGMAS = sigmas
    first_frame_within_cutoff, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(frame_1,
                                                                                          config.ORIGIN_ATOM_INDEX,
                                                                                          config.CUT_OFF)

    analytical_density_fun = num_utils.trans_invariant_density_fun(first_frame_within_cutoff, config, smooth_coefficients)

    # coefficients = ana_utils.compute_coefficients_in_dict([frame_1], sigmas, config)


    coefficients = num_utils.compute_coefficients_in_dict(analytical_density_fun, config)



    xv, yv, zv, xyz_bounds, R_x, R_y, R_z = utils_ext.generate_grid_and_bounds(frame_1, sigmas, 20, 3, 0)

    analytical_function_values = analytical_density_fun(xv,yv, zv)

    numerical_density_fun = 1
    r = np.sqrt(xv ** 2 + yv ** 2 + zv ** 2)
    thetas, phis = utils_ext.cartesian_to_spherical(xv, yv, zv, r)

    psi = np.zeros_like(xv, dtype=np.float64)
    for n in range(config.N_MAX):
        for l, m in l_m_pairs(config.L_MAX):
            # Get coefficient for (n, l, m)
            c_nlm = coefficients.get((n, l, m), 0.0)
            # Compute radial and angular parts

            R_n = ana_utils.dvr_basis_function(n, r, config.N_MAX)
            Y_lm = num_utils.Y_lm_real_scipy(l, m, thetas, phis)
            # Add contribution to total function
            psi += c_nlm * R_n * Y_lm



    diff = analytical_function_values - psi
    count = np.sum(abs(diff) > 0.1)
    print(f"Number of elements in diff > 0.1: {count}")
    # print(diff[diff>0.1])
    assert np.allclose(analytical_function_values, psi, 1e-2, 1e-2), "coefficients not equal"

# test_density_fun_with_numerical_coefficients()

def test_density_fun_with_analytical_coefficients():
    coordinates = [
        [0.0, 0.0, 0.0],
        [0.804, -0.310, 0.107],
        [0.085, 0.804, 0.607],
        [0.889, 0.494, 0.714]
    ]

    frame_1 = np.array(coordinates, dtype=np.float64)

    sigmas = utils_ext.get_sigmas(frame_1)
    config = Configuration()
    config.SIGMAS = sigmas
    first_frame_within_cutoff, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(frame_1,
                                                                                          config.ORIGIN_ATOM_INDEX,
                                                                                          config.CUT_OFF)

    analytical_density_fun = num_utils.trans_invariant_density_fun(first_frame_within_cutoff, config,
                                                                   smooth_coefficients)

    coefficients = ana_utils.compute_coefficients_in_dict([frame_1], sigmas, config)

    # coefficients = num_utils.compute_coefficients_in_dict(analytical_density_fun, config)

    xv, yv, zv, xyz_bounds, R_x, R_y, R_z = utils_ext.generate_grid_and_bounds(frame_1, sigmas, 20, 3, 0)

    analytical_function_values = analytical_density_fun(xv, yv, zv)

    numerical_density_fun = 1
    r = np.sqrt(xv ** 2 + yv ** 2 + zv ** 2)
    thetas, phis = utils_ext.cartesian_to_spherical(xv, yv, zv, r)

    psi = np.zeros_like(xv, dtype=np.float64)
    for n in range(config.N_MAX):
        for l, m in l_m_pairs(config.L_MAX):
            # Get coefficient for (n, l, m)
            c_nlm = coefficients.get((n, l, m), 0.0)
            # Compute radial and angular parts

            R_n = ana_utils.dvr_basis_function(n, r, config.N_MAX)
            Y_lm = num_utils.Y_lm_real_scipy(l, m, thetas, phis)
            # Add contribution to total function
            psi += c_nlm * R_n * Y_lm

    diff = analytical_function_values - psi
    count = np.sum(abs(diff) > 0.1)
    print(f"Number of elements in diff > 0.1: {count}")
    # print(diff[diff>0.1])
    assert np.allclose(analytical_function_values, psi, 1e-2, 1e-2), "coefficients not equal"


test_density_fun_with_analytical_coefficients()