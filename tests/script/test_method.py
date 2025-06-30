from pathlib import Path
from time import time

import chirpy as cp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import simpson as simps
from scipy.spatial.transform import Rotation as R
from scipy.special import legendre
from scipy.special import roots_legendre

import analytical_utils as ana_utils
import numerical_utils as num_utils
from tests.script import utils_ext
from tests.script.analytical_utils import Y_bar_l_m
from tests.script.configuration import Configuration
from tests.script.numerical_utils import Y_lm_real_scipy
from tests.script.utils_ext import l_m_pairs


def test_spherical_integral():
    import sympy as sp

    # Define symbols
    a, r, rij, theta, phi = sp.symbols('a r rij theta phi', real=True, positive=True)

    # Define the integrand
    k = 2 * a * r * rij
    integrand = sp.exp(-k * sp.cos(theta)) * sp.sin(theta) #   * sp.exp(-a * (r**2+rij ** 2))

    # First integrate over theta
    theta_integral = sp.integrate(integrand, (theta, 0, sp.pi))

    # Then integrate over phi (trivial)
    phi_integral = sp.integrate(1, (phi, 0, 2 * sp.pi))

    # Total result
    total_integral = sp.simplify(phi_integral * theta_integral)

    # Print result
    sp.pprint(total_integral, use_unicode=True)
# test_spherical_integral()

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

# test_if_analytical_coefficients_are_equal()

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
    # coordinates = [
    #     [0.0, 0.0, 0.0],
    #     [0.804, -0.310, 0.107],
    #     [0.085, 0.804, 0.607],
    #     [0.889, 0.494, 0.714]
    # ]
    #
    # frame_1 = np.array(coordinates, dtype=np.float64)
    #
    # sigmas = utils_ext.get_sigmas(frame_1)
    # config = Configuration()
    # config.SIGMAS = sigmas
    # first_frame_within_cutoff, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(frame_1,
    #                                                                                       config.ORIGIN_ATOM_INDEX,
    #                                                                                       config.CUT_OFF)
    #
    # analytical_density_fun = num_utils.trans_invariant_density_fun(first_frame_within_cutoff, config,
    #                                                                smooth_coefficients)
    #
    # coefficients = ana_utils.compute_coefficients_in_dict([frame_1], sigmas, config)
    #
    # # coefficients = num_utils.compute_coefficients_in_dict(analytical_density_fun, config)
    #
    # xv, yv, zv, xyz_bounds, R_x, R_y, R_z = utils_ext.generate_grid_and_bounds(frame_1, sigmas, 20, 3, 0)
    #
    # analytical_function_values = analytical_density_fun(xv, yv, zv)
    #
    # numerical_density_fun = 1
    # r = np.sqrt(xv ** 2 + yv ** 2 + zv ** 2) # ????
    # thetas, phis = utils_ext.cartesian_to_spherical(xv, yv, zv, r)
    #
    # psi = np.zeros_like(xv, dtype=np.float64)
    # for n in range(config.N_MAX):
    #     for l, m in l_m_pairs(config.L_MAX):
    #         # Get coefficient for (n, l, m)
    #         c_nlm = coefficients.get((n, l, m), 0.0)
    #         # Compute radial and angular parts
    #         # x = 2 * r / config.CUT_OFF - 1  # map r in [0, rc] to x in [-1, 1]
    #         R_n = ana_utils.dvr_basis_function(n, r, config.N_MAX)
    #         Y_lm = num_utils.Y_lm_real_scipy(l, m, thetas, phis)
    #         # Add contribution to total function
    #         psi += c_nlm * R_n * Y_lm
    #
    # diff = analytical_function_values - psi
    # count = np.sum(abs(diff) > 0.5)
    # print(f"Number of elements in diff > 0.1: {count}")
    # # print(diff[diff>0.1])
    # assert np.allclose(analytical_function_values, psi, 1e-2, 1e-2), "coefficients not equal"

    frame = np.array([
        [0.0, 0.0, 0.0],
        # [0.000, 0.000, 1.000],
        # [0.000, 1.000, 0.000],
        # [1.000, 0.000, 0.000],
        [0.804, -0.310, 0.107],
        [0.085, 0.804, 0.607],
        [0.889, 0.494, 0.714]
    ])

    # frame = np.array(coordinates, dtype=np.float64)

    # exclude origin atom from len
    sigmas = np.ones(len(frame) - 1) * 0.25

    config = Configuration()
    config.SIGMAS = sigmas

    # SJ CHANGED CODE: frame_within_cutoff contains RELATIVE distances
    frame_within_cutoff = utils_ext.filter_atoms_within_cutoff(
        frame,
        config.ORIGIN_ATOM_INDEX,
        config.CUT_OFF
    )

    s = time()
    xv, yv, zv, xyz_bounds, R_x, R_y, R_z = utils_ext.generate_grid_and_bounds(
        frame, sigmas, 5, config.CUT_OFF, 0)
    dx = xv[1, 0, 0] - xv[0, 0, 0]
    dy = yv[0, 1, 0] - yv[0, 0, 0]
    dz = zv[0, 0, 1] - zv[0, 0, 0]
    print("dx", dx)
    print("generate_grid_and_bounds", time() - s)

    s = time()
    analytical_density_fun = num_utils.trans_invariant_density_fun(
        frame_within_cutoff,
        config
    )
    print("trans_invariant_density_fun", time() - s)

    if False:
        # ----- NEW CODE (no significant speedup for 4 atom, maybe for more?)
        s = time()
        analytical_density_fun_NEW = trans_invariant_density_fun_NEW(
            frame_within_cutoff,
            config,
            smooth_coefficients
        )
        analytical_density_fun_NEW(R_x, R_y, R_z)
        print(time() - s)

        assert np.allclose(analytical_density_fun(R_x, R_y, R_z),
                           analytical_density_fun_NEW(R_x, R_y, R_z)), "FAIL"
        # ---- END OF TESTING NEW CODE

    s = time()
    coefficients = ana_utils.compute_coefficients_in_dict([frame_within_cutoff], config)
    print("compute_coefficients_in_dict", time() - s)

    #     for _l in range(config.L_MAX+1):
    #         for _m in range(-_l, _l + 1):
    #             _clm = 0.
    #             for _n in range(config.DVR_BASIS_NUM):
    #                 _clm += coefficients[(_n, _l, _m)]
    #             plt.bar((2*(_l-1) + 1) + (2*_l+1)+_m, _clm)
    #     plt.show()
    #
    # exit(0)

    # coefficients = \
    # num_utils.compute_coefficients_in_dict(analytical_density_fun, config)

    s = time()
    analytical_function_values = analytical_density_fun(xv, yv, zv)
    print("analytical_density_fun", time() - s)

    numerical_density_fun = 1
    print(xv.shape)
    s = time()
    r = np.sqrt(xv ** 2 + yv ** 2 + zv ** 2)  # ????
    thetas, phis = utils_ext.cartesian_to_spherical(xv, yv, zv, r)
    x = 2 * r / config.CUT_OFF - 1  # map r in [0, rc] to x in [-1,1]
    print("numerical density init", time() - s)

    s = time()

    # --- chirpy parallel processing
    _nlm = np.array([(_n, _l, _m)
                     for _n in range(config.N_MAX + 1)
                     for _l in range(config.L_MAX + 1)
                     for _m in range(-_l, _l + 1)
                     ])

    x_nodes, w_weights = roots_legendre(config.DVR_BASIS_NUM)

    # Map nodes from [-1, 1] to [0, r_cutoff]
    r_vals = 0.5 * config.CUT_OFF * (x_nodes + 1)  # shape: (N,)

    def _func(nlm):
        _n, _l, _m = nlm
        c_nlm = coefficients.get((_n, _l, _m), 0.0)
        R_n = ana_utils.dvr_basis_function(_n, x, config.DVR_BASIS_NUM)
        # --- renormalisation of DVR basis (just as the weights)
        R_n /= r_vals[_n] * np.sqrt(config.CUT_OFF / 2.)
        Y_lm = Y_lm_real_scipy(_l, _m, thetas, phis)
        return c_nlm * R_n * Y_lm

    JOB = cp.core.PALARRAY(_func,
                           _nlm,
                           n_cores=6
                           )
    S = JOB.run()
    print(S.shape)
    psi = S.sum(axis=0)
    # --- serial code
    # psi = np.zeros_like(xv, dtype=np.float64)
    # for _n in tqdm.tqdm(range(config.N_MAX+1)):
    #     for _l, _m in l_m_pairs(config.L_MAX):
    #         # Get coefficient for (n, l, m)
    #         c_nlm = coefficients.get((_n, _l, _m), 0.0)
    #         # Compute radial and angular parts
    #         R_n = dvr_basis_function(_n, x, config.DVR_BASIS_NUM)
    #         Y_lm = Y_lm_real_scipy(_l, _m, thetas, phis)
    #         # Add contribution to total function
    #         psi += c_nlm * R_n * Y_lm
    #         # --- if radial part disabled:
    #         # psi += c_nlm * Y_lm
    #         # OR:
    #         # psi += c_nlm * Y_lm  * np.exp(-r**2/(2 * 0.5**2))

    print("calculate psi", time() - s)

    # --- clean from noise at boundaries
    _ind = analytical_function_values >= 1.E-4
    psi[~_ind] *= 0.0

    # --- quick an dirty chirpy test
    cp.volume.ScalarField(
        data=analytical_function_values,
        cell_vec_aa=np.array([[dx, 0., 0.],
                              [0., dy, 0.],
                              [0., 0., dz]]),
        pos_aa=frame,
        numbers=len(frame) * [1, ],
        origin_aa=np.array([xv, yv, zv])[0, 0, 0],
    ).write("reference.cube")

    cp.volume.ScalarField(
        data=psi,
        cell_vec_aa=np.array([[dx, 0., 0.],
                              [0., dy, 0.],
                              [0., 0., dz]]),
        pos_aa=frame,
        numbers=len(frame) * [1, ],
        origin_aa=np.array([xv, yv, zv])[0, 0, 0],
    ).write("test.cube")

    # --- end chirpy test

    diff = analytical_function_values - psi
    count = np.sum(abs(diff[_ind]) / analytical_function_values[_ind] > 0.05)
    print(f"Number of elements with rel diff > 0.05: {count} = "
          f"{count / np.sum(_ind) * 100}%")
    # print(diff[diff>0.1])
    assert np.allclose(analytical_function_values,
                       psi, 1e-2, 1e-2), "coefficients not equal"

# if __name__ == '__main__':

    # test_density_fun_with_analytical_coefficients()


def test_scaled_DVR_orthogonality():


    # Parameters
    N = 10  # number of DVR basis functions
    a, b = 0.0, 5.0  # new interval

    # 1. Get Gauss-Legendre nodes and weights on [-1, 1]
    x_nodes, w_weights = leggauss(N)

    # 2. Define orthonormalized Legendre polynomial basis on [-1, 1]
    def phi_j_standard(j, x):
        Pj = legendre(j)
        return np.sqrt((2 * j + 1) / 2) * Pj(x)

    # 3. Construct DVR basis on [-1, 1]
    x_dense = np.linspace(-1, 1, 1000)
    chi_std = np.zeros((N, len(x_dense)))

    for alpha in range(N):
        for j in range(N):
            phi_j_x = phi_j_standard(j, x_dense)
            phi_j_node = phi_j_standard(j, x_nodes[alpha])
            chi_std[alpha] += phi_j_x * phi_j_node
        chi_std[alpha] *= np.sqrt(w_weights[alpha])

    # 4. Check orthonormality on [-1, 1]
    overlap_std = np.zeros((N, N))
    for m in range(N):
        for n in range(N):
            overlap_std[m, n] = simps(chi_std[m] * chi_std[n], x_dense)

    # 5. Map nodes and weights to [a, b]
    x_mapped = 0.5 * (b - a) * x_nodes + 0.5 * (b + a)
    w_mapped = 0.5 * (b - a) * w_weights

    # 6. Construct mapped DVR basis on [a, b]
    x_dense_mapped = 0.5 * (b - a) * x_dense + 0.5 * (b + a)
    chi_mapped = np.zeros((N, len(x_dense_mapped)))

    for alpha in range(N):
        for j in range(N):
            # return to 1???
            phi_j_x = phi_j_standard(j, (2 * x_dense_mapped - (b + a)) / (b - a))
            phi_j_node = phi_j_standard(j, x_nodes[alpha])
            chi_mapped[alpha] += phi_j_x * phi_j_node
        chi_mapped[alpha] *= np.sqrt(w_mapped[alpha])

    # 7. Check overlap on [a, b]
    overlap_mapped = np.zeros((N, N))
    for m in range(N):
        for n in range(N):
            overlap_mapped[m, n] = simps(chi_mapped[m] * chi_mapped[n], x_dense_mapped)

    print(overlap_std, overlap_mapped)

# test_scaled_DVR_orthogonality()



def test_discrete_orthogonal():
    from scipy.special import eval_legendre, roots_legendre

    # Parameters
    N = 6  # Number of DVR basis functions
    a, b = 1.0, 5.0  # New interval
    jacobian_norm = np.sqrt((b - a) / 2)  # sqrt(1.5)
    # Original Gauss-Legendre nodes and weights on [-1, 1]
    x_nodes_std, w_std = roots_legendre(N)

    # Mapping to [a, b]
    x_nodes_mapped = 0.5 * (b - a) * x_nodes_std + 0.5 * (b + a)
    weight_scaled = 0.5 * (b - a) * w_std

    # Normalized Legendre polynomial φ_n(x) on [-1, 1]
    def phi_n(n, x):
        norm = np.sqrt((2 * n + 1) / 2)
        return norm * eval_legendre(n, x)

    # Construct DVR basis function ψ_j(x) in [-1, 1]
    def dvr_basis_function(j, x_eval, N):
        x_nodes, w_nodes = roots_legendre(N)
        return sum(
            phi_n(n, x_eval) * phi_n(n, x_nodes[j])
            for n in range(N)
        ) * np.sqrt(w_nodes[j])  / jacobian_norm

    # Evaluate DVR basis functions ψ_j(x) at Gauss-Legendre points x_nodes_std
    # Note: Even though the final integration is over [a, b], DVR is defined on [-1, 1]
    psi_matrix = np.zeros((N, N))
    for j in range(N):
        psi_matrix[j, :] = dvr_basis_function(j, x_nodes_std, N)

    # Construct overlap matrix in mapped interval using Jacobian-scaled weights
    S_dvr_mapped = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S_dvr_mapped[i, j] = np.sum(weight_scaled * psi_matrix[i, :] * psi_matrix[j, :])

    print(S_dvr_mapped)


# test_discrete_orthogonal()
import math
from sympy import S
def test_CG():
    # List of test cases: (l, l1, l2, m, m1, m2, expected_value, description)
    test_cases = [
        # Test Case 1: Spin-1/2 coupling to l=1, m=1
        (1, S(1) / 2, S(1) / 2, 1, S(1) / 2, S(1) / 2, 1.0,
         "Spin-1/2 coupling to l=1, m=1, pure state"),

        # Test Case 2: Spin-1/2 coupling to l=0, m=0
        (0, S(1) / 2, S(1) / 2, 0, S(1) / 2, S(-1) / 2, math.sqrt(1 / 2),  # ≈ 0.7071067811865475
         "Spin-1/2 coupling to l=0, m=0, singlet state"),

        # Test Case 3: Invalid m conservation (m1 + m2 ≠ m)
        (1, S(1) / 2, S(1) / 2, S(1) / 2, S(1) / 2, S(1) / 2, 0.0,
         "Invalid: m1 + m2 = 1 ≠ m = 1/2"),

        # Test Case 4: l1=1, l2=1, l=1, m=1
        (1, 1, 1, 1, 1, 0, math.sqrt(1 / 2),  # ≈ 0.7071067811865475
         "Integer momenta coupling to l=1, m=1"),

        # Test Case 5: l1=1, l2=1, l=2, m=2
        (2, 1, 1, 2, 1, 1, 1.0,
         "Integer momenta coupling to l=2, m=2, pure state"),

        # Test Case 6: Invalid triad (l > l1 + l2)
        (3, 1, 1, 0, 0, 0, 0.0,
         "Invalid: l=3 > l1 + l2 = 2"),

        # Test Case 7: Spin-1/2 coupling to l=1, m=0
        (1, S(1) / 2, S(1) / 2, 0, S(1) / 2, S(-1) / 2, math.sqrt(1 / 2),  # ≈ 0.7071067811865475
         "Spin-1/2 coupling to l=1, m=0"),


    ]

    # Run and verify test cases
    for i, (l, l1, l2, m, m1, m2, expected, desc) in enumerate(test_cases, 1):
        result = utils_ext.clebsch_gordan(l1, l2, m1, m2, l, m)
        # Allow small numerical tolerance for floating-point comparison
        is_correct = abs(result - expected) < 1e-10
        print(f"Test Case {i}: {desc}")
        print(f"Input: l1={l1}, l2={l2}, l={l},  m={m}, m1={m1}, m2={m2}")
        print(f"Calculated: {result:.10f}, Expected: {expected:.10f}, {'PASS' if is_correct else 'FAIL'}")
        print()
test_CG()