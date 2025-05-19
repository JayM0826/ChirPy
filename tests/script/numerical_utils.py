# built-in module
from functools import partial

# third-party module
import numpy as np
import plotly.graph_objects as go
import scipy
from sympy.functions.special.spherical_harmonics import Ynm, Ynm_c, Znm
import sympy as sp
from scipy.special import eval_legendre, roots_legendre, sph_harm_y
from sympy.utilities.decorator import deprecated

# local module
import utils_ext
from tests.script.configuration import Configration
from tests.script.utils_ext import l_m_pairs


def spherical_integral(f_cartesion, real_spherical_harmonics, config:Configration, n, r):
    # order can be dynamic based on r, because when r is small, no need to sample too many points
    x, y, z = config.LEBEDEV_POINTS * r  # scaled sample points
    values = f_cartesion(x, y, z) * real_spherical_harmonics(config.LEBEDEV_THETA, config.LEBEDEV_PHI)
    surface_integral = np.sum(config.LEBEDEV_WEIGHTS * values)
    # TODO here is not right
    x_mapped = (r / 3) - 1  # inverse mapping
    result = dvr_basis_function(n, x_mapped, config.N_MAX)
    return surface_integral * r ** 2 * result


# Orthonormal Legendre polynomial function on [-1, 1]
def phi_n(n, x):
    weight_func = 1 # Polynomial dependent, for definition, see Eq. 2.1. in Light, J. C., & Carrington, T. (2007). Discrete-Variable Representations and their Utilization (pp. 263–310). John Wiley & Sons, Ltd. https://doi.org/10.1002/9780470141731.ch4
    norm = np.sqrt((2 * n + 1) / 2)
    return norm * eval_legendre(n, x) * np.sqrt(weight_func)

# Construct DVR basis function ψ_j(x)
def dvr_basis_function(j, grid_x, LEGENDRE_ORDER_NUM):
    root_x, weight_x = roots_legendre(LEGENDRE_ORDER_NUM)
    return sum(phi_n(n, root_x[j]) * phi_n(n, grid_x) for n in range(LEGENDRE_ORDER_NUM)) * np.sqrt(weight_x[j])


def project_density_to_sph_harmonics(density_fun, real_spherical_harmonics, config, n):
    """
    the basis must be orthogonal
    """
    integral, err = scipy.integrate.quad(partial(spherical_integral, density_fun, real_spherical_harmonics, config, n), 0,
                                         config.CUT_OFF)
    # print(f"integration_sph_coefficient : Integral result: {integral}, error={err}")
    return integral


def trans_invariant_density_fun(atom_positions, config, smooth_coefficients):
    relative_distances = atom_positions - atom_positions[
        config.ORIGIN_ATOM_INDEX]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    # NB: r must be an array
    relative_xs = relative_distances[:, 0]
    relative_ys = relative_distances[:, 1]
    relative_zs = relative_distances[:, 2]
    total_result = [
        lambda x, y, z, rx=rx, ry=ry, rz=rz, sigma=sigma, pf=1 / (-2 * sigma ** 2), smooth_coefficient=smooth_coeff: (
                smooth_coefficient * (config.ONE_OVER_TWO_PI_POWER_3DIV2 / np.power(sigma, 3))
                * np.exp(pf * ((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2))
        )
        for rx, ry, rz, sigma, smooth_coeff
        in zip(relative_xs, relative_ys, relative_zs, config.SIGMAS, smooth_coefficients)
        if not (rx == 0 and ry == 0 and rz == 0)  # ignore the origin atom
    ]

    return lambda x, y, z: np.sum([gaussian_fun(x, y, z) for gaussian_fun in total_result], axis=0)


def compute_sph_coefficients(density_fun, config:Configration):
    coeffs = []
    for n in range(config.N_MAX):
        for l, m in l_m_pairs(config.L_MAX):
                real_spherical_harmonics = Y_lm_real_fun_scipy(l, m)
                # real_spherical_harmonics = Y_lm_real_sympy(l, m)

                # real_spherical_harmonics()
                coeff = project_density_to_sph_harmonics(density_fun, real_spherical_harmonics, config, n)
                coeffs.append(coeff)
    return np.asarray(coeffs)


def plot_LCAO(coefficients, l_max, cutoff):
    i = 0
    theta, phi = utils_ext.theta_phi_meshgrid()

    values = np.zeros(shape=theta.shape)
    for l in np.arange(0, l_max + 1):
        for m in np.arange(-l, l + 1):
            if abs(coefficients[i]) > 1e-10:
                values += Y_lm_real_fun_sympy(l, m)(theta, phi) * coefficients[i]
            # plot_spherical_harmonics_for_scipy(l, m, coefficients[i])
            i += 1

    plot_LCAO_spherical_harmonics(values, *utils_ext.unit_spherical_to_cartesian(theta, phi, values) * cutoff, cutoff)


def radial_basis_gn():
    return 1


def compute_whole_grid_distribution(trajectory, sigmas, config):
    """
        number_per_unit_distance: the grid number per unit distance
        trajectory shape should be (No_frame, No_atoms, No_coordiantes)
    """
    # memory-friendly
    print(f"There will be {len(trajectory)} frames need to calculate...")
    # the grid shape should be the same for all the frames, otherwise we cannot add them up
    first_frame = trajectory[0][:, 0:3]
    filtered_first_frame, _ = utils_ext.filter_atoms_within_cutoff(first_frame, config.ORIGIN_ATOM_INDEX,
                                                                   config.CUT_OFF)
    xv, yv, zv, bounds, R_x, R_y, R_z = utils_ext.generate_grid_and_bounds(filtered_first_frame, sigmas,
                                                                           config.NUMBER_PER_UNIT_DISTANCE,
                                                                           config.CUT_OFF, config.ORIGIN_ATOM_INDEX)
    print(f"grid detail:shape={xv.shape} bounds={bounds}")
    total_density_result = np.zeros_like(xv)
    i = 0
    # here use the first frame of the trajectory
    for atom_positions in trajectory[:1, :, 0:3]:
        i += 1

        # ****************************************************************
        # calculate the density of frame i
        # ****************************************************************
        print(f"Now start to calculate the density of frame {i}")
        filtered_first_frame, smooth_coefficients = utils_ext.filter_atoms_within_cutoff(atom_positions,
                                                                                         config.ORIGIN_ATOM_INDEX,
                                                                                         config.CUT_OFF)
        density_fun = trans_invariant_density_fun(filtered_first_frame, config, smooth_coefficients)
        new_density_result = density_fun(xv, yv, zv)
        if not isinstance(new_density_result, np.ndarray):
            print("new_density_result is NOT a NumPy array")
            new_density_result = np.full(xv.shape, new_density_result)
        print(
            f"Integration of density of frame {i} over all intervals is {scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(new_density_result, R_z), R_y), R_x)}"
            f", it should be equal to the number of atoms:{len(atom_positions)}")

        total_density_result += new_density_result
        print(f"The calculation of the density of frame {i} is over")
        print(
            f"So sum over all the density of frame, we got the distribution and the shape is {total_density_result.shape}")
        print(
            f"Integration of density of frame all over all intervals is {scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(total_density_result, R_z), R_y), R_x)}")
        # ****************************************************************
        # calculate the density of frame i over
        # ****************************************************************

        # ****************************************************************
        # plot the linear combination of real spherical harmonics function
        # ****************************************************************
        coefficients = compute_sph_coefficients(density_fun, config)
        # print(coefficients)
        utils_ext.print_format_nlm_coefficients(coefficients, config.N_MAX, config.L_MAX)
        # plot_LCAO(coefficients, config.L_MAX, config.CUT_OFF)
        # ****************************************************************
        # ****************************************************************
        # ****************************************************************

        # TODO it later
        # theta, phi = utils_ext.cartesian_to_spherical(xv, yv, zv)
        # backwards_check(new_density_result, coefficients, theta, phi, config.L_MAX)

    return total_density_result, R_x, R_y, R_z, coefficients




def backwards_check(density_result, coefficients, theta, phi, l_max):
    result = np.zeros_like(density_result)
    i = 0
    for l in np.arange(0, l_max + 1, dtype=int):
        for m in np.arange(-l, l + 1, dtype=int):
            result += Y_lm_real_fun_sympy(l, m)(theta, phi) * coefficients[i]
            i += 1

    result_ = np.allclose(density_result, result, rtol=1e-2, atol=1e-2)
    print(f"backwards_check is {result_}")

    diff_norm = np.linalg.norm(density_result - result)
    print(f"Norm of difference: {diff_norm}")

# @deprecated("The implementation of sympy.Znm is not same as wiki:https://en.wikipedia.org/wiki/Spherical_harmonics#Real%20form, please use scipy Y_lm instead")
def Y_lm_real_fun_sympy(l, m):
    """
        build real spherical harmonics

        Be careful: the implementation of Znm is not same as wiki:
         https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
         https://en.wikipedia.org/wiki/Spherical_harmonics#Real%20form

        Parameters:
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.
        Returns:
        function: Znm(theta[0,pi], phi[0,2pi])
        """
    theta_sym, phi_sym = sp.symbols('theta phi', real=True)
    Y_lm = Znm(l, m, theta_sym, phi_sym).expand(func=True)
    f = sp.lambdify((theta_sym, phi_sym), Y_lm, 'numpy')
    return lambda theta, phi: f(theta, phi).real
    # return Y, theta_sym, phi_sym

def Y_lm_real_fun_scipy(l, m):
    """
    Same as wikipedia: https://en.wikipedia.org/wiki/Spherical_harmonics#Real%20form

    In SciPy sph_harm, the order is :
        m,
        l,
        phi : array_like
           Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
        theta : array_like
           Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.


    -------------------below is important---------------------------
    the order and definition of parameters are the same as sympy
    # In SciPy's sph_harm_y(here used)
        the order is :
        l,
        m,
        theta : ArrayLike[float]
            Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
        phi : ArrayLike[float]
            Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    Spherical harmonics. They are defined as

    .. math::

        Y_n^m(\theta,\phi) = \sqrt{\frac{2 n + 1}{4 \pi} \frac{(n - m)!}{(n + m)!}}
            P_n^m(\cos(\theta)) e^{i m \phi}

    where :math:`P_n^m` are the (unnormalized) associated Legendre polynomials.

    Note that SciPy's spherical harmonics include the Condon-Shortley
    phase [2]_ because it is part of `sph_legendre_p`.




    If you need to derive formulas (for example, manually expanding a spherical wave), use sympy's Znm.
    If you need to perform numerical computations (such as plotting molecular orbitals or doing acoustic simulations),
    use the scipy version (the one that includes the (−1^m) factor!).
    """
    def fun(theta, phi):

        if m == 0:
            return sph_harm_y(l, 0, theta, phi).real
        elif m > 0:
            return np.sqrt(2) * (-1) ** m * sph_harm_y(l, m, theta, phi).real   # Even
        else:
            return np.sqrt(2) * np.power(-1., m) * np.imag(sph_harm_y(l, -m, theta, phi))  # Odd
    return fun

def Y_lm_real_scipy(l, m, theta, phi):
    """
    In SciPy sph_harm, the order is :
        m,
        l,
        phi : array_like
           Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
        theta : array_like
           Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.


    -------------------below is important---------------------------
    the order and definition of parameters are the same as sympy
    # In SciPy's sph_harm_y(here used)
        the order is :
        l,
        m,
        theta : ArrayLike[float]
            Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
        phi : ArrayLike[float]
            Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    Spherical harmonics. They are defined as

    .. math::

        Y_n^m(\theta,\phi) = \sqrt{\frac{2 n + 1}{4 \pi} \frac{(n - m)!}{(n + m)!}}
            P_n^m(\cos(\theta)) e^{i m \phi}

    where :math:`P_n^m` are the (unnormalized) associated Legendre polynomials.

    Note that SciPy's spherical harmonics include the Condon-Shortley
    phase [2]_ because it is part of `sph_legendre_p`.




    f you need to derive formulas (for example, manually expanding a spherical wave), use sympy's Znm.
    If you need to perform numerical computations (such as plotting molecular orbitals or doing acoustic simulations),
    use the scipy version (the one that includes the (−1^m) factor!).
    """
    if m == 0:
        return sph_harm_y(l, 0, theta, phi).real
    elif m > 0:
        return np.sqrt(2) * (-1) ** m * sph_harm_y(l, m, theta, phi).real  # Even
    else:
        return np.sqrt(2) * np.power(-1., m) * np.imag(sph_harm_y(l, -m, theta, phi))  # Odd
    # Y = sph_harm_y(l, abs(m), theta, phi)
    #
    # # Linear combination of Y_l,m and Y_l,-m to create the real form.
    # if m < 0:
    #     Y = np.sqrt(2) * (-1.)**m * Y.imag
    # elif m > 0:
    #     Y = np.sqrt(2) * (-1.)**m * Y.real
    # else:
    #     Y = Y.real
    # return Y


def plot_spherical_harmonics_for_sympy(l, m, CUT_OFF, num=100):
    """
    l : [0, +oo]
    m : [-l, l]
      function: Znm(theta[0,pi], phi[0,2pi])
    """
    # Grids of polar and azimuthal angles
    theta, phi = utils_ext.theta_phi_meshgrid(num)

    Z_lm = Y_lm_real_fun_sympy(l, m)
    values = Z_lm(theta, phi)
    if not isinstance(values, np.ndarray):
        print("values is NOT a NumPy array")
        values = np.full(theta.shape, values)
    # https://en.wikipedia.org/wiki/File:Rotating_spherical_harmonics.gif
    # x, y, z = utils_ext.spherical_to_cartesian(theta, phi, CUT_OFF)

    x, y, z = utils_ext.unit_spherical_to_cartesian(theta, phi, values) * CUT_OFF

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=values, colorscale='Viridis')])
    fig.update_layout(
        title=f"real spherical harmonics of sympy Y_{l}_{m}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[-CUT_OFF, CUT_OFF], autorange=False),
            yaxis=dict(range=[-CUT_OFF, CUT_OFF], autorange=False),
            zaxis=dict(range=[-CUT_OFF, CUT_OFF], autorange=False),
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    fig.show()


def plot_LCAO_spherical_harmonics(values, xs, ys, zs, CUT_OFF):
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs,
                             surfacecolor=values,
                             colorscale='Viridis',
                             showscale=True,
                             hoverinfo=None
                             ))

    # Add Scatter3d overlay for hover
    # Flatten arrays for Scatter3d
    x_flat = xs.flatten()
    y_flat = ys.flatten()
    z_flat = zs.flatten()
    values_flat = values.flatten()

    fig.add_trace(
        go.Scatter3d(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            mode='markers+text',
            marker=dict(size=5, opacity=0.2),
            hovertemplate=
            'x: %{x:.2f}<br>' +
            'y: %{y:.2f}<br>' +
            'z: %{z:.2f}<br>' +
            'LC_density(x,y,z): %{customdata:.2f}<extra></extra>',
            customdata=values_flat  # Use customdata for f(x, y, z)
        )
    )

    fig.update_layout(
        title=f"linear combination of real spherical harmonics",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[-CUT_OFF, CUT_OFF], autorange=False),
            yaxis=dict(range=[-CUT_OFF, CUT_OFF], autorange=False),
            zaxis=dict(range=[-CUT_OFF, CUT_OFF], autorange=False),
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    fig.show()


def plot_spherical_harmonics_for_scipy(l, m, coefficient, CUT_OFF, num=100):
    """
    l : [0, +oo]
    m : [-l, l]
    ref: https://scipython.com/blog/visualizing-the-real-forms-of-the-spherical-harmonics/
    """
    theta, phi = utils_ext.theta_phi_meshgrid(num)

    values = Y_lm_real_scipy(l, m, theta, phi) * coefficient
    # values1 = Y_lm_real(l, m)(theta, phi)
    x, y, z = utils_ext.unit_spherical_to_cartesian(theta, phi, values) * CUT_OFF

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=values, colorscale='Viridis')])
    fig.update_layout(
        title=f"real spherical harmonics of scipy Y_{l}_{m}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',

            xaxis=dict(range=[-CUT_OFF, CUT_OFF], autorange=False),
            yaxis=dict(range=[-CUT_OFF, CUT_OFF], autorange=False),
            zaxis=dict(range=[-CUT_OFF, CUT_OFF], autorange=False),
            aspectratio=dict(x=1, y=1, z=1)

        )
    )
    fig.show()

# for l in np.arange(0, 4):
#     for m in np.arange(-l, l+1):
#         plot_spherical_harmonics_for_sympy(l, m, 5)
#         plot_spherical_harmonics_for_scipy(l, m,  1, 5)

# x = 0.3466097
# y = -0.1928484
# z = 0.1096611
# r = np.sqrt(x**2+y**2+z**2)
#
# theta, phi = cartesian_to_spherical(x, y, z, r)
# Z_lm = Y_lm_real(1, 1)
# values = Z_lm(theta, phi)
# print(values)
# print(real_sph_harm(1, 1, theta, phi))
# def Y_0_0(a):
#     return 5 # return a constant arbitrarily
# print(Y_0_0(np.arange(0, 5))) # output: 5 not [5,5,5,5,5]
