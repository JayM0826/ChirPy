import math as math
from functools import partial

import numpy as np
import plotly.graph_objects as go
import scipy
from sympy.functions.special.spherical_harmonics import Ynm, Ynm_c, Znm
from sympy import Symbol, integrate
import sympy as sp

ONE_OVER_TWO_PI_POWER_1DIV2 = 1 / np.power(2 * np.pi, 0.5)
ONE_OVER_TWO_PI_POWER_3DIV2 = np.power(ONE_OVER_TWO_PI_POWER_1DIV2, 3)
NUMBER_PER_UNIT_DISTANCE = 5
ORIGIN_ATOM_INDEX = 0
# CUT_OFF = 2. # unit angstrom: 10**(-10) m
CUT_OFF = 5  # unit angstrom: 10**(-10) m
SIGMA = 1
L_MAX = 6
LEBEDEV_ORDER = 131
LEBEDEV_POINTS, LEBEDEV_WEIGHTS = scipy.integrate.lebedev_rule(LEBEDEV_ORDER)


def coupute_XYZ_bounds(atom_3D_positions, sigmas, origin_index=ORIGIN_ATOM_INDEX, cutoff=CUT_OFF):
    """
    atom_3D_positions:
    atom_3D_position = np.array([[0,     0,    0],
                                 [2,     2,    2],
                                 [-1,   -1,   -1]])
    sigmas = np.array([2, 3, 4])#, isotropic for now

    return the bounds like (-x, x, -y, y, -z, z)

    """
    # shape = (N, 3), N means #atoms, 3 means relative distance of (x and y, z)
    if cutoff != np.inf:
        x_upper, y_upper, z_upper = cutoff, cutoff, cutoff,
        x_lower, y_lower, z_lower = - cutoff, - cutoff, - cutoff
        return tuple(int(math.ceil(x)) for x in (x_lower, x_upper, y_lower, y_upper, z_lower, z_upper))

    # otherwise
    atom_3D_relative_positions = atom_3D_positions - atom_3D_positions[origin_index]

    max_values = np.max((atom_3D_relative_positions), axis=0)
    min_values = np.min((atom_3D_relative_positions), axis=0)
    # here we use 6(without scientific proof) sigma so that the accuracy is higher
    x_upper, y_upper, z_upper = max_values + np.max(sigmas) * 6
    x_lower, y_lower, z_lower = min_values - np.max(sigmas) * 6
    # easy to use linspace with int
    return tuple(int(math.ceil(x)) for x in (x_lower, x_upper, y_lower, y_upper, z_lower, z_upper))


def plot_iosfurface(xv, yv, zv, values):
    """
    the shape of the parameters should be (N, M, L), N=M=L is possible
    """
    print(f"the shape of xv={xv.shape},yv={yv.shape}, zv={zv.shape}")
    fig = go.Figure(data=go.Isosurface(
        x=xv.flatten(),
        y=yv.flatten(),
        z=zv.flatten(),
        value=values.flatten(),
        isomin=np.min(values),
        isomax=np.max(values),
        surface_count=20,
        colorscale='Viridis',
        showscale=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    fig.update_layout(
        title='3D Isosurface Plot',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
    )

    fig.show()
    print(70 * "*")


def gaussian_fun_3D(sigmas, relative_distances_3D, smooth_coefficient):
    # NB: r must be an array
    relative_xs = relative_distances_3D[:, 0]
    relative_ys = relative_distances_3D[:, 1]
    relative_zs = relative_distances_3D[:, 2]
    total_result = [
        lambda x, y, z, rx=rx, ry=ry, rz=rz, sigma=sigma, pf=1 / (-2 * sigma ** 2), smooth_coefficient=smooth_coeff: (
                smooth_coefficient * (ONE_OVER_TWO_PI_POWER_3DIV2 / np.power(sigma, 3))
                * np.exp(pf * ((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2))
        )
        for rx, ry, rz, sigma, smooth_coeff
        in zip(relative_xs, relative_ys, relative_zs, sigmas, smooth_coefficient)
        if not (rx == 0 and ry == 0 and rz == 0) # ignore the origin atom
    ]

    def combined_function(x, y, z):
        return np.sum([gaussian_fun(x, y, z) for gaussian_fun in total_result], axis=0)

    return combined_function


def cartesian_to_spherical(x, y, z, r=1):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (theta, phi).

    Parameters:
    x (float or np.ndarray): x-coordinate(s).
    y (float or np.ndarray): y-coordinate(s).
    z (float or np.ndarray): z-coordinate(s).

    Returns:
    tuple: (theta, phi)
        - theta (float or np.ndarray): Polar angle in [0, pi].
        - phi (float or np.ndarray): Azimuthal angle in [0, 2*pi).
    """
    # Handle the case where r = 0 to avoid division by zero
    # At the origin, set theta = 0, phi = 0 as a convention
    theta = np.where(r == 0, 0.0, np.arccos(z / np.where(r == 0, 1.0, r)))

    # Compute phi using atan2, which handles all quadrants correctly
    phi = np.arctan2(y, x)

    # Ensure phi is in [0, 2*pi)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    return theta, phi

LEBEDEV_THETA, LEBEDEV_PHI = cartesian_to_spherical(*LEBEDEV_POINTS)

def integrand(f_cartesion, spherical_harmonics_Zlm_sph_version, r):
    # order can be dynamic based on r, because when r is small, no need to sample too many points
    x, y, z = LEBEDEV_POINTS * r  # scaled sample points
    values = f_cartesion(x, y, z) * spherical_harmonics_Zlm_sph_version(LEBEDEV_THETA, LEBEDEV_PHI)
    surface_integral = np.sum(LEBEDEV_WEIGHTS * values)
    return surface_integral * r**2 * radial_basis_gn()


def integration_sph(density_fun, real_spherical_harmonics, r):
    integral, err = scipy.integrate.quad(partial(integrand, density_fun, real_spherical_harmonics), 0, r)
    print(f"integration_sph_coefficient : Integral result: {integral}, error={err}")
    return integral


def trans_invariant_density_3D(atom_positions, sigma_3D, smooth_efficient, r_cutoff=CUT_OFF,
                               origin_index=ORIGIN_ATOM_INDEX):
    relative_distances_3D = atom_positions - atom_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    density_fun = gaussian_fun_3D(sigma_3D, relative_distances_3D, smooth_efficient)
    return density_fun


def compute_sph_coefficients(density_fun, r_cutoff):
    coeffs = []
    for l in np.arange(0, L_MAX + 1, dtype=int):
        for m in np.arange(-l, l + 1, dtype=int):
            real_spherical_harmonics = Y_lm_real(l, m)
            coeff = integration_sph(density_fun, real_spherical_harmonics, r_cutoff)
            coeffs.append(coeff)
    return np.asarray(coeffs)


def compute_whole_grid_density(atom_positions, sigmas, xv, yv, zv, smooth_efficient):
    atom_positions = atom_positions[:, 0:3]
    density_fun = trans_invariant_density_3D(atom_positions, sigmas, smooth_efficient)
    print_format_coefficients(compute_sph_coefficients(density_fun, CUT_OFF))
    return density_fun(xv, yv, zv)

def radial_basis_gn():
    return 1


def print_format_coefficients(coefficients):

    print(300* "*")
    print(f"{100 *'*'}COEFFICIENTS{140 *'*'}")
    print(300 * "*")


    # Format the array to 10 decimal places
    coefficients = np.around(coefficients, decimals=10)

    # Print the array with indentation to form a triangle
    print("[")
    start_idx = 0
    max_width = L_MAX * 2 + 1  # Maximum width (for l=6, m=-6 to 6)
    for l in range(L_MAX + 1):  # l from 0 to 6
        # Number of elements in this row (2l+1)
        row_length = 2 * l + 1
        # Extract the elements for this row
        row = coefficients[start_idx:start_idx + row_length]
        # Calculate indentation to center the row
        indent = (max_width - row_length) // 2
        indent_str = "               " * indent
        # Format the row
        formatted_row = f"{l}:{indent_str}{',  '.join(f'{x:.10f}' for x in row)}"
        print(f"{formatted_row},")

        start_idx += row_length
    print("]")
    print(300* "*")
    print(300 * "*")
def generate_grid_and_bounds(atom_positions, sigmas, number_per_unit_distance=NUMBER_PER_UNIT_DISTANCE,
                             origin_index=ORIGIN_ATOM_INDEX):
    """
       number_per_unit_distance: the grid number per unit distance
    """
    bounds = coupute_XYZ_bounds(atom_positions, sigmas, origin_index)
    R_x = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0]) * number_per_unit_distance)
    R_y = np.linspace(bounds[2], bounds[3], (bounds[3] - bounds[2]) * number_per_unit_distance)
    R_z = np.linspace(bounds[4], bounds[5], (bounds[5] - bounds[4]) * number_per_unit_distance)

    xv, yv, zv = np.meshgrid(R_x, R_y, R_z, indexing='ij')
    return xv, yv, zv, bounds, R_x, R_y, R_z


def compute_whole_grid_distribution(trajectory, sigmas, number_per_unit_distance=NUMBER_PER_UNIT_DISTANCE,
                                    cutoff=CUT_OFF):
    """
        number_per_unit_distance: the grid number per unit distance
        trajectory shape should be (No_frame, No_atoms, No_coordiantes)
    """
    # memory-friendly
    print(f"There will be {len(trajectory)} frames need to calculate...")
    # the grid shape should be the same for all the frames, otherwise we cannot add them up
    first_frame = trajectory[0][:, 0:3]
    filtered_first_frame, _ = filter_atoms_within_cutoff(first_frame)
    xv, yv, zv, bounds, R_x, R_y, R_z = generate_grid_and_bounds(filtered_first_frame, sigmas, number_per_unit_distance)
    print(f"grid detail:shape={xv.shape} bounds={bounds}")
    total_result = np.ndarray(xv.shape)
    i = 1
    # here just use the first frame of the trajectory
    for atom_positions in trajectory[:1, :, 0:3]:
        print(f"Now start to calculate the density of frame {i}")
        filtered_first_frame, smooth_efficient = filter_atoms_within_cutoff(atom_positions, cutoff=cutoff)
        new_result = compute_whole_grid_density(filtered_first_frame, sigmas, xv, yv, zv, smooth_efficient)
        # be careful: the order of simpson is R_z, R_x,R_y
        print(
            f"Integration of density of frame {i} over all intervals is {scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(new_result, R_z), R_y), R_x)}"
            f", it should be equal to the number of atoms:{len(atom_positions)}")

        total_result += new_result
        print(f"The calculation of the density of frame {i} is overnn")
        i += 1

    print(f"So sum over all the density of frame, we got the distribution and the shape is {total_result.shape}")
    print(
        f"Integration of density of frame all over all intervals is {scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(total_result, R_z), R_y), R_x)}")

    return total_result, R_x, R_y, R_z


def get_sigmas(atoms):
    # TODO build a dict and store the sigmas for the atoms and retrieve them when needed
    # now just return the same sigmas blindly
    return np.ones(len(atoms)) * SIGMA


def filter_atoms_within_cutoff(positions, origin_atom_index=ORIGIN_ATOM_INDEX, cutoff=CUT_OFF):
    """
    Filter atoms within a cutoff distance from a central atom.

    Parameters:
    - positions: Nx3 array of (x, y, z) coordinates.
    - center_atom_idx: Index of the central atom.
    - cutoff: Cutoff radius (e.g., in Å).

    Returns:
    - Filtered array of qualifying atoms (excluding center if desired).
    """
    origin_pos = positions[origin_atom_index]
    relative_distance = np.sqrt(np.sum((positions - origin_pos) ** 2, axis=1))
    qualified_indices = np.where(relative_distance <= (cutoff))[0]

    qualified_atom_positions = positions[qualified_indices]
    # smooth_efficient = smooth_function(relative_distance[qualified_indices], cutoff)
    # TODO in fact there is no smooth if use the below code instead of the above line
    smooth_efficient = np.ones(len((qualified_atom_positions)))
    return qualified_atom_positions, smooth_efficient


def smooth_function(relative_distance, cutoff=CUT_OFF):
    r"""
    Comparison
    The function is defined as:

     f_{\text{cut}}(r) = \begin{cases} 0.5 \left[ 1 + \cos\left( \pi \frac{r}{r_{\text{cut}}} \right) \right] & \text{if } r < r_{\text{cut}}, \\ 0 & \text{otherwise}. \end{cases}

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Mollifier
    .. [2] https://math.stackexchange.com/questions/1618981/cutoff-function-vs-mollifiers
    """
    relative_distance = np.asarray(relative_distance)
    smooth_coefficient = np.zeros_like(relative_distance, dtype=float)
    mask = relative_distance < cutoff
    # smooth
    smooth_coefficient[mask] = 0.5 * (1 + np.cos(np.pi * relative_distance[mask] / cutoff))
    print(relative_distance, smooth_coefficient)
    return smooth_coefficient


def Y_lm_real(l, m):
    """
        build real spherical harmonics
        Parameters:
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.
        Returns:
        function: Znm(theta, phi)
        """
    theta_sym, phi_sym = sp.symbols('theta phi', real=True)
    Y_lm = Znm(l, m, theta_sym, phi_sym).expand(func=True)
    f = sp.lambdify((theta_sym, phi_sym), Y_lm, 'numpy')
    return lambda theta, phi: f(theta, phi).real
    # return Y, theta_sym, phi_sym




def backwards_check(pho, l_max = 10):
    result = 0
    coeff = 1.
    for l in np.arange(0, l_max + 1, dtype=int):
        for m in np.arange(-l, l, dtype=int):
            pass
            # result += Y_lm_real(l, m)[0] * coeff

    if (pho == result):
        return True
    else:
        return False
