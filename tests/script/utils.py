import math as math

import numpy as np

import plotly.graph_objects as go
from sympy import Symbol, integrate, sin, pi
from sympy.functions.special.spherical_harmonics import Ynm
from scipy.stats import multivariate_normal
from scipy import integrate
ONE_OVER_TWO_PI_POWER_1DIV2 = 1 / np.power(2 * np.pi, 0.5)
ONE_OVER_TWO_PI_POWER_3DIV2 = np.power(ONE_OVER_TWO_PI_POWER_1DIV2, 3)
NUMBER_PER_UNIT_DISTANCE = 5
ORIGIN_ATOM_INDEX = 0
CUT_OFF = 3. # unit angstrom: 10**(-10) m
SIGMA = 0.666

def coupute_XYZ_bounds(atom_3D_positions, sigmas, origin_index=ORIGIN_ATOM_INDEX):
    """
    atom_3D_positions:
    atom_3D_position = np.array([[0,     0,    0],
                                 [2,     2,    2],
                                 [-1,   -1,   -1]])
    sigmas = np.array([2, 3, 4])#, isotropic for now

    return the bounds like (-x, x, -y, y, -z, z)

    """
    # shape = (N, 3), N means #atoms, 3 means relative distance of (x and y, z)
    atom_3D_relative_positions = atom_3D_positions - atom_3D_positions[origin_index]

    max_values = np.max((atom_3D_relative_positions), axis=0)
    min_values = np.min((atom_3D_relative_positions), axis=0)
    # here we use 6(without scientific proof) sigma so that the accuracy is higher
    x_upper, y_upper, z_upper = max_values + np.max(sigmas) * 6
    x_lower, y_lower, z_lower = min_values - np.max(sigmas) * 6
    # easy to use linspace with int
    return tuple(int(math.ceil(x)) for x in (x_lower, x_upper, y_lower, y_upper, z_lower, z_upper))


def integrate_spherical_fun_analytically(n, m):
    # Spherical coordinates: dV = ρ² sin(φ) dρ dθ dφ
    # Expected variables: ρ (radius), θ (azimuth), φ (polar angle)
    # if len(symbols) != 2:
    #     raise ValueError("Spherical coordinates require 2 variables (θ, φ)")
    theta = Symbol('theta')
    phi = Symbol('phi')

    Y = Ynm(n, m, theta, phi)
    integrand = Y * sin(theta)

    phi_integral = integrate(integrand, (phi, 0, 2 * pi))
    total_integral = integrate(phi_integral, (theta, 0, pi))
    return total_integral
    # Integrate with respect to each variable
    # for var in reversed(symbols):
    #     if var not in bounds:
    #         raise ValueError(f"Bounds for {var} not provided")
    #     lower, upper = bounds[var]
    #     result = sympy.integrate(result, (var, lower, upper))
    # return total_integral


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

def gaussian_fun_alt_3D_sequentially(sigmas, relative_distances_3D, xv, yv, zv):
    # NB: r must be an array
    relative_xs = relative_distances_3D[:, 0]
    relative_ys = relative_distances_3D[:, 1]
    relative_zs = relative_distances_3D[:, 2]
    total_result = np.zeros(shape=xv.shape)
    for relative_x, relative_y, relative_z, sigma in zip(relative_xs, relative_ys, relative_zs, sigmas):
        # here we assume the sigma for all 3 directions are same, obviously they can be different
        prefactor = 1 / (-2 * sigma ** 2)
        guassion_funciton = lambda x, y, z: ((ONE_OVER_TWO_PI_POWER_3DIV2 * 1. / np.power(sigma, 3))
                                  * np.exp(prefactor * (x - relative_x) ** 2
                                           + prefactor * (y - relative_y) ** 2
                                           + prefactor * (z - relative_z) ** 2))
        total_result += guassion_funciton(xv, yv, zv)

    return total_result

def gaussian_fun_alt_3D_sequentially1(sigmas, relative_distances_3D, smooth_efficient):
    # NB: r must be an array
    relative_xs = relative_distances_3D[:, 0]
    relative_ys = relative_distances_3D[:, 1]
    relative_zs = relative_distances_3D[:, 2]
    total_result = [
        lambda x, y, z, rx=rx, ry=ry, rz=rz, s=s, pf=1 / (-2 * s ** 2), smooth_efficient=smooth_coeff: (
                smooth_efficient * (ONE_OVER_TWO_PI_POWER_3DIV2 * 1. / np.power(s, 3))
                * np.exp(pf * (x - rx) ** 2 + pf * (y - ry) ** 2 + pf * (z - rz) ** 2)
        )
        for rx, ry, rz, s, smooth_coeff in zip(relative_xs, relative_ys, relative_zs, sigmas, smooth_efficient)
    ]

    def combined_function(x, y, z):
        return np.sum([gaussian_fun(x, y, z) for gaussian_fun in total_result], axis=0)

    return combined_function

def trans_invariant_density_alt_3D(atom_positions, sigma_3D, xv, yv, zv, r_cutoff=CUT_OFF, origin_index=ORIGIN_ATOM_INDEX):
    # TODO use different origin
    relative_distances_3D = atom_positions - atom_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    return gaussian_fun_alt_3D_sequentially(sigma_3D, relative_distances_3D, xv, yv, zv)

def trans_invariant_density_alt_3D1(atom_positions, sigma_3D, smooth_efficient,  r_cutoff=CUT_OFF, origin_index=ORIGIN_ATOM_INDEX):
    # TODO use different origin
    relative_distances_3D = atom_positions - atom_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    density_sequentially = gaussian_fun_alt_3D_sequentially1(sigma_3D, relative_distances_3D, smooth_efficient)
    return lambda x, y, z: density_sequentially(x, y, z)


def trans_invariant_density_alt1_3D(atom_positions, sigma_3D, r_cutoff=np.inf, origin_index=0):
    """
    be careful to use below code, it is just used for how to use multivariate_normal,
    """
    relative_distances_3D = atom_positions - atom_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    # density_sequentially = gaussian_fun_alt_3D_sequentially(sigma_3D, relative_distances_3D)
    covs = np.eye(3) * np.array(sigma_3D)[:, np.newaxis, np.newaxis]
    density_sequentially = np.array([multivariate_normal(mean=relative_distances_3D[i], cov=covs[i]) for i in range(len(sigma_3D))])


    def sum_over(x, y, z):
        # TODO refactor below code, it is low efficient
        pos = np.stack([x, y, z], axis=-1)
        mixed_pdf = 0
        for mvn in density_sequentially:
            # mixed_pdf += mvn.pdf([x, y ,z])
            mixed_pdf += mvn.pdf(pos)

        return mixed_pdf

    return lambda x, y, z: sum_over(x, y, z)

def compute_whole_grid_density(atom_positions, sigmas, xv, yv, zv, smooth_efficient):
    atom_positions = atom_positions[:, 0:3]
    # return trans_invariant_density_alt_3D(atom_positions, sigmas, xv, yv, zv)
    return trans_invariant_density_alt_3D1(atom_positions, sigmas, smooth_efficient)(xv, yv, zv)

def generate_grid_and_bounds(atom_positions, sigmas, number_per_unit_distance=NUMBER_PER_UNIT_DISTANCE, origin_index=ORIGIN_ATOM_INDEX):
    """
       number_per_unit_distance: the grid number per unit distance
    """
    bounds = coupute_XYZ_bounds(atom_positions, sigmas, origin_index)
    R_x = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0]) * number_per_unit_distance)
    R_y = np.linspace(bounds[2], bounds[3], (bounds[3] - bounds[2]) * number_per_unit_distance)
    R_z = np.linspace(bounds[4], bounds[5], (bounds[5] - bounds[4]) * number_per_unit_distance)

    xv, yv, zv = np.meshgrid(R_x, R_y, R_z, indexing='ij')
    return xv, yv, zv, bounds, R_x, R_y, R_z


def compute_whole_grid_distribution(trajectory, sigmas, number_per_unit_distance=NUMBER_PER_UNIT_DISTANCE, cutoff=CUT_OFF):
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
    for atom_positions in trajectory[:, :, 0:3]:
        print(f"Now start to calculate the density of frame {i}")
        filtered_first_frame, smooth_efficient = filter_atoms_within_cutoff(atom_positions, cutoff=cutoff)
        new_result = compute_whole_grid_density(filtered_first_frame, sigmas, xv, yv, zv, smooth_efficient)
        # be careful: the order of simpson is R_z,R_x,R_y
        print(f"Integration of density of frame {i} over all intervals is {integrate.simpson(integrate.simpson(integrate.simpson(new_result, R_z), R_y), R_x)}"
              f", it should be equal to the number of atoms:{len(atom_positions)}")
        total_result += new_result
        print(f"The calculation of the density of frame {i} is overnn")
        i += 1

    print(f"So sum over all the density of frame, we got the distribution and the shape is {total_result.shape}")
    print(f"Integration of density of frame all over all intervals is {integrate.simpson(integrate.simpson(integrate.simpson(total_result, R_z), R_y), R_x)}")

    return total_result, R_x, R_y, R_z

def get_sigmas(atoms):
    # TODO build a dict and store the sigmas for the atoms and retrieve them when needed
    # now just return the same sigmas blindly
    return np.ones(len(atoms)) * SIGMA


def filter_atoms_within_cutoff(positions, origin_atom_index = ORIGIN_ATOM_INDEX, cutoff=CUT_OFF):
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
    smooth_efficient = smooth_function(relative_distance[qualified_indices], cutoff)
    return qualified_atom_positions, smooth_efficient


def smooth_function(relative_distance, cutoff = CUT_OFF):
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
    smooth_efficient = np.zeros_like(relative_distance, dtype=float)
    mask = relative_distance < cutoff
    # smooth
    smooth_efficient[mask] = 0.5 * (1 + np.cos(np.pi * relative_distance[mask] / cutoff))
    print(relative_distance, smooth_efficient)
    return smooth_efficient