import math as math

import numpy as np
import plotly.graph_objects as go
from sympy import Symbol, integrate, sin, pi
from sympy.functions.special.spherical_harmonics import Ynm

ONE_OVER_TWO_PI_POWER_1DIV2 = 1 / np.power(2 * np.pi, 0.5)
ONE_OVER_TWO_PI_POWER_3DIV2 = np.power(ONE_OVER_TWO_PI_POWER_1DIV2, 3)
NUMBER_PER_UNIT_DISTANCE = 5


def coupute_XYZ_bounds(atom_3D_positions, sigmas, origin_index=0):
    """
    atom_3D_positions:
    atom_3D_position = np.array([[0,     0,    0],
                                 [2,     2,    2],
                                 [-1,   -1,   -1]])
    sigmas = np.array([2, 3, 4])#, isotropic for now

    return the bounds like (-x, x, -y, y, -z, z)

    """
    atom_3D_relative_positions = atom_3D_positions - atom_3D_positions[
        origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)

    max_values = np.max((atom_3D_relative_positions), axis=0)
    min_values = np.min((atom_3D_relative_positions), axis=0)
    x_upper, y_upper, z_upper = max_values + np.max(
        sigmas) * 6  # here we use 6(without scientific proof) sigma so that the accuracy is higher
    x_lower, y_lower, z_lower = min_values - np.max(
        sigmas) * 6  # here we use 6(without scientific proof) sigma so that the accuracy is higher
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


def gaussian_fun_alt_3D_sequentially(sigmas, relative_distances_3D):
    # NB: r must be an array
    relative_xs = relative_distances_3D[:, 0]
    relative_ys = relative_distances_3D[:, 1]
    relative_zs = relative_distances_3D[:, 2]
    total_result = []
    for relative_x, relative_y, relative_z, sigma in zip(relative_xs, relative_ys, relative_zs, sigmas):
        # here we assume the sigma for all 3 directions are same, obviously they can be different
        prefactor = 1 / (-2 * sigma ** 2)
        result = lambda x, y, z: ((ONE_OVER_TWO_PI_POWER_3DIV2 * 1. / np.power(sigma, 3))
                                  * np.exp(prefactor * (x - relative_x) ** 2
                                           + prefactor * (y - relative_y) ** 2
                                           + prefactor * (z - relative_z) ** 2))
        total_result.append(result)

    # Combine all functions into one
    def combined_function(x, y, z):
        result = 0.
        for f in total_result:
            result += f(x, y, z)
        # return sum(f(x, y, z) for f in total_result)
        return result
    return combined_function


def trans_invariant_density_alt_3D(atom_positions, sigma_3D, r_cutoff=np.inf, origin_index=0):
    # TODO use different origin
    relative_distances_3D = atom_positions - atom_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    density_sequentially = gaussian_fun_alt_3D_sequentially(sigma_3D, relative_distances_3D)
    # return density_sequentially
    return lambda x, y, z: density_sequentially(x, y, z)


def compute_whole_grid_density(atom_positions, sigmas, xv, yv, zv):
    atom_positions = atom_positions[:, 0:3]
    return trans_invariant_density_alt_3D(atom_positions, sigmas)(xv, yv, zv)


def generate_grid_and_bounds(atom_positions, sigmas, number_per_unit_distance=NUMBER_PER_UNIT_DISTANCE):
    """
       number_per_unit_distance: the grid number per unit distance
    """
    bounds = coupute_XYZ_bounds(atom_positions, sigmas)
    R_x = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0]) * number_per_unit_distance)
    R_y = np.linspace(bounds[2], bounds[3], (bounds[3] - bounds[2]) * number_per_unit_distance)
    R_z = np.linspace(bounds[4], bounds[5], (bounds[5] - bounds[4]) * number_per_unit_distance)

    xv, yv, zv = np.meshgrid(R_x, R_y, R_z)
    return xv, yv, zv, bounds


def compute_whole_grid_distribution(trajectory, sigmas, number_per_unit_distance=NUMBER_PER_UNIT_DISTANCE):
    """
        number_per_unit_distance: the grid number per unit distance
        trajectory shape should be (No_frame, No_atoms, No_coordiantes)
    """
    # memory-friendly
    print(f"There will be {len(trajectory)} frames need to calculate...")

    xv, yv, zv, bounds = generate_grid_and_bounds(trajectory[0][:, 0:3], sigmas, number_per_unit_distance)
    print(f"grid detail:shape={xv.shape} bounds={bounds}")
    total_result = np.ndarray(xv.shape)
    i = 1
    for atom_positions in trajectory[:, :, 0:3]:
        print(f"Now start to calculate the density of frame {i}")
        i += 1
        # A
        new_result = compute_whole_grid_density(atom_positions, sigmas, xv, yv, zv)
        # B
        voxel = 1
        # test_integral_0 = voxel*scipy.integrate.simps(scipy.integrate.simps((scipy.integrate.simps((new_result)))))
        # test_integral_1 = integrate_XYZ_numerically()...

        total_result += new_result
        print(f"The calculation of the density of frame {i} is over\n\n")

    print(f"So sum over all the density of frame, we got the distribution and the shape is {total_result.shape}")
    return total_result
