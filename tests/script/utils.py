import math as math

import numpy as np
import copy

import plotly.graph_objects as go
from sympy import Symbol, integrate, sin, pi
from sympy.functions.special.spherical_harmonics import Ynm
from scipy.stats import multivariate_normal
from scipy import integrate
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


def trans_invariant_density_alt_3D(atom_positions, sigma_3D, xv, yv, zv, r_cutoff=np.inf, origin_index=0):
    # TODO use different origin
    relative_distances_3D = atom_positions - atom_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    return gaussian_fun_alt_3D_sequentially(sigma_3D, relative_distances_3D, xv, yv, zv)

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

def compute_whole_grid_density(atom_positions, sigmas, xv, yv, zv):
    atom_positions = atom_positions[:, 0:3]
    return trans_invariant_density_alt_3D(atom_positions, sigmas, xv, yv, zv)


def generate_grid_and_bounds(atom_positions, sigmas, number_per_unit_distance=NUMBER_PER_UNIT_DISTANCE):
    """
       number_per_unit_distance: the grid number per unit distance
    """
    bounds = coupute_XYZ_bounds(atom_positions, sigmas)
    R_x = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0]) * number_per_unit_distance)
    R_y = np.linspace(bounds[2], bounds[3], (bounds[3] - bounds[2]) * number_per_unit_distance)
    R_z = np.linspace(bounds[4], bounds[5], (bounds[5] - bounds[4]) * number_per_unit_distance)

    xv, yv, zv = np.meshgrid(R_x, R_y, R_z, indexing='ij')
    return xv, yv, zv, bounds, R_x, R_y, R_z


def compute_whole_grid_distribution(trajectory, sigmas, number_per_unit_distance=NUMBER_PER_UNIT_DISTANCE):
    """
        number_per_unit_distance: the grid number per unit distance
        trajectory shape should be (No_frame, No_atoms, No_coordiantes)
    """
    # memory-friendly
    print(f"There will be {len(trajectory)} frames need to calculate...")

    xv, yv, zv, bounds, R_x, R_y, R_z = generate_grid_and_bounds(trajectory[0][:, 0:3], sigmas, number_per_unit_distance)
    print(f"grid detail:shape={xv.shape} bounds={bounds}")
    total_result = np.ndarray(xv.shape)
    i = 1
    for atom_positions in trajectory[0:1, :, 0:3]:
        print(f"Now start to calculate the density of frame {i}")
        new_result = compute_whole_grid_density(atom_positions, sigmas, xv, yv, zv)
        # be careful: the order of simpson is R_z,R_x,R_y
        print(f"Integration of density of frame {i} over all intervals is {integrate.simpson(integrate.simpson(integrate.simpson(new_result, zv[0,0,:]), xv[0, :, 0]), yv[:, 0, 0])}"
              f", it should be equal to the number of atoms:{len(atom_positions)}")
        total_result += new_result
        print(f"The calculation of the density of frame {i} is over\n\n")
        i += 1

    print(f"So sum over all the density of frame, we got the distribution and the shape is {total_result.shape}")
    print(f"Integration of density of frame all over all intervals is {integrate.simpson(integrate.simpson(integrate.simpson(total_result, R_z), R_y), R_x)}")

    return total_result, R_x, R_y, R_z

def get_sigmas(atoms):
    # TODO build a dict and store the sigmas for the atoms and retrieve them when needed
    # now just return 1s blindly
    return np.ones(len(atoms))

