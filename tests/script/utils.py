import math as math

import numpy as np
import plotly.graph_objects as go
import scipy as scipy


def compute_normalisation_constant_to_N(integral_result, N=1):
    """
    Compute a normalisation constant to N, default = 1
    """
    return N / integral_result

def integrate_XYZ_numerically(fun, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper):
    result, error = scipy.integrate.tplquad(fun, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
    print("Numerical integral:", result)
    return result, error


def coupute_XYZ_bounds(atom_3D_positions, sigmas, origin_index=0):
    """
    atom_3D_positions:
    atom_3D_position = np.array([[0,     0,    0],
                                 [2,     2,    2],
                                 [-1,   -1,   -1]])
    sigmas = np.array([2, 3, 4])#, isotropic for now

    return the bounds like (-x, x, -y, y, -z, z)

    """
    atom_3D_relative_positions  = atom_3D_positions - atom_3D_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)

    max_values = np.max((atom_3D_relative_positions), axis=0)
    min_values = np.min((atom_3D_relative_positions), axis=0)
    x_upper, y_upper, z_upper = max_values + np.max(sigmas) * 6 # here we use 6 sigma so that the accuracy is higher
    x_lower, y_lower, z_lower = min_values - np.max(sigmas) * 6 # here we use 6 sigma so that the accuracy is higher
    # easy to use linspace with int
    return tuple(int(math.ceil(x)) for x in (x_lower, x_upper, y_lower, y_upper, z_lower, z_upper))


# def integrate_XYZ_analytically(*symbols, fun, coordinate_system="XYZ", bounds):
#     """
#         Compute a volume integral using SymPy in XYZ coordinate system
#
#         Parameters:
#         - *symbols: Variables of integration (e.g., x, y, z)
#         - fun: Function to integrate
#         - bound: Dictionary of bounds for each variable (e.g., {x: (0,1)}); defaults to None, implying (-oo, oo)
#
#         Returns:
#         - The evaluated integral
#
#         x, y, z = sympy.symbols('x y z')
#         result_xyz = integrate_XYZ_analytically(x, y,
#                               fun=x + y + z,
#                               coordinate_system="XYZ",
#                               bounds={x: (0, 1), y: (0, 2)})  # z defaults to (-oo, oo)
#         print("Cartesian result (partial bounds):", result_xyz)
#
#         TODO for Spherical system
#         # Spherical example: Volume integral with default bounds
#         # rho, theta, phi = sympy.symbols('rho theta phi')
#         # result_sph = integrate_3D(rho, theta, phi,
#         #                       fun=1,  # Just computing volume
#         #                       coordinate_system="SPH")  # All bounds default
#         # print("Spherical result (default bounds):", result_sph)
#         """
#     # Validate input
#     if len(symbols) > 3:
#         raise ValueError("Maximum 3 variables allowed for volume integral")
#     if not bounds:
#         raise ValueError("Integration bounds must be provided")
#
#     # bounds_complemented = complete_integration_bound(bounds, symbols)
#
#     # Handle different coordinate systems
#     if coordinate_system.upper() == "XYZ":
#         # Cartesian coordinates: dV = dx dy dz
#         # Cartesian coordinates: dV = dx dy dz
#         result = fun
#         for var in reversed(symbols):  # Integrate from inner to outer
#             if var not in bounds:
#                 raise ValueError(f"Bounds for {var} not provided")
#             lower, upper = bounds[var]
#             result = sympy.integrate(result, (var, lower, upper))
#         return result
#     pass
    # if coordinate_system == "ρθφ":
    # else:
    #     # Spherical coordinates: dV = ρ² sin(φ) dρ dθ dφ
    #     # Expected variables: ρ (radius), θ (azimuth), φ (polar angle)
    #     if len(symbols) != 3:
    #         raise ValueError("Spherical coordinates require 3 variables (ρ, θ, φ)")
    #
    #     # Multiply by Jacobian (ρ² sin(φ))
    #     result = fun * symbols[0] ** 2 * sympy.sin(phi)
    #
    #     # Integrate with respect to each variable
    #     for var in reversed(symbols):
    #         if var not in bounds:
    #             raise ValueError(f"Bounds for {var} not provided")
    #         lower, upper = bounds[var]
    #         result = sympy.integrate(result, (var, lower, upper))
    #     return result

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

def gaussian_fun_alt_3D(sigma_3D, relative_distances_3D):
    # NB: r must be an array
    relative_xs = relative_distances_3D[:, 0]
    relative_ys = relative_distances_3D[:, 1]
    relative_zs = relative_distances_3D[:, 2]

    return lambda x, y, z: np.exp(1/(-2 * sigma_3D[0]**2) * (x - relative_xs[:, np.newaxis,np.newaxis, np.newaxis]) ** 2
                               + 1/(-2 * sigma_3D[1]**2) * (y - relative_ys[:, np.newaxis,np.newaxis, np.newaxis]) ** 2
                               + 1/(-2 * sigma_3D[2]**2) * (z - relative_zs[:, np.newaxis,np.newaxis, np.newaxis]) ** 2)

def trans_invariant_density_alt_3D(atom_positions, sigma_3D, r_cutoff = np.inf, origin_index = 0):
    relative_distances_3D  = atom_positions - atom_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    density = gaussian_fun_alt_3D(sigma_3D, relative_distances_3D)
    return lambda z, y, x: np.sum(density(x, y, z), axis=0)


def trans_invariant_density_alt_3D(atom_positions, sigma_3D, r_cutoff = np.inf, origin_index = 0):
    """
        be careful, the order of the return function is z,y,x
    """
    relative_distances_3D  = atom_positions - atom_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    density = gaussian_fun_alt_3D(sigma_3D, relative_distances_3D)
    return lambda z, y, x: np.sum(density(x, y, z), axis=0)

def trans_invariant_density_3D_normalized(fun, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, N=1):
    """
    be careful, the order of the return function is z,y,x
    """
    result, error = integrate_XYZ_numerically(fun, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
    print("!!!be careful, the parameters order of the return function is z,y,x!!!")
    return lambda z, y, x : compute_normalisation_constant_to_N(result, N) * fun(z,y,x)

def compute_whole_grid_density(atom_positions, sigmas, xv, yv, zv, bounds):
    """
    number_per_unit_distance: the grid number per unit distance
    """
    atom_positions = atom_positions[:, 0:3]
    density_function_3D = trans_invariant_density_alt_3D(atom_positions, sigmas)
    normalized_density_function = trans_invariant_density_3D_normalized(density_function_3D, *bounds, len(sigmas))
    values = normalized_density_function(zv, yv, xv)
    # print(f"density = {values} shape={values.shape}")
    return values

def generate_grid_and_bounds(atom_positions, sigmas, number_per_unit_distance = 10):
    bounds = coupute_XYZ_bounds(atom_positions, sigmas)
    R_x = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0]) * number_per_unit_distance)
    R_y = np.linspace(bounds[2], bounds[3], (bounds[3] - bounds[2]) * number_per_unit_distance)
    R_z = np.linspace(bounds[4], bounds[5], (bounds[5] - bounds[4]) * number_per_unit_distance)

    xv, yv, zv = np.meshgrid(R_x, R_y, R_z)
    return xv, yv, zv, bounds
def compute_whole_grid_distribution(trajectory, number_per_unit_distance = 10):
    """
        number_per_unit_distance: the grid number per unit distance
        trajectory shape should be (No_frame, No_atoms, No_coordiantes)
    """
    # memory-friendly
    print(f"There will be {len(trajectory)} frames need to calculate...")
    frame_1 = trajectory[0][:, 0:3]
    sigmas = np.ones(len(frame_1))
    xv, yv, zv, bounds = generate_grid_and_bounds(frame_1, sigmas, number_per_unit_distance)
    print(f"grid detail:shape={xv.shape} bounds={bounds}")
    result = np.ndarray(xv.shape)
    i = 1
    for atom_positions  in trajectory[:, :, 0:3]:
        print(f"Now start to calculate the density of frame {i}")
        i += 1
        new_result = compute_whole_grid_density(atom_positions, sigmas, xv, yv, zv, bounds)
        result += new_result
        print(f"The calculation of the density of frame {i} is over\n\n")

    print(f"So sum over all the density of frame, we got the distribution and the shape is {result.shape}")
    return result



def complete_integration_bound(bounds, symbols, coordinate_system="XYZ"):
    """
    ignore this method for now
    """
    # Set default bounds for all symbols to (-oo, oo), then update with provided bounds
    import sympy as sympy
    if coordinate_system.upper() == "XYZ":
        bounds_new = {var: (-sympy.oo, sympy.oo) for var in symbols}
        for symbol in symbols:
            if symbol in bounds:
                bounds_new[symbol] = bounds[symbol]
        return bounds_new
    else:
        # Spherical coordinates: dV = ρ² sin(φ) dρ dθ dφ
        if len(symbols) != 3:
            raise ValueError("Spherical coordinates require 3 variables (ρ, θ, φ)")
        # Adjust default bounds for spherical: ρ ≥ 0, θ in [0, 2π], φ in [0, π]
        sph_defaults = {symbols[0]: (0, sympy.oo), symbols[1]: (0, 2*sympy.pi), symbols[2]: (0, sympy.pi)}
        bounds_new = sph_defaults.copy()
        for symbol in symbols:
            if symbol in bounds:
                bounds_new[symbol] = bounds[symbol]
        return bounds_new

