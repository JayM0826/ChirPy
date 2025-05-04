import math as math

import numpy as np
import plotly.graph_objects as go
import scipy as scipy
import sympy as sympy
from narwhals.utils import deprecated
from sympy import Symbol, integrate, sin, pi, conjugate
from sympy.functions.special.spherical_harmonics import Ynm

ONE_OVER_TWO_PI_POWER_1DIV2 = 1 / np.power(2 * np.pi, 0.5)
ONE_OVER_TWO_PI_POWER_3DIV2 = np.power(ONE_OVER_TWO_PI_POWER_1DIV2, 3)

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

def compute_normalisation_constant_to_N(integral_result, N=1):
    """
    Compute a normalisation constant to N, default = 1
    """
    return N / integral_result

def trans_invariant_density_3D_normalized1(fun, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, N=1):
    """
    be careful, the order of the return function is z,y,x
    """
    result, error = integrate_XYZ_numerically(fun, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
    print("!!!be careful, the parameters order of the return function is z,y,x!!!")
    return lambda z, y, x : compute_normalisation_constant_to_N(result, N) * fun(z,y,x)


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

def integrate_XYZ_numerically(fun, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper):
    result, error = scipy.integrate.tplquad(fun, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
    # result, error = 1, 1
    print("Numerical integral:", result)
    return result, error

def gaussian_fun_alt_3D(sigmas, relative_distances_3D):
    # NB: r must be an array
    relative_xs = relative_distances_3D[:, 0]
    relative_ys = relative_distances_3D[:, 1]
    relative_zs = relative_distances_3D[:, 2]
    # please check below
    return lambda x, y, z: (ONE_OVER_TWO_PI_POWER_3DIV2 * np.power(sigmas, 1.5)
                            * np.exp(1/(-2 * sigmas[:, 0]**2) * (x - relative_xs[:, np.newaxis,np.newaxis, np.newaxis]) ** 2
                               + 1/(-2 * sigmas[:, 1]**2) * (y - relative_ys[:, np.newaxis,np.newaxis, np.newaxis]) ** 2
                               + 1/(-2 * sigmas[:, 2]**2) * (z - relative_zs[:, np.newaxis,np.newaxis, np.newaxis]) ** 2))

from scipy.stats import multivariate_normal
def trans_invariant_density_3D(atom_positions, sigma_3D, r_cutoff=np.inf, origin_index=0):
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

def integrate_spherical_fun_analytically(n, m):
    # Spherical coordinates: dV = ρ² sin(φ) dρ dθ dφ
    # Expected variables: ρ (radius), θ (azimuth), φ (polar angle)
    # if len(symbols) != 2:
    #     raise ValueError("Spherical coordinates require 2 variables (θ, φ)")
    Y, phi, theta = Yml(m, n)
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


def Yml(m, l):
    theta = Symbol('theta')
    phi = Symbol('phi')
    Y = Ynm(m, l, theta, phi)
    return Y, phi, theta


@staticmethod
def integrate_spherical_fun(fun, radius, theta_lower=0, theta_upper=np.pi, phi_lower=0, phi_upper=np.pi * 2):
    # Spherical coordinates: dV = ρ² sin(0) dρ dθ dφ
    # Expected variables: ρ (radius), θ (polar angle), φ (azimuthal angle)
    # θ: From 0 to π, covering the full polar angle. φ: From 0 to 2π, covering the full azimuthal angle.
    # if len(symbols) != 2:
    #     raise ValueError("Spherical coordinates require 2 variables (θ, φ)")
    # Example Cartesian grid
    Nx, Ny, Nz = 50, 50, 50
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0
    z_min, z_max = -3.0, 3.0
    R_x = np.linspace(x_min, x_max, Nx)
    R_y = np.linspace(y_min, y_max, Ny)
    R_z = np.linspace(z_min, z_max, Nz)

    # Create a 3D Cartesian grid
    X, Y, Z = np.meshgrid(R_x, R_y, R_z, indexing='ij')

    # Example function: new_result(x, y, z) = exp(-(x^2 + y^2 + z^2))
    new_result = np.exp(-(X ** 2 + Y ** 2 + Z ** 2))

    # Perform the Cartesian integration
    int_z = integrate.simpson(new_result, R_z, axis=2)
    int_y = integrate.simpson(int_z, R_y, axis=1)
    int_x = integrate.simpson(int_y, R_x, axis=0)
    print("Cartesian integration result:", int_x)

    # Convert to spherical coordinates
    # Determine r_max
    r_max = np.sqrt(x_max ** 2 + y_max ** 2 + z_max ** 2)  # Maximum radius to cover the box

    # Spherical grid
    Nr, Ntheta, Nphi = 50, 50, 50
    R_r = np.linspace(0, r_max, Nr)
    R_theta = np.linspace(0, np.pi, Ntheta)
    R_phi = np.linspace(0, 2 * np.pi, Nphi)

    # Create a 3D spherical grid
    r, theta, phi = np.meshgrid(R_r, R_theta, R_phi, indexing='ij')

    # Convert spherical coordinates to Cartesian
    X_sph = r * np.sin(theta) * np.cos(phi)
    Y_sph = r * np.sin(theta) * np.sin(phi)
    Z_sph = r * np.cos(theta)

    # Evaluate new_result on the spherical grid
    # First, check if the point is within the Cartesian box
    mask = (X_sph >= x_min) & (X_sph <= x_max) & \
           (Y_sph >= y_min) & (Y_sph <= y_max) & \
           (Z_sph >= z_min) & (Z_sph <= z_max)

    # Compute new_result(r, theta, phi)
    new_result_sph = np.zeros_like(r)
    new_result_sph[mask] = np.exp(-(X_sph[mask] ** 2 + Y_sph[mask] ** 2 + Z_sph[mask] ** 2))

    # Include the Jacobian: r^2 * sin(theta)
    jacobian = r ** 2 * np.sin(theta)
    integrand = new_result_sph * jacobian

    # Integrate over phi, theta, r
    int_phi = integrate.simpson(integrand, R_phi, axis=2)
    int_theta = integrate.simpson(int_phi, R_theta, axis=1)
    int_r = integrate.simpson(int_theta, R_r, axis=0)

    print("Spherical integration result:", int_r)



def trans_invariant_density_alt_3D(atom_positions, sigma_3D, xv, yv, zv, r_cutoff=CUT_OFF, origin_index=ORIGIN_ATOM_INDEX):
    # TODO use different origin
    relative_distances_3D = atom_positions - atom_positions[origin_index]  # shape = (N, 2), N means #atoms, 2 means relative distance of (x and y)
    return gaussian_fun_3D(sigma_3D, relative_distances_3D, xv, yv, zv)

def gaussian_fun_3D(sigmas, relative_distances_3D, xv, yv, zv):
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


def convert_spherical_harmonic_to_cartesian(l, m):
    """
    Convert a real spherical harmonic Z_{l,m}(theta, phi) to its Cartesian form.

    Parameters:
    l (int): Degree of the spherical harmonic (non-negative integer).
    m (int): Order of the spherical harmonic (-l <= m <= l).

    Returns:
    tuple: (angular_form, cartesian_form) where:
        - angular_form is the expression in terms of theta, phi.
        - cartesian_form is the expression in terms of x, y, z, r.
    """
    # Define angular coordinates
    theta, phi = sp.symbols('theta phi')

    # Define Cartesian coordinates
    x, y, z, r = sp.symbols('x y z r')

    # Compute the real spherical harmonic using Znm
    Y_lm_real = Znm(l, m, theta, phi)


    # Define substitutions for spherical to Cartesian coordinates
    substitutions = {
        sp.sin(theta): sp.sqrt(x ** 2 + y ** 2) / r,
        sp.cos(theta): z / r,
        sp.sin(phi): y / sp.sqrt(x ** 2 + y ** 2),
        sp.cos(phi): x / sp.sqrt(x ** 2 + y ** 2),
    }

    # Handle trigonometric functions of 2*phi and 2*theta
    sin_2phi = 2 * substitutions[sp.sin(phi)] * substitutions[sp.cos(phi)]
    sin_2phi = sin_2phi.simplify()  # 2*x*y/(x^2 + y^2)

    cos_2phi = substitutions[sp.cos(phi)] ** 2 - substitutions[sp.sin(phi)] ** 2
    cos_2phi = cos_2phi.simplify()  # (x^2 - y^2)/(x^2 + y^2)

    sin_2theta = 2 * substitutions[sp.sin(theta)] * substitutions[sp.cos(theta)]
    sin_2theta = sin_2theta.simplify()  # 2*sqrt(x^2 + y^2)*z/r^2

    # Additional substitutions for higher-order terms
    additional_subs = {
        sp.sin(2 * phi): sin_2phi,
        sp.cos(2 * phi): cos_2phi,
        sp.sin(2 * theta): sin_2theta,
        sp.sin(theta) ** 2: (x ** 2 + y ** 2) / r ** 2,
        sp.cos(theta) ** 2: z ** 2 / r ** 2,
    }

    # Combine all substitutions
    substitutions.update(additional_subs)

    # Substitute and simplify
    Y_lm_real = Y_lm_real.expand(func=True)
    Y_lm_cartesian = Y_lm_real.subs(substitutions)

    # Adjust for the r^l factor (spherical harmonics are defined on r=1)
    # Multiply by r^l to get the homogeneous polynomial form, then divide by r^l in the final expression
    Y_lm_cartesian = Y_lm_cartesian * r ** l

    # Simplify again
    Y_lm_cartesian = Y_lm_cartesian.simplify()
    # Evaluate symbolic constants numerically



    return Y_lm_real, Y_lm_cartesian


def evaluate_cartesian_form(cartesian_form, x_val, y_val, z_val, method='sympy'):
    """
    Evaluate the Cartesian form of a spherical harmonic at a given (x, y, z) point.

    Parameters:
    cartesian_form: SymPy expression in terms of x, y, z, r.
    x_val (float): x-coordinate.
    y_val (float): y-coordinate.
    z_val (float): z-coordinate.
    method (str): 'sympy' for symbolic evaluation, 'numpy' for numerical evaluation.

    Returns:
    float: The value of the spherical harmonic at the given point.
    """
    # Compute r = sqrt(x^2 + y^2 + z^2)
    r_val = np.sqrt(x_val ** 2 + y_val ** 2 + z_val ** 2)
    # Define symbols
    x, y, z, r = sp.symbols('x y z r')
    if method == 'sympy':


        # Substitute the values
        value = cartesian_form.subs({x: x_val, y: y_val, z: z_val, r: r_val})

        # Convert to numerical value
        value = float(value.evalf())

    elif method == 'numpy':
        # Convert SymPy expression to a NumPy function
        f = sp.lambdify((x, y, z, r), cartesian_form.evalf(), 'numpy')

        # Evaluate using NumPy
        value = f(x_val, y_val, z_val, r_val)

    else:
        raise ValueError("Method must be 'sympy' or 'numpy'")

    return value


# Main script to compute and evaluate spherical harmonics
def compute_and_evaluate_spherical_harmonics(l, m, x_val, y_val, z_val):
    """
    Compute the Cartesian form of a spherical harmonic and evaluate it at (x, y, z).

    Parameters:
    l (int): Degree of the spherical harmonic.
    m (int): Order of the spherical harmonic.
    x_val (float): x-coordinate.
    y_val (float): y-coordinate.
    z_val (float): z-coordinate.

    Returns:
    None (prints the results).
    """
    # Compute the angular and Cartesian forms
    angular_form, cartesian_form = convert_spherical_harmonic_to_cartesian(l, m)

    # Print the forms
    print(f"\nY_{l},{m}:")
    print("Angular form:")
    sp.pprint(angular_form)
    print("Cartesian form:")
    sp.pprint(cartesian_form)

    # Evaluate using SymPy
    # value_sympy = evaluate_cartesian_form(cartesian_form, x_val, y_val, z_val, method='sympy')
    # print(f"\nValue at (x={x_val}, y={y_val}, z={z_val}) using SymPy: {value_sympy}")

    # Evaluate using NumPy
    value_numpy = evaluate_cartesian_form(cartesian_form, x_val, y_val, z_val, method='numpy')
    print(f"Value at (x={x_val}, y={y_val}, z={z_val}) using NumPy: {value_numpy}")

#
# # Test the tool for l=2, m=-2 to 2 at (x, y, z) = (1, 2, 3)
# x_val, y_val, z_val = 1.0, 2.0, 3.0
# for m in range(-2, 3):
#     compute_and_evaluate_spherical_harmonics(2, m, x_val, y_val, z_val)
# a = evaluate_on_unit_sphere(2,-1, 1, 1)
# print(a)



def evaluate_on_unit_sphere(l, m):
    """
    Evaluate a spherical harmonic at a point (theta, phi) on the unit sphere.

    Parameters:
    l (int): Degree of the spherical harmonic.
    m (int): Order of the spherical harmonic.
    theta (float): Polar angle.
    phi (float): Azimuthal angle.
    Returns:
    float: Value of the spherical harmonic.
    """
    Y_lm, theta_sym, phi_sym = Y_lm_real(l, m)
    f = sp.lambdify((theta_sym, phi_sym), Y_lm, 'numpy')
    return lambda theta, phi: f(theta, phi).real

if __name__ == '__main__':
    integrate_spherical_fun(1,2)

import time

import chirpy as cp
import numpy as np
from chirpy.classes.volume import ScalarField

import numerical_utils as utils

# density_data = soap_3d.example_density()
#
# # print(system.__dict__)
#
# cell_vec_aa = np.array([[2.0, 0.0, 0.0],
#                         [0.0, 2.0, 0.0],
#                         [0.0, 0.0, 2.0]])
# print(70 * "*")
# scalar_field = cp.classes.volume.ScalarField(data=density_data, cell_vec_aa=cell_vec_aa)
# print(70 * "*")
# scalar_field.print_info()

"""
output:

–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
ScalarField 
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
CELL  200.00000  200.00000  200.00000   90.00000   90.00000   90.00000
CUBIC
-----------------------------------------------------------------------------
 A    200.00000    0.00000    0.00000
 B      0.00000  200.00000    0.00000
 C      0.00000    0.00000  200.00000

"""

# from scipy.integrate import quad, lebedev_rule
# import numpy as np
#
# def f_spherical(r, theta, phi):
#     x = r * np.sin(theta) * np.cos(phi)
#     y = r * np.sin(theta) * np.sin(phi)
#     z = r * np.cos(theta)
#     return x**2 + y**2
#
# def integrand(r):
#     order = 11
#     points, weights = lebedev_rule(order)
#     x, y, z = points
#     theta = np.arccos(z)
#     phi = np.arctan2(y, x)
#     values = f_spherical(r, theta, phi)
#     surface_integral = np.sum(weights * values)
#     return r**2 * surface_integral
#
# R = 2.0
# integral, _ = quad(integrand, 0, R)
# print(f"体积分结果: {integral}")
obj = cp.trajectory.XYZ("/home/jim92/PycharmProjects/ChirPy/tests/tartrate.xyz")
system = obj.expand()
print(70 * "*")
print(70 * "*")
print(70 * "*")
print(70 * "*")

system.print_info()
print(system.data.shape)
atom_positions = system.data[-1][:, 0:3]
# density = utils.compute_whole_grid_density(atom_positions)
frame_1 = system.data[0][:, 0:3]
sigmas = utils.get_sigmas(frame_1)
distribution, R_x, R_y, R_z = utils.compute_whole_grid_distribution(system.data, sigmas, )
print(f"x step size:{R_x[1] - R_x[0]}, len of R_x={len(R_x)}, and result = {len(R_x) * (R_x[1] - R_x[0])}")
print(f"y step size:{R_y[1] - R_y[0]}, len of R_y={len(R_y)}, and result = {len(R_y) * (R_y[1] - R_y[0])}")
print(f"z step size:{R_z[1] - R_z[0]}, len of R_z={len(R_z)}, and result = {len(R_z) * (R_z[1] - R_z[0])}")
# distribution /= len(system.data) # ???

# print(f"distribution is {np.sum(distribution)}")
# utils.plot_iosfurface(xv, yv, zv, values)
print(70 * "*")

scalar_field = ScalarField(data=distribution, origin_aa=system.pos_aa[0, 0] + np.array([R_x[0], R_y[0], R_z[0]]),
                           pos_aa=system.pos_aa[0],
                           numbers=cp.constants.symbols_to_numbers(system.symbols),
                           cell_vec_aa=np.array(
                               [[R_x[1] - R_x[0], 0, 0], [0, R_y[1] - R_y[0], 0], [0, 0, R_z[1] - R_z[0]]]))
# scalar_field = ScalarField(data=distribution, origin_aa=system.pos_aa[0,0], pos_aa=system.pos_aa[0], numbers=cp.constants.symbols_to_numbers(system.symbols),
# cell_vec_aa=np.array([[1./2, 0, 0],[0, 1./2, 0],[0, 0, 1./2]]), grid_x=R_x, grid_y=R_y, grid_z=R_z)
scalar_field.print_info()
scalar_field.write(f"distribution_{time.time_ns()}.cube")

print(scalar_field.integral(volume_unit='aa**3'))
print(scalar_field.voxel)
print(scalar_field.integral() / scalar_field.voxel)

# import numpy as np
# from sympy import symbols, lambdify, expand
# from sympy.physics.hydrogen import Ynm
# import plotly.graph_objects as go
#
# # Step 1: Define the spherical harmonic symbolically
# theta, phi = symbols('theta phi')
# Y_11 = Ynm(1, 1, theta, phi)
#
# # # Step 2: Fully expand the expression
# # Y_11_expanded = expand(Y_11)
# # print("Expanded Y_11:", Y_11_expanded)  # Verify: -sqrt(3)*exp(I*phi)*sin(theta)/(2*sqrt(2)*sqrt(pi))
#
# # Step 3: Convert to numerical function
# Y_11_func = lambdify((theta, phi), Y_11, modules=['numpy', {'Ynm': Ynm}])
#
# # Step 4: Test with a single value to ensure it works
# test_val = Y_11_func(np.pi/2, 0)
# print("Test value at (pi/2, 0):", test_val)  # Should be ≈ -0.345 + 0j
#
# # Step 5: Create spherical grid
# theta_vals = np.linspace(0, np.pi, 100)
# phi_vals = np.linspace(0, 2 * np.pi, 100)
# Theta, Phi = np.meshgrid(theta_vals, phi_vals)
#
# # Step 6: Evaluate over the grid
# Y_11_vals = Y_11_func(Theta, Phi)
# Y_11_real = np.real(Y_11_vals)  # Take real part for plotting
#
# # Step 7: Convert to Cartesian
# # Change: Evaluate Ynm numerically before taking the absolute value
# Y_11_real_num = np.array([[Ynm(1, 1, th, ph).evalf() for ph in phi_vals] for th in theta_vals], dtype=np.complex128) # Evaluate Ynm numerically
# r = np.abs(Y_11_real_num) # Now take absolute value
# X = r * np.sin(Theta) * np.cos(Phi)
# Y = r * np.sin(Theta) * np.sin(Phi)
# Z = r * np.cos(Theta)
#
#
# # Step 8: Plot with Plotly
# # Change: Use Y_11_real_num.real instead of Y_11_real for surfacecolor
# fig = go.Figure(data=[go.Surface(
#     x=X, y=Y, z=Z,
#     surfacecolor=Y_11_real_num.real,  # Use the real part of the numerically evaluated Ynm
#     colorscale='Viridis',
#     colorbar=dict(title='Re(Y_1^1)')
# )])
# fig.update_layout(
#     title='Spherical Harmonic Y_1^1',
#     scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectratio=dict(x=1, y=1, z=1)),
#     width=700, height=700
# )
# fig.show()
#
# from sympy import symbols, exp, integrate, erf, sqrt, pi
#
# # Define symbolic variables, coefficients, and means
# x, y, z = symbols('x y z')
# a, b, c = symbols('a b c', positive=True)  # Positive for convergence
# mu_x, mu_y, mu_z = symbols('mu_x mu_y mu_z')
#
# # Define the 3D Gaussian with non-zero means
# f = exp(-(a * (x - mu_x)**2 + b * (y - mu_y)**2 + c * (z - mu_z)**2))
#
# # Antiderivative with respect to x (y and z as constants)
# F_x = integrate(f, x)
# print("Antiderivative with respect to x:")
# print(F_x)
#
# # Antiderivative with respect to y (x and z as constants)
# F_y = integrate(f, y)
# print("\nAntiderivative with respect to y:")
# print(F_y)
#
# # Antiderivative with respect to z (x and y as constants)
# F_z = integrate(f, z)
# print("\nAntiderivative with respect to z:")
# print(F_z)
#
#
#
#
#
# from sympy import symbols, exp, integrate, erf, sqrt, pi
# print(70 * "*")
#
# x, y, z, sigma_x, sigma_y, sigma_z, mu_x, mu_y, mu_z = symbols('x y z sigma_x sigma_y sigma_z mu_x mu_y mu_z', positive=True)
# f_norm = (1 / ((2 * pi)**(3/2) * sigma_x * sigma_y * sigma_z)) * exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2) + (z - mu_z)**2 / (2 * sigma_z**2)))
# F_x_norm = integrate(f_norm, x)
# print(F_x_norm)
import numpy as np



