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