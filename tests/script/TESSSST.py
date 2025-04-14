import time

import chirpy as cp
import numpy as np
from chirpy.classes.volume import ScalarField

import utils as utils


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

system = cp.trajectory.XYZ("tartrate.xyz").expand()
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

scalar_field = ScalarField(data=distribution, origin_aa=system.pos_aa[0,0] + np.array([R_x[0], R_y[0], R_z[0]]), pos_aa=system.pos_aa[0],
                           numbers=cp.constants.symbols_to_numbers(system.symbols),
                           cell_vec_aa=np.array([[R_x[1] - R_x[0], 0, 0],[0, R_y[1]- R_y[0], 0],[0, 0, R_z[1]- R_z[0]]]))
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
