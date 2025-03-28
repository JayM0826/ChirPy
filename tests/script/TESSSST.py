import time

import chirpy as cp
import numpy as np

import utils as utils
from src.chirpy.classes.volume import ScalarField



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
a = np.random.rand(2,3)
print(a)
print(70  * "*")
v = np.random.rand(2,3)
print(v)
print(70  * "*")


print(np.sum(a + v))
print(70  * "*")


system = cp.trajectory.XYZ("tartrate.xyz").expand()
system.print_info()
atom_positions = system.data[-1][:, 0:3]
# density = utils.compute_whole_grid_density(atom_positions)
frame_1 = system.data[0][:, 0:3]
sigmas = np.ones(len(frame_1))
distribution = utils.compute_whole_grid_distribution(system.data, sigmas, )
# utils.plot_iosfurface(xv, yv, zv, values)
print(70 * "*")
scalar_field = ScalarField(data=distribution, origin_aa=system.pos_aa[0,0], pos_aa=system.pos_aa[0], numbers=cp.constants.symbols_to_numbers(system.symbols), cell_vec_aa=np.array([[1./2,0,0],[0,1./2,0],[0,0,1./2]]))
scalar_field.write(f"distribution_{time.time_ns()}.cube")
scalar_field.print_info()
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