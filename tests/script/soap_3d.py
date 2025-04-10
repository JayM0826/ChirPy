# -*- coding: utf-8 -*-
"""SOAP_3D.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17EOlWoDiq_5gg5_X_QR0IMMOLkX7Nmtk
"""

import numpy as np

import temp as temp
import utils as utils




@staticmethod
def example_density():
    R_x = np.linspace(-5, 5, 100)
    R_y = np.linspace(-5, 5, 100)
    R_z = np.linspace(-5, 5, 100)
    xv, yv, zv = np.meshgrid(R_x, R_y, R_z, indexing='ij')
    T_3D = np.array([0.1, 0.1, 0.1])
    ############################  x      y,    z
    atom_3D_position = np.array([[0, 0, 0],
                                 [2, 2, 2],
                                 [-1, -1, -1]])
    sigmas = np.array([2, 3, 4])
    density_function_3D = utils.trans_invariant_density_alt_3D1(atom_3D_position, sigmas)
    # how to determine the upper bound and lower bound: 3-sigma rule
    N = len(atom_3D_position)
    bounds = utils.coupute_XYZ_bounds(atom_3D_position, sigmas)
    print(f"the integration bounds for x,y,z are {bounds}")
    # print(f"Normalization constant = {utils.compute_normalisation_constant_to_N(result, 3)}")
    result, error = temp.integrate_XYZ_numerically(density_function_3D, *bounds)
    print(f"Numerical integral:{result}, it should be equal to {N}")
    result = density_function_3D(xv, yv, zv)
    print(np.sum(result))

    # return result




@staticmethod
def plot_isosurface():
    # the order is important(z,y,x)
    R_x = np.linspace(-5, 5, 100)
    R_y = np.linspace(-5, 5, 100)
    R_z = np.linspace(-5, 5, 100)
    xv, yv, zv = np.meshgrid(R_x, R_y, R_z, indexing="ij")
    T_3D = np.array([0.1, 0.1, 0.1])
    ############################  x      y,    z
    atom_3D_position = np.array([[0, 0, 0],
                                 [2, 2, 2],
                                 [-1, -1, -1]])
    sigmas = np.array([2, 3, 4])
    density_function_3D = utils.trans_invariant_density_alt_3D(atom_3D_position + T_3D, sigmas, xv, yv, zv)

    # values = density_function_3D(xv, yv, zv)
    # print(np.sum(values))
    utils.plot_iosfurface(xv, yv, zv, density_function_3D)

if __name__ == '__main__':
    result = example_density()
    # plot_isosurface()
# plot_isosurface()