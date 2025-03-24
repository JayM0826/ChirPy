# do not use below method, because sympy can not get a value when the function got complicated
# Define the Gaussian density symbolically
# def gaussian_fun_alt_3D_sympy(x, y, z, x0, y0, z0, sigma_3D):
#     """
#         Compute symbolic 3D Gaussian function for a single point
#         x, y, z: symbolic variables
#         x0, y0, z0: coordinates of Gaussian center
#         sigma_3D: array-like with [sigma_x, sigma_y, sigma_z]
#         """
#     return sympy.exp(
#         -((x - x0) ** 2) / (2 * sigma_3D[0] ** 2) +
#         -((y - y0) ** 2) / (2 * sigma_3D[1] ** 2) +
#         -((z - z0) ** 2) / (2 * sigma_3D[2] ** 2)
#     )
#
# def trans_invariant_density_alt_3D_sympy(atom_positions, sigma_3D, r_cutoff=np.inf, origin_index=0):
#     """
#     Create symbolic translation-invariant density function
#     atom_positions: numpy array of shape (N, 3) with atom coordinates
#     sigma_3D: array-like with [sigma_x, sigma_y, sigma_z]
#     """
#     x, y, z = sympy.symbols('x y z')
#     relative_distances_3D = atom_positions - atom_positions[origin_index]
#
#     density = 0
#     for pos in relative_distances_3D:
#         density += gaussian_fun_alt_3D_sympy(
#             x, y, z,
#             float(pos[0]), float(pos[1]), float(pos[2]),
#             sigma_3D
#         )
#
#     # density += gaussian_fun_alt_3D_sympy(
#     #     x, y, z,
#     #     float(relative_distances_3D[0][0]), float(relative_distances_3D[0][1]), float(relative_distances_3D[0][2]),
#     #     sigma_3D
#     # )
#     return density