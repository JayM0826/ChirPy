import numpy as np
from scipy.special import roots_legendre, eval_legendre
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# x_grid, w = roots_legendre(N)
#
# f_tilde = np.zeros_like(x_plot)
# # Reconstruct DVR density
# for j in range(N):
#     cj = np.sqrt(w[j]) * function(x_grid[j])
#
#     dvr_j = dvr_function(j, x_plot, N, x_grid, w)
#     # Plot basis function (optional)
#     line = \
#     ax.plot(x_plot, dvr_j * np.sqrt(w[j]), color='k', alpha=(1.2 - j / N) / 1.2 * 0.2, lw=1.5, ls=':', zorder=-100)[0]
#     extra_lines.append(line)

#
# --- radial part
def phi_n(n, x):
    """
    Orthonormal Legendre polynomial function on [-1, 1]
    n: base 0
    """
    weight_func = 1
    norm = np.sqrt((2 * n + 1) / 2)
    return norm * eval_legendre(n, x) * np.sqrt(weight_func)

# Construct DVR basis function ψ_j(x)
def dvr_basis_function(j, sample_points_plot, LEGENDRE_ORDER_NUM):
    """
    sample_points_plot must be in [-1, 1]
    """
    root_x, weight_x = roots_legendre(LEGENDRE_ORDER_NUM)
    return sum(phi_n(n, sample_points_plot) * phi_n(n, root_x[j])
               for n in range(LEGENDRE_ORDER_NUM)) * np.sqrt(weight_x[j])

# Original function to reconstruct
def original_function(x):
    return x ** 30 - x ** 12 + x ** 24 + x ** 3 - 1



# Reconstruct function using DVR basis
def reconstruct_function(x, LEGENDRE_ORDER_NUM):
    root_x, weight_x = roots_legendre(LEGENDRE_ORDER_NUM)

    f_values = original_function(root_x)
    result = np.zeros_like(x)
    for j in range(LEGENDRE_ORDER_NUM):
        cj = np.sqrt(weight_x[j]) * f_values[j]
        result += cj * dvr_basis_function(j, x, LEGENDRE_ORDER_NUM)

    # print(f_values[0])
    # print(dvr_basis_function(1, root_x[1], LEGENDRE_ORDER_NUM))
    # print(result[0])
    return result



# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
x_plot = np.linspace(-1, 1, 1000)
y_original = original_function(x_plot)
line_original, = ax.plot(x_plot, y_original, 'b-', label='Original fun')
# Initialize with zeros of the same shape as x_plot
line_reconstructed, = ax.plot(x_plot, np.zeros_like(x_plot), 'r--', label='DVR Reconstruction')
ax.set_xlim(-1, 1)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('DVR Reconstruction, N = 0')
ax.legend()
ax.grid(True)

# Animation update function
def update(N):
    y_reconstructed = reconstruct_function(x_plot, N)
    line_reconstructed.set_ydata(y_reconstructed)
    ax.set_title(f"DVR Reconstruction, N = {N}")
    return line_reconstructed,

# Create animation
max_N = 50
ani = FuncAnimation(fig, update, frames=range(2, max_N + 1), interval=500, blit=True)

# Save animation
ani.save('dvr_convergence.gif', writer='pillow')

# Show plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

# Define the polynomial function: f(x) = 3x^4 - 2x^2 + x
def f(x):
    return 3 * x**4 - 2 * x**2 + x

#

# import numpy as np
# from scipy.spatial.transform import Rotation as R
#
# # Step 1: Original frame
# frame = np.array([
#     [0.0, 0.0, 0.0],
#     [0.500, 0.0, 0.866],
#     [0.250, 1.000, -0.433],
#     [0.750, 1.000, -1.299]
# ])
#
# # Step 2: Generate a random SO(3) rotation
# rotation = R.random()
# R_matrix = rotation.as_matrix()  # 3x3 rotation matrix
#
# # Step 3: Apply the rotation to each point in the frame
# rotated_frame = frame @ R_matrix.T  # Transpose because we apply row-wise
#
# # Output results
# print("Original Frame:\n", frame)
# print("\nRotation Matrix:\n", R_matrix)
# print("\nRotated Frame:\n", rotated_frame)
