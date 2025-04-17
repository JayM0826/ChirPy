# built-in module
import math as math

import chirpy
import numpy as np
import plotly.graph_objects as go
import pyfiglet
import pyfiglet
import datetime
import platform
# import torch
from colorama import Fore, Style, init
import time
from PIL import Image


def print_format_coefficients(coefficients, l_max):
    print(300 * "*")
    print(f"{100 * '*'}COEFFICIENTS{140 * '*'}")
    print(300 * "*")

    # Format the array to 10 decimal places
    coefficients = np.around(coefficients, decimals=10)

    # Print the array with indentation to form a triangle
    print("[")
    start_idx = 0
    max_width = l_max * 2 + 1  # Maximum width (e.g., l=6, m=-6 to 6)
    for l in range(l_max + 1):
        # Number of elements in this row (2l+1)
        row_length = 2 * l + 1
        # Extract the elements for this row
        row = coefficients[start_idx:start_idx + row_length]
        # Calculate indentation to center the row
        indent = (max_width - row_length) // 2
        indent_str = "               " * indent
        # Format the row
        formatted_row = f"{l}:{indent_str}{',  '.join(f'{x:.10f}' for x in row)}"
        print(f"{formatted_row},")

        start_idx += row_length
    print("]")
    print(300 * "*")
    print(300 * "*")


def cartesian_to_spherical(x, y, z, r=1):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (theta, phi).

    Parameters:
    x (float or np.ndarray): x-coordinate(s).
    y (float or np.ndarray): y-coordinate(s).
    z (float or np.ndarray): z-coordinate(s).

    Returns:
    tuple: (theta, phi)
        - theta (float or np.ndarray): Polar angle in [0, pi].
        - phi (float or np.ndarray): Azimuthal angle in [0, 2*pi).
    """
    # Handle the case where r = 0 to avoid division by zero
    # At the origin, set theta = 0, phi = 0 as a convention
    theta = np.where(r == 0, 0.0, np.arccos(z / np.where(r == 0, 1.0, r)))

    # Compute phi using atan2, which handles all quadrants correctly
    phi = np.arctan2(y, x)

    # Ensure phi is in [0, 2*pi)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    return theta, phi


def unit_spherical_to_cartesian(theta, phi, r):
    """
    r: real spherical harmonics value on (theta, phi)
    return the x,y,z based on the unit sphere. If it needs to be scaled, multiply the specific radius.
     It should not change the value because real spherical harmonics fun is independent of radius.
    """
    xyz = np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)])
    return np.abs(r) * xyz


def get_relative_path(BASE_PATH, *path_segments):
    """
    Build a path relative to the project root.
    data_path = con.get_relative_path("data", "input.csv")
    print(data_path)   # /path/to/project/data/input.csv, it is an absloute path
    """
    return BASE_PATH.joinpath(*path_segments).resolve()


def generate_grid_and_bounds(atom_positions, sigmas, number_per_unit_distance, cutoff,
                             origin_index):
    """
       number_per_unit_distance: the grid number per unit distance
    """
    bounds = coupute_XYZ_bounds(atom_positions, sigmas, cutoff, origin_index)
    R_x = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0]) * number_per_unit_distance)
    R_y = np.linspace(bounds[2], bounds[3], (bounds[3] - bounds[2]) * number_per_unit_distance)
    R_z = np.linspace(bounds[4], bounds[5], (bounds[5] - bounds[4]) * number_per_unit_distance)

    xv, yv, zv = np.meshgrid(R_x, R_y, R_z, indexing='ij')
    return xv, yv, zv, bounds, R_x, R_y, R_z


def coupute_XYZ_bounds(atom_3D_positions, sigmas, cutoff, origin_atom_index):
    """
    atom_3D_positions:
    atom_3D_position = np.array([[0,     0,    0],
                                 [2,     2,    2],
                                 [-1,   -1,   -1]])
    sigmas = np.array([2, 3, 4])#, isotropic for now

    return the bounds like (-x, x, -y, y, -z, z)

    """
    # shape = (N, 3), N means #atoms, 3 means relative distance of (x and y, z)
    if cutoff != np.inf:
        x_upper, y_upper, z_upper = cutoff, cutoff, cutoff,
        x_lower, y_lower, z_lower = - cutoff, - cutoff, - cutoff
        return tuple(int(math.ceil(x)) for x in (x_lower, x_upper, y_lower, y_upper, z_lower, z_upper))

    # otherwise
    atom_3D_relative_positions = atom_3D_positions - atom_3D_positions[origin_atom_index]

    max_values = np.max((atom_3D_relative_positions), axis=0)
    min_values = np.min((atom_3D_relative_positions), axis=0)
    # here we use 6(without scientific proof) sigma so that the accuracy is higher
    x_upper, y_upper, z_upper = max_values + np.max(sigmas) * 6
    x_lower, y_lower, z_lower = min_values - np.max(sigmas) * 6
    # easy to use linspace with int
    return tuple(int(math.ceil(x)) for x in (x_lower, x_upper, y_lower, y_upper, z_lower, z_upper))


def filter_atoms_within_cutoff(positions, origin_atom_index, cutoff):
    """
    Filter atoms within a cutoff distance from a central atom.

    Parameters:
    - positions: Nx3 array of (x, y, z) coordinates.
    - center_atom_idx: Index of the central atom.
    - cutoff: Cutoff radius (e.g., in Å).

    Returns:
    - Filtered array of qualifying atoms (excluding center if desired).
    """
    origin_pos = positions[origin_atom_index]
    relative_distance = np.sqrt(np.sum((positions - origin_pos) ** 2, axis=1))
    qualified_indices = np.where(relative_distance <= (cutoff))[0]

    qualified_atom_positions = positions[qualified_indices]
    # smooth_coefficients = smooth_function(relative_distance[qualified_indices], cutoff)
    # TODO in fact there is no smooth if use the below code instead of the above line
    smooth_coefficients = np.ones(len((qualified_atom_positions)))
    return qualified_atom_positions, smooth_coefficients


def smooth_function(relative_distance, cutoff):
    r"""
    Comparison
    The function is defined as:

     f_{\text{cut}}(r) = \begin{cases} 0.5 \left[ 1 + \cos\left( \pi \frac{r}{r_{\text{cut}}} \right) \right] & \text{if } r < r_{\text{cut}}, \\ 0 & \text{otherwise}. \end{cases}

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Mollifier
    .. [2] https://math.stackexchange.com/questions/1618981/cutoff-function-vs-mollifiers
    """
    relative_distance = np.asarray(relative_distance)
    smooth_coefficient = np.zeros_like(relative_distance, dtype=float)
    mask = relative_distance < cutoff
    # smooth
    smooth_coefficient[mask] = 0.5 * (1 + np.cos(np.pi * relative_distance[mask] / cutoff))
    print(relative_distance, smooth_coefficient)
    return smooth_coefficient


def get_sigmas(atoms):
    # TODO build a dict and store the sigmas for the atoms and retrieve them when needed
    # now just return the same sigmas blindly
    return np.ones(len(atoms)) * 1


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


def theta_phi_meshgrid(num=100):
    """
    num: how many grid points per pi radian
    """
    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, num)
    phi = np.linspace(0, 2 * np.pi, 2 * num)
    # Create a 2-D meshgrid of (theta, phi) angles.
    return np.meshgrid(theta, phi)


#

def print_art_text(str="QUANTUM"):
    ascii_banner = pyfiglet.figlet_format(str)
    print(ascii_banner)


def print_banner(str):
    print(f"{'=' * 60}")

    # ASCII title
    banner = pyfiglet.figlet_format(str)


    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    python_version = platform.python_version()
    system_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    # cuda_status = "Available ✅" if torch.cuda.is_available() else "Not Available ❌"
    # device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"


    print(banner)
    print(f"{'=' * 60}")
    print(f"{Style.BRIGHT}{Fore.YELLOW}CHIRPY SIMULATION FRAMEWORK {chirpy.__version__}")
    print(f"{Fore.YELLOW}Start Time:     {now}")
    print(f"{Fore.YELLOW}Python Version: {python_version}")
    print(f"{Fore.YELLOW}System Info:    {system_info}")
    # print(f"CUDA Support:   {cuda_status}")
    # print(f"GPU Device:     {device_name}")
    print(f"{'=' * 60}")
    print(f"{Fore.BLACK}")
    print()


def loading_step(name, success=True, delay=0.89):
    print(f"{Fore.CYAN}Loading {name}...", end="")
    time.sleep(delay)
    if success:
        print(f"{Fore.GREEN} ✅")
    else:
        print(f"{Fore.RED} ❌")

    # -*- coding: utf-8 -*-





def print_Gauss(path='Gauss.jpg', max_width=70):
    def rgb_to_ansi(r, g, b, char='  '):
        return f"\x1b[48;2;{r};{g};{b}m  \x1b[0m"
    img = Image.open(path)
    img = img.convert("RGB")

    # 缩小图像宽度，保持终端友好
    w, h = img.size
    aspect_ratio = h / w
    new_w = min(w, max_width)
    new_h = int(new_w * aspect_ratio * 0.7)
    img = img.resize((new_w, new_h))

    for y in range(new_h):
        for x in range(new_w):
            r, g, b = img.getpixel((x, y))
            print(rgb_to_ansi(r, g, b), end="")
        print()