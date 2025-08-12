# built-in module
import datetime
import functools
import math as math
import platform
import time
import warnings

import chirpy
import numpy as np
import plotly.graph_objects as go
import pyfiglet
from PIL import Image
# import torch
from colorama import Fore, Style
from scipy.special import sph_harm_y
from sympy import S
from sympy.physics.quantum.cg import CG


def print_format_nlm_coefficients(coefficients, n_max, l_max):
    for n in range(n_max):
        print(f"{100 * '*'}n={n}: COEFFICIENTS{140 * '*'}")
        index_start = n * (l_max ** 2)
        index_end = (n + 1) * (l_max ** 2)
        print_format_lm_coefficients(coefficients[index_start:index_end], l_max)


def print_format_lm_coefficients(coefficients, l_max):
    print(300 * "*")
    print(f"{100 * '*'}COEFFICIENTS{140 * '*'}")
    print(300 * "*")

    # Format the array to 10 decimal places
    coefficients = np.around(coefficients, decimals=10)

    # Print the array with indentation to form a triangle
    print("[")
    start_idx = 0
    max_width = (l_max - 1) * 2 + 1  # Maximum width (e.g., l=6, m=-6 to 6)
    for l in range(l_max):
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
    r:=norm(x, y, z), default 1(unit sphere)
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


def generate_grid_and_bounds(atom_positions, sigmas, number_per_unit_distance,
                             cutoff, origin_index):
    """
       number_per_unit_distance: the grid number per unit distance
    """
    # xyz_bounds = [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper]
    xyz_bounds = coupute_XYZ_bounds(
        atom_positions,
        sigmas,
        cutoff,
        origin_index
    )
    x_linspace = np.linspace(
        xyz_bounds[0],
        xyz_bounds[1],
        int((xyz_bounds[1] - xyz_bounds[0]) * number_per_unit_distance)
    )
    y_linspace = np.linspace(
        xyz_bounds[2],
        xyz_bounds[3],
        int((xyz_bounds[3] - xyz_bounds[2]) * number_per_unit_distance)
    )

    z_linspace = np.linspace(
        xyz_bounds[4],
        xyz_bounds[5],
        int((xyz_bounds[5] - xyz_bounds[4]) * number_per_unit_distance)
    )

    x_meshgrid, y_meshgrid, z_meshgrid = np.meshgrid(
        x_linspace,
        y_linspace,
        z_linspace,
        indexing='ij'
    )
    return (x_meshgrid, y_meshgrid, z_meshgrid, xyz_bounds, x_linspace,
            y_linspace, z_linspace)


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
        return (x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
#        return tuple(int(math.ceil(x)) for x in (
#                                                 x_lower,
#                                                 x_upper,
#                                                 y_lower,
#                                                 y_upper,
#                                                 z_lower,
#                                                 z_upper
#                                                 ))
#
    # otherwise
    atom_3D_relative_positions = atom_3D_positions - atom_3D_positions[
            origin_atom_index
            ]

    max_values = np.max((atom_3D_relative_positions), axis=0)
    min_values = np.min((atom_3D_relative_positions), axis=0)
    # here we use 6(without scientific proof) sigma so that the accuracy
    # is highe
    x_upper, y_upper, z_upper = max_values + np.max(sigmas) * 6
    x_lower, y_lower, z_lower = min_values - np.max(sigmas) * 6
    # easy to use linspace with int

    return tuple(int(math.ceil(x)) for x in (
                                             x_lower,
                                             x_upper,
                                             y_lower,
                                             y_upper,
                                             z_lower,
                                             z_upper
                                             ))


def filter_atoms_within_cutoff(positions, origin_atom_index, cutoff):
    """
    Filter atoms within a cutoff distance from a central atom.

    Parameters:
    - positions: Nx3 array of (x, y, z) coordinates.
    - center_atom_idx: Index of the central atom.
    - cutoff: Cutoff radius (e.g., in Å).

    Returns:
    - return a fully new filtered array of qualified atoms
      (including origin atom).
    """
    # origin_pos = positions[origin_atom_index]
    # relative_distance = np.sqrt(np.sum((positions-origin_pos) ** 2, axis=1))
    relative_positions = np.delete(
        positions - positions[origin_atom_index],
        origin_atom_index,
        axis=0
    )

    # qualified_indices = np.where(relative_distance <= (cutoff))[0]
    # qualified_atom_positions = positions[qualified_indices]

    # SJ CHANGED CODE: return relative distances directly so we do not have to
    # recompute them. smoothing routine needs to be updated
    _selection = np.linalg.norm(relative_positions, axis=1) <= cutoff
    qualified_atom_positions = relative_positions[_selection]

    return qualified_atom_positions


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
    print(f"{Fore.WHITE}")
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


def l_m_pairs(l_max):
    """
    generate (l, m) list of spherical harmonics
    """
    return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]


def n_l_m_pairs(n_max):
    """
    Note that here the relation between n and (l,m) is different from quantum mechanics.
    The quantum num(n,l,m) for  quantum mechanics have physical meaning but here is just as index of basis.
    Maybe this method should be deleted!!!
    generate (n, l, m) list of spherical harmonics
    """
    return [(n, l, m) for n in range(1, n_max + 1) for l in range(n) for m in range(-l, l + 1)]


def compute_cos_sin_angle_multiples(cos_phi, sin_phi, max_angular):
    """
    useful for computing harmonics, especially in associated Legendre polynomial
    """
    cos_sin_m_phi = np.zeros((max_angular + 1, 2))  # Each row: [cos(mφ), sin(mφ)]

    for m in range(max_angular + 1):
        if m == 0:
            cos_sin_m_phi[m] = [1.0, 0.0]
        elif m == 1:
            cos_sin_m_phi[m] = [-cos_phi, -sin_phi]
        else:
            cos_sin_m_phi[m] = (
                    -2.0 * cos_phi * cos_sin_m_phi[m - 1] - cos_sin_m_phi[m - 2]
            )

    return cos_sin_m_phi


def deprecated(reason):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(f"{func.__name__} is deprecated: {reason}",
                          category=DeprecationWarning,
                          stacklevel=2)
            return func(*args, **kwargs)

        return wrapped

    return decorator





def clebsch_gordan(l1, l2, m1, m2, l, m):
    result = CG(S(l1), S(m1), S(l2), S(m2), S(l), S(m)).doit()
    return float(result) if result else 0.0



def calculate_bispectrum(c_nlm: dict, l1: int, l2: int, l3: int,
                         sum_over_n=True):
    """proof of principle, needs more efficient code"""

    if not sum_over_n:
        # --- Todo: implement n-range selection
        raise NotImplementedError("Please set sum_over_n=True")

    n_values = sorted(set(_n for _n, _l, _m in c_nlm))

    bl1l2l3 = 0.0
    for _n in n_values:
        for _m1 in range(-l1, l1 + 1):
            for _m2 in range(-l2, l2 + 1):
                M = _m1 + _m2
                _m3 = -M
                if not (-l3 <= _m3 <= l3):
                    continue
                _CG = CG(
                    S(l1), S(_m1),
                    S(l2), S(_m2),
                    S(l3), S(M)
                ).doit()
                _N = CG(
                    S(l3), S(M),
                    S(l3), S(_m3),
                    S(0), S(0)
                ).doit()
                if _CG != 0:
                    bl1l2l3 += complex(_N * _CG) * c_nlm[(_n, l1, _m1)] * c_nlm[(_n, l2, _m2)] * c_nlm[(_n, l3, _m3)]

    if (abs(bl1l2l3.imag) > 1e-2):
        print(l1, l2, l3)
    return bl1l2l3


def calculate_aggregated_bispectrum(coefficients, max_plot_l=5, L_MAX=20):
    """
    Calculate the aggregated bispectrum summing over n for each l, l1, l2 combination.

    Args:
        coefficients: Dictionary with (n, l, m) as key and coefficient value
        max_plot_l: Maximum l value to consider for bispectrum (default=5)
        L_MAX: Maximum l value in coefficients (default=20)

    Returns:
        bispectrum: Dictionary with (l, l1, l2) as key and aggregated bispectrum value over n
    """

    # Compute bispectrum for each n

    bispectrum_aggregated = {}
    for l in range(max_plot_l + 1):
        for l1 in range(max_plot_l + 1):
            for l2 in range(max_plot_l + 1):
                bispectrum_aggregated[(l, l1, l2)] = calculate_bispectrum(coefficients, l1, l2, l)

    return bispectrum_aggregated


def calculate_similarity_metric(power_chi, power_chi_prime, max_l=20):
    """
    Calculate the similarity metric d(chi, chi') = sqrt(2 - 2 * p(chi) * p(chi')).

    Args:
        power_chi: Dictionary with l as key and power spectrum value for chi
        power_chi_prime: Dictionary with l as key and power spectrum value for chi'
        max_l: Maximum l value to consider (default=5)

    Returns:
        similarity: The computed similarity metric
    """
    # Extract power spectrum values for l from 0 to max_l
    l_values = list(range(max_l + 1))
    p_chi = np.array([power_chi.get(l, 0) for l in l_values])
    p_chi_prime = np.array([power_chi_prime.get(l, 0) for l in l_values])

    # Normalize to unit length
    norm_chi = np.sqrt(np.sum(p_chi ** 2))
    norm_chi_prime = np.sqrt(np.sum(p_chi_prime ** 2))
    if norm_chi > 0:
        p_chi = p_chi / norm_chi
    if norm_chi_prime > 0:
        p_chi_prime = p_chi_prime / norm_chi_prime

    # Compute dot product
    dot_product = np.dot(p_chi, p_chi_prime)

    # Calculate similarity metric
    similarity = np.sqrt(2 - 2 * dot_product)
    return similarity


def calculate_aggregated_power_spectrum(coefficients, max_l=20):
    # Calculate aggregated power spectrum summing over m and n for each l
    power_by_l = {}
    for (n, l, m), coeff in coefficients.items():
        if l <= max_l:
            if l not in power_by_l:
                power_by_l[l] = 0
            power_by_l[l] += np.real(np.conjugate(coeff) * coeff)  # Sum over m and n
    return power_by_l




def Y_lm_real_scipy(l, m, theta, phi):
    """
    In SciPy sph_harm, the order is :
        m,
        l,
        phi : array_like
           Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
        theta : array_like
           Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.


    -------------------below is important---------------------------
    the order and definition of parameters are the same as sympy
    # In SciPy's sph_harm_y(here used)
        the order is :
        l,
        m,
        theta : ArrayLike[float]
            Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
        phi : ArrayLike[float]
            Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    Spherical harmonics. They are defined as

    .. math::

        Y_n^m(\theta,\phi) = \sqrt{\frac{2 n + 1}{4 \pi} \frac{(n - m)!}{(n + m)!}}
            P_n^m(\cos(\theta)) e^{i m \phi}

    where :math:`P_n^m` are the (unnormalized) associated Legendre polynomials.

    Note that SciPy's spherical harmonics include the Condon-Shortley
    phase [2]_ because it is part of `sph_legendre_p`.




    f you need to derive formulas (for example, manually expanding a spherical wave), use sympy's Znm.
    If you need to perform numerical computations (such as plotting molecular orbitals or doing acoustic simulations),
    use the scipy version (the one that includes the (−1^m) factor!).
    """
    if m == 0:
        return sph_harm_y(l, 0, theta, phi).real
    elif m > 0:
        return np.sqrt(2) * (-1) ** m * sph_harm_y(l, m, theta, phi).real  # Even
    else:
        return np.sqrt(2) * np.power(-1., m) * np.imag(sph_harm_y(l, -m, theta, phi))  # Odd
    # Y = sph_harm_y(l, abs(m), theta, phi)
    #
    # # Linear combination of Y_l,m and Y_l,-m to create the real form.
    # if m < 0:
    #     Y = np.sqrt(2) * (-1.)**m * Y.imag
    # elif m > 0:
    #     Y = np.sqrt(2) * (-1.)**m * Y.real
    # else:
    #     Y = Y.real
    # return Y




def numerical_density_at_nlm(c_nlm, dvr_basis_fun, config, r_nodes, x_vals, theta_vals, phi_vals, n, l, m):
    R_n = dvr_basis_fun(n, x_vals, config.DVR_BASIS_NUM)
    R_n /= r_nodes[n] * np.sqrt(config.CUT_OFF / 2.)
    Y_lm = sph_harm_y(l, m, theta_vals, phi_vals)
    return c_nlm * R_n * Y_lm
