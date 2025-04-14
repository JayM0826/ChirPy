import sympy as sp
from sympy.functions.special.spherical_harmonics import Znm,Ynm
import numpy as np

theta, phi = sp.symbols('theta phi', real=True)


def density_Z_0_0(n=0, m=0):
    # Define angular coordinates

    Z_n_m = Znm(n, m, theta, phi)
    # print(f"Z_{n},{m} in spherical form: {Z_n_m}")
    # Define Cartesian coordinates
    x, y, z = sp.symbols('x y z', real=True)
    r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
    # Define substitutions
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

    Y_lm_real = Z_n_m.subs(substitutions)
    Y_lm_real = Y_lm_real.simplify()

    Y_lm_real = Y_lm_real.expand(func=True)

    # Combine all substitutions
    substitutions.update(additional_subs)
    # Substitute and simplify
    Y_lm_real = Y_lm_real.subs(substitutions)
    Y_lm_real = Y_lm_real.simplify()


    # Apply substitutions
    Y_lm = Y_lm_real.simplify()
    # print(f"Z_{n},{m} in Cartesian form: {Y_lm}")

    Y_lm = Y_lm.expand(func=True)

    Y_lm = Y_lm.subs(substitutions)
    # Y_lm = Y_lm.evalf()
    density_Z_00 = sp.lambdify((x, y, z), Y_lm, 'numpy')
    # print(f"Z_{n},{m} in Cartesian form: {Y_lm}")
    return density_Z_00
# Z_1,0 in Cartesian form: sqrt(3)*z/(2*sqrt(pi)*sqrt(x**2 + y**2 + z**2))

# f_Z_00 = density_Z_0_0(1,0)
# # Test the function
# points = [
#     (-1, 2, 0),
#     (0, 1, 2),
#     (0, 3, 1),
#     (0, 0, 1),  # Test the origin
# ]
# for x_val, y_val, z_val in points:
#     value = f_Z_00(x_val, y_val, z_val)
#     print(f"Z_00({x_val}, {y_val}, {z_val}) = {value}")
