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


system = cp.trajectory.XYZ("tartrate.xyz").expand()
system.print_info()
atom_positions = system.data[-1][:, 0:3]
# density = utils.compute_whole_grid_density(atom_positions)
distribution = utils.compute_whole_grid_distribution(system.data, )
# utils.plot_iosfurface(xv, yv, zv, values)
print(70 * "*")

scalar_field = ScalarField(data=distribution, cell_vec_aa=np.array([[1,0,0],[0,1,0],[0,0,1]]))
scalar_field.write("distribution.cube")
scalar_field.print_info()




