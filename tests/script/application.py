import time

import chirpy as cp
import numpy as np





import utils as utils
import configuration
import utils_ext
from chirpy.classes.volume import ScalarField

class Application:

    default_config = configuration.Configration()
    def __init__(self, filename, config=default_config):
        self.config = config
        self.filename = filename
        self.filepath = utils_ext.get_relative_path(self.config.BASE_PATH, filename).as_posix()


    def run(self):
        system = cp.trajectory.XYZ(self.filepath).expand()
        print(70 * "*")
        print(70 * "*")
        print(70 * "*")
        print(70 * "*")

        system.print_info()
        print(system.data.shape)
        atom_positions = system.data[-1][:, 0:3]
        # density = utils.compute_whole_grid_density(atom_positions)
        frame_1 = system.data[0][:, 0:3]

        sigmas = utils_ext.get_sigmas(frame_1)
        self.config.SIGMAS = sigmas
        distribution, R_x, R_y, R_z = utils.compute_whole_grid_distribution(system.data, sigmas, self.config)
        print(f"x step size:{R_x[1] - R_x[0]}, len of R_x={len(R_x)}, and result = {len(R_x) * (R_x[1] - R_x[0])}")
        print(f"y step size:{R_y[1] - R_y[0]}, len of R_y={len(R_y)}, and result = {len(R_y) * (R_y[1] - R_y[0])}")
        print(f"z step size:{R_z[1] - R_z[0]}, len of R_z={len(R_z)}, and result = {len(R_z) * (R_z[1] - R_z[0])}")
        # distribution /= len(system.data) # ???

        # print(f"distribution is {np.sum(distribution)}")
        # utils.plot_iosfurface(xv, yv, zv, values)
        print(70 * "*")

        scalar_field = ScalarField(data=distribution,
                                   origin_aa=system.pos_aa[0, 0] + np.array([R_x[0], R_y[0], R_z[0]]),
                                   pos_aa=system.pos_aa[0],
                                   numbers=cp.constants.symbols_to_numbers(system.symbols),
                                   cell_vec_aa=np.array(
                                       [[R_x[1] - R_x[0], 0, 0], [0, R_y[1] - R_y[0], 0], [0, 0, R_z[1] - R_z[0]]]))
        # scalar_field = ScalarField(data=distribution, origin_aa=system.pos_aa[0,0], pos_aa=system.pos_aa[0], numbers=cp.constants.symbols_to_numbers(system.symbols),
        # cell_vec_aa=np.array([[1./2, 0, 0],[0, 1./2, 0],[0, 0, 1./2]]), grid_x=R_x, grid_y=R_y, grid_z=R_z)
        scalar_field.print_info()
        scalar_field.write(f"distribution_{time.time_ns()}.cube")

        print(scalar_field.integral(volume_unit='aa**3'))
        print(scalar_field.voxel)
        print(scalar_field.integral() / scalar_field.voxel)

    def exit(self):
        for attr in list(self.__dict__):
            delattr(self, attr)
        del self


if __name__ == '__main__':
    utils_ext.print_Gauss()
    utils_ext.print_banner("Gauß  START")
    app = Application("tartrate.xyz")
    # app.run()
    app.exit()
    del app
    utils_ext.print_art_text("Gauß  REST")

    