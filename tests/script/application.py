import time

import chirpy as cp
import numpy as np





import numerical_utils as num_utils
import analytical_utils as ana_utils

import configuration
import utils_ext
from chirpy.classes.volume import ScalarField

from tests.script import plot_util


class Application:

    default_config = configuration.Configuration()
    def __init__(self, filename, config=default_config):
        self.config = config
        self.filename = filename
        self.filepath = utils_ext.get_relative_path(self.config.BASE_PATH, filename).as_posix()


    def run_numerically(self):
        system = cp.trajectory.XYZ(self.filepath).expand()
        print(70 * "*")
        print(70 * "*")
        print(70 * "*")
        print(70 * "*")

        system.print_info()
        print(system.data.shape)
        frame_1 = system.data[0][:, 0:3]

        sigmas = utils_ext.get_sigmas(frame_1)
        self.config.SIGMAS = sigmas
        distribution, R_x, R_y, R_z, coefficients, density_fun = num_utils.compute_whole_grid_distribution(system.data, sigmas, self.config)
        print(f"x step size:{R_x[1] - R_x[0]}, len of R_x={len(R_x)}, and result = {len(R_x) * (R_x[1] - R_x[0])}")
        print(f"y step size:{R_y[1] - R_y[0]}, len of R_y={len(R_y)}, and result = {len(R_y) * (R_y[1] - R_y[0])}")
        print(f"z step size:{R_z[1] - R_z[0]}, len of R_z={len(R_z)}, and result = {len(R_z) * (R_z[1] - R_z[0])}")
        # distribution /= len(system.data) # ???

        # print(f"distribution is {np.sum(distribution)}")
        # utils.plot_iosfurface(xv, yv, zv, values)
        print(70 * "*")

        # scalar_field = ScalarField(data=distribution,
        #                            origin_aa=system.pos_aa[0, 0] + np.array([R_x[0], R_y[0], R_z[0]]),
        #                            pos_aa=system.pos_aa[0],
        #                            numbers=cp.constants.symbols_to_numbers(system.symbols),
        #                            cell_vec_aa=np.array(
        #                                [[R_x[1] - R_x[0], 0, 0], [0, R_y[1] - R_y[0], 0], [0, 0, R_z[1] - R_z[0]]]))
        # # scalar_field = ScalarField(data=distribution, origin_aa=system.pos_aa[0,0], pos_aa=system.pos_aa[0], numbers=cp.constants.symbols_to_numbers(system.symbols),
        # # cell_vec_aa=np.array([[1./2, 0, 0],[0, 1./2, 0],[0, 0, 1./2]]), grid_x=R_x, grid_y=R_y, grid_z=R_z)
        # scalar_field.print_info()
        # scalar_field.write(f"distribution_{time.time_ns()}.cube")
        #
        # print(scalar_field.integral(volume_unit='aa**3'))
        # print(scalar_field.voxel)
        # print(scalar_field.integral() / scalar_field.voxel)
        return coefficients, density_fun

    def exit(self):
        for attr in list(self.__dict__):
            delattr(self, attr)
        del self

    def run_analytically(self):
        system = cp.trajectory.XYZ(self.filepath).expand()
        print(70 * "*")
        print(70 * "*")
        print(70 * "*")
        print(70 * "*")

        system.print_info()
        print(system.data.shape)

        frame_1 = system.data[0][:, 0:3]
        sigmas = utils_ext.get_sigmas(frame_1) * 0.25
        self.config.SIGMAS = sigmas
        coefficients = ana_utils.compute_coefficients_in_dict(system.data, self.config)
        # utils_ext.print_format_nlm_coefficients(coefficients, self.config.N_MAX, self.config.L_MAX)
        utils_ext.plot_coefficients(coefficients)

        power_spectrum = utils_ext.calculate_aggregated_power_spectrum(coefficients)
        plot_util.plot_power_spectrum(power_spectrum, 6)

        return coefficients

    def run_analytically_test(self):
        system = cp.trajectory.XYZ(self.filepath).expand()
        print(70 * "*")
        print(70 * "*")
        print(70 * "*")
        print(70 * "*")

        system.print_info()
        print(system.data.shape)

        # add the zero - line(e.g., plt.hlines(0, 0, len(frames), ls='--', colors='k', lw = 1))
        pseudoscalar_keys = [
                  (2, 3, 4),
                  (3, 4, 6),
                  (2, 5, 6)
        ]
        frame_i = 0
        from collections import defaultdict
        bispectra_by_key = defaultdict(list)
        for frame in system.data[:, 0:3]:
            frame_i += 1
            # sigmas = utils_ext.get_sigmas(frame) * 0.25
            sigmas = np.array([1.4, 1.0, 1.0,1.4,1.0,1.0]) * 0.15
            self.config.SIGMAS = sigmas
            frame_within_cutoff = utils_ext.filter_atoms_within_cutoff(
                frame,
                self.config.ORIGIN_ATOM_INDEX,
                self.config.CUT_OFF
            )
            coefficients = ana_utils.compute_coefficients_in_dict([frame_within_cutoff], self.config)
            # utils_ext.print_format_nlm_coefficients(coefficients, self.config.N_MAX, self.config.L_MAX)
            # plot_util.plot_coefficients(coefficients)

            # bispectrum = utils_ext.calculate_aggregated_bispectrum(coefficients)
            for pseudoscalar_key in pseudoscalar_keys:
                bispectrum_for_key = utils_ext.calculate_bispectrum(coefficients, *pseudoscalar_key)
                print(f"bispectrum = {bispectrum_for_key} when pseudoscalar_key={pseudoscalar_key} in frame {frame_i}")
                bispectra_by_key[pseudoscalar_key].append(bispectrum_for_key)
            # plot_util.plot_power_spectrum(power_spectrum, 6)
            print("\n\n\n")


        plot_util.plot_bispectra_path(pseudoscalar_keys, bispectra_by_key, frame_i, "Bispectrum Evolution across achiral path")
        return coefficients

if __name__ == '__main__':
    utils_ext.print_Gauss()
    utils_ext.print_banner("Gauß  START")
    # app = Application("achiral_path.xyz")

    app = Application("achiral_path.xyz")
    # coeff_numerical = app.run_numerically()



    coeff_analytical = app.run_analytically_test()


    app.exit()
    del app



    