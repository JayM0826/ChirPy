#!/usr/bin/env python
#------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
#------------------------------------------------------
 

import argparse
import numpy as np

from chirpy.classes import system


def main():
    '''Unit Cell parametres are taken from fn1 if needed'''
    parser = argparse.ArgumentParser(
            description="Convert any supported input into XYZ format",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("fn",
                        help="file (xyz.pdb,xvibs,...)"
                        )
    parser.add_argument("--center_coords",
                        action='store_true',
                        help="Center Coordinates in cell centre or at origin \
                                (cell_aa parametre overrides default origin)",
                        default=False
                        )
    parser.add_argument("--sort",
                        action='store_true',
                        help="Alphabetically sort entries",
                        default=False
                        )
    parser.add_argument("--cell_aa_deg",
                        nargs=6,
                        help="Orthorhombic cell parametres a b c al be ga in \
                                angstrom/degree",
                        default=[0.0, 0.0, 0.0, 90., 90., 90.]
                        )
    parser.add_argument("-f",
                        help="Output file name",
                        default='out.xyz'
                        )
    args = parser.parse_args()
    args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)

    system.Molecule(**vars(args)).XYZData.write(args.f, fmt='xyz')
#    system.Molecule(**vars(args)).sort_atoms.XYZData.write(args.f, fmt='xyz')


if __name__ == "__main__":
    main()
