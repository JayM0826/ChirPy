#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.9.0
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2020 Sascha Jähnigen
#
#
# ------------------------------------------------------


import argparse
import numpy as np
from chirpy.classes import system
from chirpy.topology import mapping


def main():
    '''Unit Cell parametres are taken from fn if needed'''
    parser = argparse.ArgumentParser(
            description="Find Methyl Group (like) subgroups in molecule",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument(
            "fn",
            help="file (xy,pdb,xvibs,...)"
            )
    parser.add_argument(
            "--hetatm",
            action='store_true',
            help="Include heteroatoms into search \
            (e.g. finding NH3+ groups; default: False)",
            default=False
            )
    parser.add_argument(
            "--cell_aa_deg",
            nargs=6,
            help="Use custom cell parametres a b c al be ga in \
                    angstrom/degree",
            default=None,
            )
    args = parser.parse_args()

    if args.cell_aa_deg is None:
        del args.cell_aa_deg
    else:
        args.cell_aa_deg = np.array(args.cell_aa_deg).astype(float)

    _load = system.Supercell(args.fn).XYZ
    mapping.find_methyl_groups(_load.data, _load.symbols, **vars(args))


if(__name__ == "__main__"):
    main()
