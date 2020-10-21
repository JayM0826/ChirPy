#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy
#
#    A buoyant python package for analysing supramolecular
#    and electronic structure, chirality and dynamics.
#
#
#  Developers:
#    2010-2016  Arne Scherrer
#    since 2014 Sascha Jähnigen
#
#  https://hartree.chimie.ens.fr/sjaehnigen/chirpy.git
#
# ------------------------------------------------------


import numpy as np
from itertools import islice
from .generators import _reader, _open


def _cube(frame, **kwargs):
    '''Kernel for processing cube frame.'''
    comments = (next(frame).strip(), next(frame).strip())

    _cellinfo = list(zip(*[_l.strip().split() for _l in islice(frame, 4)]))
    cell_vec_au = np.zeros((3, 3))
    origin_au = np.zeros((3,))
    n_atoms, n_x, n_y, n_z = map(int, _cellinfo[0])
    origin_au[0], *cell_vec_au[:, 0] = map(float, _cellinfo[1])
    origin_au[1], *cell_vec_au[:, 1] = map(float, _cellinfo[2])
    origin_au[2], *cell_vec_au[:, 2] = map(float, _cellinfo[3])

    _atominfo = list(zip(*[_l.strip().split()
                           for _l in islice(frame, abs(n_atoms))]))
    pos_au = np.zeros((abs(n_atoms), 3))
    numbers = tuple(map(int, _atominfo[0]))
    # dummy = map(int, _atominfo[1])
    pos_au[:, 0] = list(map(float, _atominfo[2]))
    pos_au[:, 1] = list(map(float, _atominfo[3]))
    pos_au[:, 2] = list(map(float, _atominfo[4]))

    _nlines = 6 + abs(n_atoms) + (int(n_z / 6) + 1) * n_y * n_x
    if n_atoms < 0:
        next(frame)
        _nlines += 1
        n_atoms *= -1

    if kwargs.get('n_lines') != _nlines:
        raise ValueError('Inconsistent CUBE file!')

    data = []
    for _l in frame:
        data.extend(_l.strip().split())
    try:
        data = np.array(data).reshape(n_x, n_y, n_z).astype(float)
    except ValueError:
        raise ValueError('Tried to read broken or incomplete file!')

    return data, origin_au, cell_vec_au, pos_au, numbers, comments


def cubeIterator(FN, **kwargs):
    '''Iterator for xyzReader
       Usage: next() returns data, symbols, comments of
       current frame'''
    _kernel = _cube

    with _open(FN, 'r', **kwargs) as _f:
        _f.readline()
        _f.readline()
        _natoms = int(_f.readline().strip().split()[0])
        _nx = int(_f.readline().strip().split()[0])
        _ny = int(_f.readline().strip().split()[0])
        _nz = int(_f.readline().strip().split()[0])
        _nlines = 6 + abs(_natoms) + (int(_nz / 6) + 1) * _ny * _nx

        if _natoms < 0:
            _nlines += 1

    return _reader(FN, _nlines, _kernel, **kwargs)


def cubeReader(FN, **kwargs):
    '''Read complete XYZ file at once'''
    data, origin_au, cell_vec_au, pos_au, numbers, comments = \
        zip(*cubeIterator(FN, **kwargs))

    return np.array(data), origin_au[0], cell_vec_au[0],\
        np.array(pos_au), numbers[0], list(comments)
