# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

import numpy as np

class ColorLookupTable():
    def __init__(self):
        self._by_index = {}
        self._by_name = {}

    def n_entries(self):
        return len(self._by_index)

    def load_from_file_freesurfer(self, filename):
        with open(filename, 'r') as fid:
            F = fid.read()
            F = F.split('\r\n')
        self._by_index = {}
        self._by_name = {}
        for line in F:
            if not line == '':
                index, name, r, g, b, a = line.split()
                r = np.uint8(r)
                g = np.uint8(g)
                b = np.uint8(b)
                a = np.uint8(a)
                self._by_index[str(index)] = {'name': name, 'r': r, 'g': g, 'b': b, 'a': a}
                self._by_name[str(name)] = {'index': index, 'r': r, 'g': g, 'b': b, 'a': a}

    def index_to_rgba(self, index):
        entry = self._by_index[str(index)]
        return entry['r'], entry['g'], entry['b'], entry['a']

    def name_to_rgba(self, name):
        entry = self._by_name[str(name)]
        return entry['r'], entry['g'], entry['b'], entry['a']

    def index_to_name(self, index):
        entry = self._by_index[str(index)]
        return entry['name']

    def has_index(self, index):
        return str(index) in self._by_index.keys()

    def default_rgba(self):
        if self.has_index(0):
            return self.index_to_rgba(0)
        else:
            return np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0)

    def convert_ndarray_to_rgba(self, array):
        unique = np.unique(np.uint32(array))
        R = np.zeros(array.shape)
        G = np.zeros(array.shape)
        B = np.zeros(array.shape)
        A = np.zeros(array.shape)
        for index in unique:
            if self.has_index(index):
                r, g, b, a = self.index_to_rgba(index)
            else:
                r, g, b, a = self.default_rgba()
            I = np.where(array == index)
            R[I] = r
            G[I] = g
            B[I] = b
            A[I] = a
        return R, G, B, A


def load_freesurfer_lut_file(filename):
    lut = ColorLookupTable()
    lut.load_from_file_freesurfer(filename)
    return lut
