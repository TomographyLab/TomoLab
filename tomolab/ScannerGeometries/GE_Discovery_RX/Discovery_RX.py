# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

from ...Reconstruction.PET.PET_meshing import Michelogram
import numpy as np

class Discovery_RX(object):
    def __init__(self):
        self.model = "Discovery_RX"
        self.manufacturer = "GE Healthcare"
        self.version = "n.d."
        self.supports_listmode = False
        self.uses_meshing = True

        self.michelogram = Michelogram(
            n_rings=24, span=3, max_ring_difference=1)

        self.N_u = 367
        self.N_v = self.michelogram.segments_sizes.max()  # 47 ???
        self.size_u = 1.90735 * self.N_u
        self.size_v = 3.34 * 47
        self.N_azimuthal = self.michelogram.n_segments  # 11
        self.N_axial = 315
        self.angles_azimuthal = np.float32([0.0])
            #[-0.482, -0.373, -0.259, -0.180, -0.105, 0.0, 0.105, 0.180,
        # 0.259, 0.373, 0.482])
        self.angles_axial = np.float32(
            np.linspace(
                0,
                np.pi -
                np.pi /
                self.N_axial,
                self.N_axial))

        self.scale_activity = 8.58e-05 #TODO: check this

        self.activity_N_samples_projection_DEFAULT = 300
        self.activity_N_samples_backprojection_DEFAULT = 300
        self.activity_sample_step_projection_DEFAULT = 2.0
        self.activity_sample_step_backprojection_DEFAULT = 2.0
        #self.activity_shape_DEFAULT = [367,367,47]
        #self.activity_size_DEFAULT = float32(
        #    [1.90735, 1.90735, 3.34]) * float32(self.activity_shape_DEFAULT)
        self.activity_shape_DEFAULT = [128,128,47]
        self.activity_size_DEFAULT = np.float32(
            [5.46875, 5.46875, 3.34]) * np.float32(self.activity_shape_DEFAULT)

        self.attenuation_N_samples_projection_DEFAULT = 300
        self.attenuation_N_samples_backprojection_DEFAULT = 300
        self.attenuation_sample_step_projection_DEFAULT = 2.0
        self.attenuation_sample_step_backprojection_DEFAULT = 2.0
        #self.attenuation_shape_DEFAULT = [344, 344, 127]
        #self.attenuation_size_DEFAULT = float32(
        #        [1.90735,1.90735,3.34]) * float32(
        # self.attenuation_shape_DEFAULT)
        self.attenuation_shape_DEFAULT = [128,128,47]
        self.attenuation_size_DEFAULT = np.float32(
                [5.46875,5.46875,3.34]) * np.float32(self.activity_shape_DEFAULT)