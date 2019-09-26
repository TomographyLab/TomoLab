# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

import numpy as np

'''
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
'''


class Discovery_RX():
    def __init__(self):
        self.model = "Discovery_RX"
        self.manufacturer = "GE Healthcare"
        self.version = "2019"
        self.supports_listmode = False
        self.uses_meshing = True

        self.michelogram = Michelogram(n_rings=24, span=3, max_ring_difference=1)

        self.resolution = np.float32((3.15, 3.15, 3.27))
        self.pixel_size = self.resolution[0]
        self.slice_thickness = self.resolution[2]

        self.N_u = 331
        self.N_v = self.michelogram.segments_sizes.max()  # 127
        self.size_u = self.pixel_size * self.N_u
        self.size_v = self.slice_thickness * self.N_v
        self.N_azimuthal = self.michelogram.n_segments  # 11
        self.N_axial = 315
        self.angles_azimuthal = np.float32([-0.15872557, -0.14429598, -0.12986638, -0.11543678, -0.10100719,
                                            -0.08657759, -0.07214799, -0.05771839, -0.04328879, -0.0288592 ,
                                            -0.0144296 ,  0.        ,  0.0144296 ,  0.0288592 ,  0.04328879,
                                             0.05771839,  0.07214799,  0.08657759,  0.10100719,  0.11543678,
                                             0.12986638,  0.14429598,  0.15872557])
        self.angles_axial = (-4.625834 + np.arange(self.N_axial)*0.5714286)*(np.pi/180.0)
        self.scale_activity = 8.58e-05

        self.activity_N_samples_projection_DEFAULT = 300
        self.activity_N_samples_backprojection_DEFAULT = 300
        self.activity_sample_step_projection_DEFAULT = 2.0
        self.activity_sample_step_backprojection_DEFAULT = 2.0
        self.activity_shape_DEFAULT = np.float32([128, 128, 47])
        self.activity_size_DEFAULT = self.resolution * self.activity_shape_DEFAULT

        self.attenuation_N_samples_projection_DEFAULT = 300
        self.attenuation_N_samples_backprojection_DEFAULT = 300
        self.attenuation_sample_step_projection_DEFAULT = 2.0
        self.attenuation_sample_step_backprojection_DEFAULT = 2.0
        self.attenuation_shape_DEFAULT = np.float32([128, 128, 47])
        self.attenuation_size_DEFAULT = self.resolution * self.attenuation_shape_DEFAULT

        self.listmode = None
        self.supports_listmode = False

        self.physiology = None
        self.supports_physiology = False