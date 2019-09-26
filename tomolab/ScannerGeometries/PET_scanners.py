# -*- coding: utf-8 -*-
# occiput
# Harvard University, Martinos Center for Biomedical Imaging
# Aalto University, Department of Computer Science


# Here is the library of PET scanners! Access to listmode data is provided by an external Package.
# Don't be put off, the mechanism is quite simple.

__all__ = ["Generic", "Brain_PET", "Biograph_mMR",
           "Discovery_RX", "get_scanner_by_name"]


from ..Reconstruction.PET.PET_meshing import Michelogram
import numpy as np


# Import scanner definitions
from ..ScannerGeometries.Siemens_Biograph_mMR.Biograph_mMR import Biograph_mMR
try:
    from ..ScannerGeometries.Siemens_Brain_PET import Brain_PET
except:
    Brain_PET = None
try:
    from ..ScannerGeometries.GE_Discovery_RX.Discovery_RX import Discovery_RX
except:
    Discovery_RX = None


class Generic(object):
    def __init__(self):
        self.model = "Generic PET Scanner"
        self.manufacturer = "Occiput's immagination"
        self.version = "1.0"
        self.supports_listmode = False
        self.uses_meshing = False

        self.N_u = 128
        self.N_v = 128
        self.size_u = 2.0 * 128
        self.size_v = 2.0 * 128
        self.N_azimuthal = 5
        self.N_axial = 120
        self.angles_azimuthal = np.float32([-0.5, -0.25, 0.0, 0.25, 0.5])
        self.angles_axial = np.float32(np.linspace(0, np.pi - np.pi / self.N_axial, self.N_axial))

        self.scale_activity = 1.0

        self.activity_N_samples_projection_DEFAULT = 150
        self.activity_N_samples_backprojection_DEFAULT = 150
        self.activity_sample_step_projection_DEFAULT = 2.0
        self.activity_sample_step_backprojection_DEFAULT = 2.0
        self.activity_shape_DEFAULT = [128, 128, 128]
        self.activity_size_DEFAULT = np.float32([2.0, 2.0, 2.0]) * np.float32(
            self.activity_shape_DEFAULT
        )

        self.attenuation_N_samples_projection_DEFAULT = 150
        self.attenuation_N_samples_backprojection_DEFAULT = 150
        self.attenuation_sample_step_projection_DEFAULT = 2.0
        self.attenuation_sample_step_backprojection_DEFAULT = 2.0
        self.attenuation_shape_DEFAULT = [128, 128, 128]
        self.attenuation_size_DEFAULT = np.float32([2.0, 2.0, 2.0]) * np.float32(
            self.attenuation_shape_DEFAULT
        )

        self.listmode = None
        self.physiology = None



class Brain_PET(Generic):
    """
    Not implemented, imported as generic scanner
    """


def get_scanner_by_name(name):
    if name == "Generic":
        scanner = Generic()
    elif name == "Brain_PET" or name == "BrainPET":
        scanner =  Brain_PET()
    elif name == "Siemens_Biograph_mMR" or \
         name == "Siemens_Biograph_mMR" or \
         name == "mMR" or \
         name == "Siemens_mMR" or \
         name == "Biograph_mMR":
        scanner =  Biograph_mMR()
    elif name == "Discovery_RX" or \
         name == "GE_Discovery_RX" or \
         name == "GE_RX" or \
         name == "RX" or \
         name == "DRX":
        scanner =  Discovery_RX()
    else:
        scanner =  None
    return scanner