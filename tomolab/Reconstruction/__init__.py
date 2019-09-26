# -*- coding: utf-8 -*-
# occiput
# Harvard University, Martinos Center for Biomedical Imaging
# Aalto University, Department of Computer Science

__all__ = ['MR','PET','SPECT',
           'PET_Static_Scan',
           ]

from . import MR
from . import PET
from . import SPECT

from .PET import PET_Static_Scan
#from .SPECT import *
#from .MR import *
