# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa


try:
    from ...ScannerGeometries.Siemens_Biograph_mMR import Biograph_mMR_Physiology
except:
    Biograph_mMR_Physiology = None

try:
    from ...ScannerGeometries.Siemens_Brain_PET import Brain_PET_Physiology
except:
    Brain_PET_Physiology = None