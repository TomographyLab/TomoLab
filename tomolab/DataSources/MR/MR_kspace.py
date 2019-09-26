# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa



from ...DataSources.FileSources.nifti import load_nifti


def import_kspace(filename):
    return load_nifti(filename)
