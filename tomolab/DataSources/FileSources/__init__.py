# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

__all__ = ['guess_file_type_by_name',
           'load_nifti',
           'load_dicom', 'load_dicom_series','load_multiple_dicom_series',
           'load_mask', 'load_freesurfer_lut_file',
           'download_Dropbox',
           'load_interfile', ]


from .dicom import load_dicom_series, load_dicom, load_multiple_dicom_series
from .nifti import load_nifti
from .freesurfer import load_mask, load_freesurfer_lut_file
from .web import download_Dropbox
from .interfile import load_interfile


def guess_file_type_by_name(filename):
    if filename.endswith("h5"):
        return "h5"
    elif filename.endswith(".l.hdr"):
        return "interfile_listmode_header"
    if filename.endswith(".l"):
        return "interfile_listmode_data"
    if filename.endswith(".h33") or filename.endswith(".s.hdr"):
        return "interfile_projection_header"
    elif filename.endswith(".a") or filename.endswith(".s"):
        return "interfile_projection_data"
    elif filename.endswith(".v.hdr"):
        return "interfile_volume_header"
    elif filename.endswith(".v"):
        return "interfile_volume_data"
    elif filename.endswith(".nii.gz"):
        return "nifti_compressed"
    elif filename.endswith(".nii"):
        return "nifti"
    elif filename.endswith(".mat"):
        return "mat"
    elif filename.endswith(".img") or filename.endswith(".ima"):
        return "dicom_volume"
    else:
        return "unknown"



