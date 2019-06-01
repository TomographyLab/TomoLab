# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisae


__all__ = ['load_dicom', 'load_dicom_series']

import os
import pydicom
import numpy as np

try:
    import dcmstack
    dcmstack_available = True
except:
    dcmstack_available = False

from glob import glob

from ...Core.Conversion import nipy2occiput
from ...Visualization.Visualization import ProgressBar

##############################################################################
# We need to override this method of dcmstack class
# in order to allow for float32 datatype for the
# volume array. High emission images in Bq/cc units
# cannot be contrainted in the limited ammisible range
# of int16. Moreover, there's no point in unsing integer
# values for emission data
#-----------------------------------------------------------------------------
def override_dcmstack_get_data(self):
    import numpy as np
    '''Get an array of the voxel values.

    Returns
    -------
    A numpy array filled with values from the DICOM data sets' pixels.

    Raises
    ------
    InvalidStackError
        The stack is incomplete or invalid.
    '''
    # Create a numpy array for storing the voxel data
    stack_shape = self.get_shape()
    stack_shape = tuple(list(stack_shape) + ((5 - len(stack_shape)) * [1]))
    vox_array = np.empty(stack_shape, np.float32)

    # Fill the array with data
    n_vols = 1
    if len(stack_shape) > 3:
        n_vols *= stack_shape[3]
    if len(stack_shape) > 4:
        n_vols *= stack_shape[4]
    files_per_vol = len(self._files_info) // n_vols
    file_shape = self._files_info[0][0].nii_img.get_shape()
    for vec_idx in range(stack_shape[4]):
        for time_idx in range(stack_shape[3]):
            if files_per_vol == 1 and file_shape[2] != 1:
                file_idx = vec_idx * (stack_shape[3]) + time_idx
                vox_array[:, :, :, time_idx, vec_idx] = \
                    self._files_info[file_idx][0].nii_img.get_data()
            else:
                for slice_idx in range(files_per_vol):
                    file_idx = (vec_idx * (stack_shape[3] * stack_shape[2]) +
                                time_idx * (stack_shape[2]) + slice_idx)
                    vox_array[:, :, slice_idx, time_idx, vec_idx] = \
                        self._files_info[file_idx][0].nii_img.get_data()[:, :, 0]
    # Trim unused time/vector dimensions
    if stack_shape[4] == 1:
        vox_array = vox_array[..., 0]
        if stack_shape[3] == 1:
            vox_array = vox_array[..., 0]

    return vox_array

if dcmstack_available:
    dcmstack.DicomStack.get_data = override_dcmstack_get_data
##############################################################################


def load_dicom(search_path, extension='IMA'):
    progress_bar = ProgressBar(title='Reading src')
    progress_bar.set_percentage(0.1)
    if (not dcmstack_available):
        progress_bar.set_percentage(100.0)
        raise Exception("Pleast install dcmstack from https://github.com/moloney/dcmstack/tags")
    else:
        search_string = search_path + '/*.' + extension
        src_paths = glob(search_string)
        stacks = dcmstack.parse_and_stack(src_paths)
        images = []
        for k, key in enumerate(stacks.keys()):
            stack = stacks[key]
            img = nipy2occiput(stack.to_nifti(embed_meta=True))
            images.append(img)
            progress_bar.set_percentage((k + 1) * 100.0 / len(stacks.keys()))
        progress_bar.set_percentage(100.0)
        return images


def load_dicom_series(path, files_start_with=None, files_end_with=None,
                      exclude_files_end_with=('.dat', '.txt', '.py', '.pyc', '.nii', '.gz')):
    """Rudimentary file to load_interfile dicom serie from a directory. """
    N = 0
    paths = []
    slices = []
    files = os.listdir(path)
    progress_bar = ProgressBar(title='Reading src')
    progress_bar.set_percentage(0.1)

    for k, file_name in enumerate(files):
        file_valid = True
        if files_start_with is not None:
            if not file_name.startswith(files_start_with):
                file_valid = False
        if files_end_with is not None:
            if not file_name.endswith(files_end_with):
                file_valid = False
        for s in exclude_files_end_with:
            if file_name.endswith(s):
                file_valid = False
        if file_valid:
            # print(file_name)
            full_path = path + os.sep + file_name
            # read moco information from files
            paths.append(full_path)
            f = pydicom.read_file(full_path)
            slice = f.pixel_array
            slices.append(slice)
            N += 1
            instance_number = f.get(0x00200013).value
            creation_time = f.get(0x00080013).value
            # print "Instance number:    ",instance_number
            # print "Creation time:      ",creation_time
        progress_bar.set_percentage((k + 1) * 100.0 / len(files))

    progress_bar.set_percentage(100.0)
    array = np.zeros((slices[0].shape[0], slices[0].shape[1], N), dtype=np.float32)
    for i in range(N):
        slice = np.float32(slices[i])  # FIXME: handle other data types
        array[:, :, i] = slice
        # return numpy2occiput(array)
    return array


