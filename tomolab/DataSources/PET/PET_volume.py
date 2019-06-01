# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

# Import an interfile volume as an Image3D and export.


from ...Transformation.Transformations import Transform_Scale
from ...DataSources.FileSources.interfile import load_interfile

import os
import numpy as np

__all__ = ["import_interfile_volume", "export_interfile_volume"]


def import_interfile_volume_data(
    headerfile="", datafile=""
):  # FIXME: this should be in the Interfile package
    F = load_interfile(headerfile)
    if "matrix size[1]" in F:
        Nx = F["matrix size[1]"]["value"]
        Ny = F["matrix size[2]"]["value"]
        Nz = F["matrix size[3]"]["value"]
    else:
        Nx = F["matrix size [1]"]["value"]
        Ny = F["matrix size [2]"]["value"]
        Nz = F["matrix size [3]"]["value"]
    if datafile == "":
        datafile1 = headerfile.replace(
            headerfile.split(os.sep)[-1], F["name of data file"]["value"]
        )
        datafile2 = headerfile.replace(".v.hdr", ".v")
        datafile2 = datafile2.replace(".h33", ".v")
        datafile3 = headerfile.replace(".h33", ".v")
        try:
            data = np.fromfile(datafile1, dtype=np.float32)
        except:
            try:
                data = np.fromfile(datafile2, dtype=np.float32)
            except:
                try:
                    data = np.fromfile(datafile3, dtype=np.float32)
                except:
                    print("Data file not found.")
    else:
        data = np.fromfile(datafile, dtype=np.float32)
    data = data.reshape([Nz, Ny, Nx])
    data = np.asfortranarray(data.swapaxes(0, 2))
    data = np.transpose(data, [1, 0, 2])
    data = data[::-1, :, :]
    return data


def import_interfile_volume(headerfile="", datafile=""):
    from ...Core import Image3D

    # Load ndarray data
    data = import_interfile_volume_data(headerfile, datafile)
    # Load other information - e.g. pixels size
    F = load_interfile(headerfile)
    if "scale factor (mm/pixel) [1]" in F:
        pixsize_x = F["scale factor (mm/pixel) [1]"]["value"]
        pixsize_y = F["scale factor (mm/pixel) [2]"]["value"]
        pixsize_z = F["scale factor (mm/pixel) [3]"]["value"]
    # Create Image3D
    T_pix_to_world = Transform_Scale(
        np.int32([pixsize_x, pixsize_y, pixsize_z]), map_from="pixels_PET", map_to="world"
    )
    volume = Image3D(data=data, affine=T_pix_to_world, space="world")
    return volume


def export_interfile_volume(data_file_name, data):
    data = data[::-1, :, :]
    data = np.transpose(data, [1, 0, 2])
    data = data.swapaxes(0, 2)
    data = np.asarray(data, dtype=np.float32, order="C")
    data.tofile(data_file_name)
