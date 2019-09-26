# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

# Import an interfile projection as a PET_Projection object of occiput and export.


__all__ = [
    "import_interfile_projection",
    "export_interfile_projection",
    "import_h5f_projection",
]


from ..FileSources import interfile
import h5py
import os
import numpy as np


def import_interfile_projection_data(
    headerfile="", datafile="", load_time=False
):  # FIXME: this should be in the Interfile package
    F = interfile.load_interfile(headerfile)

    if "matrix size[1]" in F:
        N_planes = F["matrix size[1]"]["value"]
        N_axial = F["matrix size[2]"]["value"]
        N_sinograms = F["matrix size[3]"]["value"]
    else:
        N_planes = F["matrix size [1]"]["value"]
        N_axial = F["matrix size [2]"]["value"]
        N_sinograms = F["matrix size [3]"]["value"]
    if datafile == "":
        datafile1 = headerfile.replace(
            headerfile.split(os.sep)[-1], F["name of data file"]["value"]
        )
        datafile2 = headerfile.replace(".s.hdr", ".s")
        datafile2 = datafile2.replace(".h33", ".a")
        datafile3 = headerfile.replace(".h33", ".s")
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
    data = data.reshape([N_sinograms, N_axial, N_planes])
    if load_time:
        try:
            duration = np.int32([0, F["image duration"]["value"]]) * 1000
        except:
            print(
                "Unable to load_interfile image (sinogram) duration. This may determine an incorrect scale and use of randoms and scatter when reconstructing. Set .time_bins manually. "
            )
            duration = np.int32([0, 0])
    else:
        duration = np.int32([0, 0])
    return data, duration


def import_interfile_projection(
    headerfile,
    binning,
    michelogram,
    datafile="",
    invert=False,
    vmin=0.00,
    vmax=1e10,
    load_time=False,
):
    from ...Reconstruction.PET.PET_projection import PET_Projection, PET_Projection_Sparsity

    data, duration = import_interfile_projection_data(
        headerfile, datafile, load_time=load_time
    )
    N_planes = data.shape[2]
    N_axial = data.shape[1]
    N_sinograms = data.shape[0]

    N_azim = michelogram.n_segments
    z_elements = np.int32([michelogram.segments_sizes[0]])
    for i in range((N_azim - 1) // 2):
        z_elements = np.concatenate(
            [
                z_elements,
                np.int32(
                    [
                        michelogram.segments_sizes[i + 1],
                        michelogram.segments_sizes[i + 1],
                    ]
                ),
            ]
        )

    projection = np.zeros(
        [binning.N_axial, binning.N_azimuthal, binning.N_u, binning.N_v],
        dtype=np.float32,
        order="C",
    )
    indexes_azim = [5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 0]
    index0 = 0

    for i in range(N_azim):
        index1 = index0 + z_elements[i]
        data_azim = data[index0:index1, :, :]
        for j in range(N_axial):
            plane = np.zeros([projection.shape[2], projection.shape[3]], order="F")
            offset = (projection.shape[3] - (index1 - index0)) // 2
            plane[:, offset : offset + index1 - index0] = (
                data_azim[:, j, :].squeeze().transpose()
            )
            projection[j, indexes_azim[i], :, :] = np.fliplr(plane)
        index0 += z_elements[i]

    # flip azimuthally - this makes it consistent with Occiput's routines that import from Petlink32 listmode
    projection2 = np.zeros(projection.shape, dtype=np.float32, order="C")
    projection2[:, 5, :, :] = projection[:, 5, :, :]
    projection2[:, 4, :, :] = projection[:, 6, :, :]
    projection2[:, 3, :, :] = projection[:, 7, :, :]
    projection2[:, 2, :, :] = projection[:, 8, :, :]
    projection2[:, 1, :, :] = projection[:, 9, :, :]
    projection2[:, 0, :, :] = projection[:, 10, :, :]
    projection2[:, 6, :, :] = projection[:, 4, :, :]
    projection2[:, 7, :, :] = projection[:, 3, :, :]
    projection2[:, 8, :, :] = projection[:, 2, :, :]
    projection2[:, 9, :, :] = projection[:, 1, :, :]
    projection2[:, 10, :, :] = projection[:, 0, :, :]

    sparsity = PET_Projection_Sparsity(
        binning.N_axial, binning.N_azimuthal, binning.N_u, binning.N_v
    )
    ## invert, except where values are 0
    # if there are broken detectors, disable them (set sensitivity to 0)
    if invert:
        projection_fixed = projection2.copy()
        projection_fixed[projection_fixed < vmin] = 0.0
        projection_fixed[projection_fixed > vmax] = vmax

        projection_inv = np.zeros(projection2.shape)
        projection_inv[projection_fixed != 0] = (
            1.0 / projection_fixed[projection_fixed != 0]
        )

        P = PET_Projection(
            binning, projection_inv, sparsity.offsets, sparsity.locations, duration
        )
    else:
        P = PET_Projection(
            binning, projection2, sparsity.offsets, sparsity.locations, duration
        )
    return P


def export_interfile_projection(
    sinogram_data_file, projection_data, binning, michelogram, invert=False
):
    # projection_data has size  N_axial, N_azimuthal, N_u, N_v      dtype=float32     order="C"
    # export as  N_sinograms, N_axial, N_planes

    # FIXME: need to flip azimuthally as in import_interfile

    N_planes = binning.N_u
    N_axial = binning.N_axial
    N_sinograms = (
        np.int32(michelogram.segments_sizes[0])
        + 2 * np.int32(michelogram.segments_sizes[1::]).sum()
    )
    data = np.zeros([N_sinograms, N_axial, N_planes], dtype=np.float32, order="C")

    N_azim = michelogram.n_segments
    z_elements = np.int32([michelogram.segments_sizes[0]])
    for i in range((N_azim - 1) // 2):
        z_elements = np.concatenate(
            [
                z_elements,
                np.int32(
                    [
                        michelogram.segments_sizes[i + 1],
                        michelogram.segments_sizes[i + 1],
                    ]
                ),
            ]
        )
    indexes_azim = [5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 0]
    index0 = 0

    for i in range(N_azim):
        index1 = index0 + z_elements[i]
        data_azim = np.zeros(
            [index1 - index0, data.shape[1], data.shape[2]], dtype=np.float32, order="C"
        )
        for j in range(N_axial):
            plane = np.fliplr(projection_data[j, indexes_azim[i], :, :])
            offset = (projection_data.shape[3] - (index1 - index0)) // 2
            data_azim[:, j, :] = plane[:, offset : offset + index1 - index0].transpose()
        data[index0:index1, :, :] = data_azim
        index0 += z_elements[i]

    if invert:
        data2 = np.zeros([N_sinograms, N_axial, N_planes], dtype=np.float32, order="C")
        data2[data != 0] = 1.0 / data[data != 0]
    else:
        data2 = data
    data2.tofile(sinogram_data_file)


def import_h5f_projection(filename):
    from ...Reconstruction.PET.PET_projection import PET_Projection, Binning
    h5f = h5py.File(filename, "r")
    offsets = np.asarray(h5f["offsets"], order="F")
    offsets = np.ascontiguousarray(offsets)
    locations = np.asarray(h5f["locations"], order="F")
    locations = np.asfortranarray(locations)
    try:
        data = np.asarray(h5f["data"], order="F")
    except:
        data = np.asarray(
            h5f["static_data"], order="F"
        )  # compatibility with first version - deprecate at some point
    try:
        time_bins = np.asarray(h5f["time_bins"], order="F")
    except:
        time_bins = np.int32(
            [0, 0]
        )  # compatibility with first version - deprecate at some point
    binning = Binning(
        {
            "n_axial": np.int32(h5f["n_axial"]),
            "n_azimuthal": np.int32(h5f["n_azimuthal"]),
            "angles_axial": np.float32(h5f["angles_axial"]),
            "angles_azimuthal": np.float32(h5f["angles_azimuthal"]),
            "size_u": np.float32(h5f["size_u"]),
            "size_v": np.float32(h5f["size_v"]),
            "n_u": np.int32(h5f["n_u"]),
            "n_v": np.int32(h5f["n_v"]),
        }
    )
    h5f.close()
    data = np.ascontiguousarray(np.float32(data))
    return PET_Projection(binning, data, offsets, locations, time_bins)
