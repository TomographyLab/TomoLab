# -*- coding: utf-8 -*-
# occiput
# Harvard University, Martinos Center for Biomedical Imaging
# Aalto University, Department of Computer Science


# If you are looking for PET reconstruction, this is where to start.
# The objects defined here provide abstractions for Static and Dynamic PET reconstruction,
# abstracting the scanner geometries and vendor models and providing an interface to the
# software tools for projection, backprojection and reconstruction.
# Occiput enables reconstruction both from list-mode and from sinogram data using GPU
# acceleration.


import numpy as np
import os
import copy
from scipy import ndimage

from ...Core.Print import array_to_string, print_percentage,\
    millisec_to_min_sec, pretty_print_large_number
from ...DataSources.FileSources import interfile, guess_file_type_by_name
from ...DataSources.PET.PET_sinogram import import_interfile_projection, import_h5f_projection
from ...DataSources.PET.PET_volume import import_interfile_volume
from ...DataSources.MR.MR_vNAV import load_vnav_mprage
from ...Transformation.Transformations import Transform_Identity
from .PET_projection import PET_Projection, Binning, PET_Projection_Sparsity
from .PET_projection import display_PET_Projection_geometry
from .PET_raytracer import ProjectionParameters, BackprojectionParameters
from ...ScannerGeometries.PET_scanners import *  # Generic, get_scanner_by_name
from .PET_profiler import ReconstructionProfiler
from ...DataSources.PET import PET_listmode
from ...Core import Image3D
from ...Transformation.Transformations import RigidTransform, Transform_Scale
from ...Core.Errors import  UnexpectedParameter, FileNotFound
from ...Core.NiftyRec import PET_project_compressed, PET_backproject_compressed
from .PET_subsets import SubsetGenerator
from ...Visualization.Visualization import ProgressBar
from ...Visualization import Colors as C

# Set verbose level
# This is a global setting for occiput. There are 3 levels of verbose:
# high, low, no_printing
from ...global_settings import *
set_verbose_no_printing()

try:
    import pylab
except BaseException:
    has_pylab = False
else:
    has_pylab = True

# Default parameters
DEFAULT_SUBSET_SIZE = 24
DEFAULT_RECON_ITERATIONS = 10
DEFAULT_N_TIME_BINS = 15
EPS = 1e-6

def f_continuous(var):
    """Makes an nd_array Fortran-contiguous. """
    if isinstance(var, np.ndarray):
        if not var.flags.f_contiguous:
            var = np.asarray(var, order="F")
    else:
        if hasattr(var, "data"):
            if isinstance(var.data, np.ndarray):
                if not var.data.flags.f_contiguous:
                    var.data = np.asarray(var.data, order="F")
    return var

class PET_Static_Scan:
    """PET Static Scan. """

    def __init__(self):
        # by default, use GPU.
        self.use_gpu(True)
        # set scanner geometry and load_interfile interface
        self.set_scanner('Generic')
        # memoization of activity.
        self.activity = None
        # memoization of attenuation.
        self.attenuation = None
        # memoization of attenuation projection.
        self.attenuation_projection = None
        # sensitivity is a permanent parameter.
        self.sensitivity = None
        # measurement: prompts. Initialized as empty data structure.
        self.prompts = None
        self.randoms = None
        self.scatter = None
        # normalization volume - for all projections - memoize
        self._normalization = (None)
        # If True, the normalization volume needs to be recomputed
        self._need_normalization_update = (True)
        self.use_compression(False)

        self.set_transform_scanner_to_world(
            Transform_Identity(
                map_from="scanner",
                map_to="world"))
        self.profiler = ReconstructionProfiler()
        self.resolution = self.activity_size / self.activity_shape
        self.pixel_size = self.resolution[:2]
        self.slice_thickness = self.resolution[2]

    ################################################################################

    def _get_sparsity(self):
        if self.prompts is None:
            # in this case returns sparsity pattern for uncompressed projection
            sparsity = PET_Projection_Sparsity(
                self.binning.N_axial,
                self.binning.N_azimuthal,
                self.binning.N_u,
                self.binning.N_v,
            )
        else:
            sparsity = self.prompts.sparsity
        return sparsity
        
    def _make_Image3D_activity(self, data=None):
        shape = np.asarray(self.activity_shape)
        size  = np.asarray(self.activity_size)
        t_scanner_to_world = self.transform_scanner_to_world
        t_pix_to_scanner = Transform_Scale(
            size / shape, map_from="pixels_PET_Static", map_to="scanner"
        )
        t_pix_to_world = t_scanner_to_world.left_multiply(t_pix_to_scanner)
        image = Image3D(data=data, affine=t_pix_to_world, space="world")
        return image

    def _make_Image3D_attenuation(self, data=None):
        shape = np.asarray(self.attenuation_shape)
        size = np.asarray(self.attenuation_size)
        t_scanner_to_world = self.transform_scanner_to_world
        t_pix_to_scanner = Transform_Scale(
            size / shape, map_from="pixels_PET_Static", map_to="scanner"
        )
        t_pix_to_world = t_scanner_to_world.left_multiply(t_pix_to_scanner)
        image = Image3D(data=data, affine=t_pix_to_world, space="world")
        return image
    
    
    ################################################################################

    def set_transform_scanner_to_world(self, transform):
        # FIXME: verify that the transform maps from 'scanner' to 'world'
        self.transform_scanner_to_world = transform

    def set_activity_shape(self, activity_shape):
        activity_shape = np.asarray(activity_shape)
        if not len(activity_shape) == 3:
            print("Invalid activity shape")  # FIXME: raise invalid input error
        else:
            self.activity_shape = activity_shape
        try:
            self.set_activity_resolution()
        except:
            pass

    def set_activity_size(self, activity_size):
        activity_size = np.asarray(activity_size)
        if not len(activity_size) == 3:
            print("Invalid activity size")  # FIXME: raise invalid input error
        else:
            self.activity_size = activity_size
        self._adapt_line_step_size_activity()
        try:
            self.set_activity_resolution()
        except:
            pass

    def set_activity_resolution(self):
        self.resolution = self.activity_size / self.activity_shape
        self.pixel_size = self.resolution[:2]
        self.slice_thickness = self.resolution[2]

    def set_activity_scale(self, scale):
        self.scale_activity = scale

    def set_attenuation_shape(self, attenuation_shape):
        attenuation_shape = np.asarray(attenuation_shape)
        if not len(attenuation_shape) == 3:
            print("Invalid attenuation shape")  # FIXME: raise invalid input error
        else:
            self.attenuation_shape = attenuation_shape

    def set_attenuation_size(self, attenuation_size):
        attenuation_size = np.asarray(attenuation_size)
        if not len(attenuation_size) == 3:
            print("Invalid attenuation size")  # FIXME: raise invalid input error
        else:
            self.attenuation_size = attenuation_size
        self._adapt_line_step_size_attenuation()

    def _adapt_line_step_size_activity(
        self
    ):  # FIXME: move this calculation in the raytracer
        if not hasattr(self, "activity_size"):
            activity_size = np.float32([0, 0, 0])
        elif self.activity_size is None:
            activity_size = np.float32([0, 0, 0])
        else:
            activity_size = np.float32(self.activity_size)
        diagonal = np.sqrt((activity_size ** 2).sum())
        self.activity_projection_parameters.sample_step = (
            diagonal / self.activity_projection_parameters.N_samples
        )
        self.activity_backprojection_parameters.sample_step = (
            diagonal / self.activity_backprojection_parameters.N_samples
        )

    def _adapt_line_step_size_attenuation(
        self
    ):  # FIXME: move this calculation in the raytracer
        if not hasattr(self, "attenuation_size"):
            attenuation_size = np.float32([0, 0, 0])
        elif self.attenuation_size is None:
            attenuation_size = np.float32([0, 0, 0])
        else:
            attenuation_size = np.float32(self.attenuation_size)
        diagonal = np.sqrt((attenuation_size ** 2).sum())
        self.attenuation_projection_parameters.sample_step = (
            diagonal / self.attenuation_projection_parameters.N_samples
        )
        self.attenuation_backprojection_parameters.sample_step = (
            diagonal / self.attenuation_backprojection_parameters.N_samples
        )

    def set_binning(self, binning):
        if isinstance(binning, Binning):
            self.binning = binning
        else:
            self.binning = Binning(binning)
        self._subsets_generator = SubsetGenerator(
            self.binning.N_azimuthal, self.binning.N_axial
        )
        return self.binning

    def set_scanner(self, scanner):
        try:
            self.scanner = get_scanner_by_name(scanner)
        except BaseException:
            raise NotImplementedError

        self.activity_projection_parameters = ProjectionParameters()
        self.activity_backprojection_parameters = BackprojectionParameters()
        self.activity_projection_parameters.N_samples = (
            self.scanner.activity_N_samples_projection_DEFAULT
        )
        self.activity_projection_parameters.sample_step = (
            self.scanner.activity_sample_step_projection_DEFAULT
        )
        self.activity_backprojection_parameters.N_samples = (
            self.scanner.activity_N_samples_backprojection_DEFAULT
        )
        self.activity_backprojection_parameters.sample_step = (
            self.scanner.activity_sample_step_backprojection_DEFAULT
        )

        self.set_activity_shape(self.scanner.activity_shape_DEFAULT)
        self.set_activity_size(self.scanner.activity_size_DEFAULT)

        self.activity_projection_parameters.gpu_acceleration = self._use_gpu
        self.activity_backprojection_parameters.gpu_acceleration = self._use_gpu

        self.attenuation_projection_parameters = ProjectionParameters()
        self.attenuation_backprojection_parameters = BackprojectionParameters()
        self.attenuation_projection_parameters.N_samples = (
            self.scanner.attenuation_N_samples_projection_DEFAULT
        )
        self.attenuation_projection_parameters.sample_step = (
            self.scanner.attenuation_sample_step_projection_DEFAULT
        )
        self.attenuation_backprojection_parameters.N_samples = (
            self.scanner.attenuation_N_samples_backprojection_DEFAULT
        )
        self.attenuation_backprojection_parameters.sample_step = (
            self.scanner.attenuation_sample_step_backprojection_DEFAULT
        )

        self.set_attenuation_shape(self.scanner.attenuation_shape_DEFAULT)
        self.set_attenuation_size(self.scanner.attenuation_size_DEFAULT)

        self.attenuation_projection_parameters.gpu_acceleration = self._use_gpu
        self.attenuation_backprojection_parameters.gpu_acceleration = self._use_gpu

        binning = Binning()
        binning.size_u = self.scanner.size_u
        binning.size_v = self.scanner.size_v
        binning.N_u = self.scanner.N_u
        binning.N_v = self.scanner.N_v
        binning.N_axial = self.scanner.N_axial
        binning.N_azimuthal = self.scanner.N_azimuthal
        binning.angles_axial = self.scanner.angles_axial
        binning.angles_azimuthal = self.scanner.angles_azimuthal
        self.binning = binning

        self.set_activity_scale(self.scanner.scale_activity)

        self._subsets_generator = SubsetGenerator(
            self.binning.N_azimuthal, self.binning.N_axial
        )

    def use_gpu(self, use_it):
        self._use_gpu = use_it

    def use_compression(self, use_it):
        self._use_compression = use_it
        if not use_it:
            if self.prompts is not None:
                if self.prompts.is_compressed():
                    self.set_prompts(self.prompts.uncompress_self())
            if self.randoms is not None:
                if self.randoms.is_compressed():
                    self.set_randoms(self.randoms.uncompress_self())
            if self.sensitivity is not None:
                if self.sensitivity.is_compressed():
                    self.set_sensitivity(self.sensitivity.uncompress_self())
            if self.scatter is not None:
                if self.scatter.is_compressed():
                    self.set_scatter(self.scatter.uncompress_self())
        else:
            if hasattr(self, "_use_compression"):
                if self._use_compression is False and use_it is True:
                    # FIXME
                    # print "Not able to compress once uncompressed.
                    # Please implement PET_Projection.uncompress_self() to "
                    # print "enable this functionality. "
                    return
            if self.prompts is not None:
                if not self.prompts.is_compressed():
                    self.set_prompts(self.prompts.compress_self())
            if self.randoms is not None:
                if not self.randoms.is_compressed():
                    self.set_randoms(self.randoms.compress_self())
            if self.sensitivity is not None:
                if not self.sensitivity.is_compressed():
                    self.set_sensitivity(self.sensitivity.compress_self())
            if self.scatter is not None:
                if not self.scatter.is_compressed():
                    self.set_scatter(self.scatter.compress_self())

    ####################################################################################################################

    def _load_static_measurement(self, time_bin=None):
        if time_bin is None:
            Rp = self.scanner.listmode.get_measurement_static_prompt()
            Rd = self.scanner.listmode.get_measurement_static_delay()
        else:
            Rp = self.scanner.listmode.get_measurement_prompt(time_bin)
            Rd = self.scanner.listmode.get_measurement_delay(time_bin)
        time_start = Rp["time_start"]
        time_end = Rp["time_end"]

        time_bins = np.int32(np.linspace(time_start, time_end, 2))
        prompts = PET_Projection(
            self.binning, Rp["counts"], Rp["offsets"], Rp["locations"], time_bins
        )
        randoms = PET_Projection(
            self.binning, Rd["counts"], Rd["offsets"], Rd["locations"], time_bins
        )
        if self._use_compression:
            self.set_prompts(prompts)
            self.set_randoms(randoms)
        else:
            # print "Uncompressing"
            self.set_prompts(prompts.uncompress_self())
            self.set_randoms(randoms.uncompress_self())

    def import_listmode(
            self,
            filename,
            datafile=None,
            time_range_ms=(0, None),
            display_progress=True,
            print_debug=True,
    ):
        """Load measurement data from a listmode file. """
        if print_debug:
            print(f"- Loading static PET data from listmode file: {filename}")
        hdr = interfile.load_interfile(filename)

        # 2) Guess the path of the listmode data file, if not specified or mis-specified;
        #  1 - see if the specified listmode data file exists
        if datafile is not None:
            datafile = datafile.replace("/", os.path.sep).replace(
                "\\", os.path.sep
            )  # cross platform compatibility
            if not os.path.exists(datafile):
                raise FileNotFound("listmode data", datafile)
        #  2 - if the listmode data file is not specified, try with the name (and full path) contained in the listmode header
        datafile = hdr["name of data file"]["value"]
        datafile = datafile.replace("/", os.path.sep).replace("\\", os.path.sep)  # cross platform compatibility
        if not os.path.exists(datafile):
            #  3 - if it doesn't exist, look in the same path as the header file for the listmode
            #      data file with name specified in the listmode header file
            datafile = (
                    os.path.split(filename)[0] + os.path.sep + os.path.split(datafile)[-1]
            )
            if not os.path.exists(datafile):
                #  4 - if it doesn't exist, look in the same path as the header file for the listmode data
                #      file with same name as the listmode header file, replacing the extension: ".l.hdr -> .l"
                if filename.endswith(".l.hdr"):
                    datafile = filename.replace(".l.hdr", ".l")
                    if not os.path.exists(datafile):
                        raise FileNotFound("listmode data", datafile)
                #  5 - if it doesn't exist, look in the same path as the header file for the listmode data
                #      file with same name as the listmode header file, replacing the extension: ".hdr -> .l"
                elif filename.endswith(".hdr"):
                    datafile = filename.replace(".hdr", ".l")
                    if not os.path.exists(datafile):
                        raise FileNotFound("listmode data", datafile)

        # 3) Determine duration of the acquisition
        n_packets = hdr["total listmode word counts"]["value"]
        scan_duration = hdr["image duration"]["value"] * 1000  # milliseconds

        # 4) determine scanner parameters
        n_radial_bins = hdr["number of projections"]["value"]
        n_angles = hdr["number of views"]["value"]
        n_rings = hdr["number of rings"]["value"]
        max_ring_diff = hdr["maximum ring difference"]["value"]
        n_sinograms = (n_rings + 2 * n_rings * max_ring_diff - max_ring_diff ** 2 - max_ring_diff)

        # Determine the time binning
        time_range_0 = time_range_ms[0]
        if time_range_ms[1] is not None:
            time_range_1 = time_range_ms[1]
        else:
            time_range_1 = scan_duration
        time_bins = np.int32(np.linspace(time_range_0, time_range_1, 2))

        # Display information
        if print_debug:
            print(" - Number of packets:    %d       " % n_packets)
            print(" - Scan duration:        %d [sec] " % (scan_duration / 1000.0))
            print(" - Listmode data file:   %s       " % datafile)
            print(" - Listmode header file: %s       " % filename)
            print(" - Number of time bins:  %d       " % (len(time_bins) - 1))
            print(" - Time start:           %.2f [sec] " % (time_range_0 / 1000.0))
            print(" - Time end:             %.2f [sec] " % (time_range_1 / 1000.0))
            print(" - time_bins:            %s       " % str(time_bins))
            print(" - n_radial_bins:        %d       " % n_radial_bins)
            print(" - n_angles:             %d       " % n_angles)
            print(" - n_angles:             %d       " % n_sinograms)

        if display_progress:
            progress_bar = ProgressBar(
                color=C.LIGHT_RED, title="Decoding listmode ...")
            progress_callback = progress_bar.set_percentage
        else:
            def progress_callback(value):
                if value == 1.0:
                    print(value, "/", 100)
                if (np.int32(value) / 10) * 10 == value:
                    print(value, "/", 100)

        # Load the listmode data
        M = self.scanner.michelogram

        R = self.scanner.listmode.load_listmode(
            datafile,
            n_packets,
            time_bins,
            self.binning,
            n_radial_bins,
            n_angles,
            n_sinograms,
            M.span,
            M.segments_sizes,
            M.michelogram_sinogram,
            M.michelogram_plane,
            progress_callback,
        )
        progress_callback(100)
        print("Done!")

        # Load static measurement data
        self._load_static_measurement()

        # Free structures listmode data
        self.scanner.listmode.free_memory()

    ####################################################################################################################

    def import_prompts(self, filename, datafile=""):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_projection_header":
            projection = import_interfile_projection(
                filename,
                self.binning,
                self.scanner.michelogram,
                datafile,
                load_time=True,
            )
        elif filetype == "h5":
            projection = import_h5f_projection(filename)
        else:
            print("PET.import_prompts: file type unknown. ")
            return
        projection.data = np.float32(projection.data)
        if self._use_compression is False:
            projection = projection.uncompress_self()
        self.set_prompts(projection)

    def import_attenuation(
        self, filename, datafile="", filename_hardware="", datafile_hardware=""
    ):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_volume_header":
            volume = import_interfile_volume(filename, datafile)
        elif filetype == "nifti":
            print(
                "Nifti attenuation file not supported. Everything is ready to implement this, please implement it. "
            )
            # FIXME: if nifti files are used, sum the hardware image using resampling in the common space
        elif filetype == "h5":
            print(
                "H5 attenuation file not supported. Everything is ready to implement this, please implement it. "
            )
            # FIXME: if h5 files are used, sum the hardware image using resampling in the common space
        elif filetype == "mat":
            print(
                "Matlab attenuation file not supported. Everything is ready to implement this, please implement it. "
            )
        else:
            print(
                (
                    "PET.import_attenuation: file type of %s unknown. Unable to load_interfile attenuation tomogram. "
                    % filename
                )
            )
            return
        if filename_hardware != "":
            filetype = guess_file_type_by_name(filename_hardware)
            if filetype == "interfile_volume_header":
                volume_hardware = import_interfile_volume(
                    filename_hardware, datafile_hardware
                )
            else:
                print("File type of %s unknown. Unable to load_interfile hardware attenuation tomogram. "
                    % filename_hardware)
            volume.data = volume.data + volume_hardware.data
        volume.data = np.float32(volume.data)
        self.set_attenuation(volume)

    def import_attenuation_projection(self, filename, datafile=""):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_projection_header":
            projection = import_interfile_projection(
                filename,
                self.binning,
                self.scanner.michelogram,
                datafile,
                load_time=True,
            )
        elif filetype == "h5":
            projection = import_h5f_projection(filename)
        else:
            print("PET.import_attenuation_projection: file type unknown. ")
            return
        projection.data = np.float32(projection.data)
        if self._use_compression is False:
            projection = projection.uncompress_self()
        self.set_attenuation_projection(projection)

    # FIXME: when importing, compress if compression is enabled

    def import_sensitivity(self, filename, datafile="", vmin=0.00, vmax=1e10):
        filetype = guess_file_type_by_name(filename)
        if filetype == "h5":
            sensitivity = import_h5f_projection(filename)
        elif filetype == "interfile_projection_header":
            sensitivity = import_interfile_projection(
                filename,
                self.binning,
                self.scanner.michelogram,
                datafile,
                True,
                vmin,
                vmax,
            )
            if self.prompts is not None:
                # FIXME: sensitivity loaded from interfile with some
                # manufacturers has non-zero value
                # where there are no detectors - set to zero where data is zero
                # (good approx only for long acquisitions).
                # See if there is a better way to handle this.
                sensitivity.data[self.prompts.data == 0] = 0
            else:  # FIXME: see comment two lines up
                print(
                    "Warning: If loading real scanner data, please "
                    "load_interfile prompts before loading the sensitivity. \n"
                    "Ignore this message if this is a simulation. \n"
                    "See the source code for more info. ")
        elif filetype == "mat":
            print("Sensitivity from Matlab not yet implemented. "
                  "All is ready, please spend 15 minutes and implement. ")
            return
        else:
            print("File type unknown. ")
            return
        sensitivity.data = np.float32(sensitivity.data)
        if self._use_compression is False:
            sensitivity = sensitivity.uncompress_self()
        self.set_sensitivity(sensitivity)

    def import_scatter(self, filename, datafile="", duration_ms=None):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_projection_header":
            projection = import_interfile_projection(
                filename, self.binning, self.scanner.michelogram, datafile
            )
        elif filetype == "h5":
            projection = import_h5f_projection(filename)
        else:
            print("PET.import_scatter: file type unknown. ")
            return
        projection.data = np.float32(projection.data)
        if self._use_compression is False:
            projection = projection.uncompress_self()
        self.set_scatter(projection, duration_ms)

    def import_randoms(self, filename, datafile=""):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_projection_header":
            projection = import_interfile_projection(
                filename, self.binning, self.scanner.michelogram, datafile
            )
        elif filetype == "h5":
            projection = import_h5f_projection(filename)
        else:
            print("PET.import_randoms: file type unknown. ")
            return
        projection.data = np.float32(projection.data)
        if self._use_compression is False:
            projection = projection.uncompress_self()
        self.set_randoms(projection)

    ###########################################################################

    def set_prompts(self, prompts):
        if isinstance(prompts, PET_Projection):
            self.prompts = prompts
            self.prompts.data = np.float32(self.prompts.data)
            # update self.sparsity (self.sparsity exists to store sparsity
            # information in case there is no prompts data)
            self.sparsity = self.prompts.sparsity
        # self.set_binning(prompts.get_binning()) #FIXME: check if it is compatible with the scanner
        elif self.prompts is not None:
            prompts = PET_Projection(
                self.prompts.get_binning(),
                prompts,
                self.prompts.sparsity.offsets,
                self.prompts.sparsity.locations,
                self.prompts.get_time_bins(),
            )
            self.prompts = prompts
            self.prompts.data = np.float32(self.prompts.data)
        else:
            print(
                "Prompts data should be an instance of PET_Projection or an array whose dimension"
            )
            print("matches the sparsity pattern of the current projection data. ")
            # FIXME: raise input error and to a try-except when creating the instance of PET_Projection

    def set_scatter(self, scatter, duration_ms=None):
        self.scatter = scatter
        self.scatter.data = np.float32(self.scatter.data)
        if duration_ms is not None:
            self.scatter.time_bins = np.int32([0, duration_ms])

    def simulate_scatter(self, activity=None, attenuation=None):
        can_simulate = True
        if not hasattr(self.scanner, "scatter_simulator"):
            can_simulate = False
        elif self.scanner.scatter_simulator is None:
            can_simulate = False
        if not can_simulate:
            print("The selected scanner interface does not expose a scatter simultor. ")
            return None
        if activity is None:
            activity = self.activity
        if attenuation is None:
            attenuation = self.attenuation
        scatter_projection = self.scanner.scatter_simulator.simulate(
            activity, attenuation)
        return scatter_projection

    def tail_fit_scatter(self, scatter_projection):
        print("Not implemented. Please implement tail fitting. ")
        return None

    def set_randoms(self, randoms):
        if isinstance(randoms, PET_Projection):
            self.randoms = randoms
            self.randoms.data = np.float32(self.randoms.data)
            self.sparsity_delay = (
                self.randoms.sparsity
            )  # update self.sparsity (self.sparsity exists to store
            # sparsity information in case there is not randoms data)
            # self.set_binning(randoms.get_binning())   #FIXME: make sure binning is consistent with randoms
        elif self.randoms is not None:
            randoms = PET_Projection(
                self.randoms.get_binning(),
                randoms,
                self.randoms.sparsity.offsets,
                self.randoms.sparsity.locations,
                self.randoms.get_time_bins(),
            )
            self.randoms = randoms
        else:
            print(
                "Delay randoms data should be an instance of PET_Projection or an array whose dimension"
            )
            print("matches the sparsity pattern of the current projection data. ")
            # FIXME: raise input error and to a try-except when creating the instance of PET_Projection

    def set_sensitivity(self, sensitivity):
        # FIXME: verify type: PET_projection or nd_array (the latter only in full sampling mode)
        self.sensitivity = sensitivity
        self.sensitivity.data = np.float32(self.sensitivity.data)

    def set_attenuation_projection(self, attenuation_projection):
        self.attenuation_projection = attenuation_projection
        self.attenuation_projection.data = np.float32(self.attenuation_projection.data)

    def set_attenuation(self, attenuation):
        self.attenuation = attenuation
        self.attenuation.data = np.float32(self.attenuation.data)



    ####################################################################################################################

    def get_prompts(self):
        return self.prompts

    def get_scatter(self):
        return self.scatter

    def get_randoms(self):
        return self.randoms

    def get_sensitivity(self):
        return self.sensitivity

    def get_attenuation_projection(self):
        return self.attenuation_projection

    def get_attenuation(self):
        return self.attenuation

    def get_activity(self):
        return self.activity

    def get_normalization(
        self,
        attenuation_times_sensitivity=None,
        transformation=None,
        sparsity=None,
        duration_ms=None,
        subsets_matrix=None,
        epsilon=None,
    ):
        # FIXME: memoization
        if attenuation_times_sensitivity is None:
            attenuation_times_sensitivity = (
                self.sensitivity
            )  # FIXME: include attenuation here - memoization mumap proj
        if (
            np.isscalar(attenuation_times_sensitivity)
            or attenuation_times_sensitivity is None
        ):
            attenuation_times_sensitivity = np.ones(self.prompts.data.shape)
        if duration_ms is None:
            duration_ms = self.prompts.get_duration()
        duration_sec = duration_ms / 1000.0
        alpha = self.scale_activity
        normalization = self.backproject_activity(
            attenuation_times_sensitivity * duration_sec * alpha,
            transformation,
            subsets_matrix,
        )
        return normalization

    ####################################################################################################################

    def export_prompts(self, filename):
        self.get_prompts().save_to_file(filename)

    def export_sensitivity(self, filename):
        if self.sensitivity is None:
            print("Sensitivity has not been loaded")
        else:
            self.get_sensitivity().save_to_file(filename)

    def export_scatter(self, filename):
        self.get_randoms().save_to_file(filename)

    def export_randoms(self, filename):
        self.get_randoms().save_to_file(filename)

    def export_attenuation_projection(self, filename):
        self.get_attenuation_projection().save_to_file(filename)

    ####################################################################################################################

    def project_attenuation(
            self,
            attenuation=None,
            unit='inv_cm',
            transformation=None,
            sparsity=None,
            subsets_matrix=None,
            exponentiate=True
    ):

        self.profiler.tic()
        if attenuation is None:
            attenuation = self.attenuation
        if isinstance(attenuation, np.ndarray):
            attenuation_data = f_continuous(np.float32(attenuation))
        else:
            attenuation_data = f_continuous(np.float32(attenuation.data))
        self.profiler.rec_project_make_continuous()

        if not list(attenuation_data.shape) == list(self.attenuation_shape):
            raise UnexpectedParameter(
                "Attenuation must have the same shape as self.attenuation_shape")

        # By default, the center of the imaging volume is at the center of the
        # scanner
        tx = 0.5 * \
             (self.attenuation_size[0] - self.attenuation_size[0] / self.attenuation_shape[0])
        ty = 0.5 * \
             (self.attenuation_size[1] - self.attenuation_size[1] / self.attenuation_shape[1])
        tz = 0.5 * \
             (self.attenuation_size[2] - self.attenuation_size[2] / self.attenuation_shape[2])
        if transformation is None:
            transformation = RigidTransform((tx, ty, tz, 0, 0, 0))
        else:
            transformation = copy.copy(transformation)
            transformation.x = transformation.x + tx
            transformation.y = transformation.y + ty
            transformation.z = transformation.z + tz

        # Scale according to the unit measure of the specified attenuation. It is assumed that the attenuation map
        # is constant in a voxel, with the value specified in 'attenuation', of
        # unit measure 'unit'.
        if unit == 'inv_mm':
            invert = False
            scale = 1.0
        elif unit == 'inv_cm':
            invert = False
            scale = 10.0
        elif unit == 'mm':
            invert = True
            scale = 1.0
        elif unit == 'cm':
            invert = True
            scale = 10.0
        else:
            print(
                "Unit measure unknown. Assuming inv_cm. Keep track of the unit measures! ")
            invert = False
            scale = 10.0

        if invert:
            attenuation_data = 1.0 / (attenuation_data + EPS)
        step_size_mm = self.attenuation_projection_parameters.sample_step
        step_size = step_size_mm / scale

        # Optionally project with a sparsity pattern not equal to sparsity associated to the loaded prompts data
        # Note: if prompts have not been loaded, self._get_sparsity() assumes
        # no compression.

        if sparsity is None:
            sparsity = self._get_sparsity()

        # Optionally project only to a subset of projection planes
        if subsets_matrix is None:
            sparsity_subset = sparsity
            self.profiler.tic()
            angles = self.binning.get_angles()
            self.profiler.rec_project_get_angles()
        else:
            self.profiler.tic()
            sparsity_subset = sparsity.get_subset(subsets_matrix)
            self.profiler.rec_project_get_subset_sparsity()
            self.profiler.tic()
            angles = self.binning.get_angles(subsets_matrix)
            self.profiler.rec_project_get_angles()

        offsets = sparsity_subset.offsets
        locations = sparsity_subset.locations
        activations = np.ones(
            [angles.shape[1], angles.shape[2]], dtype="uint32")

        # Call the raytracer
        self.profiler.tic()
        projection_data, timing = PET_project_compressed(attenuation_data, None,
                                                         offsets, locations, activations,
                                                         angles.shape[2], angles.shape[1], angles,
                                                         self.binning.N_u, self.binning.N_v,
                                                         self.binning.size_u,self.binning.size_v,
                                                         self.attenuation_size[0], self.attenuation_size[1],self.attenuation_size[2],
                                                         0.0, 0.0, 0.0,
                                                         transformation.x, transformation.y, transformation.z,
                                                         transformation.theta_x, transformation.theta_y,transformation.theta_z,
                                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                         self.attenuation_projection_parameters.gpu_acceleration,
                                                         self.attenuation_projection_parameters.N_samples,
                                                         self.attenuation_projection_parameters.sample_step,
                                                         self.attenuation_projection_parameters.background_attenuation,
                                                         0.0,
                                                         self.attenuation_projection_parameters.truncate_negative_values,
                                                         self.attenuation_projection_parameters.direction,
                                                         self.attenuation_projection_parameters.block_size)

        self.profiler.rec_project_projection()
        self.profiler.rec_projection(timing)

        # Fix scale and exponentiate
        if exponentiate:
            self.profiler.tic()
            projection_data = np.exp(-projection_data * step_size)
            self.profiler.rec_project_exponentiate()
        else:
            self.profiler.tic()
            projection_data = projection_data * step_size
            self.profiler.rec_project_scale()

        # Create object PET_Projection: it contains the raw projection data and
        # the description of the projection geometry and sparsity pattern.
        # Projection of the attenuation does not have timing information
        time_bins = np.int32([0, 0])
        self.profiler.tic()
        projection = PET_Projection(
            self.binning,
            projection_data,
            sparsity.offsets,
            sparsity.locations,
            time_bins,
            subsets_matrix)
        self.profiler.rec_project_wrap()
        self.set_attenuation_projection(projection)
        return projection

    def backproject_attenuation(
            self,
            projection,
            unit="inv_cm",
            transformation=None,
            sparsity=None,
            subsets_matrix=None):
        if isinstance(projection, np.ndarray):
            projection_data = np.float32(projection)
        else:
            projection_data = np.float32(projection.data)

        # By default, the center of the imaging volume is at the center of the
        # scanner
        tx = 0.5 * \
            (self.attenuation_size[0] - self.attenuation_size[0] / self.attenuation_shape[0])
        ty = 0.5 * \
            (self.attenuation_size[1] - self.attenuation_size[1] / self.attenuation_shape[1])
        tz = 0.5 * \
            (self.attenuation_size[2] - self.attenuation_size[2] / self.attenuation_shape[2])
        if transformation is None:
            transformation = RigidTransform((tx, ty, tz, 0, 0, 0))
        else:
            transformation = copy.copy(transformation)
            transformation.x = transformation.x + tx
            transformation.y = transformation.y + ty
            transformation.z = transformation.z + tz

        # Scale according to the unit measure of the specified attenuation. It is assumed that the attenuation map
        # is constant in a voxel, with the value specified in 'attenuation', of
        # unit measure 'unit'.
        if unit == 'inv_mm':
            invert = False
            scale = 1.0
        elif unit == 'inv_cm':
            invert = False
            scale = 10.0
        elif unit == 'mm':
            invert = True
            scale = 1.0
        elif unit == 'cm':
            invert = True
            scale = 10.0
        else:
            print(
                "Unit measure unknown. Assuming inv_cm. Keep track of the unit measures! ")
            invert = False
            scale = 10.0

        if invert:
            projection_data = np.float32(1.0 / (projection_data + EPS))
        step_size_mm = self.attenuation_projection_parameters.sample_step
        step_size = step_size_mm / scale

        if sparsity is None:
            sparsity = self._get_sparsity()

        if isinstance(projection, np.ndarray):
            projection_data = np.float32(projection)
            offsets = sparsity.offsets
            locations = sparsity.locations
            angles = self.binning.get_angles(subsets_matrix)
            activations = np.ones(
                [self.binning.N_azimuthal, self.binning.N_axial], dtype="uint32")
        else:
            projection_data = np.float32(projection.data)
            offsets = projection.sparsity.offsets
            locations = projection.sparsity.locations
            angles = projection.get_angles()
            activations = np.ones(
                [projection.sparsity.N_azimuthal, projection.sparsity.N_axial], dtype="uint32")

        # Call ray-tracer
        backprojection_data, timing = PET_backproject_compressed(projection_data, None, offsets, locations, activations,
                                                                 angles.shape[2], angles.shape[1], angles,
                                                                 self.binning.N_u, self.binning.N_v,
                                                                 self.binning.size_u, self.binning.size_v,
                                                                 self.attenuation_shape[0], self.attenuation_shape[1],
                                                                 self.attenuation_shape[2],
                                                                 self.attenuation_size[0], self.attenuation_size[1],
                                                                 self.attenuation_size[2],
                                                                 0.0, 0.0, 0.0,
                                                                 transformation.x, transformation.y, transformation.z,
                                                                 transformation.theta_x, transformation.theta_y,
                                                                 transformation.theta_z,
                                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                                 self.attenuation_backprojection_parameters.gpu_acceleration,
                                                                 self.attenuation_backprojection_parameters.N_samples,
                                                                 self.attenuation_backprojection_parameters.sample_step,
                                                                 self.attenuation_backprojection_parameters.background_attenuation,
                                                                 0.0,
                                                                 self.attenuation_backprojection_parameters.direction,
                                                                 self.attenuation_backprojection_parameters.block_size)

        self.profiler.rec_backprojection(timing)
        backprojection_data = backprojection_data * step_size

        # Set the correct scale - unit measure and return Image3D
        # FIXME: set scale for requested unit measure
        return self._make_Image3D_attenuation(backprojection_data)

    def project_activity(
            self,
            activity,
            unit="Bq/mm3",
            transformation=None,
            sparsity=None,
            subsets_matrix=None,
    ):

        self.profiler.tic()
        if isinstance(activity, np.ndarray):
            activity_data = f_continuous(np.float32(activity))
        else:
            activity_data = f_continuous(np.float32(activity.data))
        self.profiler.rec_project_make_continuous()

        # By default, the center of the imaging volume is at the center of the scanner; no rotation
        tx = 0.5 * (
                self.activity_size[0] - self.activity_size[0] / self.activity_shape[0]
        )
        ty = 0.5 * (
                self.activity_size[1] - self.activity_size[1] / self.activity_shape[1]
        )
        tz = 0.5 * (
                self.activity_size[2] - self.activity_size[2] / self.activity_shape[2]
        )
        if transformation is None:
            transformation = RigidTransform((tx, ty, tz, 0, 0, 0))
        else:
            transformation = copy.copy(transformation)
            transformation.x = transformation.x + tx
            transformation.y = transformation.y + ty
            transformation.z = transformation.z + tz

        # Optionally project with a sparsity pattern not equal to sparsity associated to the loaded prompts data
        # Note: if prompts have not been loaded, self._get_sparsity() assumes no compression.
        if sparsity is None:
            sparsity = self._get_sparsity()

        # Optionally project only to a subset of projection planes
        if subsets_matrix is None:
            self.profiler.tic()
            sparsity_subset = sparsity
            angles = self.binning.get_angles()
            self.profiler.rec_project_get_angles()
        else:
            self.profiler.tic()
            sparsity_subset = sparsity.get_subset(subsets_matrix)
            self.profiler.rec_project_get_subset_sparsity()
            self.profiler.tic()
            angles = self.binning.get_angles(subsets_matrix)
            self.profiler.rec_project_get_angles()

        scale = 1.0 # FIXME: change this according to the input unit measure - check how this is done in project_attenuation
        step_size_mm = self.activity_projection_parameters.sample_step
        step_size = step_size_mm / scale

        offsets = sparsity_subset.offsets
        locations = sparsity_subset.locations
        activations = np.ones([angles.shape[1], angles.shape[2]], dtype="uint32")

        # print locations[:,0:20]
        # print locations.flags
        # print sparsity.locations[:,0:20]
        # print sparsity.locations.flags

        # print "project activity"
        # print "activity",activity_data.shape
        # print "offsets",offsets.shape
        # print "locations",locations.shape
        # print "activations",activations.shape
        # print "angles.shape",angles.shape

        # Call the raytracer
        self.profiler.tic()
        projection_data, timing = PET_project_compressed(
            activity_data,None,offsets,locations,activations,
            angles.shape[2],angles.shape[1],angles,
            self.binning.N_u,self.binning.N_v,
            self.binning.size_u,self.binning.size_v,
            self.activity_size[0],self.activity_size[1],self.activity_size[2],
            0.0,0.0,0.0,
            transformation.x,transformation.y,transformation.z,
            transformation.theta_x,transformation.theta_y,transformation.theta_z,
            0.0,0.0,0.0,0.0,0.0,0.0,
            self.activity_projection_parameters.gpu_acceleration,
            self.activity_projection_parameters.N_samples,
            self.activity_projection_parameters.sample_step,
            self.activity_projection_parameters.background_activity,
            0.0,
            self.activity_projection_parameters.truncate_negative_values,
            self.activity_projection_parameters.direction,
            self.activity_projection_parameters.block_size,
        )

        self.profiler.rec_project_projection()
        self.profiler.rec_projection(timing)

        # Create object PET_Projection: it contains the raw projection data and the description of the projection geometry
        # and sparsity pattern.
        time_bins = np.int32([0, 1000.0])  # 1 second - projection returns a rate - by design

        self.profiler.tic()
        projection_data = projection_data * step_size
        self.profiler.rec_project_scale()

        self.profiler.tic()
        projection = PET_Projection(
            self.binning,
            projection_data,
            sparsity.offsets,
            sparsity.locations,
            time_bins,
            subsets_matrix,
        )
        self.profiler.rec_project_wrap()

        # Optionally scale by sensitivity, attenuation, time, global sensitivity
        # if attenuation is not None:
        #    projection.data = projection.data * attenuation.data #attenuation.compress_as(projection).data
        # if self.sensitivity is not None:
        #    projection.data = projection.data * self.sensitivity.data.reshape(projection_data.shape)
        # self.sensitivity.compress_as(projection).data

        return projection

    def backproject_activity(
            self,
            projection,
            transformation=None,
            subsets_matrix=None
    ):
        # By default, the center of the imaging volume is at the center of the scanner
        tx = 0.5 * (
                self.activity_size[0] - self.activity_size[0] / self.activity_shape[0]
        )
        ty = 0.5 * (
                self.activity_size[1] - self.activity_size[1] / self.activity_shape[1]
        )
        tz = 0.5 * (
                self.activity_size[2] - self.activity_size[2] / self.activity_shape[2]
        )
        if transformation is None:
            transformation = RigidTransform((tx, ty, tz, 0, 0, 0))
        else:
            transformation = copy.copy(transformation)
            transformation.x = transformation.x + tx
            transformation.y = transformation.y + ty
            transformation.z = transformation.z + tz

        if not isinstance(projection, np.ndarray):
            if not subsets_matrix is None:
                self.profiler.tic()
                projection_subset = projection.get_subset(subsets_matrix)
                self.profiler.rec_backpro_get_subset()
                self.profiler.tic()
                sparsity_subset = projection_subset.sparsity
                angles = projection_subset.get_angles()
                self.profiler.rec_backpro_get_angles()
                projection_data = np.float32(projection_subset.data)
            else:
                self.profiler.tic()
                projection_subset = projection
                sparsity_subset = projection.sparsity
                angles = projection_subset.get_angles()
                self.profiler.rec_backpro_get_angles()
                projection_data = np.float32(projection_subset.data)
        else:
            sparsity = self._get_sparsity()
            if not subsets_matrix is None:
                # doesn't look right
                self.profiler.tic()
                indexes = subsets_matrix.flatten() == 1
                projection_data = np.float32(
                    projection.swapaxes(0, 1).reshape(
                        (
                            sparsity.N_axial * sparsity.N_azimuthal,
                            self.binning.N_u,
                            self.binning.N_v,
                        )
                    )[indexes, :, :]
                )
                self.profiler.rec_backpro_get_subset_data()
                self.profiler.tic()
                sparsity_subset = sparsity.get_subset(subsets_matrix)
                self.profiler.rec_backpro_get_subset_sparsity()
                self.profiler.tic()
                angles = self.binning.get_angles(subsets_matrix)
                self.profiler.rec_backpro_get_angles()
            else:
                self.profiler.tic()
                sparsity_subset = sparsity
                angles = self.binning.get_angles()
                self.profiler.rec_backpro_get_angles()
                projection_data = np.float32(projection)

        offsets = sparsity_subset.offsets
        locations = sparsity_subset.locations
        activations = np.ones([sparsity_subset.N_azimuthal,
                               sparsity_subset.N_axial], dtype="uint32")

        scale = 1.0  # FIXME: change this according to the input unit measure - check how this is done in project_attenuation
        step_size_mm = self.activity_projection_parameters.sample_step
        step_size = step_size_mm / scale

        # print "backproject activity"
        # print "projection_data",projection_data.shape
        # print "offsets",offsets.shape
        # print "locations",locations.shape
        # print "activations",activations.shape
        # print "angles",angles.shape
        # print angles[:,0,0:5]
        # print offsets[0:3,0:5]
        # print locations[:,0:5]
        # time.sleep(0.2)

        # Call ray-tracer
        self.profiler.tic()
        backprojection_data, timing = PET_backproject_compressed(
            projection_data,None,offsets,locations,activations,
            angles.shape[2],angles.shape[1],angles,
            self.binning.N_u,self.binning.N_v,
            self.binning.size_u,self.binning.size_v,
            self.activity_shape[0],self.activity_shape[1],self.activity_shape[2],
            self.activity_size[0],self.activity_size[1],self.activity_size[2],
            0.0,0.0,0.0,
            transformation.x,transformation.y,transformation.z,
            transformation.theta_x,transformation.theta_y,transformation.theta_z,
            0.0,0.0,0.0,0.0,0.0,0.0,
            self.activity_backprojection_parameters.gpu_acceleration,
            self.activity_backprojection_parameters.N_samples,
            self.activity_backprojection_parameters.sample_step,
            self.activity_backprojection_parameters.background_activity,
            0.0,
            self.activity_backprojection_parameters.direction,
            self.activity_backprojection_parameters.block_size,
        )

        self.profiler.rec_backpro_backprojection()
        self.profiler.rec_backprojection(timing)

        self.profiler.tic()
        backprojection_data = backprojection_data * step_size

        self.profiler.rec_backpro_scale()

        self.profiler.tic()
        backprojection = self._make_Image3D_activity(data=backprojection_data)
        self.profiler.rec_backpro_wrap()

        return backprojection

    def get_gradient_activity(
            self,
            activity,
            attenuation=None,
            unit_activity="Bq/mm3",
            transformation_activity=None,
            sparsity=None,
            duration_ms=None,
            subset_size=None,
            subset_mode="random",
            subsets_matrix=None,
            azimuthal_range=None,
            separate_additive_terms=False,
            epsilon=None,
    ):
        # Optionally use only a subset of the projections - use, in order: subsets_matrix; subset_size, subset_mode and az_range
        if subsets_matrix is None:
            if subset_size is not None:
                if subset_size >= 0:
                    subsets_matrix = self._subsets_generator.new_subset(
                        subset_mode, subset_size, azimuthal_range
                    )

        # Optionally use the specified value of epsilon (small number added to the denominator is divisions)
        if epsilon is None:
            epsilon = EPS

        if attenuation is None:
            attenuation = 1.0

        if self.prompts is None:
            print("self.prompts is None, please set prompts. ")
            return
            # FIXME : throw an error

        # By default use the timing information stored in the prompts, however optionally enable overriding
        if duration_ms is None:
            duration_ms = self.prompts.get_duration()
        duration_sec = duration_ms / 1000.0

        # Precompute attenuation*sensitivity - FIXME: do it only is the subset, same for other calculation in proj space
        alpha = self.scale_activity
        prompts = self.prompts
        randoms = self.randoms
        scatter = self.scatter
        sensitivity = self.sensitivity
        if randoms is None:
            randoms = 0.0
        if scatter is None:
            scatter = 0.0
        if sensitivity is None or sensitivity is 1.0:
            att_sens = attenuation
        else:
            att_sens = sensitivity * attenuation

        att_sens = att_sens.get_subset(subsets_matrix)

        # Compute the firt term of the gradient: backprojection of the sensitivity of the scanner
        # If it is requested that the gradient is computed using all the projection measurements, use the
        # memoized normalization. self.get_normalization() takes care of memoization.
        #        gradient_term1 = self.get_normalization(att_sens, transformation_activity, sparsity, duration_ms, subsets_matrix, epsilon=epsilon)
        norm = PET_Projection(self.binning, data=1.0, subsets_matrix=subsets_matrix)
        gradient_term1 = self.backproject_activity(norm)
        print("the two lines above (1033-1034 static.py) are temporary")

        # Compute the second term of the gradient: backprojection of the ratio between the measurement and the projection of
        # current activity estimate... Ordinary Poisson to include scatter and randoms.
        projection = self.project_activity(
            activity,
            unit=unit_activity,
            transformation=transformation_activity,
            sparsity=sparsity,
            subsets_matrix=subsets_matrix,
        )
        prompts_subset = prompts.get_subset(subsets_matrix)
        gradient_term2 = self.backproject_activity(
            prompts_subset
            / (
                    projection
                    + randoms / (att_sens * alpha * duration_sec + epsilon)
                    + scatter / (attenuation * alpha * duration_sec + epsilon)
                    + epsilon
            ),
            transformation=transformation_activity,
        )

        if separate_additive_terms:
            return (gradient_term1, gradient_term2, subsets_matrix)
        else:
            gradient = gradient_term1 + gradient_term2
            return (gradient, subsets_matrix)

    def get_gradient_attenuation(
            self,
            attenuation,
            activity,
            sparsity=None,
            duration_ms=None,
            subset_size=None,
            subset_mode="random",
            subsets_matrix=None,
            azimuthal_range=None,
            epsilon=None,
    ):
        # Optionally use only a subset of the projections - use, in order: subsets_matrix; subset_size, subset_mode and az_range
        if subsets_matrix is None:
            if subset_size is not None:
                if subset_size >= 0:
                    subsets_matrix = self._subsets_generator.new_subset(
                        subset_mode, subset_size, azimuthal_range
                    )

        # Optionally use the specified value of epsilon (small number added to the denominator is divisions)
        if epsilon is None:
            epsilon = EPS

        if attenuation is None:
            attenuation = 1.0

        if self.prompts is None:
            print("self.prompts is None, please set prompts. ")
            return
            # FIXME : throw an error

        # By default use the timing information stored in the prompts, however optionally enable overriding
        if duration_ms is None:
            duration_ms = self.prompts.get_duration()
        duration_sec = duration_ms / 1000.0

        # Precompute attenuation*sensitivity - FIXME: do it only is the subset, same for other calculation in proj space
        alpha = self.scale_activity
        prompts = self.prompts
        randoms = self.randoms
        scatter = self.scatter
        sensitivity = self.sensitivity
        if randoms is None:
            randoms = 0.0
        if scatter is None:
            scatter = 0.0
        if sensitivity is None:
            sensitivity = 1.0

        attenuation_projection = self.project_attenuation(
            attenuation,
            unit="inv_cm",
            transformation=None,
            sparsity=sparsity,
            subsets_matrix=subsets_matrix,
            exponentiate=True,
        )

        # FIXME: transformation = None
        pr_activity = (
                self.project_activity(
                    activity,
                    transformation=None,
                    sparsity=sparsity,
                    subsets_matrix=subsets_matrix,
                )
                * sensitivity
                * attenuation_projection
                * duration_sec
                * alpha
        )
        gradient = self.backproject_attenuation(
            pr_activity
            - prompts
            / (
                    randoms / (pr_activity + epsilon)
                    + scatter / (pr_activity / (sensitivity + epsilon) + epsilon)
                    + 1
            ),
            unit="inv_cm",
            transformation=None,
            sparsity=sparsity,
            subsets_matrix=subsets_matrix,
        )
        return gradient

    def estimate_activity_and_attenuation(
            self,
            activity=None,
            attenuation=None,
            iterations=DEFAULT_RECON_ITERATIONS,
            sparsity=None,
            subset_size=DEFAULT_SUBSET_SIZE,
            subset_mode="random",
            epsilon=None,
            subsets_matrix=None,
            azimuthal_range=None,
            show_progressbar=True,
    ):
        # FIXME: save time: don't compute twice the proj of the attenuation
        activity = self._make_Image3D_activity(data=np.ones(self.activity_shape, dtype=np.float32, order="F"))
        attenuation = self._make_Image3D_attenuation(data=np.zeros(self.attenuation_shape, dtype=np.float32, order="F"))

        if show_progressbar:
            progress_bar = ProgressBar()
            progress_bar.set_percentage(0.1)
        for iteration in range(iterations):
            activity = self.estimate_activity(
                activity,
                attenuation,
                1,
                sparsity,
                subset_size,
                subset_mode,
                epsilon,
                subsets_matrix,
                azimuthal_range,
                show_progressbar=False,
            )
            attenuation = self.estimate_attenuation(
                activity,
                attenuation,
                1,
                sparsity,
                subset_size,
                subset_mode,
                epsilon,
                subsets_matrix,
                azimuthal_range,
                show_progressbar=False,
            )
            if show_progressbar:
                progress_bar.set_percentage((iteration + 1) * 100.0 / iterations)
        if show_progressbar:
            progress_bar.set_percentage(100.0)
        return (activity, attenuation)

    def estimate_attenuation(
            self,
            activity=None,
            attenuation=None,
            iterations=DEFAULT_RECON_ITERATIONS,
            sparsity=None,
            subset_size=DEFAULT_SUBSET_SIZE,
            subset_mode="random",
            epsilon=None,
            subsets_matrix=None,
            azimuthal_range=None,
            show_progressbar=True,
    ):
        if show_progressbar:
            progress_bar = ProgressBar()
            progress_bar.set_percentage(0.1)

        if attenuation is None:
            attenuation = self._make_Image3D_attenuation(data=np.zeros(self.attenuation_shape, dtype=np.float32, order="F"))

        for iteration in range(iterations):
            attenuation = attenuation + self.get_gradient_attenuation(
                attenuation,
                activity,
                sparsity,
                duration_ms=None,
                subset_size=subset_size,
                subset_mode=subset_mode,
                subsets_matrix=subsets_matrix,
                azimuthal_range=azimuthal_range,
                epsilon=epsilon,
            )
            if show_progressbar:
                progress_bar.set_percentage((iteration + 1) * 100.0 / iterations)
        if show_progressbar:
            progress_bar.set_percentage(100.0)
        return attenuation

    def estimate_activity(
            self,
            activity=None,
            attenuation=None,
            iterations=DEFAULT_RECON_ITERATIONS,
            sparsity=None,
            subset_size=DEFAULT_SUBSET_SIZE,
            subset_mode="random",
            epsilon=None,
            subsets_matrix=None,
            azimuthal_range=None,
            show_progressbar=True,
    ):
        # Optionally use the specified value of epsilon (small number added to the denominator is divisions)
        if epsilon is None:
            epsilon = EPS

        if self.prompts is None:
            print("self.prompts is None, please set prompts. ")
            return
            # FIXME : throw an error

        duration_ms = self.prompts.get_duration()

        if show_progressbar:
            progress_bar = ProgressBar()
            progress_bar.set_percentage(0.1)

        # print "Projection of the attenuation. "
        if attenuation is None:
            attenuation = self.attenuation
        if attenuation is not None:
            self.attenuation_projection = self.project_attenuation(
                attenuation
            )  # FIXME: now it's only here that this is defined
        else:
            self.attenuation_projection = 1.0

        if activity is None:
            activity = self._make_Image3D_activity(data=np.ones(self.activity_shape, dtype=np.float32, order="F"))

        # FIXME: use transformation - also notice that transformation_activity is always set to None here

        self._time_profiling_reset()
        for iteration in range(iterations):
            [gradient1, gradient2, subsets_matrix] = self.get_gradient_activity(
                activity,
                self.attenuation_projection,
                transformation_activity=None,
                sparsity=sparsity,
                duration_ms=duration_ms,
                subset_size=subset_size,
                subset_mode=subset_mode,
                subsets_matrix=subsets_matrix,
                azimuthal_range=azimuthal_range,
                separate_additive_terms=True,
                epsilon=epsilon,
            )
            activity = activity * gradient2 / (gradient1 + epsilon)
            if show_progressbar:
                progress_bar.set_percentage((iteration + 1) * 100.0 / iterations)
        if show_progressbar:
            progress_bar.set_percentage(100.0)
        return activity

    def mlem_reconstruction(
            self,
            iterations=10,
            activity=None,
            attenuation_projection=None,
            transformation=None,
            azimuthal_range=None,
            show_progressbar=True,
            title_progressbar=None,
            SaveAll=False,
            SaveDisk=False,
            savepath="",
            debug=False,
    ):
        if show_progressbar:
            if title_progressbar is None:
                title_progressbar = "MLEM Reconstruction"
            progress_bar = ProgressBar(
                color=C.LIGHT_BLUE, title=title_progressbar)
            progress_bar.set_percentage(0.0)

        if activity is None:
            activity = self._make_Image3D_activity(data=np.ones(self.activity_shape, dtype=np.float32, order="F"))

        if self.sensitivity is None:
            sensitivity = self.prompts.copy()
            sensitivity.data = 0.0 * sensitivity.data + 1
            self.set_sensitivity(sensitivity)

        if SaveAll:
            activity_all = np.ones(
                (self.activity_shape[0],
                 self.activity_shape[1],
                 self.activity_shape[2],
                 iterations),
                dtype=np.float32)

        self.profiler.reset()
        for i in range(iterations):
            if not show_progressbar:
                if iterations >= 15:
                    if i == iterations - 1:
                        print("iteration ", (i + 1), "/", iterations)
                    elif i + 1 == 1:
                        print("iteration ", (i + 1), "/", iterations)
                    elif (np.int32(i + 1) / 5) * 5 == i + 1:
                        print("iteration ", (i + 1), "/", iterations)
                else:
                    print("iteration ", (i + 1), "/", iterations)
            subsets_matrix = None
            activity = self.recon_step(activity,
                                       subsets_matrix,
                                       attenuation_projection,
                                       transformation,
                                       debug)

            if SaveAll:
                activity_all[:, :, :, i] = np.flip(activity.data, (0, 1))

            if SaveDisk:
                activity.save_to_file(savepath + 'activity_recon_%d.nii' % i)

            if show_progressbar:
                progress_bar.set_percentage((i + 1) * 100.0 / iterations)

        return activity

    def osem_reconstruction(
            self,
            iterations=10,
            activity=None,
            attenuation_projection=None,
            subset_mode="ordered_axial",
            #subset_size=64,
            subset_number = 21,
            transformation=None,
            azimuthal_range=None,
            show_progressbar=True,
            title_progressbar=None,
            SaveAll=False,
            SaveDisk=False,
            savepath="",
            debug = False,
    ):

        subset_size = self.binning.N_axial // subset_number  # number of projection lines per subset
        iterations *= subset_number
        print(f'subset number: {subset_number}')
        print(f'subset size: {subset_size}')
        print(f'iterations tot: {iterations}')

        if show_progressbar:
            if title_progressbar is None:
                title_progressbar = "OSEM Reconstruction"
            progress_bar = ProgressBar(
                color=C.LIGHT_BLUE, title=title_progressbar)
            progress_bar.set_percentage(0.0)

        if activity is None:
            activity = self._make_Image3D_activity(data=np.ones(self.activity_shape,
                                                                dtype=np.float32, order="F"))

        if self.sensitivity is None:
            sensitivity = self.prompts.copy()
            sensitivity.data = 0.0 * sensitivity.data + 1
            self.set_sensitivity(sensitivity)

        if SaveAll:
            activity_all = np.ones(
                (self.activity_shape[0],
                 self.activity_shape[1],
                 self.activity_shape[2],
                 iterations),
                dtype=np.float32)

        subsets_generator = SubsetGenerator(self.binning.N_azimuthal,
                                            self.binning.N_axial)

        self.profiler.reset()
        for i in range(iterations):
            if not show_progressbar:
                if iterations >= 15:
                    if i == iterations - 1:
                        print("iteration ", (i + 1), "/", iterations)
                    elif i + 1 == 1:
                        print("iteration ", (i + 1), "/", iterations)
                    elif (np.int32(i + 1) / 5) * 5 == i + 1:
                        print("iteration ", (i + 1), "/", iterations)
                else:
                    print("iteration ", (i + 1), "/", iterations)

            subsets_matrix = subsets_generator.new_subset(subset_mode, subset_size, azimuthal_range)
            activity = self.recon_step(activity,
                                       subsets_matrix,
                                       attenuation_projection,
                                       transformation,
                                       debug)

            if SaveAll:
                activity_all[:, :, :, i] = np.flip(activity.data, (0, 1))

            if SaveDisk:
                activity.save_to_file(savepath + 'activity_recon_%d.nii' % i)

            if show_progressbar:
                progress_bar.set_percentage((i + 1) * 100.0 / iterations)

        if SaveAll:
            return activity, activity_all
        else:
            return activity


    def recon_step(
            self,
            activity,
            subsets_matrix=None,
            attenuation_projection=None,
            transformation=None,
            gradient_prior_type=None,
            gradient_prior_args=(),
            debug = False,
    ):
        epsilon = 1e-08

        self.profiler.rec_iteration()
        self.profiler.tic()
        prompts = self.prompts
        if self._use_compression:
            prompts = prompts.uncompress_self()
        self.profiler.rec_uncompress()

        duration_ms = prompts.get_duration()
        if duration_ms is None:
            print( "Acquisition duration unknown (self.prompts.time_bins undefined); assuming 60 minutes. ")
            duration_ms = 1000 * 60 * 60
        duration = duration_ms / 1000.0
        alpha = self.scale_activity

        if attenuation_projection is not None:
            self.profiler.tic()
            attenuation_projection = attenuation_projection.get_subset(subsets_matrix)
            self.profiler.rec_get_subset_attenuation()
        elif self.attenuation_projection is not None:
            self.profiler.tic()
            attenuation_projection = self.attenuation_projection.get_subset(
                subsets_matrix
            )
            self.profiler.rec_get_subset_attenuation()
        elif self.attenuation is not None:
            print("Projecting attenuation")
            self.attenuation_projection = self.project_attenuation(self.attenuation)
            self.profiler.tic()
            attenuation_projection = self.attenuation_projection.get_subset(
                subsets_matrix
            )
            self.profiler.rec_get_subset_attenuation()
            print("Done")
        else:
            attenuation_projection = 1.0

        if self.sensitivity is not None:
            self.profiler.tic()
            sens_x_att = self.sensitivity.get_subset(subsets_matrix)
            self.profiler.rec_get_subset_sensitivity()
            self.profiler.tic()
            sens_x_att = sens_x_att * attenuation_projection
            self.profiler.rec_compose_various()
        else:
            sens_x_att = attenuation_projection
        if np.isscalar(sens_x_att):
            sens_x_att = sens_x_att * np.ones(prompts.data.shape, dtype=np.float32)

        if self.randoms is not None:
            randoms = self.randoms
            if self._use_compression:
                self.profiler.tic()
                randoms = randoms.uncompress_self()
                self.profiler.rec_uncompress()
            self.profiler.tic()
            randoms = randoms.get_subset(subsets_matrix)
            self.profiler.rec_get_subset_randoms()
            self.profiler.tic()
            randoms = (randoms + epsilon) / \
                      (sens_x_att * alpha * duration + epsilon)
            self.profiler.rec_compose_randoms()

        if self.scatter is not None:
            self.profiler.tic()
            mscatter = self.scatter.get_subset(subsets_matrix)
            self.profiler.rec_get_subset_scatter()
            self.profiler.tic()
            mscatter = (mscatter + epsilon) / \
                       (attenuation_projection * alpha * duration + epsilon)
            self.profiler.rec_compose_scatter()

            # Scale scatter: this is used in dynamic and kinetic imaging,
            # when scatter is calculated using the ativity for a time period
            # longer than the current frame:
            if self.scatter.get_duration() is not None:
                if self.scatter.get_duration() > 1e-6:
                    self.profiler.tic()
                    mscatter = mscatter * duration / self.scatter.get_duration()
                    self.profiler.rec_compose_scatter()

        gradient_prior_func = None
        if gradient_prior_type is not None:
            if gradient_prior_type == "smooth":
                gradient_prior_func = self.smoothness_prior
            elif gradient_prior_type == "kinetic":
                gradient_prior_func = self.kinetic_model_prior
            elif gradient_prior_type == "both":
                gradient_prior_func = self.kinetic_plus_smoothing_prior

        # print duration, alpha
        self.profiler.tic()
        norm = self.backproject_activity(
            sens_x_att * alpha * duration,
            transformation=transformation)
        if gradient_prior_func is not None:
            update2 = norm + epsilon - \
                      gradient_prior_func(activity, *gradient_prior_args)
        else:
            update2 = norm + epsilon
        self.profiler.rec_backprojection_norm_total()

        self.profiler.tic()
        projection = self.project_activity(
            activity,
            subsets_matrix=subsets_matrix,
            transformation=transformation)
        self.profiler.rec_projection_activity_total()

        self.profiler.tic()
        p = prompts.get_subset(subsets_matrix)
        self.profiler.rec_get_subset_prompts()

        if self.randoms is not None:
            if self.scatter is not None:
                self.profiler.tic()
                s = (projection + randoms + mscatter + epsilon)
                self.profiler.rec_compose_various()
            else:
                self.profiler.tic()
                s = (projection + randoms + epsilon)
                self.profiler.rec_compose_various()
                self.profiler.tic()
        else:
            if self.scatter is not None:
                self.profiler.tic()
                s = (projection + mscatter + epsilon)
                self.profiler.rec_compose_various()
            else:
                self.profiler.tic()
                s = (projection + epsilon)
                self.profiler.rec_compose_various()

        self.profiler.tic()
        update1 = self.backproject_activity(
            p / s, transformation=transformation)
        self.profiler.rec_backprojection_activity_total()

        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=[20, 3])
            plt.subplot(1, 3, 1), plt.imshow(attenuation_projection.data[:, 0, :, 64], cmap='hot')
            plt.colorbar(), plt.title('attenuation_projection')
            plt.subplot(1, 3, 2), plt.imshow(sens_x_att.data[:, 0, :, 64], cmap='hot')
            plt.colorbar(), plt.title('sens_x_att')
            # plt.subplot(1, 3, 3), plt.imshow(mscatter.data[:, 0, :, 64] / s.data[:, 0, :, 64], cmap='hot')
            # plt.colorbar()
            plt.show()
            plt.figure(figsize=[20, 3])
            plt.subplot(1, 3, 1), plt.imshow(projection.data[:, 0, :, 64], cmap='hot')
            plt.colorbar(), plt.title('projection (estimate)')
            plt.subplot(1, 3, 2), plt.imshow(randoms.data[:, 0, :, 64], cmap='hot')
            plt.colorbar(), plt.title('randoms')
            plt.subplot(1, 3, 3), plt.imshow(mscatter.data[:, 0, :, 64] / s.data[:, 0, :, 64], cmap='hot')
            plt.colorbar(), plt.title('mscatter')
            plt.show()
            plt.figure(figsize=[20, 3])
            plt.subplot(1, 3, 1), plt.imshow(p.data[:, 0, :, 64], cmap='hot')
            plt.colorbar(), plt.title('update_num')
            plt.subplot(1, 3, 2), plt.imshow(s.data[:, 0, :, 64], cmap='hot')
            plt.colorbar(), plt.title('update_den')
            plt.subplot(1, 3, 3), plt.imshow(p.data[:, 0, :, 64] / s.data[:, 0, :, 64], cmap='hot')
            plt.colorbar(), plt.title('update_proj')
            plt.show()
            plt.figure(figsize=[15, 3])
            plt.subplot(1, 3, 1), plt.imshow(activity.data[:, :, 64], cmap='hot')
            plt.colorbar(), plt.title('activity_old')
            plt.subplot(1, 3, 2), plt.imshow(update1.data[:, :, 64], cmap='hot')
            plt.colorbar(), plt.title('update')
            plt.subplot(1, 3, 3), plt.imshow(update2.data[:, :, 64], cmap='hot')
            plt.colorbar(), plt.title('norm')
            plt.show()

        self.profiler.tic()
        # activity = activity - gradient1
        activity = (activity / update2) * update1
        self.profiler.rec_update()

        return activity

    ################ Gradient prior terms for OSL version of osem_step #####################à

    def kinetic_model_prior(self, activity_, model, sigma, sf):
        # derivative of a gaussian prior that enforces similarity between recon
        # and fitting
        gradient = (model - sf * activity_.data) / sigma ** 2
        return gradient

    def smoothness_prior(self, activity_, importance):
        # kernel = ones((3,3,1))
        # kernel[1,1] = -8.0
        kernel = np.asarray([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])  # 3D laplacian operator
        gradient = ndimage.convolve(
            activity_.data, kernel, mode='constant', cval=0.0)
        return importance * gradient

    def kinetic_plus_smoothing_prior(
            self,
            activity_,
            sources,
            sigma,
            sf,
            importance
    ):
        return self.kinetic_model_prior(activity_, sources, sigma, sf) + \
               self.smoothness_prior(activity_, importance)

    ####################################################################################################################

    def quick_inspect(
            self,
            figshape=None,
            index_axial=None,
            index_azimuthal=None,
            index_bin=None):

        if index_axial is None: index_axial = self.binning.N_axial // 2
        if index_azimuthal is None: index_azimuthal = self.binning.N_azimuthal // 2
        if index_bin is None: index_bin = self.binning.N_v // 2
        if figshape is not None and figshape == "default":
            figshape = (15, 5)
        else:
            figshape = (15, 5)

        """Plot a slice of the prompts, randoms and scatter, approapriately scaled to
        verify if the relative scales are correct. """
        if self.randoms is not None and not np.isscalar(self.randoms):
            randoms = self.randoms.to_nd_array()[
                      index_axial, index_azimuthal, :, index_bin]
        else:
            randoms = 0.0

        if self.prompts is not None and not np.isscalar(self.prompts):
            prompts = self.prompts.to_nd_array()[
                      index_axial, index_azimuthal, :, index_bin]
        else:
            prompts = 0.0

        if self.sensitivity is not None:
            if not np.isscalar(self.sensitivity):
                sensitivity = self.sensitivity.to_nd_array()[
                              index_axial, index_azimuthal, :, index_bin]
            else:
                sensitivity = self.sensitivity
        else:
            sensitivity = 1.0

        if self.scatter is not None and not np.isscalar(self.scatter):
            scatter = self.scatter.to_nd_array()[
                      index_axial, index_azimuthal, :, index_bin]
            if self.scatter.get_duration() is not None:
                if self.scatter.get_duration() > 1e-6:
                    if self.prompts.get_duration() is not None:
                        if self.prompts.get_duration() > 1e-6:
                            print(
                                'Scatter duration: %f' %
                                self.scatter.get_duration())
                            print(
                                'Prompts duration: %f' %
                                self.prompts.get_duration())
                            scatter = scatter * self.prompts.get_duration() / self.scatter.get_duration()
        else:
            scatter = 0.0

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=figshape)
            plt.plot(randoms + sensitivity * scatter, 'r',
                     label='rand + sens*scatt')
            plt.plot(prompts - (randoms + sensitivity * scatter), 'g',
                     label='prompts - (rand + sens*scatt')
            plt.plot(prompts,
                     label='prompts')
            plt.legend(fontsize=10)
            plt.xlim((0, self.binning.N_u))
            plt.show()
        except:
            print(
                "quick_inspect uses matplotlib.pyplot to display imaging data. Please install matplotlib. ")

    def brain_crop(self, bin_range=(100, 240)):
        if self._use_compression is True:
            print("Projection cropping currently only works with uncompressed data. ")
            print(
                "In order to enable cropping, please complete the implementation of PET_Projection.get_subset()"
            )
            print("Now PET_Projection.get_subset() only works with uncompressed data. ")
            return
        if hasattr(self, "_cropped"):
            return
        A = bin_range[0]
        B = bin_range[1]
        self.binning.size_u = (1.0 * self.binning.size_u) / self.binning.N_u * (B - A)
        self.binning.N_u = B - A
        if self.prompts is not None:
            self.prompts = self.prompts.crop((A, B))
        if self.randoms is not None:
            self.randoms = self.randoms.crop((A, B))
        if self.scatter is not None:
            self.scatter = self.scatter.crop((A, B))
        if self.sensitivity is not None:
            self.sensitivity = self.sensitivity.crop((A, B))
        self._cropped = True


    def display_geometry(self):
        return display_PET_Projection_geometry()

    def __repr__(self):
        s = "Static PET acquisition:  \n"
        s = s + " - Time_start:                   %s \n" % millisec_to_min_sec(
            self.prompts.get_time_start()
        )
        s = s + " - Time_end:                     %s \n" % millisec_to_min_sec(
            self.prompts.get_time_end()
        )
        s = s + " - Duration:                     %s \n" % millisec_to_min_sec(
            self.prompts.get_time_end() - self.prompts.get_time_start()
        )
        s = s + " - N_counts:                     %d \n" % self.prompts.get_integral()
        s = (
                s
                + " - N_locations:                  %d \n"
                % self.prompts.sparsity.get_N_locations()
        )
        # s = s+" - compression_ratio:            %d \n"%self.prompts.sparsity.compression_ratio
        # s = s+" - listmode_loss:                %d \n"%self.prompts.sparsity.listmode_loss
        s = s + " = Scanner: \n"
        s = s + "     - Name:                     %s \n" % self.scanner.model
        s = s + "     - Manufacturer:             %s \n" % self.scanner.manufacturer
        s = s + "     - Version:                  %s \n" % self.scanner.version
        s = s + " * Binning: \n"
        s = s + "     - N_axial bins:             %d \n" % self.binning.N_axial
        s = s + "     - N_azimuthal bins:         %d \n" % self.binning.N_azimuthal
        s = s + "     - Angles axial:             %s \n" % array_to_string(
            self.binning.angles_axial
        )
        s = s + "     - Angles azimuthal:         %s \n" % array_to_string(
            self.binning.angles_azimuthal
        )
        s = s + "     - Size_u:                   %f \n" % self.binning.size_u
        s = s + "     - Size_v:                   %f \n" % self.binning.size_v
        s = s + "     - N_u:                      %s \n" % self.binning.N_u
        s = s + "     - N_v:                      %s \n" % self.binning.N_v
        return s

    def _repr_html_(self):
        if not has_ipy_table:
            return "Please install ipy_table."
        if self.scanner is not None:
            table_data = [
                ["Time_start", millisec_to_min_sec(self.prompts.get_time_start())],
                ["Time_end", millisec_to_min_sec(self.prompts.get_time_end())],
                [
                    "Duration",
                    millisec_to_min_sec(
                        self.prompts.get_time_end() - self.prompts.get_time_start()
                    ),
                ],
                ["N_counts", pretty_print_large_number(self.prompts.get_integral())],
                [
                    "N_locations",
                    pretty_print_large_number(self.prompts.sparsity.get_N_locations),
                ],
                # ['compression_ratio',print_percentage(self.compression_ratio)],
                # ['listmode_loss',self.listmode_loss],
                ["Scanner Name", self.scanner.model],
                ["Scanner Manufacturer", self.scanner.manufacturer],
                ["Scanner Version", self.scanner.version],
            ]
        else:
            table_data = [
                ["Time_start", millisec_to_min_sec(self.prompts.get_time_start())],
                ["Time_end", millisec_to_min_sec(self.prompts.get_time_end())],
                [
                    "Duration",
                    millisec_to_min_sec(
                        self.prompts.get_time_end() - self.prompts.get_time_start()
                    ),
                ],
                ["N_counts", pretty_print_large_number(self.prompts.get_integral())],
                [
                    "N_locations",
                    pretty_print_large_number(self.prompts.sparsity.get_N_locations()),
                ],
            ]
            # ['compression_ratio',print_percentage(self.compression_ratio)],
            # ['listmode_loss',self.listmode_loss], ]
        table = ipy_table.make_table(table_data)
        table = ipy_table.apply_theme("basic_left")
        # table = ipy_table.set_column_style(0, color='lightBlue')
        table = ipy_table.set_global_style(float_format="%3.3f")
        return table._repr_html_()


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################


class PET_Multi2D_Scan(PET_Static_Scan):
    def __init__(self):
        self.n_slices = 0
        self.scatter_duration = None
        self.prompts_duration = None
        self.attenuation_projection = None
        PET_Static_Scan.__init__(self)

    def set_activity_size(self, activity_size):
        self.activity_size = (activity_size[0], activity_size[1], self.n_slices)

    def set_activity_shape(self, activity_shape):
        self.activity_shape = (activity_shape[0], activity_shape[1], self.n_slices)

    def set_attenuation_projection(self, attenuation_projection):
        self.attenuation_projection = attenuation_projection

    def set_prompts_duration(self, duration_array):
        self.prompts_duration = duration_array

    def set_scatter_duration(self, duration_array):
        self.scatter_duration = duration_array

    def set_number_of_slices(self, n_slices):
        self.n_slices = n_slices
        self.binning.N_azimuthal = 1
        self.binning.angles_azimuthal = np.int32([0])
        self.binning.N_v = self.n_slices + 1
        self.binning.size_v = self.n_slices
        self.set_binning(self.binning)
        if self.scatter_duration is None:
            self.scatter_duration = np.ones([self.n_slices + 1])
        if self.prompts_duration is None:
            self.prompts_duration = np.ones([self.n_slices + 1])

    def osem_reconstruction(
        self,
        iterations=10,
        activity=None,
        subset_mode="random",
        subset_size=64,
        transformation=None,
    ):
        if activity is None:
            activity = self._make_Image3D_activity(
                np.ones(self.activity_shape, dtype=np.float32, order="F")
            )
            subsets_generator = SubsetGenerator(
                self.binning.N_azimuthal, self.binning.N_axial
            )

        if self.sensitivity is None:
            sensitivity = self.prompts.copy()
            sensitivity.data = 0.0 * sensitivity.data + 1
            self.set_sensitivity(sensitivity)

        for i in range(iterations):
            print(i)
            subsets_matrix = subsets_generator.new_subset(subset_mode, subset_size)
            activity = self.osem_step(activity, subsets_matrix, transformation)
        return activity

    def mlem_reconstruction(self, iterations=10, activity=None, transformation=None):
        if activity is None:
            activity = self._make_Image3D_activity(
                np.ones(self.activity_shape, dtype=np.float32, order="F")
            )

        if self.sensitivity is None:
            self.set_sensitivity(1.0)

        for i in range(iterations):
            print(i)
            subsets_matrix = None
            activity = self.osem_step(activity, subsets_matrix, transformation)
        return activity

    def osem_step(self, activity, subsets_matrix, transformation):
        epsilon = 1e-08

        prompts = self.prompts
        if self._use_compression:
            prompts = prompts.uncompress_self()
        prompts = prompts.get_subset(subsets_matrix)

        # make matrices of duration arrays
        prompts_duration = np.tile(
            self.prompts_duration / 1000.0,
            (prompts.data.shape[0], prompts.data.shape[1], prompts.data.shape[2], 1),
        )
        scatter_duration = np.tile(
            self.scatter_duration / 1000.0,
            (prompts.data.shape[0], prompts.data.shape[1], prompts.data.shape[2], 1),
        )
        alpha = self.scale_activity

        if self.attenuation_projection is not None:
            attenuation_projection = self.attenuation_projection.get_subset(
                subsets_matrix
            )
        else:
            attenuation_projection = 1.0

        if self.sensitivity is not None:
            if np.isscalar(self[t].sensitivity):
                sens_x_att = self[t].sensitivity * attenuation_projection
            else:
                sens_x_att = (
                    self.sensitivity.get_subset(subsets_matrix) * attenuation_projection
                )
        else:
            sens_x_att = attenuation_projection

        if self.randoms is not None:
            randoms = self.randoms
            if self._use_compression:
                randoms = randoms.uncompress_self()
            randoms = (randoms.get_subset(subsets_matrix) + epsilon) / (
                sens_x_att * alpha * prompts_duration + epsilon
            )

        if self.scatter is not None:
            mscatter = (self.scatter.get_subset(subsets_matrix) + epsilon) / (
                attenuation_projection * alpha * prompts_duration + epsilon
            )
            # Scale scatter: this is used in dynamic and kinetic imaging, when scatter is calculated using the ativity for a time period longer than the current frame:
            mscatter = mscatter * prompts_duration / scatter_duration

        norm = self.backproject_activity(
            sens_x_att * alpha * prompts_duration, transformation=transformation
        )

        projection = self.project_activity(
            activity, subsets_matrix=subsets_matrix, transformation=transformation
        )

        if self.randoms is not None:
            if self.scatter is not None:
                update1 = self.backproject_activity(
                    prompts / (projection + randoms + mscatter + epsilon),
                    transformation=transformation,
                )
            else:
                update1 = self.backproject_activity(
                    prompts / (projection + randoms + epsilon),
                    transformation=transformation,
                )

        else:
            if self.scatter is not None:
                update1 = self.backproject_activity(
                    prompts / (projection + mscatter + epsilon),
                    transformation=transformation,
                )
            else:
                update1 = self.backproject_activity(
                    prompts / (projection + epsilon), transformation=transformation
                )

        # activity = activity - gradient1
        activity = (activity / (norm + epsilon)) * update1

        return activity

    def quick_inspect(self, index_axial=0, index_slice=0):
        index_azimuthal = 0
        if self.randoms is not None and not np.isscalar(self.randoms):
            randoms = self.randoms.to_nd_array()[
                index_axial, index_azimuthal, :, index_slice + 1
            ]
        else:
            randoms = 0.0
        if self.prompts is not None and not np.isscalar(self.prompts):
            prompts = self.prompts.to_nd_array()[
                index_axial, index_azimuthal, :, index_slice + 1
            ]
        else:
            prompts = 0.0
        if self.sensitivity is not None:
            if not np.isscalar(self.sensitivity):
                sensitivity = self.sensitivity.to_nd_array()[
                    index_axial, index_azimuthal, :, index_slice + 1
                ]
            else:
                sensitivity = self.sensitivity
        else:
            sensitivity = 1.0
        if self.scatter is not None and not np.isscalar(self.scatter):
            scatter = self.scatter.to_nd_array()
            prompts_duration = np.tile(
                self.prompts_duration / 1000.0,
                (
                    self.prompts.data.shape[0],
                    self.prompts.data.shape[1],
                    self.prompts.data.shape[2],
                    1,
                ),
            )
            scatter_duration = np.tile(
                self.scatter_duration / 1000.0,
                (
                    self.prompts.data.shape[0],
                    self.prompts.data.shape[1],
                    self.prompts.data.shape[2],
                    1,
                ),
            )
            scatter = scatter * prompts_duration / scatter_duration
            scatter = scatter[index_axial, index_azimuthal, :, index_slice + 1]
        else:
            scatter = 0.0
        if has_pylab:
            pylab.plot(prompts - randoms)
            pylab.hold(1)
            pylab.plot(sensitivity * scatter, "g")
        else:
            print(
                "quick_inspect uses Pylab to display imaging data. Please install Pylab. "
            )


class PET_Dynamic_Scan(PET_Static_Scan):
    """PET Dynamic Scan. This is useful for motion correction and for kinetic imaging, or both. """

    def __init__(self):
        self._dynamic = []  # Sequence of static scans, one per time bin
        self.time_bins = []  # Time binning
        self.static = None
        PET_Static_Scan.__init__(self)

    def import_listmode(
        self,
        hdr_filename,
        time_range_ms=[0, None],
        data_filename=None,
        motion_files_path=None,
        display_progress=False,
    ):
        """Load prompts data from a listmode file. """
        # Optionally load_interfile motion information:
        if motion_files_path:
            vNAV = load_vnav_mprage(motion_files_path)
            self.__motion_events = vNAV
            if time_range_ms[1] is not None:
                raise ValueError(
                    "Either time_bins or motion_files_path should be defined, not both. "
                )
            time_range_ms = (
                self.__motion_events.extract_motion_events().sum()
            )  # FIXME: this is not right, implement binning of sinogram according to motion events (this requires modifying the C code that does the binning and passing the right parameters: the list_mode trigger number corresponding to the beginning and end of each sinogram)

        print_debug("- Loading dynamic PET data from listmode file " + str(hdr_filename))
        hdr = interfile.load_interfile(hdr_filename)
        # Extract information from the listmode header

        # 1) Guess the path of the listmode data file, if not specified or mis-specified;
        #  1 - see if the specified listmode data file exists
        if data_filename is not None:
            data_filename = data_filename.replace("/", os.path.sep).replace(
                "\\", os.path.sep
            )  # cross platform compatibility
            if not os.path.exists(data_filename):
                raise FileNotFound("listmode data", data_filename)
        #  2 - if the listmode data file is not specified, try with the name (and full path) contained in the listmode header
        data_filename = hdr["name of data file"]["value"]
        data_filename = data_filename.replace("/", os.path.sep).replace(
            "\\", os.path.sep
        )  # cross platform compatibility
        if not os.path.exists(data_filename):
            #  3 - if it doesn't exist, look in the same path as the header file for the listmode data file with name specified in the listmode header file
            data_filename = (
                os.path.split(hdr_filename)[0]
                + os.path.sep
                + os.path.split(data_filename)[-1]
            )
            if not os.path.exists(data_filename):
                #  4 - if it doesn't exist, look in the same path as the header file for the listmode data file with same name as the listmode header file, replacing the extension: ".l.hdr -> .l"
                if hdr_filename.endswith(".l.hdr"):
                    data_filename = hdr_filename.replace(".l.hdr", ".l")
                    if not os.path.exists(data_filename):
                        raise FileNotFound("listmode data", data_filename)
                #  5 - if it doesn't exist, look in the same path as the header file for the listmode data file with same name as the listmode header file, replacing the extension: ".hdr -> .l"
                elif hdr_filename.endswith(".hdr"):
                    data_filename = hdr_filename.replace(".hdr", ".l")
                    if not os.path.exists(data_filename):
                        raise FileNotFound("listmode data", data_filename)

        # 2) Determine the duration of the acquisition
        n_packets = hdr["total listmode word counts"]["value"]
        scan_duration = hdr["image duration"]["value"] * 1000  # milliseconds

        # 3) determine scanner parameters
        n_radial_bins = hdr["number of projections"]["value"]
        n_angles = hdr["number of views"]["value"]
        n_rings = hdr["number of rings"]["value"]
        max_ring_diff = hdr["maximum ring difference"]["value"]
        n_sinograms = (
            n_rings + 2 * n_rings * max_ring_diff - max_ring_diff ** 2 - max_ring_diff
        )

        # Determine the time binning
        if time_range_ms[1] is None:
            time_range_ms = np.int32(np.linspace(0, scan_duration, DEFAULT_N_TIME_BINS + 1))
        elif np.isscalar(
            time_range_ms
        ):  # time_bins in this case indicates the number of time bins
            time_range_ms = np.int32(np.linspace(0, scan_duration, time_range_ms + 1))

        # Display information
        print_debug(" - Number of packets:    %d       " % n_packets)
        print_debug(" - Scan duration:        %d [sec] " % (scan_duration / 1000.0))
        print_debug(" - Listmode data file:   %s       " % data_filename)
        print_debug(" - Listmode header file: %s       " % hdr_filename)
        print_debug(" - Number of time bins:  %d       " % (len(time_range_ms) - 1))
        print_debug(" - Time start:           %f [sec] " % (time_range_ms[0] / 1000.0))
        print_debug(" - Time end:             %f [sec] " % (time_range_ms[-1] / 1000.0))
        print_debug(" - time_range_ms:        %s       " % str(time_range_ms))
        print_debug(" - n_radial_bins:        %d       " % n_radial_bins)
        print_debug(" - n_angles:             %d       " % n_angles)
        print_debug(" - n_angles:             %d       " % n_sinograms)

        if display_progress:
            progress_bar = ProgressBar()
            progress_callback = progress_bar.set_percentage
        else:

            def progress_callback(value):
                pass

        # Load the listmode data
        M = self.scanner.michelogram
        R = self.scanner.listmode.load_listmode(
            data_filename,
            n_packets,
            time_range_ms,
            self.binning,
            n_radial_bins,
            n_angles,
            n_sinograms,
            M.span,
            M.segments_sizes,
            M.michelogram_sinogram,
            M.michelogram_plane,
            progress_callback,
        )
        if display_progress:
            progress_bar.set_percentage(100)

        #        self.dynamic_inflation = R['dynamic_inflation']

        N_time_bins = R["N_time_bins"]
        time_start = R["time_start"]
        time_end = R["time_end"]
        # self.time_bins = time_bins[time_start:N_time_bins+1]  #the actual time bins are less than the requested time bins, truncate time_bins
        self.time_bins = time_range_ms

        # Make list of PET_Static_Scan objects, one per bin
        self._dynamic = []
        for t in range(N_time_bins):
            PET_t = PET_Static_Scan()
            PET_t.use_compression(self._use_compression)
            PET_t.use_gpu(self._use_gpu)
            PET_t.set_scanner(self.scanner.__class__)
            PET_t.set_binning(self.binning)
            PET_t._load_static_measurement(t)
            # make list of static scans
            self._dynamic.append(PET_t)
            # also make one attribut for each static scan
            setattr(self, "frame%d" % t, self._dynamic[t])
            # set activity shape and size and attenuation shape and size
            PET_t.set_activity_size(self.activity_size)
            PET_t.set_activity_shape(self.activity_shape)
            PET_t.set_attenuation_size(self.activity_size)
            PET_t.set_attenuation_shape(self.activity_shape)

        # Make a global PET_Static_Scan object
        self.static = PET_Static_Scan()
        self.static.use_compression(self._use_compression)
        self.static.use_gpu(self._use_gpu)
        self.static.set_scanner(self.scanner.__class__)
        self.static.set_binning(self.binning)
        self.static._load_static_measurement()
        self.static.set_activity_size(self.activity_size)
        self.static.set_activity_shape(self.activity_shape)
        self.static.set_attenuation_size(self.activity_size)
        self.static.set_attenuation_shape(self.activity_shape)

        # Free structures listmode data
        self.scanner.listmode.free_memory()

        # Construct ilang model
        self._construct_ilang_model()

    def import_prompts(self):
        print(
            "Not implemented: should load_interfile prompts for multiple time frames and also set self.static.prompts to the integral."
        )

    def import_randoms(self):
        print(
            "Not implemented: should load_interfile randoms for multiple time frames and also set self.static.randoms to the integral."
        )

    def export_prompts(self):
        print("Not implemented.")

    def export_randoms(self):
        print("Not implemented.")

    def get_prompts(self):
        prompts = []
        for frame in self:
            prompts.append(frame.prompts)
        return prompts

    def set_prompts(self, prompts_list):
        N_time_bins = len(prompts_list)
        if len(self) == N_time_bins:
            print(
                "PET_Dynamic_Scan.set_prompts(): Number of sinograms matches current setup; no re-initialization."
            )
            for t in range(N_time_bins):
                self[t].set_prompts(prompts_list[t])
        else:
            print(
                "PET_Dynamic_Scan.set_prompts(): Number of sinograms does not match current setup; re-initialization."
            )
            self._dynamic = []
            total_prompts = prompts_list[0] * 0
            for t in range(N_time_bins):
                PET_t = PET_Static_Scan()
                PET_t.use_compression(self._use_compression)
                PET_t.use_gpu(self._use_gpu)
                PET_t.set_scanner(self.scanner.__class__)
                PET_t.set_binning(self.binning)
                PET_t.set_prompts(prompts_list[t])
                self._dynamic.append(PET_t)
                setattr(self, "frame%d" % t, self._dynamic[t])
                # FIXME: remove excess frames if new N_time_bins is less than previous
                PET_t.set_activity_size(self.activity_size)
                PET_t.set_activity_shape(self.activity_shape)
                PET_t.set_attenuation_size(self.activity_size)
                PET_t.set_attenuation_shape(self.activity_shape)
                total_prompts += prompts_list[t]

            # Make a global PET_Static_Scan object
            self.static = PET_Static_Scan()
            self.static.use_compression(self._use_compression)
            self.static.use_gpu(self._use_gpu)
            self.static.set_scanner(self.scanner.__class__)
            self.static.set_binning(self.binning)
            self.static.set_prompts(total_prompts)
            self.static.set_activity_size(self.activity_size)
            self.static.set_activity_shape(self.activity_shape)
            self.static.set_attenuation_size(self.activity_size)
            self.static.set_attenuation_shape(self.activity_shape)

            # Construct ilang model
            self._construct_ilang_model()

    def get_randoms(self):
        randoms = []
        for frame in self:
            randoms.append(frame.randoms)
        return randoms

    def set_randoms(self, randoms_list):
        N_time_bins = len(randoms_list)
        if len(self) == N_time_bins:
            print(
                "PET_Dynamic_Scan.set_randoms(): Number of sinograms matches current setup; no re-initialization."
            )
            for t in range(N_time_bins):
                self[t].set_randoms(randoms_list[t])
        else:
            print(
                "PET_Dynamic_Scan.set_randoms(): Number of sinograms does not match current setup; re-initialization."
            )
            self._dynamic = []
            total_randoms = randoms_list[0] * 0
            for t in range(N_time_bins):
                PET_t = PET_Static_Scan()
                PET_t.use_compression(self._use_compression)
                PET_t.use_gpu(self._use_gpu)
                PET_t.set_scanner(self.scanner.__class__)
                PET_t.set_binning(self.binning)
                PET_t.set_randoms(randoms_list[t])
                self._dynamic.append(PET_t)
                setattr(self, "frame%d" % t, self._dynamic[t])
                # FIXME: remove excess frames if new N_time_bins is less than previous
                PET_t.set_activity_size(self.activity_size)
                PET_t.set_activity_shape(self.activity_shape)
                PET_t.set_attenuation_size(self.activity_size)
                PET_t.set_attenuation_shape(self.activity_shape)
                total_randoms += randoms_list[t]

            # Make a global PET_Static_Scan object
            self.static = PET_Static_Scan()
            self.static.use_compression(self._use_compression)
            self.static.use_gpu(self._use_gpu)
            self.static.set_scanner(self.scanner.__class__)
            self.static.set_binning(self.binning)
            self.static.set_randoms(total_randoms)
            self.static.set_activity_size(self.activity_size)
            self.static.set_activity_shape(self.activity_shape)
            self.static.set_attenuation_size(self.activity_size)
            self.static.set_attenuation_shape(self.activity_shape)

            # Construct ilang model
            self._construct_ilang_model()

    def get_scatter(self):
        return self.static.scatter

    def set_scatter(self, scatter, duration_ms=None):
        self.scatter = scatter
        self.static.set_scatter(scatter, duration_ms)
        for frame in range(len(self)):
            self[frame].set_scatter(scatter, duration_ms)

    def get_sensitivity(self):
        return self.static.sensitivity

    def set_sensitivity(self, sensitivity):
        # FIXME: verify type: PET_projection or nd_array (the latter only in full sampling mode)
        self.sensitivity = sensitivity
        self.static.set_sensitivity(sensitivity)
        for frame in range(len(self)):
            self[frame].set_sensitivity(sensitivity)

    def get_attenuation(self):
        return self.static.attenuation

    def set_attenuation(self, attenuation):
        # FIXME: verify type
        self.static.set_attenuation(attenuation)
        for frame in range(len(self)):
            self[frame].set_attenuation(attenuation)

    def get_attenuation_projection(self):
        return self.static.attenuation_projection

    def set_attenuation_projection(self, attenuation_projection):
        # FIXME: verify type
        self.attenuation_projection = attenuation_projection
        self.static.set_attenuation_projection(attenuation_projection)
        for frame in range(len(self)):
            self[frame].set_attenuation_projection(attenuation_projection)

    def use_compression(self, use_it):
        self._use_compression = use_it

    def set_activity_shape(self, activity_shape):
        if not len(activity_shape) == 3:
            print("Invalid activity shape")  # FIXME: raise invalid input error
        else:
            self.activity_shape = activity_shape
            if hasattr(self, "static"):
                if self.static is not None:
                    self.static.set_activity_shape(activity_shape)
            for frame in range(len(self)):
                self[frame].set_activity_shape(activity_shape)

    def set_activity_size(self, activity_size):
        if not len(activity_size) == 3:
            print("Invalid activity size")  # FIXME: raise invalid input error
        else:
            self.activity_size = activity_size
            if hasattr(self, "static"):
                if self.static is not None:
                    self.static.set_activity_size(activity_size)
            for frame in range(len(self)):
                self[frame].set_activity_size(activity_size)

    def set_attenuation_shape(self, attenuation_shape):
        if not len(attenuation_shape) == 3:
            print("Invalid attenuation shape")  # FIXME: raise invalid input error
        else:
            self.attenuation_shape = attenuation_shape
            if hasattr(self, "static"):
                if self.static is not None:
                    self.static.set_attenuation_shape(attenuation_shape)
            for frame in range(len(self)):
                self[frame].set_attenuation_shape(attenuation_shape)

    def set_attenuation_size(self, attenuation_size):
        if not len(attenuation_size) == 3:
            print("Invalid attenuation size")  # FIXME: raise invalid input error
        else:
            self.attenuation_size = attenuation_size
            if hasattr(self, "static"):
                if self.static is not None:
                    self.static.set_attenuation_size(attenuation_size)
            for frame in range(len(self)):
                self[frame].set_attenuation_size(attenuation_size)

    def brain_crop(self, bin_range=(100, 240)):
        if self._use_compression is True:
            print("Cropping currently only works with uncompressed data. ")
            return
        self.static.brain_crop(bin_range)
        for frame in range(len(self)):
            self[frame].brain_crop(bin_range)

    def osem_reconstruction(
        self,
        iterations=10,
        activity=None,
        subset_mode="random",
        subset_size=64,
        transformations=None,
    ):
        for frame in range(len(self)):
            if activity is not None:
                activity_init = activity[frame]
            else:
                if hasattr(self[frame], "activity"):
                    activity_init = self[frame].activity
                else:
                    activity_init = None
            if transformations is not None:
                transformation = transformations[frame]
            else:
                transformation = None
            print(("Reconstructing frame %d/%d" % (frame, len(self))))
            activity_recon = self[frame].osem_reconstruction(
                iterations=iterations,
                activity=activity_init,
                subset_mode=subset_mode,
                subset_size=subset_size,
                transformation=transformation,
            )
            self[frame].activity = activity_recon

    def osem_reconstruction_4D(
        self,
        iterations=10,
        activity=None,
        subset_mode="random",
        subset_size=64,
        transformations=None,
    ):
        if activity is None:
            activity = self._make_Image3D_activity(
                np.ones(self.activity_shape, dtype=np.float32, order="F")
            )

        if self.sensitivity is None:
            self.set_sensitivity(1.0)

        subsets_generator = SubsetGenerator(
            self.binning.N_azimuthal, self.binning.N_axial
        )

        for i in range(iterations):
            print(i)
            activity = self.osem_step_4D(
                activity, subsets_generator, subset_mode, subset_size, transformations
            )
        return activity

    def osem_step_4D(
        self,
        activity,
        subsets_generator,
        subset_mode="random",
        subset_size=64,
        transformations=None,
    ):
        epsilon = 1e-08
        norm = self._make_Image3D_activity(
            np.zeros(self.activity_shape, dtype=np.float32, order="F")
        )
        update1 = self._make_Image3D_activity(
            np.zeros(self.activity_shape, dtype=np.float32, order="F")
        )

        for t in range(len(self)):
            subsets_matrix = subsets_generator.new_subset(subset_mode, subset_size)

            prompts = self[t].prompts
            if self[t]._use_compression:
                prompts = prompts.uncompress_self()

            duration_ms = prompts.get_duration()
            if duration_ms is None:
                duration_ms = 1000 * 60 * 60
            duration = duration_ms / 1000.0
            alpha = self[t].scale_activity

            # Compute projections of the attenuation map on the fly; too memory consuming to precompute the
            # projection for each time frame - precomputing makes PET_Static_Scan more efficient, but in Dynamic, it
            # requires too much memory in practical applications.

            if self[t].attenuation is not None:
                attenuation_projection = self[t].project_attenuation(self[t].attenuation)
                attenuation_projection = attenuation_projection.get_subset(
                    subsets_matrix
                )
            else:
                attenuation_projection = 1.0

            if self[t].sensitivity is not None:
                if np.isscalar(self[t].sensitivity):
                    sens_x_att = self[t].sensitivity * attenuation_projection
                else:
                    sens_x_att = (
                        self[t].sensitivity.get_subset(subsets_matrix)
                        * attenuation_projection
                    )
            else:
                sens_x_att = attenuation_projection
            if np.isscalar(sens_x_att):
                sens_x_att = sens_x_att * np.ones(prompts.data.shape, dtype=np.float32)

            if self[t].randoms is not None:
                randoms = self.randoms
                if self[t]._use_compression:
                    randoms = randoms.uncompress_self()
                randoms = (randoms.get_subset(subsets_matrix) + epsilon) / (
                    sens_x_att * alpha * duration + epsilon
                )

            if self[t].scatter is not None:
                mscatter = (self.scatter.get_subset(subsets_matrix) + epsilon) / (
                    attenuation_projection * alpha * duration + epsilon
                )
                # Scale scatter: this is used in dynamic and kinetic imaging, when scatter is calculated using the ativity for a time period longer than the current frame:
                if self[t].scatter.get_duration() is not None:
                    if self[t].scatter.get_duration() > 1e-6:
                        mscatter = mscatter * duration / self[t].scatter.get_duration()

            norm += self[t].backproject_activity(
                sens_x_att * alpha * duration, transformation=transformations[t]
            )

            projection = self[t].project_activity(
                activity,
                subsets_matrix=subsets_matrix,
                transformation=transformations[t],
            )

            if self[t].randoms is not None:
                if self[t].scatter is not None:
                    update1 += self[t].backproject_activity(
                        prompts.get_subset(subsets_matrix)
                        / (projection + randoms + mscatter + epsilon),
                        transformation=transformations[t],
                    )
                else:
                    update1 += self[t].backproject_activity(
                        prompts.get_subset(subsets_matrix)
                        / (projection + randoms + epsilon),
                        transformation=transformations[t],
                    )

            else:
                if self[t].scatter is not None:
                    update1 += self[t].backproject_activity(
                        prompts.get_subset(subsets_matrix)
                        / (projection + mscatter + epsilon),
                        transformation=transformations[t],
                    )
                else:
                    update1 += self[t].backproject_activity(
                        prompts.get_subset(subsets_matrix) / (projection + epsilon),
                        transformation=transformations[t],
                    )

        activity = (activity / (norm + epsilon)) * update1
        return activity

    def get_activity_as_array(self):
        activities = []
        for frame in range(len(self)):
            activity = self[frame].activity.data
            activities.append(activity)
        return np.asarray(activities)

    def get_2D_slices(self, slices=(62, 63, 64, 65, 66), azimuthal_index=5):
        N_time_bins = len(self)

        pet = PET_Multi2D_Scan()
        pet.use_compression(False)
        pet.set_scanner(self.scanner.__class__)
        pet.set_number_of_slices(N_time_bins)
        pet.set_activity_shape((self.activity_shape[0], self.activity_shape[1]))
        pet.set_activity_size((self.activity_size[0], self.activity_size[1]))

        # extract slice of prompts, scatter, randoms, sensitivity
        prompts = np.zeros(
            [pet.binning.N_axial, pet.binning.N_u, N_time_bins + 1], dtype=np.float32
        )
        randoms = np.zeros(
            [pet.binning.N_axial, pet.binning.N_u, N_time_bins + 1], dtype=np.float32
        )
        scatter = np.zeros(
            [pet.binning.N_axial, pet.binning.N_u, N_time_bins + 1], dtype=np.float32
        )
        sensitivity = np.ones(
            [pet.binning.N_axial, pet.binning.N_u, N_time_bins + 1], dtype=np.float32
        )
        attenuation_projection = np.ones(
            [pet.binning.N_axial, pet.binning.N_u, N_time_bins + 1], dtype=np.float32
        )
        for t in range(N_time_bins):
            if hasattr(self[t], "prompts"):
                if self[t].prompts is not None:
                    prompts[:, :, t + 1] = (
                        self[t]
                        .prompts.to_nd_array()[:, azimuthal_index, :, slices]
                        .sum(0)
                        .squeeze()
                    )
        for t in range(N_time_bins):
            if hasattr(self[t], "randoms"):
                if self[t].randoms is not None:
                    randoms[:, :, t + 1] = (
                        self[t]
                        .randoms.to_nd_array()[:, azimuthal_index, :, slices]
                        .sum(0)
                        .squeeze()
                    )
        # FIXME: implement scatter scale in PET_Static_Scan and use it here;
        # this will reduce memory for scatter by storing scatter projection only once.
        for t in range(N_time_bins):
            if hasattr(self[t], "scatter"):
                if self[t].scatter is not None:
                    scatter[:, :, t + 1] = (
                        self[t]
                        .scatter.to_nd_array()[:, azimuthal_index, :, slices]
                        .sum(0)
                        .squeeze()
                    )
        for t in range(N_time_bins):
            if hasattr(self[t], "sensitivity"):
                if self[t].sensitivity is not None:
                    sensitivity[:, :, t + 1] = (
                        self[t]
                        .sensitivity.to_nd_array()[:, azimuthal_index, :, slices]
                        .mean(0)
                        .squeeze()
                    )

        # project attenuation and extract slice of the projection
        # FIXME: implement .set_attenuation_projection() in PET_Static_SCAN and use in recon if loaded
        if hasattr(self, "attenuation_projection"):
            if self.attenuation_projection is not None:
                att = (
                    self.attenuation_projection.to_nd_array()[
                        :, azimuthal_index, :, slices
                    ]
                    .mean(0)
                    .squeeze()
                )
                for t in range(N_time_bins):
                    attenuation_projection[:, :, t + 1] = att

        prompts = PET_Projection(pet.binning, prompts)
        randoms = PET_Projection(pet.binning, randoms)
        sensitivity = PET_Projection(pet.binning, sensitivity)
        scatter = PET_Projection(pet.binning, scatter)
        attenuation_projection = PET_Projection(pet.binning, attenuation_projection)

        pet.set_prompts(prompts)
        pet.set_scatter(scatter)
        pet.set_randoms(randoms)
        pet.set_sensitivity(sensitivity)
        pet.set_attenuation_projection(attenuation_projection)

        prompts_duration = np.ones([N_time_bins + 1])
        scatter_duration = np.ones([N_time_bins + 1])
        for t in range(N_time_bins):
            prompts_duration[1 + t] = self[t].prompts.get_duration()
            scatter_duration[1 + t] = self[t].scatter.get_duration()
        pet.set_prompts_duration(prompts_duration)
        pet.set_scatter_duration(scatter_duration)

        return pet

    '''
    def _construct_ilang_model(self):
        # define the ilang probabilistic model
        self.ilang_model = PET_Dynamic_Poisson(self)
        # construct the Directed Acyclical Graph
        self.ilang_graph = ProbabilisticGraphicalModel(["lambda", "alpha", "counts"])
        # self.ilang_graph.set_nodes_given(['counts','alpha'],True)
        # self.ilang_graph.add_dependence(self.ilang_model,{'lambda':'lambda','alpha':'alpha','z':'counts'})
    '''

    def __repr__(self):
        """Display information about Dynamic_PET_Scan"""
        s = "Dynamic PET acquisition:  \n"
        s = s + " - N_time_bins:                  %d \n" % len(self.time_bins)
        s = s + " - Time_start:                   %s \n" % millisec_to_min_sec(
            self.time_bins[0]
        )
        s = s + " - Time_end:                     %s \n" % millisec_to_min_sec(
            self.time_bins[-1]
        )
        s = (
            s
            + " - N_counts:                     %d \n"
            % self.static.prompts.get_integral()
        )
        s = (
            s
            + " - N_locations:                  %d \n"
            % self.static.prompts.sparsity.get_N_locations()
        )
        s = (
            s
            + " - compression_ratio:            %d \n"
            % self.static.prompts.get_compression_ratio()
        )
        #        s = s+" - dynamic_inflation:            %d \n"%self.dynamic_inflation
        s = (
            s
            + " - listmode_loss:                %d \n"
            % self.static.prompts.get_listmode_loss()
        )
        s = s + " - Mean time bin duration:       %d [sec] \n" % 0  # FIXME
        if self.scanner is not None:
            s = s + " * Scanner: \n"
            s = s + "     - Name:                     %s \n" % self.scanner.model
            s = s + "     - Manufacturer:             %s \n" % self.scanner.manufacturer
            s = s + "     - Version:                  %s \n" % self.scanner.version
        if self.binning is not None:
            s = s + " * Binning: \n"
            s = s + "     - N_axial bins:             %d \n" % self.binning.N_axial
            s = s + "     - N_azimuthal bins:         %d \n" % self.binning.N_azimuthal
            # s = s+"     - Angles axial step:        %f \n"%self.binning.angles_axial
            # s = s+"     - Angles azimuthal:         %f \n"%self.binning.angles_azimuthal
            s = s + "     - Size_u:                   %f \n" % self.binning.size_u
            s = s + "     - Size_v:                   %f \n" % self.binning.size_v
            s = s + "     - N_u:                      %s \n" % self.binning.N_u
            s = s + "     - N_v:                      %s \n" % self.binning.N_v
        return s

    def _repr_html_(self):
        if not has_ipy_table:
            return "Please install ipy_table."
        table_data = [
            ["N_time_bins", len(self.time_bins)],
            ["Time_start", millisec_to_min_sec(self.time_bins[0])],
            ["Time_end", millisec_to_min_sec(self.time_bins[-1])],
            ["Duration", millisec_to_min_sec(self.time_bins[-1] - self.time_bins[0])],
            ["N_counts", pretty_print_large_number(self.static.prompts.get_integral())],
            [
                "N_locations",
                pretty_print_large_number(
                    self.static.prompts.sparsity.get_N_locations()
                ),
            ],
            [
                "compression_ratio",
                print_percentage(self.static.prompts.get_compression_ratio()),
            ],
            #        ['dynamic_inflation',self.dynamic_inflation],
            ["listmode_loss", self.static.prompts.get_listmode_loss()],
        ]
        if self.scanner:
            table_data += [
                ["Name", self.scanner.model],
                ["Manufacturer", self.scanner.manufacturer],
                ["Version", self.scanner.version],
            ]
        table = ipy_table.make_table(table_data)
        table = ipy_table.apply_theme("basic_left")
        # table = ipy_table.set_column_style(0, color='lightBlue')
        table = ipy_table.set_global_style(float_format="%3.3f")
        return table._repr_html_()

    def __iter__(self):
        """This method makes the object iterable. """
        return iter(self._dynamic)

    def __getitem__(self, i):
        """This method makes the object addressable like a list. """
        return self._dynamic[i]

    def __len__(self):
        return len(self._dynamic)


class PET_Cyclic_Scan(PET_Dynamic_Scan):
    """PET Cyclic Scan. This is useful for respiratory gated imaging and for cardiac gated imaging (or both)."""

    def __init__(self):
        PET_Dynamic_Scan.__init__(self)

    def import_listmode(
        self,
        hdr_filename,
        time_range_matrix_ms,
        data_filename=None,
        display_progress=False,
    ):
        """Load cyclic measurement data from a listmode file. """
        print_debug("- Loading static PET data from listmode file " + str(hdr_filename))
        hdr = interfile.load_interfile(hdr_filename)
        # Extract information from the listmode header

        # 1) Guess the path of the listmode data file, if not specified or mis-specified;
        #  1 - see if the specified listmode data file exists
        if data_filename is not None:
            data_filename = data_filename.replace("/", os.path.sep).replace(
                "\\", os.path.sep
            )  # cross platform compatibility
            if not os.path.exists(data_filename):
                raise FileNotFound("listmode data", data_filename)
        #  2 - if the listmode data file is not specified, try with the name (and full path)
        #      contained in the listmode header
        data_filename = hdr["name of data file"]["value"]
        data_filename = data_filename.replace("/", os.path.sep).replace(
            "\\", os.path.sep
        )  # cross platform compatibility
        if not os.path.exists(data_filename):
            #  3 - if it doesn't exist, look in the same path as the header file for the listmode data
            #      file with name specified in the listmode header file
            data_filename = (
                os.path.split(hdr_filename)[0]
                + os.path.sep
                + os.path.split(data_filename)[-1]
            )
            if not os.path.exists(data_filename):
                #  4 - if it doesn't exist, look in the same path as the header file for the listmode data
                #      file with same name as the listmode header file, replacing the extension: ".l.hdr -> .l"
                if hdr_filename.endswith(".l.hdr"):
                    data_filename = hdr_filename.replace(".l.hdr", ".l")
                    if not os.path.exists(data_filename):
                        raise FileNotFound("listmode data", data_filename)
                #  5 - if it doesn't exist, look in the same path as the header file for the listmode data
                #      file with same name as the listmode header file, replacing the extension: ".hdr -> .l"
                elif hdr_filename.endswith(".hdr"):
                    data_filename = hdr_filename.replace(".hdr", ".l")
                    if not os.path.exists(data_filename):
                        raise FileNotFound("listmode data", data_filename)

        # 2) Determine duration of the acquisition
        n_packets = hdr["total listmode word counts"]["value"]
        scan_duration = hdr["image duration"]["value"] * 1000  # milliseconds

        # 3) determine scanner parameters
        n_radial_bins = hdr["number of projections"]["value"]
        n_angles = hdr["number of views"]["value"]
        n_rings = hdr["number of rings"]["value"]
        max_ring_diff = hdr["maximum ring difference"]["value"]
        n_sinograms = (
            n_rings + 2 * n_rings * max_ring_diff - max_ring_diff ** 2 - max_ring_diff
        )
        n_frames = time_range_matrix_ms.shape[0]
        n_cycles = time_range_matrix_ms.shape[1]

        # 4) Display information
        print_debug(" - Number of packets:    %d       " % n_packets)
        print_debug(" - Scan duration:        %d [sec] " % (scan_duration / 1000.0))
        print_debug(" - Listmode data file:   %s       " % data_filename)
        print_debug(" - Listmode header file: %s       " % hdr_filename)
        print_debug(" - n_frames :            %d       " % n_frames)
        print_debug(" - n_cycles :            %d       " % n_cycles)
        print_debug(" - n_radial_bins:        %d       " % n_radial_bins)
        print_debug(" - n_angles:             %d       " % n_angles)
        print_debug(" - n_angles:             %d       " % n_sinograms)

        if display_progress:
            progress_bar = ProgressBar()
            progress_callback = progress_bar.set_percentage
        else:

            def progress_callback(value):
                pass

        # Load the listmode data
        M = self.scanner.michelogram
        R = self.scanner.listmode.load_listmode_cyclic(
            data_filename,
            time_range_matrix_ms,
            self.binning,
            n_radial_bins,
            n_angles,
            n_sinograms,
            M.span,
            M.segments_sizes,
            M.michelogram_sinogram,
            M.michelogram_plane,
            n_packets,
            progress_callback,
        )
        progress_callback(100)

        self._dynamic = []
        for t in range(n_frames):
            PET_t = PET_Static_Scan()
            PET_t.use_compression(self._use_compression)
            PET_t.use_gpu(self._use_gpu)
            PET_t.set_scanner(self.scanner.__class__)
            PET_t.set_binning(self.binning)
            PET_t._load_static_measurement(t)
            # make list of static scans
            self._dynamic.append(PET_t)
            # also make one attribut for each static scan
            setattr(self, "frame%d" % t, self._dynamic[t])
            # set activity shape and size and attenuation shape and size
            PET_t.set_activity_size(self.activity_size)
            PET_t.set_activity_shape(self.activity_shape)
            PET_t.set_attenuation_size(self.activity_size)
            PET_t.set_attenuation_shape(self.activity_shape)

        # Make a global PET_Static_Scan object
        self.static = PET_Static_Scan()
        self.static.use_compression(self._use_compression)
        self.static.use_gpu(self._use_gpu)
        self.static.set_scanner(self.scanner.__class__)
        self.static.set_binning(self.binning)
        self.static._load_static_measurement()
        self.static.set_activity_size(self.activity_size)
        self.static.set_activity_shape(self.activity_shape)
        self.static.set_attenuation_size(self.activity_size)
        self.static.set_attenuation_shape(self.activity_shape)

        # Free structures listmode data
        self.scanner.listmode.free_memory()

        # Construct ilang model
        self._construct_ilang_model()
