# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

# Interface to the ray-tracer for projection and back-projection.
# The interface to the ray-tracer for scatter simulation is in another file.

from ...Core.Errors import UnknownParameter

DEFAULT_PROJECTION_PARAMETERS = {
    "N_samples":300,
    "sample_step":2.0,
    "background_activity":0.0,
    "background_attenuation":0.0,
    "truncate_negative_values":0,
    "gpu_acceleration":1,
    "direction":5, # affects performance only, between 1 and 6; 2 and 5 are normally the best values
    "block_size":1024, # affects performance only
    }

DEFAULT_BACKPROJECTION_PARAMETERS = {
    "N_samples":300,
    "sample_step":2.0,
    "background_activity":0.0,
    "background_attenuation":0.0,
    "truncate_negative_values":0,
    "gpu_acceleration":1,
    "direction":5, # affects performance only, between 1 and 6; 2 and 5 are normally the best value
    "block_size":1024, # affects performance only
    }


class ProjectionParameters:
    """Data structure containing the parameters of a projector for PET. """

    default_parameters = DEFAULT_PROJECTION_PARAMETERS

    def __init__(self,parameters = None):
        self._initialised = 0
        self.name = "Unknown binning name"
        if parameters is None:
            self.load_from_dictionary(self.default_parameters)
        elif type(parameters) == dict:
            self.load_from_dictionary(parameters)
        elif type(parameters) in [list,tuple]:
            if len(parameters) == len(list(self.default_parameters.keys())):
                self.N_samples = parameters[0]
                self.sample_step = parameters[1]
                self.background_activity = parameters[2]
                self.background_attenuation = parameters[3]
                self.truncate_negative_values = parameters[4]
                self.gpu_acceleration = parameters[5]
                self.direction = parameters[6]
                self.block_size = parameters[7]
            else:
                raise UnknownParameter(
                        "Parameter %s specified for ProjectionParameters is not compatible. "
                        % str(parameters)
                        )
        else:
            raise UnknownParameter(
                    "Parameter %s specified for ProjectionParameters is not compatible. "
                    % str(parameters)
                    )

    def load_from_dictionary(self,dictionary):
        self.N_samples = dictionary[
            "N_samples"
        ]  # Number of samples along a line when computing line integrals
        self.sample_step = dictionary[
            "sample_step"
        ]  # distance between consecutive points along a line when computing line integrals (this is in the same unit measure as the size of the imaging volume (activity_size and attenuation_size))
        self.background_activity = dictionary[
            "background_activity"
        ]  # Activity in voxels outside of the imaging volume
        self.background_attenuation = dictionary[
            "background_attenuation"
        ]  # Attenuation in voxels outside of the imaging volume
        self.truncate_negative_values = dictionary[
            "truncate_negative_values"
        ]  # If set to 1, eventual negative values obtained when projecting are set to 0
        # (This is meant to remove eventual unwanted small negative values due to FFT-based smoothing within the projection algorithm
        # - note that numerical errors may produce small negative numbers when doing FFT-IFFT even if the function is all positive )
        self.gpu_acceleration = dictionary[
            "gpu_acceleration"
        ]  # Whether to use GPU acceleration (if available) or not
        self.direction = dictionary[
            "direction"
        ]  # Direction parameter for the projector and b-proj., it affects performance only; 2 and 5 are normally the best values.
        self.block_size = dictionary[
            "block_size"
        ]  # Number of blocks per thread for the GPU accelerated projector and b-proj. It affects performance only. Typical values: 256, 512, 768

    def __repr__(self):
        s = "PET Projection Parameters: \n"
        s = s + " - N_samples:                %d \n" % self.N_samples
        s = s + " - sample_step:              %f \n" % self.sample_step
        s = s + " - background_activity:      %f \n" % self.background_activity
        s = s + " - background_attenuation:   %f \n" % self.background_attenuation
        s = s + " - truncate_negative_values: %f \n" % self.truncate_negative_values
        s = s + " - gpu_acceleration:         %d \n" % self.gpu_acceleration
        s = s + " - direction:                %d \n" % self.direction
        s = s + " - block_size:               %d \n" % self.block_size
        return s

class BackprojectionParameters(ProjectionParameters):
    """Data structure containing the parameters of a projector for PET. """

    default_parameters = DEFAULT_BACKPROJECTION_PARAMETERS
    # Note: this seems unfinished, but it inherits all that is necessary from ProjectionParameters

    def __repr__(self):
        s = "PET Backprojection Parameters: \n"
        s = s + " - N_samples:                %d \n" % self.N_samples
        s = s + " - sample_step:              %f \n" % self.sample_step
        s = s + " - background_activity:      %f \n" % self.background_activity
        s = s + " - background_attenuation:   %f \n" % self.background_attenuation
        s = s + " - truncate_negative_values: %f \n" % self.truncate_negative_values
        s = s + " - gpu_acceleration:         %d \n" % self.gpu_acceleration
        s = s + " - direction:                %d \n" % self.direction
        s = s + " - block_size:               %d \n" % self.block_size
        return s