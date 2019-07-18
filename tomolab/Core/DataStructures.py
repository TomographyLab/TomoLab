# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

"""Occiput Core. Here are defined the Core occiput data structures:
Image3D, ImageND, Grid3D, GridND and spatial transformations."""

__all__ = ['make_grid', 'transform_grid', 'grid_from_box_and_affine',
           'resample_image_on_grid','GridND', 'Grid3D', 'Image3D', 'ImageND']

import nibabel
import copy
import numpy as np

from ..global_settings import printoptions, get_default_background, \
                              is_gpu_enabled, has_ipy_table, ipy_table

from ..Visualization.Visualization import VolumeRenderer,  TriplanarView, TriplanarViewInteractive
from tomolab.Core.NiftyRec import INTERPOLATION_LINEAR, INTERPOLATION_POINT, \
    TR_resample_grid, TR_transform_grid, TR_grid_from_box_and_affine

from ..Transformation.Transformations import Transform_Affine, Transform_6DOF, \
                                       Transform_Identity, Transform_Rotation, \
                                       Transform_Scale, Transform_Translation
#from ..DataSources.FileSources.Files import guess_file_type_by_name
#from ..DataSources.PET.volume import import_interfile_volume


class GridND(object):
    def __init__(
            self, data=None, space="", is_uniform=True, is_affine=True,
            is_axis_aligned=True
    ):
        self.ndim = None
        self.__set_data(data)
        self.space = space
        self.__clear_cache()
        self.__is_uniform = is_uniform
        self.__is_affine = is_affine
        self.__is_axis_aligned = is_axis_aligned

    def min(self):
        if self.__min is None:
            self.__min = eval("self.data%s" % (".min(0)" * self.ndim))
        return self.__min

    def max(self):
        if self.__max is None:
            self.__max = eval("self.data%s" % (".max(0)" * self.ndim))
        return self.__max

    def span(self):
        if self.__span is None:
            self.__span = self.max() - self.min()
        return self.__span

    def center(self, use_corners_only=True):
        if self.__center is None or (
                self.__use_corners_only != use_corners_only):
            if use_corners_only:
                corners = self.corners()
                center = corners.mean(1)
            else:
                center = None
                # FIXME: implement
            self.__center = center
            self.__use_corners_only = use_corners_only
        return self.__center

    def mean_distance_from_point(self, point, use_corners_only=True):
        if use_corners_only:
            corners = self.corners()
            dist = corners - np.tile(
                np.asarray(point).reshape(3, 1), [1, corners.shape[1]]
            )
            dist = np.sqrt((dist * dist)).sum(1) / corners.shape[1]
        else:
            dist = None
            # FIXME: implement
        return dist

    def mean_dist_from_center(self):
        if self.__mean_dist_center is None:
            center = self.center()
            self.__mean_dist_center = self.mean_distance_from_point(center)
        return self.__mean_dist_center

    def get_shape(self):
        return self.data.shape

    def is_uniform(self):
        """Returns True if the grid is uniform. """
        return self.__is_uniform
        # FIXME: change the flags when grid is transformed

    def is_affine(self):
        """Returns True if the grid is the affine transformation of a uniform grid. """
        return self.__is_affine

    def is_axis_aligned(self):
        """Returns True if the grid is uniform and aligned to the x,y,z axis. """
        return self.__is_axis_aligned

    def transform(self, affine_from_grid):
        return transform_grid(self, affine_from_grid)

    def corners(self, homogeneous_coords=False):
        if self.__corners is None:
            n_corners = 2 ** self.ndim
            corners = []
            for i in range(n_corners):
                b = eval(
                    "["
                    + bin(i)[2:].zfill(self.ndim).replace("0",
                                                          "0,").replace(
                        "1", "1,")
                    + "]"
                )
                b = (np.asarray(
                    self.data.shape[0: self.ndim]) - 1) * np.asarray(
                    b
                )
                s = str(b.tolist())[1:-1]
                corner = eval("self.data[%s,:]" % s)
                corners.append(corner)
            corners = np.asarray(corners).transpose()
            if homogeneous_coords:
                corners2 = np.ones((4, n_corners))
                corners2[0:3, :] = corners
                corners = corners2
            self.__corners = corners
        return self.__corners

    def __get_data(self):
        return self.__data

    def __set_data(self, data):
        self.__data = data
        if data is not None:
            self.ndim = self.data.ndim - 1
        else:
            self.ndim = None
        self.__clear_cache()

    def __get_shape(self):
        return self.data.shape

    def __clear_cache(self):
        self.__min = None
        self.__max = None
        self.__span = None
        self.__center = None
        self.__use_corners_only = None
        self.__mean_dist_center = None
        self.__corners = None

    def __repr__(self):
        with printoptions(precision=4):
            if self.is_axis_aligned():
                type = "axis aligned"
            elif self.is_affine():
                type = "affine"
            elif self.is_uniform():
                type = "uniform"
            else:
                type = ""
            if type != "":
                type = " (%s)" % type
            s = "Grid%dD %s: " % (int(self.ndim), type)
            s = s + "\n - space -------------------> " + str(self.space)
            s = s + "\n - shape -------------------> " + str(list(self.shape))
            s = s + "\n - min ---------------------> " + str(self.min())
            s = s + "\n - max ---------------------> " + str(self.max())
            s = s + "\n - span --------------------> " + str(self.span())
            s = s + "\n - center ------------------> " + str(self.center())
            s = (
                    s
                    + "\n - mean dist from center: --> "
                    + str(self.mean_dist_from_center())
            )
            s = s + "\n"
        return s

    def _repr_html_(self):
        if not has_ipy_table:
            return "Please install ipy_table."
        if self.is_axis_aligned():
            type = "axis aligned"
        elif self.is_affine():
            type = "affine"
        elif self.is_uniform():
            type = "uniform"
        else:
            type = ""
        if type != "":
            type = " (%s)" % type
        s = "Grid%dD %s: " % (int(self.ndim), type)

        def pretty(list):
            return str(["{0:.2f}".format(flt) for flt in list])

        table_data = [
            [s, "", "", "", "", "", ""],
            ["space", "shape", "min", "max", "span", "center", "spread"],
            [
                str(self.space),
                pretty(self.shape),
                pretty(self.min()),
                pretty(self.max()),
                pretty(self.span()),
                pretty(self.center()),
                pretty(self.mean_dist_from_center()),
            ],
        ]
        table = ipy_table.make_table(table_data)
        table = ipy_table.apply_theme("basic")
        table = ipy_table.set_cell_style(0, 0, column_span=7)
        table = ipy_table.set_cell_style(0, 0, align="center")
        table = ipy_table.set_row_style(1, color="#F7F7F7")
        table = ipy_table.set_row_style(2, color="#F7F7F7")
        table = ipy_table.set_row_style(0, color="#CCFF99")
        table = ipy_table.set_row_style(1, bold=True)
        table = ipy_table.set_row_style(1, align="center")
        table = ipy_table.set_global_style(float_format="%3.3f")
        s = table._repr_html_()
        return s

    data = property(__get_data, __set_data)
    shape = property(__get_shape)


class Grid3D(GridND):
    def __init__(self, data=None, space=""):
        GridND.__init__(self, data, space)
        self.ndim = 3


class ImageND(object):
    def __init__(self, data=None, affine=None, space="", header=None, mask_flag=0):
        self.ndim = None
        self.space = ""
        self.header = header
        if isinstance(data, str):
            self.load_from_file(data)
        else:
            self.set_data(data)
        self.set_affine(affine)
        self.set_space(space)

        self.background = get_default_background()
        self.use_gpu = is_gpu_enabled()

        self.set_mask_flag(mask_flag)

    def set_mask_flag(self, is_mask):
        self.__is_mask = is_mask

    def is_mask(self):
        """Returns True if the image is a mask (i.e. the values should be \
        interpreted as discrete labels - take note that, however, \
        that this reports the state of a flag; the data can be non-integer. )"""
        return self.__is_mask

    def set_lookup_table(self, lut):
        """Set a lookup table for visualization. """
        self.__lut = lut

    def get_lookup_table(self):
        try:
            lut = self.__lut
        except:
            lut = None
        return lut

    def set_data(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.asarray([])
        self.ndim = self.data.ndim
        self.max_val = self.max()
        self.min_val = self.min()
        self.shape = self.get_shape()
        self.size = self.get_size()

    def min(self):
        return self.data.min()  # FIXME: pass on optional parameters

    def max(self):
        return self.data.max()  # FIXME: pass on optional parameters

    def set_affine(self, affine):
        if isinstance(affine, Transform_Affine):
            self.affine = affine
        elif isinstance(affine, np.ndarray):
            self.affine = Transform_Affine(data=affine)
        elif affine is None:
            self.affine = Transform_Identity()
        else:
            print("'affine' must be an instance of Affine")
        self.affine.map_to = self.space

    def set_space(self, space):
        self.affine.map_to = space
        self.space = space
        self.affine.map_from = "index"

    def get_data(self):
        return self.data

    def get_header(self):
        print(self.header)
        return self.header

    def get_shape(self):
        return np.asarray(self.data.shape)

    def get_size(self):
        return self.data.size

    def get_world_grid(self, n_points=None):
        if n_points is None:
            n_points = self.get_shape()
        grid = grid_from_box_and_affine(self.get_world_grid_min(), self.get_world_grid_max(), n_points)
        return grid

    def get_world_grid_min(self):
        d = self.get_data()
        s = np.asarray(d.shape) - 1
        corners = np.asarray(
            [[0, 0, 0, 1], [s[0], 0, 0, 1], [0, s[1], 0, 1], [s[0], s[1], 0, 1], [0, 0, s[2], 1], [s[0], 0, s[2], 1],
             [0, s[1], s[2], 1], [s[0], s[1], s[2], 1]]).transpose()
        corners = self.affine.left_multiply(corners)
        corners = corners[0:3, :]
        m = corners.min(1)
        return m

    def get_world_grid_max(self):
        d = self.get_data()
        s = np.asarray(d.shape) - 1
        corners = np.asarray(
            [[0, 0, 0, 1], [s[0], 0, 0, 1], [0, s[1], 0, 1], [s[0], s[1], 0, 1], [0, 0, s[2], 1], [s[0], 0, s[2], 1],
             [0, s[1], s[2], 1], [s[0], s[1], s[2], 1]]).transpose()
        corners = self.affine.left_multiply(corners)
        corners = corners[0:3, :]
        M = corners.max(1)
        return M

    def get_pixel_grid(self):
        print("get_pixel_grid")
        n_points = np.uint32(self.shape)
        grid = grid_from_box_and_affine(np.asarray([0, 0, 0]), n_points - 1, n_points)
        return grid

    def transform(self, affine):
        if not isinstance(affine, Transform_Affine):
            print("Transformation must be an instance of Affine.")
            # FIXME raise error
            return None
        self.affine = affine.left_multiply(self.affine)
        # return self

    def save_to_file(self, filename):
        from .Conversion import occiput2nifti
        self.data = np.flip(self.data,(0,1))
        nii = occiput2nifti(self)
        nibabel.save(nii, filename)

    '''
    def load_from_file(self, filename, datafile=""):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_volume_header":
            volume = import_interfile_volume(filename, datafile)
        elif filetype == "dicom_volume":
            print("DICOM volume file not supported. Everything is ready to implement this, please implement it. ")
            return
        elif filetype == "nifti":
            print("Nifti volume file not supported. Everything is ready to implement this, please implement it. ")
            return
        elif filetype == "h5":
            print("H5 volume file not supported. Everything is ready to implement this, please implement it. ")
            return
        elif filetype == "mat":
            print("Matlab volume file not supported. Everything is ready to implement this, please implement it. ")
            return
        else:
            print(("ImageND.load_from_file: file type of %s unknown. Unable to load. "% filename))
            return
        self.set_data(np.float32(volume.data))
    '''

    def copy(self):
        return copy.copy(self)


class Image3D(ImageND):
    def __init__(self, data=None, affine=None, space="", header=None):
        ImageND.__init__(self, data, affine, space, header)
        self.ndim = 3

    def compute_resample_on_grid(self, grid, affine_grid_to_world=None, verify_mapping=True,
                                 interpolation_mode=INTERPOLATION_LINEAR):
        resampled_data = resample_image_on_grid(self, grid, affine_grid_to_world, verify_mapping, self.background,
                                                self.use_gpu, interpolation_mode)
        # create new Image3D object
        return Image3D(data=resampled_data)  # FIXME: perhaps just return the raw resampled data

    def compute_resample_in_box(self, box_min, box_max, box_n):
        pass

    def compute_resample_in_space_of_image(self, image):
        pass

    def compute_gradient_on_grid(self, grid, affine_grid_to_world=None, verify_mapping=True):
        resampled_data = resample_image_on_grid(self, grid, affine_grid_to_world, verify_mapping, self.background,
                                                self.use_gpu)
        # create new Image3D object
        # print resampled_data.max()
        gradient_data = np.gradient(resampled_data)  # FIXME: use NiftyPy
        return gradient_data

    def compute_gradient_in_box(self, box):
        pass

    def compute_gradient_in_space_of_image(self, image):
        pass

    def compute_gradient_pixel_space(self):
        gradient_data = np.gradient(self.data)  # FIXME: use NiftyPy
        return gradient_data

    def compute_smoothed(self, smoothing):
        pass

    def export_image(self, index, axis=0, normalise=True, scale_factor=None, shrink=None, rotate=0):
        pass

    def has_data(self):
        return self.data is not None

    def display(self, coords=None, clim=None, cmap=None, interp = None, figsize = (20,9), res=(1.0,1.0,1.0), crop=None, autotranspose=True):
        V = TriplanarView(self, res=res, crop=crop, autotranspose=autotranspose)
        return V.show(coords = coords, clim=clim, cmap=cmap, figsize = figsize, interp = interp)

    def display_interactive(self, clim=None, cmap=None, figsize = (20,9), res=(1.0,1.0,1.0), crop=None, autotranspose=True):
        V = TriplanarViewInteractive(self, res=res, crop=crop, autotranspose=autotranspose)
        return V.show(clim=clim, colormap=cmap, figsize = figsize)

    '''
    def _display_legacy(self, axis=0, shrink=None, rotate=None, subsample_slices=None, scales=None, open_browser=None):
        # The following is a quick fix: use MultipleVolumesNiftyPy if the image has small size,
        # MultipleVolumes otherwise. MultipleVolumesNiftyPy makes use of the GPU but crashes with large images.
        # NOTE that MultipleVolumes produces a different visualisation from MultipleVolumesNiftyPy:
        # it displays the raw imaging data, without accounting for the transformation to world space.
        # Modify MultipleVolumesNiftyPy so that it processes the data in sequence if it is too large for the GPU.
        # Then get rid of  MultipleVolumes.
        # if self.size <= 256**3:
        #    D = MultipleVolumesNiftyPy([self],axis,open_browser=open_browser)
        # else:
        D = MultipleVolumes([self], axis=axis, shrink=shrink, rotate=rotate, subsample_slices=subsample_slices,
                            scales=scales, open_browser=open_browser)
        return D

    def _display_with(self, other_images, axis=0, shrink=None, rotate=None, subsample_slices=None, scales=None,
                      open_browser=None):
        if isinstance(other_images, type(())):
            other_images = list(other_images)
        elif isinstance(other_images, type([])):
            other_images = other_images
        else:
            other_images = [other_images, ]
        D = MultipleVolumes([self, ] + other_images, axis=axis, shrink=shrink, rotate=rotate,
                            subsample_slices=subsample_slices, scales=scales, open_browser=open_browser)
        return D

    def _display_slice(self, axis=0, index=None, open_browser=False):
        if index is None:
            if axis == 0:
                index = np.uint32(self.shape[0] // 2)
            if axis == 1:
                index = np.uint32(self.shape[1] // 2)
            if axis == 2:
                index = np.uint32(self.shape[2] // 2)
            if axis == 0:
                a = self.data[index, :, :].reshape((self.shape[1], self.shape[2]))
            if axis == 1:
                a = self.data[:, index, :].reshape((self.shape[0], self.shape[2]))
            if axis == 2:
                a = self.data[:, :, index].reshape((self.shape[0], self.shape[1]))
        D = DisplayNode.DisplayNode()
        im = Image.fromarray(a).convert("RGB").rotate(90)
        D.display('image', im, open_browser)
        return D
    '''

    def volume_render(self, res=(1.0,1.0,1.0), crop=None, autotranspose=True):
        V = VolumeRenderer(self, res=res, crop=crop, autotranspose=autotranspose)
        return V.display()

    def quick_render(self, res=(1.0,1.0,1.0), zoom=1, flip=True):
        from ..Reconstruction import PET_Static_Scan
        shape = np.asarray(self.get_shape()*5)
        size  = np.asarray(shape * res * zoom)
        pet = PET_Static_Scan()
        pet.set_activity_shape(shape)
        pet.set_activity_size(size)
        pet.set_attenuation_shape(shape)
        pet.set_attenuation_size(size)
        tmp = self.copy()
        if tmp.min() < 0:
            tmp.data = (tmp.data - tmp.min()) / (tmp.max() - tmp.min())
        if flip:
            tmp.data = np.flip(tmp.data,(0,1,2))
        ideal_prompts = pet.project_activity(tmp)
        return ideal_prompts.quick_render()

    def _repr_html_(self):
        self.display()
        return ''

    # Overload math operators
    def _is_same_type(self, other):
        return isinstance(other, self.__class__)

    def _is_in_same_space(self, other):
        # FIXME: implement
        return True

    def _is_on_same_grid(self, other):
        # FIXME: implement
        return True

    def __add__(self, other):
        if self._is_same_type(other):
            if self._is_in_same_space(other):
                if self._is_on_same_grid(other):
                    out = self.copy()
                    out.data = out.data + other.data
                    return out
                else:
                    # FIXME: implement
                    print(
                        "SUM of images on different grids; the right hand side image must be resampled, please implement this.")
            else:
                # FIXME: raise error
                raise ("SUM of images not in the same space. It cannot be done. ")
        else:
            out = self.copy()
            out.data = out.data + other
            return out
        return None

    def __sub__(self, other):
        if self._is_same_type(other):
            if self._is_in_same_space(other):
                if self._is_on_same_grid(other):
                    out = self.copy()
                    out.data = out.data - other.data
                    return out
                else:
                    # FIXME: implement
                    print(
                        "SUB of images on different grids; the right hand side image must be resampled, please implement this.")
            else:
                # FIXME: raise error
                raise ("SUB of images not in the same space. It cannot be done. ")
        else:
            out = self.copy()
            out.data = out.data - other
            return out
        return None

    def __mul__(self, other):
        if self._is_same_type(other):
            if self._is_in_same_space(other):
                if self._is_on_same_grid(other):
                    out = self.copy()
                    out.data = out.data * other.data
                    return out
                else:
                    # FIXME: implement
                    print(
                        "MUL of images on different grids; the right hand side image must be resampled, please implement this.")
            else:
                # FIXME: raise error
                raise ("MUL of images not in the same space. It cannot be done. ")
        else:
            out = self.copy()
            out.data = out.data * other
            return out
        return None

    def __div__(self, other):
        if self._is_same_type(other):
            if self._is_in_same_space(other):
                if self._is_on_same_grid(other):
                    out = self.copy()
                    out.data = out.data / other.data
                    return out
                else:
                    # FIXME: implement
                    raise (
                        "DIV of images on different grids; the right hand side image must be resampled, please implement this.")
            else:
                # FIXME: raise error
                raise ("DIV of images not in the same space. It cannot be done. ")
        else:
            out = self.copy()
            out.data = out.data / other
            return out

    def __truediv__(self, other):
        if self._is_same_type(other):
            if self._is_in_same_space(other):
                if self._is_on_same_grid(other):
                    out = self.copy()
                    out.data = out.data / other.data
                    return out
                else:
                    # FIXME: implement
                    raise (
                        "DIV of images on different grids; the right hand side image must be resampled, please implement this.")
            else:
                # FIXME: raise error
                raise ("DIV of images not in the same space. It cannot be done. ")
        else:
            out = self.copy()
            out.data = out.data / other
            return out

    def __radd_(self, other):
        return self.__add__(other)

    def __rmul_(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        if self._is_same_type(other):
            if self._is_in_same_space(other):
                if self._is_on_same_grid(other):
                    out = self.copy()
                    out.data = other.data - out.data
                    return out
                else:
                    # FIXME: implement
                    print(
                        "SUB of images on different grids; the right hand side image must be resampled, please implement this.")
            else:
                # FIXME: raise error
                print("SUB of images not in the same space. It cannot be done. ")
        else:
            out = self.copy()
            out.data = other - out.data
            return out
        return None

    def __rdiv__(self, other):
        if self._is_same_type(other):
            if self._is_in_same_space(other):
                if self._is_on_same_grid(other):
                    out = self.copy()
                    out.data = other.data / out.data
                    return out
                else:
                    # FIXME: implement
                    print(
                        "DIV of images on different grids; the right hand side image must be resampled, please implement this.")
            else:
                # FIXME: raise error
                print("DIV of images not in the same space. It cannot be done. ")
        else:
            out = self.copy()
            out.data = other / out.data
            return out
        return None


##########################################################################################
## GPU-accelerated spatial transformations: (previously in
## occiput.Core.NiftyPy_wrap)
##########################################################################################

def make_grid(data, space, ndim):
    """Instantiate n-dimensional point cloud on GPU."""
    if ndim == 3:
        return Grid3D(data, space)
    else:
        return GridND(data, space)


def transform_grid(grid, affine_from_grid):
    """Transform n-dimensional point cloud on GPU."""
    # 1) verify if the spaces of the affine map and of the grid are compatible:
    if not affine_from_grid.can_left_multiply(grid):
        print("Affine transformation not compatible with grid. ")
        # FIXME: raise error, or warning, depending on a global setting
    # 2) transform
    transformed = TR_transform_grid(grid.data, affine_from_grid.data)
    # 3) instantiate a new grid
    grid = make_grid(transformed, affine_from_grid.map_to,
                     transformed.ndim - 1)
    return grid


def grid_from_box_and_affine(min_coords, max_coords, n_points, affine=None, space="world"):
    """Instantiate point cloud on GPU form bounding box and affine transformation
    matrix."""
    # FIXME: optionally use the affine to transform min_coords and max_coords
    data = TR_grid_from_box_and_affine(min_coords, max_coords, n_points)
    ndim = data.ndim - 1
    grid = make_grid(data, space, ndim)
    return grid


def resample_image_on_grid(
        image,
        grid,
        affine_grid_to_world=None,
        verify_mapping=True,
        background=0.0,
        use_gpu=1,
        interpolation_mode=None,
):
    """Resample image on point cloud (on GPU)."""
    if verify_mapping:
        # check if the image, the grid and the affine mapping are compatible:
        # 1)If affine_grid_to_world is not defined, verify if image and grid compatibility
        if affine_grid_to_world == None:
            if not image.affine.can_inverse_left_multiply(grid):
                print("grid and image are not compatible. ")
                # FIXME: raise exception
                return
        # 2)If affine_grid_to_world is defined, verify if image, grid
        # and affine_grid_to_world are compatible
        else:
            # a) check mapping from grid to world
            if not affine_grid_to_world.can_left_multiply(grid):
                print("grid and affine_grid_to_world are not compatible. ")
                # FIXME: raise exception
                return
            # b) check mapping from image to world
            if not image.affine.can_inverse_left_multiply(
                    affine_grid_to_world):
                print(
                    "image and affine_grid_to_world are not compatible. ")
                # FIXME: raise exception
                return
    # compute affine:
    if affine_grid_to_world == None:
        affine = image.affine
    else:
        affine = affine_grid_to_world.left_multiply(image.affine)
    # decide sampling mode
    if interpolation_mode is None:
        if image.is_mask():
            interpolation_mode = INTERPOLATION_POINT
        else:
            interpolation_mode = INTERPOLATION_LINEAR
    # resample:
    resampled_data = TR_resample_grid(
        np.float32(image.data),
        np.float32(grid.data),
        np.float32(affine.data),
        background,
        use_gpu,
        interpolation_mode,
    )
    return resampled_data
