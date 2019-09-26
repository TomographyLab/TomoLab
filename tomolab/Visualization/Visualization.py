# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

__all__ = [
    'TriplanarView',
    'TriplanarViewInteractive',
    'VolumeRenderer',
    ]

import numpy as np
import uuid
import scipy.ndimage
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from ipywidgets import interact, fixed, IntSlider, FloatProgress, FloatRangeSlider, Layout
import ipyvolume as ipv
import json
from skimage.transform import rescale
from . import Colors as C
from ..global_settings import is_PET_Projection,is_ImageND, is_in_ipynb
from PIL import Image
from .DisplayNode.DisplayNodeProxy import DisplayNode

#################################################à

def pad_and_crop(volume, res = (1.0,1.0,1.0), crop=None):
    res = tuple((np.asarray(res) / np.asarray(res).min()).astype(np.float32))
    new_volume = rescale(volume, scale=res,
                        anti_aliasing=False,
                        preserve_range=True,
                        multichannel=False,
                        order=0)
    [x,y,z] = np.asarray(new_volume.shape)
    c_max = np.max([x,y,z])

    # pad
    npad = ((np.ceil((c_max-x)/2).astype(np.int32), np.floor((c_max-x)/2).astype(np.int32)),
            (np.ceil((c_max-y)/2).astype(np.int32), np.floor((c_max-y)/2).astype(np.int32)),
            (np.ceil((c_max-z)/2).astype(np.int32), np.floor((c_max-z)/2).astype(np.int32)))
    pad_val = max(new_volume.min(), 0.0)
    new_volume = np.pad(new_volume, pad_width=npad, mode='constant', constant_values=pad_val)

    # crop
    if crop is not None:
        start = x//2-(crop//2)
        new_volume = new_volume[start:start+crop,start:start+crop,start:start+crop]
    return new_volume



class ProgressBar():
    def __init__(self, color=C.BLUE, title="Processing:"):
        self._percentage = 0.0
        self.visible = False
        if is_in_ipynb():
            self.set_display_mode("ipynb")
        else:
            self.set_display_mode("text")
        if color == C.BLUE:
            color_style = ''
        elif color == C.LIGHT_BLUE:
            color_style = 'info'
        elif color == C.RED:
            color_style = 'danger'
        elif color == C.LIGHT_RED:
            color_style = 'warning'
        elif color == C.GREEN:
            color_style = 'success'
        else:
            print('Unavailable color code. Using default "BLUE", instead. ')
            color_style = ''

        self._pb = FloatProgress(
            value=0.0,
            min=0,
            max=100.0,
            step=0.1,
            description=title,
            bar_style=color_style,
            orientation='horizontal',
            layout=Layout(padding='0px 0px 0px 10px', width='100%', height='20px')
        )
        self._pb.style.description_width = 'initial'
        self._pb.layout.border='solid 1px'

    def show(self):
        if self.mode == "ipynb":
            display(self._pb)
        self.visible = True

    def set_display_mode(self, mode="ipynb"):
        self.mode = mode

    def set_percentage(self, percentage):
        if not self.visible:
            self.show()
        if percentage < 0.0:
            percentage = 0.0
        if percentage > 100.0:
            percentage = 100.0
        percentage = int(percentage)
        self._percentage = percentage
        if self.mode == "ipynb":
            self._pb.value = self._percentage
        else:
            print("%2.1f / 100" % percentage)

    def get_percentage(self):
        return self._percentage


# TODO: add class for image fusion built on top of TriplanarView (slider for alpha channel)

class TriplanarView:
    """
    Inspect a 3D or 4D data volume, with 3 planar view in Axial,
    Sagittal, and Coronal direction.
    You can pass a np.ndarray, or an occiput 'ImageND' object,
    or a 'PET_Projection' object.

    Usage
    -----
    V = TriplanarView(volume, crop=128)
    V.show(clim=(0,15), colormap='color')
    """

    def __init__(self, volume, res=(1.0, 1.0, 1.0), crop=None, autotranspose=False):

        if isinstance(volume, np.ndarray):
            if autotranspose:
                if (volume.ndim == 3) or (volume.shape[3] == 1):
                    volume = volume.transpose((1, 0, 2))
                    volume = np.flip(volume, 2)  # U-D
                else:
                    volume = volume.transpose((2, 1, 3, 0))
                    volume = np.flip(volume, 2)  # U-D
            self.data = pad_and_crop(volume, res=res, crop=crop)
            self.plottype = 'volume'
        elif is_ImageND(volume):
            volume = volume.data
            if autotranspose:
                if (volume.ndim == 3) or (volume.shape[3] == 1):
                    volume = volume.transpose((1, 0, 2))
                    volume = np.flip(volume, 2)  # U-D
                else:
                    volume = volume.transpose((2, 1, 3, 0))
                    volume = np.flip(volume, 2)  # U-D
            self.data = pad_and_crop(volume, res=res, crop=crop)
            self.plottype = 'volume'
        elif is_PET_Projection(volume):
            volume = volume.to_nd_array()
            if autotranspose:
                volume = volume.transpose((0, 2, 3, 1))
            self.data = volume  # pad_and_crop(volume, res=res, crop=crop)
            self.plottype = 'projection'
        else:
            print("Data format unknown")
            return

        # initialise where the image handles will go
        self.image_handles = None
        self.autotranspose = autotranspose

    def show(self, coords=None, cmap=None, figsize=(15,5), clim=None, interp=None):
        """
        Plot volumetric data.

        Args
        ----
            colormap : str
                    - 'color': show a selection of colorful colormaps
                    - 'mono': show a selection of monochromatic colormaps
                    - <cmap name>: pick one from matplotlib list
            clim : list | tuple
                    lower and upper bound for colorbar (you will have a slider
                    to better refine it later)
            figsize : list | tuple
                    The figure height and width for matplotlib, in inches.
        """

        if self.plottype == 'projection':
            self.views = [
                'Projection plane [u_coord - v_coord]',
                'One-ring projection plane [v_coord - axial_angle]',
                'One-ring projection plane [u_coord - axial_angle]']
            self.directions = ['Axial_angle', 'Azimuthal_angle', 'u_coord', 'v_coord']
        elif self.plottype == 'volume':
            self.views = ['Sagittal',
                          'Coronal',
                          'Axial']
            self.directions = ['x', 'y', 'z', 't']

        data_array = self.data

        if not ((data_array.ndim == 3) or (data_array.ndim == 4)):
            raise ValueError('Input image should be 3D or 4D')

        if coords is None:
            coords = np.int32(data_array.shape) // 2
        else:
            if self.autotranspose:
                if (data_array.ndim == 3) or (data_array.shape[3] == 1):
                    coords = (coords[1],coords[0],coords[2])
                    print(1)
                else:
                    coords = (coords[2], coords[1], coords[3], coords[0])
                    print(2)
            print(coords)

        if (data_array.ndim == 3) :
            #x, y, z = coords
            t = 0  # time is fixed
        elif (data_array.ndim == 4) and (data_array.shape[3] == 1):
            x, y, z = coords[:3]
            t = 0  # time is fixed
        else:  # 4D
            #x, y, z, \
            t = coords[3]

        fig = plt.figure(figsize=figsize)
        #ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        #ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        #ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        ax1 = plt.subplot2grid((1,3), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((1,3), (0, 1), colspan=1)
        ax3 = plt.subplot2grid((1,3), (0, 2), rowspan=1)
        axes = [ax1, ax2, ax3]

        for ii, ax in enumerate(axes):
            slice_obj = 3 * [slice(None)]
            if data_array.ndim == 4:
                slice_obj += [t]
            slice_obj[ii] = coords[ii]

            ax.set_facecolor('black')
            ax.tick_params(
                axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False
            )
            img = (np.rot90(data_array[tuple(slice_obj)], k=1))
            im = ax.imshow(img, cmap=cmap, clim=clim, interpolation=interp)
            ax.set_title(self.views[ii])

            # add "cross hair"
            guide_positions = [val for jj, val in enumerate(coords)
                               if jj != ii]
            ax.axvline(x=guide_positions[0], color='red', alpha=0.8)
            ax.axhline(y=guide_positions[1], color='red', alpha=0.8)


class TriplanarViewInteractive:
    """
    Inspect a 3D or 4D data volume, with 3 planar view in Axial,
    Sagittal, and Coronal direction.
    You can pass a np.ndarray, or an occiput 'ImageND' object,
    or a 'PET_Projection' object.

    Usage
    -----
    V = TriplanarView(volume, crop=128)
    V.show(clim=(0,15), colormap='color')
    """

    def __init__(self, volume, res = (1.0,1.0,1.0), crop = None, autotranspose=False):

        if isinstance(volume, np.ndarray):
            if autotranspose:
                if (volume.ndim == 3) or (volume.shape[3] == 1):
                    volume = volume.transpose(1, 0, 2)
                    volume = np.flip(volume, 2)  # U-D
                else:
                    volume = volume.transpose(2, 1, 3, 0)
                    volume = np.flip(volume, 2)  # U-D
            self.data = pad_and_crop(volume, res=res, crop=crop)
            self.plottype = 'volume'
        elif is_ImageND(volume):
            volume = volume.data
            if autotranspose:
                if (volume.ndim == 3) or (volume.shape[3] == 1):
                    volume = volume.transpose(1, 0, 2)
                    volume = np.flip(volume, 2)  # U-D
                else:
                    volume = volume.transpose(2, 1, 3, 0)
                    volume = np.flip(volume, 2)  # U-D
            self.data = pad_and_crop(volume, res=res, crop=crop)
            self.plottype = 'volume'
        elif is_PET_Projection(volume):
            volume = volume.to_nd_array()
            if autotranspose:
                volume = volume.transpose(0, 2, 3, 1)
            self.data = volume #pad_and_crop(volume, res=res, crop=crop)
            self.plottype = 'projection'
        else:
            print("Data format unknown")
            return

        # initialise where the image handles will go
        self.image_handles = None

    def show(self, colormap=None, figsize=(20, 9), clim=None, **kwargs):
        """
        Plot volumetric data.

        Args
        ----
            colormap : str
                    - 'color': show a selection of colorful colormaps
                    - 'mono': show a selection of monochromatic colormaps
                    - <cmap name>: pick one from matplotlib list
            clim : list | tuple
                    lower and upper bound for colorbar (you will have a slider
                    to better refine it later)
            figsize : list | tuple
                    The figure height and width for matplotlib, in inches.
            mask_background : bool
                    Whether the background should be masked (set to NA).
                    This parameter only works in conjunction with the default
                    plotting function (`plotting_func=None`). It finds clusters
                    of values that round to zero and somewhere touch the edges
                    of the image. These are set to NA. If you think you are
                    missing data in your image, set this False.
        """

        cmp_def = ['viridis']
        cmp_monochrome = ['binary', 'Greys', 'gist_yarg'] + \
                         ['Blues', 'BuPu', 'PuBu', 'PuBuGn', 'BuGn'] + \
                         ['bone', 'gray', 'afmhot', 'hot']
        cmp_colorful = ['CMRmap',
                        'gist_stern',
                        'gnuplot',
                        'gnuplot2',
                        'terrain'] + ['jet',
                                      'bwr',
                                      'coolwarm',
                                      ] + ['Spectral',
                                           'seismic',
                                           'BrBG',
                                           ] + ['rainbow',
                                                'nipy_spectral',
                                                ]
        interp_methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
                           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
                           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

        # set default colormap options & add them to the kwargs
        if colormap is None:
            kwargs['colormap'] = cmp_def + ['--- monochrome ---'] + \
                cmp_monochrome + ['---  colorful  ---'] + cmp_colorful
            # kwargs['colormap'] = ['viridis'] + \
            #    sorted(m for m in plt.cm.datad if not m.endswith("_r"))
        elif isinstance(colormap, str):
            # fix cmap if only one given
            if colormap == 'mono':
                kwargs['colormap'] = cmp_monochrome
            elif colormap == 'color':
                kwargs['colormap'] = cmp_def + cmp_colorful
            else:
                kwargs['colormap'] = fixed(colormap)

        kwargs['interp'] = interp_methods


        kwargs['figsize'] = fixed(figsize)
        if self.plottype == 'projection':
            self.views = [
                'Projection plane [u_coord - v_coord]',
                'One-ring projection plane [v_coord - axial_angle]',
                'One-ring projection plane [u_coord - axial_angle]']
            self.directions = ['Axial_angle', 'v_coord', 'u_coord', 'Azimuthal_angle']
        elif self.plottype == 'volume':
            self.views = ['Sagittal',
                          'Coronal',
                          'Axial']
            self.directions = ['x', 'y', 'z', 't']

        self._default_plotter(clim, **kwargs)

    def _default_plotter(self, clim=None, mask_background=False, **kwargs):
        """
        Plot three orthogonal views.
        This is called by nifti_plotter, you shouldn't call it directly.
        """
        data_array = self.data

        if not ((data_array.ndim == 3) or (data_array.ndim == 4)):
            raise ValueError('Input image should be 3D or 4D')

        # mask the background
        if mask_background:
            # TODO: add the ability to pass 'mne' to use a default brain mask
            # TODO: split this out into a different function
            if data_array.ndim == 3:
                labels, n_labels = scipy.ndimage.measurements.label(
                    (np.round(data_array) == 0))
            else:  # 4D
                labels, n_labels = scipy.ndimage.measurements.label(
                    (np.round(data_array).max(axis=3) == 0)
                )

            mask_labels = [lab for lab in range(1, n_labels + 1)
                           if (np.any(labels[[0, -1], :, :] == lab) |
                               np.any(labels[:, [0, -1], :] == lab) |
                               np.any(labels[:, :, [0, -1]] == lab))]

            if data_array.ndim == 3:
                data_array = np.ma.masked_where(
                    np.isin(labels, mask_labels), data_array)
            else:
                data_array = np.ma.masked_where(
                    np.broadcast_to(
                        np.isin(labels, mask_labels)[:, :, :, np.newaxis],
                        data_array.shape
                    ),
                    data_array
                )

        # init sliders for the various dimensions
        for dim, label in enumerate(['x', 'y', 'z']):
            if label not in kwargs.keys():
                kwargs[label] = IntSlider(
                    value=(data_array.shape[dim] - 1) / 2,
                    min=0, max=data_array.shape[dim] - 1,
                    description=self.directions[dim],
                    continuous_update=True
                )

        if (data_array.ndim == 3) or (data_array.shape[3] == 1):
            kwargs['t'] = fixed(0)  # time is fixed
        else:  # 4D
            if self.plottype == 'image':
                desc = 'time'
            elif self.plottype == 'projection':
                desc = 'Azim_angle'
            else:
                desc = 't'
            kwargs['t'] = IntSlider(
                value=data_array.shape[3] // 2,
                min=0,
                max=data_array.shape[3] - 1,
                description=desc,
                continuous_update=True)

        if clim is None:
            clim = (data_array.min(), data_array.max())
        kwargs['clim'] = FloatRangeSlider(
            value=clim,
            min=clim[0],
            max=clim[1],
            step=0.05,
            description='Contrast',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        interact(self._plot_slices, data=fixed(data_array), **kwargs)

    def _plot_slices(self, data, x, y, z, t, clim,
                     colormap='viridis', figsize=(15, 5), interp = 'bicubic'):
        """
        Plot x,y,z slices.
        This function is called by _default_plotter
        """

        if self.image_handles is None:
            self._init_figure(data, colormap, figsize, clim, interp)
        coords = [x, y, z]

        for i, ax in enumerate(self.fig.axes):
            ax.set_title(self.views[i])
        for ii, imh in enumerate(self.image_handles):
            slice_obj = 3 * [slice(None)]
            if data.ndim == 4:
                slice_obj += [t]
            slice_obj[ii] = coords[ii]

            # update the image
            imh.set_data(np.flipud(np.rot90(data[tuple(slice_obj)], k=1)))
            imh.set_clim(clim)

            # draw guides to show selected coordinates
            guide_positions = [val for jj, val in enumerate(coords)
                               if jj != ii]
            imh.axes.lines[0].set_xdata(2 * [guide_positions[0]])
            imh.axes.lines[1].set_ydata(2 * [guide_positions[1]])

            imh.set_cmap(colormap)
            imh.set_interpolation(interp)

        return self.fig

    def _init_figure(self, data, colormap, figsize, clim, interp):

        # init an empty list
        self.image_handles = []
        # open the figure
        self.fig = plt.figure(figsize=figsize)
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        axes = [ax1, ax2, ax3]

        for ii, ax in enumerate(axes):
            ax.set_facecolor('black')
            ax.tick_params(
                axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False
            )
            # fix the axis limits
            axis_limits = [limit for jj, limit in enumerate(data.shape[:3])
                           if jj != ii]
            ax.set_xlim(0, axis_limits[0])
            ax.set_ylim(0, axis_limits[1])

            img = np.zeros(axis_limits[::-1])
            # img[1] = data_max
            im = ax.imshow(img, cmap=colormap, clim=clim, interpolation=interp)
            # add "cross hair"
            ax.axvline(x=0, color='red', alpha=0.8)
            ax.axhline(y=0, color='red', alpha=0.8)
            # append to image handles
            self.image_handles.append(im)


class MultipleVolumes:
    def __init__(self,
                 volumes,
                 axis=0,
                 shrink=256,
                 rotate=90,
                 scales=None,
                 subsample_slices=None,
                 crop=None,
                 open_browser=None):

        self.volumes = volumes
        self._axis = axis
        self._shrink = shrink
        self._rotate = rotate
        self._subsample_slices = subsample_slices
        self._scales = scales
        self._open_browser = open_browser
        self._progress_bar = ProgressBar(color = C.LIGHT_BLUE)
        self._crop = crop

    def get_data(self, volume_index):
        volume = self.volumes[volume_index]
        if isinstance(volume, np.ndarray):
            return volume
        else:
            # Image3D
            return volume.data

    def get_shape(self, volume_index):
        return self.volumes[volume_index].shape

    # display
    def display_in_browser(self, axis=None, max_size=200):
        self.display(axis, max_size, open_browser=True)

    def display(self,
                axis=None,
                shrink=None,
                rotate=None,
                subsample_slices=None,
                scales=None,
                fusion=False,
                normalize=True,
                crop=None,
                global_scale=False,
                open_browser=None):

        if axis is None:
            axis = self._axis
        if shrink is None:
            shrink = self._shrink
        if shrink is None:
            shrink = np.asarray(self.volumes[0].shape).max()
        if rotate is None:
            rotate = self._rotate
        if subsample_slices is None:
            subsample_slices = self._subsample_slices
        if subsample_slices is None:
            subsample_slices = 1
        if crop is None:
            crop = self._crop
        if scales is None:
            scales = self._scales
        if open_browser is None:
            open_browser = self._open_browser
        if open_browser is None:
            open_browser = False

        D = DisplayNode()

        self._progress_bar = ProgressBar()

        volume = self.volumes[0]  # FIXME: optionally use other grid
        # make grid of regularly-spaced points
        box_min = volume.get_world_grid_min()
        box_max = volume.get_world_grid_max()
        span = box_max - box_min
        max_span = span.max()
        n_points = np.uint32(span / max_span * shrink)
        grid = volume.get_world_grid(n_points)
        n = 0
        images = []

        if len(self.volumes) == 2 and fusion == True:
            resampleds = [self.volumes[0].compute_resample_on_grid(grid).data,
                          self.volumes[1].compute_resample_on_grid(grid).data]
            rgb = None

            for slice_index in range(n_points[axis]):
                sequence = []
                for index, resampled in enumerate(resampleds):
                    M = resampled.max()
                    m = resampled.min()
                    resampled = (resampled - m) * (1 / (M - m))

                    if scales is None:
                        scale = 1
                    else:
                        scale = scales[index]

                    if axis == 0:
                        a = np.float32(
                            resampled[slice_index, :, :].reshape(
                                (resampled.shape[1], resampled.shape[2])
                            )
                        )
                    elif axis == 1:
                        a = np.float32(
                            resampled[:, slice_index, :].reshape(
                                (resampled.shape[0], resampled.shape[2])
                            )
                        )
                    else:
                        a = np.float32(
                            resampled[:, :, slice_index].reshape(
                                (resampled.shape[0], resampled.shape[1])
                            )
                        )

                    if normalize:
                        if not global_scale:
                            if scale is None:
                                a = a * 255 / (a.max() + 1e-9)
                            else:
                                a = a * scale * 255 / (a.max() + 1e-9)
                        else:
                            if scale is None:
                                a = a * 255 / (M + 1e-9)
                            else:
                                a = a * scale * 255 / (M + 1e-9)

                    a = self.__crop_img(a, crop)

                    if rgb is None:
                        rgb = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
                    rgb[:, :, index] = a

                im = Image.fromarray(rgb, mode="RGB")

                if rotate is not None:
                    im = im.rotate(rotate, expand=True)

                images.append(im)
                self._progress_bar.set_percentage(
                    slice_index * 100.0 / (n_points[axis])
                )

        else:

            for index in range(len(self.volumes)):
                volume = self.volumes[index]
                resampled = volume.compute_resample_on_grid(grid).data
                self._resampled = resampled
                sequence = []

                if scales is None:
                    scale = 1
                else:
                    scale = scales[index]

                M = resampled.max()
                m = resampled.min()
                resampled = (resampled - m) * (1 / (M - m))

                for slice_index in range(n_points[axis]):
                    if axis == 0:
                        a = np.float32(
                            resampled[slice_index, :, :].reshape(
                                (resampled.shape[1], resampled.shape[2])
                            )
                        )
                    elif axis == 1:
                        a = np.float32(
                            resampled[:, slice_index, :].reshape(
                                (resampled.shape[0], resampled.shape[2])
                            )
                        )
                    else:
                        a = np.float32(
                            resampled[:, :, slice_index].reshape(
                                (resampled.shape[0], resampled.shape[1])
                            )
                        )

                    if normalize:
                        if not global_scale:
                            if scale is None:
                                a = a * 255 / (a.max() + 1e-9)
                            else:
                                a = a * scale * 255 / (a.max() + 1e-9)
                        else:
                            if scale is None:
                                a = a * 255 / (M + 1e-9)
                            else:
                                a = a * scale * 255 / (M + 1e-9)

                    a = self.__crop_img(a, crop)

                    lookup_table = volume.get_lookup_table()
                    im = self.__array_to_im(a, lookup_table)

                    if rotate is not None:
                        im = im.rotate(rotate, expand=True)

                    sequence.append(im)
                    n += 1
                    self._progress_bar.set_percentage(
                        n * 100.0 / (len(self.volumes) * n_points[axis])
                    )
                images.append(sequence)

        if len(images) == 1:
            return D.display("tipix", images[0], open_browser)
        else:
            return D.display("tipix", images, open_browser)

    def __crop_img(self, image, crop):
        if crop is not None:
            [x, y] = np.asarray(image.shape)
            if x > crop:
                startx = x // 2 - (crop // 2)
            else:
                startx = 0
            if y > crop:
                starty = y // 2 - (crop // 2)
            else:
                starty = 0

            return image[startx:startx + crop, starty:starty + crop]
        else:
            return image

    def __array_to_im(self, a, lookup_table):
        if lookup_table is not None:
            red, green, blue, alpha = lookup_table.convert_ndarray_to_rgba(a)
            rgb = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
            rgb[:, :, 0] = red
            rgb[:, :, 1] = green
            rgb[:, :, 2] = blue
            im = Image.fromarray(rgb, mode="RGB")
        else:
            im = Image.fromarray(a).convert("RGB")
        return im

    def _repr_html_(self):
        return self.display()._repr_html_()


#TODO: develop volume renderer based on tomolab.Reconstruction.PET.PET.project_activity
'''
        def quick_render(self, volume, scale=1.0):
            # FIXME: use the VolumeRender object in occiput.Visualization (improve it), the following is a quick fix:
            [offsets, locations] = PET_initialize_compression_structure(180, 1, 256, 256)
            if isinstance(volume, np.ndarray):
                volume = np.float32(volume)
            else:
                volume = np.float32(volume.data)
            subsets_generator = SubsetGenerator(1, 180)
            subsets_matrix = subsets_generator.all_active()
            mask = uniform_cylinder(
                volume.shape,
                volume.shape,
                [0.5 * volume.shape[0], 0.5 * volume.shape[1], 0.5 * volume.shape[2]],
                0.5 * min(volume.shape[0] - 1, volume.shape[1]),
                volume.shape[2],
                2,
                1,
                0,
            )
            volume[np.where(mask.data == 0)] = 0.0
            direction = 7
            block_size = 512
            proj, timing = PET_project_compressed(
                volume,
                None,
                offsets,
                locations,
                subsets_matrix,
                180,
                1,
                np.pi / 180,
                0.0,
                256,
                256,
                256.0,
                256.0,
                256.0,
                256.0,
                256.0,
                256.0,
                256.0,
                256.0,
                128.0,
                128.0,
                128.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
                256,
                1.5,
                0.0,
                0.0,
                0,
                direction,
                block_size,
            )
            proj[np.where(proj > proj.max() / scale)] = proj.max() / scale
            binning = Binning()
            binning.N_axial = 180
            binning.N_azimuthal = 1
            binning.angles_axial = np.float32(np.linspace(0, np.pi - np.pi / 180.0, 180))
            binning.angles_azimuthal = np.float32(np.linspace(0, 0, 1))
            binning.size_u = 256.0
            binning.size_v = 256.0
            binning.N_u = 256
            binning.N_v = 256
            projection = PET_Projection(binning, proj, offsets, locations)
            return projection.uncompress_self()
        '''

class VolumeRenderer():
    """
    3D Volume rendering of reconstructed activity image.
    You can pass a np.ndarray, or an occiput 'ImageND' object,
    or a 'PET_Projection' object.

    Usage
    -----
    V = VolumeRenderer(volume, crop=128)
    V.display()
    """

    def __init__(self, vol, res=(1.0,1.0,1.0), crop=None, autotranspose=False):

        if isinstance(vol, np.ndarray):
            volume = vol
            self.plottype = 'image'
        elif is_ImageND(vol):
            volume = vol.data
            self.plottype = 'image'
        elif is_PET_Projection(vol):
            volume = vol.to_nd_array()
            self.plottype = 'projection'
        else:
            print("Data format unknown")
            return

        volume = pad_and_crop(volume, res=res, crop=crop)
        if volume.min() <0:
                volume = (volume*1.0 - volume.min()) / (volume.max() - volume.min())
        if autotranspose:
            if self.plottype == 'image':
                volume = volume.transpose(0,2,1)
            elif self.plottype == 'projection':
                volume = volume.transpose(0, 2, 3, 1)

        self.volume = volume


    def display(self):
        return ipv.quickvolshow(self.volume)

    def _repr_html_(self):
        return self.display()


H_graph = """
<div id="__id__"></div>
<style type="text/css">
path.link {
  fill: none;
  stroke: #666;
  stroke-width: 1.5px;
}

marker#t0 {
  fill: green;
}

path.link.t0 {
  stroke: green;
}

path.link.t2 {
  stroke-dasharray: 0,2 1;
}

circle {
  fill: #ccc;
  stroke: #333;
  stroke-width: 1.5px;
}

text {
  font: 10px sans-serif;
  pointer-events: none;
}

text.shadow {
  stroke: #fff;
  stroke-width: 3px;
  stroke-opacity: .8;
}
</style>
"""

J_graph = """
<script type="text/Javascript">
require.config({paths: {d3: "http://d3js.org/d3.v3.min"}});
require(["d3"], function(d3) {

var width = 800;
var height = 500;

var graph = __graph_data__;

var nodes = graph['nodes'];
var links = graph['links'];

var nodes2 = {};
var links2 = [];

for (var i=0; i<nodes.length; i++) {
    node = nodes[i];
    nodes2[node['name']] = node;
};

for (var i=0; i<links.length; i++) {
    links2[i] = {'source':nodes2[links[i]['source']], 'target':nodes2[links[i]['target']], 'type':links[i]['type'],};
};


var force = d3.layout.force()
    .nodes(d3.values(nodes2))
    .links(links2)
    .size([width, height])
    .linkDistance(60)
    .charge(-300)
    .on("tick", tick)
    .start();

var svg = d3.select("#__id__").append("svg:svg")
    .attr("width", width)
    .attr("height", height);

// Per-type markers, as they don't inherit styles.
svg.append("svg:defs").selectAll("marker")
    .data(["t0", "t1", "t2"])
  .enter().append("svg:marker")
    .attr("id", String)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 20)
    .attr("refY", -1.5)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
  .append("svg:path")
    .attr("d", "M0,-5L10,0L0,5");

var path = svg.append("svg:g").selectAll("path")
    .data(force.links())
  .enter().append("svg:path")
    .attr("class", function(d) { return "link " + d.type; })
    .attr("marker-end", function(d) { return "url(#" + d.type + ")"; });

colors  = ['#fff','#ccc','#ee4','#dddddd','#eeeeee','#ffffc0'];
strokes = ['#333','#333','#333','#dddddd','#dddddd','#dddddd'];

var circle = svg.append("svg:g").selectAll("circle")
    .data(force.nodes())
  .enter().append("svg:circle")
    .attr("r", 10)
    .call(force.drag)
    .style('fill', function(d){return colors[d.type];})
    .style('stroke', function(d){return strokes[d.type];});

var text = svg.append("svg:g").selectAll("g")
    .data(force.nodes())
  .enter().append("svg:g");

// A copy of the text with a thick white stroke for legibility.
text.append("svg:text")
    .attr("x", 12)
    .attr("y", ".31em")
    .attr("class", "shadow")
    .text(function(d) { return d.name; });

text.append("svg:text")
    .attr("x", 12)
    .attr("y", ".31em")
    .text(function(d) { return d.name; });


// Use elliptical arc path segments to doubly-encode directionality.
function tick() {
  path.attr("d", function(d) {
    var dx = d.target.x - d.source.x,
        dy = d.target.y - d.source.y,
        dr = Math.sqrt(dx * dx + dy * dy);
    return "M" + d.source.x + "," + d.source.y + "A" + dr + "," + dr + " 0 0,1 " + d.target.x + "," + d.target.y;
  });

  circle.attr("transform", function(d) {
    return "translate(" + d.x + "," + d.y + ")";
  });

  text.attr("transform", function(d) {
    return "translate(" + d.x + "," + d.y + ")";
  });
}

});
</script>
"""


class Graph:
    def __init__(self, graph):
        self.set_graph(graph)

    def set_graph(self, g):
        self.graph = g

    def get_html(self):
        div_id = "graph_" + str(uuid.uuid4())
        H = H_graph.replace("__id__", div_id)
        J = J_graph.replace("__id__", div_id)
        J = J.replace("__graph_data__", json.dumps(self.graph))
        return H + J

    def _repr_html_(self):
        graph = HTML(self.get_html())
        return graph._repr_html_()
