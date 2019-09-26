# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

class SPECT_Projection:
    """SPECT projection object. """

    def __init__(self, data):
        self.data = data

    def get_data(self):
        """Returns the raw projection data (note that is can be accessed also as self.data ). """
        return self.data

    def save_to_file(self, filename):
        h5f = h5py.File(filename, "w")
        h5f.create_dataset("data", data=self.data)
        # h5f.create_dataset('size_x', data=size_x)
        # h5f.create_dataset('size_y', data=size_y)
        h5f.close()

    def get_integral(self):
        return self.data.sum()

    def to_image(self, data, index=0, scale=None, absolute_scale=False):
        from PIL import Image

        a = np.float32(data[:, :, index].reshape((data.shape[0], data.shape[1])))
        if scale is None:
            a = 255.0 * (a) / (a.max() + 1e-12)
        else:
            if absolute_scale:
                a = scale * (a)
            else:
                a = scale * 255.0 * (a) / (a.max() + 1e-12)
        return Image.fromarray(a).convert("RGB")

    def display_in_browser(self, axial=True, azimuthal=False, index=0, scale=None):
        self.display(
            axial=axial, azimuthal=azimuthal, index=index, scale=scale, open_browser=True
        )

    '''
    def display(self, scale=None, open_browser=False):
        data = self.data
        d = DisplayNode()
        images = []
        progress_bar = ProgressBar(
            height="6px",
            width="100%%",
            background_color=C.LIGHT_GRAY,
            foreground_color=C.GRAY,
        )
        if scale is not None:
            scale = scale * 255.0 / (data.max() + 1e-12)
        else:
            scale = 255.0 / (data.max() + 1e-12)
        N_projections = self.data.shape[2]
        N_x = self.data.shape[0]
        N_y = self.data.shape[1]
        print(
            (
                    "SPECT Projection   [N_projections: %d   N_x: %d   N_y: %d]"
                    % (N_projections, N_x, N_y)
            )
        )
        for i in range(N_projections):
            images.append(self.to_image(data, i, scale=scale, absolute_scale=True))
            progress_bar.set_percentage(i * 100.0 / N_projections)
        progress_bar.set_percentage(100.0)
        return d.display("tipix", images, open_browser)
    '''
    def _repr_html_(self):
        return self.display()._repr_html_()