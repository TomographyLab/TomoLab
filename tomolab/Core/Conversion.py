# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa


import nibabel
from .DataStructures import Image3D, ImageND
from nipy.io.nifti_ref import nifti2nipy
import numpy as np

'''
def nifti2occiput(nif):
    """Convert Nifti image to occiput ImageND image.

        Args:
            nif (Nifti): Nifti image.

        Returns:
            ImageND: occiput image.
        """
    nip = nifti2nipy(nif)
    return nipy2occiput(nip)
'''

def nipy2occiput(nib):
    """Convert nipy image to Occiput ImageND image.

        Args:
            img (nipy): nipy image.

        Returns:
            ImageND: occiput image. """
    ndim = len(nib.shape)
    affine = nib.affine
    #affine[:,-1] = np.asarray([0.0,0.0,0.0,1.0]) #TODO: check if it's legit
    if ndim == 3:
        im = Image3D(
            data=nib.get_data(), affine=affine,
            space="world", header=nib.header)
    else:
        im = ImageND(
            data=nib.get_data(), affine=affine,
            space="world", header=nib.header)
    return im


def occiput2nifti(occ):
    """Conver occiput ImageND to Nifti image.

        Args:
            occ (ImageND): occiput ImageND image.

        Returns:
            Nifti: Nifti image.
        """
    nii = nibabel.nifti1.Nifti1Image(occ.data, occ.affine.data, occ.header)
    return nii


def numpy2occiput(array):
    """Numpy ndarray to occiput ImageND image.

        Args:
            array (ndarray): numpy.ndarray.

        Returns:
            ImageND: occiput ImageND image.
        """
    if array.ndim == 3:
        im = Image3D( data=array, space="world")
    else:
        raise ("Currently only conversion of 3D arrays is supported. ")
    return im
