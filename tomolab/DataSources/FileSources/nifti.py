# occiput
# Michele Scipioni
# University of Pisa, Italy
# Jan 2018, Pisa

__all__ = ['load_nifti']


import warnings
from ...Core.Conversion import nipy2occiput


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import nibabel


def load_nifti(filename):
    nip = nibabel.load(filename)
    img = nipy2occiput(nip)
    return img