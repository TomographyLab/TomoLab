# -*- coding: utf-8 -*-
# occiput
# Harvard University, Martinos Center for Biomedical Imaging
# Aalto University, Department of Computer Science


# FIXME: perhaps move this somewhere else
try:
    import ipy_table
    has_ipy_table = True
except BaseException:
    print("Please install ipy_table (e.g. 'easy_install ipy_table') to enable ipython notebook tables. ")
    ipy_table = None
    has_ipy_table = False

# svg_write
# FIXME: perhaps move this somewhere else
try:
    import svgwrite
    has_svgwrite = True
except BaseException:
    print("Please install svgwrite (e.g. 'easy_install svgwrite') to enable svg visualisations. ")
    svgwrite = None
    has_svgwrite = False


def is_in_ipynb():
    try:
        from IPython import get_ipython
        chk = str(get_ipython()).split(".")[1]
        if chk == 'zmqshell':
            return True
        else:
            return False
    except BaseException:
        return False


def is_ImageND(obj):
    from .Core.DataStructures import Image3D, ImageND
    return obj is not None \
           and obj.__class__.__module__ == 'tomolab.Core.DataStructures' \
           and (isinstance(obj, Image3D)  or isinstance(obj, ImageND))

def is_PET_Projection(obj):
    from .Reconstruction.PET.PET_projection import PET_Projection
    return obj is not None \
           and obj.__class__.__module__ == 'tomolab.Reconstruction.PET.PET_projection' \
           and isinstance(obj, PET_Projection)

# GPU enables / disable
__use_gpu = True


def enable_gpu():
    global __use_gpu
    __use_gpu = True


def disable_gpu():
    global __use_gpu
    __use_gpu = False


def is_gpu_enabled():
    global __use_gpu
    return __use_gpu


# Set level of verbosity when printing to stdout - for debugging
__verbose = 1


def set_verbose_high():
    """Print everything - DEBUG mode"""
    global __verbose
    __verbose = 2


def set_verbose_medium():
    """Print runtime information"""
    global __verbose
    __verbose = 1


def set_verbose_low():
    """Print only important messages"""
    global __verbose
    __verbose = 0


def set_verbose_no_printing():
    """Do not print messages at all"""
    global __verbose
    __verbose = -1


def get_verbose_level():
    return __verbose


def print_debug(msg):
    """Use this for DEBUG Information"""
    if __verbose >= 2:
        print(msg)


def print_runtime(msg):
    """Use this for messages useful at runtime"""
    if __verbose >= 1:
        print(msg)


def print_important(msg):
    """Use this for important messages"""
    if __verbose >= 0:
        print(msg)


# Other print options
import contextlib as __contextlib
import numpy as __numpy

@__contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = __numpy.get_printoptions()
    __numpy.set_printoptions(*args, **kwargs)
    yield
    __numpy.set_printoptions(**original)

# Default background of images
__background = 0.0

def set_default_background(bg):
    global __background
    __background = bg


def get_default_background():
    global __background
    return __background
