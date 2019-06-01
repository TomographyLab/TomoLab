# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

__all__ = ['uniform_cylinder','uniform_sphere','uniform_spheres_ring','uniform_cylinders_ring','complex_phantom']

from ...Core import Image3D
from tomolab.Core.NiftyRec import  ET_spherical_phantom, \
                                    ET_cylindrical_phantom, \
                                    ET_spheres_ring_phantom, \
                                    ET_cylinders_ring_phantom

class InstallationError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


def uniform_sphere(
    shape=(256, 256, 256),
    size=(256, 256, 256),
    center=(128, 128, 128),
    radius=90,
    inner_value=1.0,
    outer_value=0.0,
):
    """Create volume (3D numpy array) with uniform value within a spherical region. """
    return Image3D(
        ET_spherical_phantom(shape, size, center, radius, inner_value, outer_value)
    )


def uniform_cylinder(
    shape=(256, 256, 256),
    size=(256, 256, 256),
    center=(128, 128, 128),
    radius=90,
    length=200,
    axis=2,
    inner_value=1.0,
    outer_value=0.0,
):
    """Create volume (3D numpy array) with uniform value within a spherical region. """
    return Image3D(
        ET_cylindrical_phantom(
            shape, size, center, radius, length, axis, inner_value, outer_value
        )
    )


def uniform_spheres_ring(
    shape=(256, 256, 256),
    size=(256, 256, 256),
    center=(128, 128, 128),
    ring_radius=90,
    min_radius=5,
    max_radius=30,
    N_elems=6,
    inner_value=1.0,
    outer_value=0.0,
    taper=0,
    axis=2,
):
    """Create volume (3D numpy array) with uniform value within a spherical region. """
    return Image3D(
        ET_spheres_ring_phantom(
            shape,
            size,
            center,
            ring_radius,
            min_radius,
            max_radius,
            N_elems,
            inner_value,
            outer_value,
            taper,
            axis,
        )
    )

def uniform_cylinders_ring(
    shape=(256, 256, 256),
    size=(256, 256, 256),
    center=(128, 128, 128),
    ring_radius=90,
    length=200,
    min_radius=5,
    max_radius=30,
    N_elems=6,
    inner_value=1.0,
    outer_value=0.0,
    taper=0,
    axis=2,
):
    """Create volume (3D numpy array) with uniform value within a spherical region. """
    return Image3D(
        ET_cylinders_ring_phantom(
            shape,
            size,
            center,
            ring_radius,
            length,
            min_radius,
            max_radius,
            N_elems,
            inner_value,
            outer_value,
            taper,
            axis,
        )
    )

def complex_phantom(
        shape=(128, 128, 47),
        size=(500, 500, 500),
        center=(250, 250, 250),
        radius=180,
        insert_radius = 120,
        hole_radius = 50,
        length=450,
        insert_length = 300,
        insert_min_radius=10,
        insert_max_radius=40,
        insert_N_elems=8,
        inner_value=1.0,
        insert_value=1.0,
        outer_value=0.0,
        axis=2,
):

    insert = uniform_cylinders_ring(shape=shape,
                                    size=size,
                                    center=center,
                                    ring_radius=insert_radius,
                                    min_radius=insert_min_radius,
                                    max_radius=insert_max_radius,
                                    length=insert_length,
                                    N_elems=insert_N_elems,
                                    inner_value=insert_value,
                                    outer_value=0.0,
                                    taper=0,
                                    axis=axis,
                                    )

    hole = uniform_cylinder(shape=shape,
                            size=size,
                            center=center,
                            radius=hole_radius,
                            length=length,
                            inner_value=inner_value,
                            outer_value=outer_value,
                            axis=axis,
                            )

    bg = uniform_cylinder(shape=shape,
                          size=size,
                          center=center,
                          radius=radius,
                          length=length,
                          inner_value=inner_value,
                          outer_value=outer_value,
                          axis=axis,
                          )

    activity = bg + insert - hole

    return activity