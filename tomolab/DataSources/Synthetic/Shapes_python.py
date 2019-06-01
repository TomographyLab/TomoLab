# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa


import numpy as np


def spherical_phantom(shape=(128,128,47), size=(700,700,500), center=None, radius=100, inner_value=1.0, outer_value=0.0):
    '''
    sp = spherical_phantom(shape=(128,128,47),
                       size=(500,500,500),
                       center=None,
                       radius=150,
                       inner_value=1.0,
                       outer_value=0.0)
    '''
    shape = np.asarray(shape)
    size  = np.asarray(size)
    dx,dy,dz = size/shape
    r_sq = radius**2

    if center is None:center = size / 2
    cx,cy,cz = center

    if inner_value <= 0.0:inner_value = 1e-9
    if outer_value <= 0.0:outer_value = 0.0
    image = np.zeros(shape) + outer_value

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                distance_sq = (x*dx - cx)**2 + (y*dy - cy)**2 + (z*dz - cz)**2
                if distance_sq <= r_sq:
                    image[x,y,z] = inner_value
    return image


def cylindrical_phantom(shape=(128, 128, 47),
                        size=(700, 700, 500),
                        center=None,
                        radius=100,
                        length=200,
                        inner_value=1.0,
                        outer_value=0.0,
                        axis=0):
    '''
    cp = cylindrical_phantom(shape=(128, 128, 47),
                             size=(500, 500, 500),
                             center=None,
                             radius=150,
                             length=300,
                             inner_value=1.0,
                             outer_value=0.0,
                             axis=2)
    '''

    if axis > 2:
        print('Wrong axis selected [0,1,2]')
        return None

    shape = np.asarray(shape)
    size = np.asarray(size)
    dx, dy, dz = size / shape
    r_sq = radius ** 2

    if center is None: center = size / 2
    cx, cy, cz = center

    if inner_value <= 0.0: inner_value = 1e-9
    if outer_value <= 0.0: outer_value = 0.0
    image = np.zeros(shape) + outer_value

    if (axis == 0):
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    if ((x * dx - cx) ** 2 <= (0.5 * length) ** 2):
                        distance_sq = (y * dy - cy) ** 2 + (z * dz - cz) ** 2;
                        if distance_sq <= r_sq:
                            image[x, y, z] = inner_value
    elif (axis == 1):
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    if ((y * dy - cy) ** 2 <= (0.5 * length) ** 2):
                        distance_sq = (x * dx - cx) ** 2 + (z * dz - cz) ** 2;
                        if distance_sq <= r_sq:
                            image[x, y, z] = inner_value
    elif (axis == 2):
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    if ((z * dz - cz) ** 2 <= (0.5 * length) ** 2):
                        distance_sq = (x * dx - cx) ** 2 + (y * dy - cy) ** 2;
                        if distance_sq <= r_sq:
                            image[x, y, z] = inner_value
    return image


def spheres_ring_phantom(shape=(128, 128, 47),
                         size=(700, 700, 500),
                         center=None,
                         ring_radius=100,
                         length=200,
                         min_sphere_radius=2,
                         max_sphere_radius=10,
                         N_spheres=5,
                         inner_value=1.0,
                         outer_value=0.0,
                         axis=0,
                         taper=0):
    '''
    srp = spheres_ring_phantom(shape=(128, 128, 47),
                               size=(500, 500, 500),
                               center=None,
                               length=200,
                               ring_radius=150,
                               min_sphere_radius=10,
                               max_sphere_radius=40,
                               N_spheres=7,
                               inner_value=1.0,
                               outer_value=0.0,
                               axis=2,
                               taper=100)
    '''

    if axis > 2:
        print('Wrong axis selected [0,1,2]')
        return None

    shape = np.asarray(shape)
    size = np.asarray(size)
    dx, dy, dz = size / shape
    nvox = shape[0] * shape[1] * shape[2]

    if center is None: center = size / 2
    cx, cy, cz = center

    if inner_value <= 0.0: inner_value = 1e-9
    if outer_value <= 0.0: outer_value = 0.0
    image = np.zeros(shape) + outer_value
    c = np.zeros((4, 1))

    # make one sphere at the time and sum into the output image
    for s in range(N_spheres):
        # Compute the center of the sphere:
        angle = s * 2 * np.pi / N_spheres
        current_taper = -0.5 * taper + s * taper / (N_spheres - 1)

        if (axis == 0):
            # Ring around the X axis:
            c[0] = cx + current_taper
            c[1] = cy + ring_radius
            c[2] = cz
            c[3] = 1
            rotation_m = create_rotation_matrix(theta=(angle, 0, 0), center=(cx, cy, cz), axis_order='XYZ_ROTATION')
        elif (axis == 1):
            # Ring around the Y axis:
            c[0] = cx
            c[1] = cy + current_taper
            c[2] = cz + ring_radius
            c[3] = 1
            rotation_m = create_rotation_matrix(theta=(0, angle, 0), center=(cx, cy, cz), axis_order='XYZ_ROTATION')
        elif (axis == 2):
            # Ring around the Z axis:
            c[0] = cx + ring_radius
            c[1] = cy
            c[2] = cz + current_taper
            c[3] = 1
            rotation_m = create_rotation_matrix(theta=(0, 0, angle), center=(cx, cy, cz), axis_order='XYZ_ROTATION')

        c_rotated = np.dot(rotation_m, c)

        # The radius of the sphere increases linearly from min_sphere_radius to max_sphere_radius:
        r = min_sphere_radius + s * (max_sphere_radius - min_sphere_radius) / (N_spheres - 1)

        # Make sphere:
        tmpImage = spherical_phantom(shape=shape, size=size,
                                     center=c_rotated[:3], radius=r,
                                     inner_value=inner_value, outer_value=0.0)

        # Set value inside of the spheres (inner_value):
        # for j in range(nvox):
        #    # This sets the value to 'inner_value' also in the regions where two or more spheres eventually intersect:
        #    image[j] += (1-bool(image[j])) * tmpImage.ravel[j];
        image += tmpImage

    '''
    # Set outer value in all voxels that are set to zero (note that if inner_value is set to 0, EPS is added)
    image = image.ravel()
    for i in range(nvox):
        if (image[i]==0.0):
            image[i] = outer_value
    '''

    return image.reshape(shape)


def cylinders_ring_phantom(shape=(128, 128, 47),
                           size=(700, 700, 500),
                           center=None,
                           ring_radius=100,
                           length=200,
                           min_radius=2,
                           max_radius=10,
                           N_elems=5,
                           inner_value=1.0,
                           outer_value=0.0,
                           axis=0,
                           taper=0):

    '''
    srp = cylinders_ring_phantom(shape=(128, 128, 47),
                                 size=(500, 500, 500),
                                 center=None,
                                 ring_radius=100,
                                 length=200,
                                 min_radius=10,
                                 max_radius=40,
                                 N_elems=7,
                                 inner_value=1.0,
                                 outer_value=0.0,
                                 axis=2,
                                 taper=100)
    '''

    if axis > 2:
        print('Wrong axis selected [0,1,2]')
        return None

    shape = np.asarray(shape)
    size = np.asarray(size)
    dx, dy, dz = size / shape
    nvox = shape[0] * shape[1] * shape[2]

    if center is None: center = size / 2
    cx, cy, cz = center

    if inner_value <= 0.0: inner_value = 1e-9
    if outer_value <= 0.0: outer_value = 0.0
    image = np.zeros(shape) + outer_value
    c = np.zeros((4, 1))

    # make one sphere at the time and sum into the output image
    for s in range(N_elems):
        # Compute the center of the sphere:
        angle = s * 2 * np.pi / N_elems
        current_taper = -0.5 * taper + s * taper / (N_elems - 1)

        if (axis == 0):
            # Ring around the X axis:
            c[0] = cx + current_taper
            c[1] = cy + ring_radius
            c[2] = cz
            c[3] = 1
            rotation_m = create_rotation_matrix(theta=(angle, 0, 0), center=(cx, cy, cz), axis_order='XYZ_ROTATION')
        elif (axis == 1):
            # Ring around the Y axis:
            c[0] = cx
            c[1] = cy + current_taper
            c[2] = cz + ring_radius
            c[3] = 1
            rotation_m = create_rotation_matrix(theta=(0, angle, 0), center=(cx, cy, cz), axis_order='XYZ_ROTATION')
        elif (axis == 2):
            # Ring around the Z axis:
            c[0] = cx + ring_radius
            c[1] = cy
            c[2] = cz + current_taper
            c[3] = 1
            rotation_m = create_rotation_matrix(theta=(0, 0, angle), center=(cx, cy, cz), axis_order='XYZ_ROTATION')

        c_rotated = np.dot(rotation_m, c)

        # The radius of the sphere increases linearly from min_sphere_radius to max_sphere_radius:
        r = min_radius + s * (max_radius - min_radius) / (N_elems - 1)

        # Make sphere:
        tmpImage = cylindrical_phantom(shape=shape,
                                       size=size,
                                       center=c_rotated[:3],
                                       radius=r,
                                       length=length,
                                       inner_value=inner_value,
                                       outer_value=0.0,
                                       axis=axis)

        # Set value inside of the spheres (inner_value):
        # for j in range(nvox):
        #    # This sets the value to 'inner_value' also in the regions where two or more spheres eventually intersect:
        #    image[j] += (1-bool(image[j])) * tmpImage.ravel[j];
        image += tmpImage
    '''
    # Set outer value in all voxels that are set to zero (note that if inner_value is set to 0, EPS is added)
    image = image.ravel()
    for i in range(nvox):
        if (image[i]==0.0):
            image[i] = outer_value
    '''

    return image.reshape(shape)


def create_rotation_matrix(theta, center, axis_order='XYZ_ROTATION'):
    status = 0
    theta_x, theta_y, theta_z = theta
    center_x, center_y, center_z = center

    # Initialize affine transform matrix
    s_theta_x = np.sin(theta_x)
    c_theta_x = np.cos(theta_x)
    s_theta_y = np.sin(theta_y)
    c_theta_y = np.cos(theta_y)
    s_theta_z = np.sin(theta_z)
    c_theta_z = np.cos(theta_z)

    rotation_z = np.zeros((4, 4))
    rotation_z[0, 0] = c_theta_z
    rotation_z[0, 1] = -s_theta_z
    rotation_z[1, 0] = s_theta_z
    rotation_z[1, 1] = c_theta_z
    rotation_z[2, 2] = 1.0
    rotation_z[3, 3] = 1.0

    rotation_y = np.zeros((4, 4))
    rotation_y[0, 0] = c_theta_y
    rotation_y[0, 2] = s_theta_y
    rotation_y[2, 0] = -s_theta_y
    rotation_y[2, 2] = c_theta_y
    rotation_y[1, 1] = 1.0
    rotation_y[3, 3] = 1.0

    rotation_x = np.zeros((4, 4))
    rotation_x[1, 1] = c_theta_x
    rotation_x[1, 2] = -s_theta_x
    rotation_x[2, 1] = s_theta_x
    rotation_x[2, 2] = c_theta_x
    rotation_x[0, 0] = 1.0
    rotation_x[3, 3] = 1.0

    translation = np.zeros((4, 4))
    translation[0, 3] = center_x
    translation[1, 3] = center_y
    translation[2, 3] = center_z
    translation[3, 3] = 1.0

    if axis_order == 'XYZ_ROTATION':
        transformationMatrix = np.dot(rotation_y, rotation_x)
        transformationMatrix = np.dot(rotation_z, transformationMatrix)
    elif axis_order == 'XZY_ROTATION':
        transformationMatrix = np.dot(rotation_z, rotation_x)
        transformationMatrix = np.dot(rotation_y, transformationMatrix)
    elif axis_order == 'YXZ_ROTATION':
        transformationMatrix = np.dot(rotation_x, rotation_y)
        transformationMatrix = np.dot(rotation_z, transformationMatrix)
    elif axis_order == 'YZX_ROTATION':
        transformationMatrix = np.dot(rotation_z, rotation_y)
        transformationMatrix = np.dot(rotation_x, transformationMatrix)
    elif axis_order == 'ZXY_ROTATION':
        transformationMatrix = np.dot(rotation_x, rotation_z)
        transformationMatrix = np.dot(rotation_y, transformationMatrix)
    elif axis_order == 'ZYX_ROTATION':
        transformationMatrix = np.dot(rotation_y, rotation_z)
        transformationMatrix = np.dot(rotation_x, transformationMatrix)
    else:  # default : 'XYZ_ROTATION'
        transformationMatrix = np.dot(rotation_y, rotation_x)
        transformationMatrix = np.dot(rotation_z, transformationMatrix)

    translation = np.dot(transformationMatrix, translation)

    transformationMatrix[0, 3] = center_x - translation[0, 3]
    transformationMatrix[1, 3] = center_y - translation[1, 3]
    transformationMatrix[2, 3] = center_z - translation[2, 3]

    '''
    print("=============== Input: ==============\n")
    print("Cx: %3.3f  Cy: %3.3f  Cz: %3.3f  \n"%(center_x, center_y, center_z))
    print("Rx: %3.3f  Ry: %3.3f  Rz: %3.3f  \n"%(theta_x, theta_y, theta_z))
    print("=======    Rotation matrix:    ======\n")
    for i in range(4):
        print("[%3.3f  %3.3f  %3.3f  %3.3f]\n"%(transformationMatrix[i,0],transformationMatrix[i,1],transformationMatrix[i,2],transformationMatrix[i,3]))
    print("=====================================\n")
    '''
    return transformationMatrix