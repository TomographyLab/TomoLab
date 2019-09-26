# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa

import json
import numpy as np

from ..global_settings import printoptions, has_ipy_table, ipy_table
from ..Core.Errors import UnknownParameter
from . import transformations_operations as tr

__all__ = ['Transform_Affine', 'Transform_6DOF',
           'Transform_Identity', 'Transform_Rotation', 'Transform_Scale',
           'Transform_Translation']

class RigidTransform:
    """Region of Interest. Legacy! """
    # FIXME: eliminate the class RigidTransform; use transformation matrices in Image3D instead,
    #  for activity and attenuation volumes.
    #   Use Core.Transform_6DOF or Core.Transform_Affine if required, to parameterize the projector and back_projector.

    def __init__(self, parameters=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        if type(parameters) == dict:
            self.load_from_dictionary(parameters)
        elif type(parameters) in [list, tuple]:
            if len(parameters) == 6:
                self.x = parameters[0]
                self.y = parameters[1]
                self.z = parameters[2]
                self.theta_x = parameters[3]
                self.theta_y = parameters[4]
                self.theta_z = parameters[5]
            else:
                raise UnknownParameter(
                    "Parameter %s specified for the construction of RigidTransform is not compatible. "
                    % str(parameters)
                )
        else:
            raise UnknownParameter(
                "Parameter %s specified for the construction of RigidTransform is not compatible. "
                % str(parameters)
            )

    def load_from_dictionary(self, dictionary):
        self.x = dictionary["x"]  # Translation along x
        self.y = dictionary["y"]  # Translation along y
        self.z = dictionary["z"]  # Translation along z
        self.theta_x = dictionary["theta_x"]  # Rotation around x
        self.theta_y = dictionary["theta_y"]  # Rotation around y
        self.theta_z = dictionary["theta_z"]  # Rotation around z

    def __repr__(self):
        s = "PET volume location (RigidTransform): \n"
        s = s + " - x:             %f \n" % self.x
        s = s + " - y:             %f \n" % self.y
        s = s + " - z:             %f \n" % self.z
        s = s + " - theta_x:       %f \n" % self.theta_x
        s = s + " - theta_y:       %f \n" % self.theta_y
        s = s + " - theta_z:       %f \n" % self.theta_z
        return s

    def _repr_html_(self):
        if not has_ipy_table:
            return "Please install ipy_table."
        table_data = [
            ["x", self.x],
            ["y", self.y],
            ["z", self.z],
            ["theta_x", self.theta_x],
            ["theta_y", self.theta_y],
            ["theta_z", self.theta_z],
        ]
        table = ipy_table.make_table(table_data)
        table = ipy_table.apply_theme("basic_left")
        # table = ipy_table.set_column_style(0, color='lightBlue')
        table = ipy_table.set_global_style(float_format="%3.3f")
        return table._repr_html_()

class Transform_Affine(object):
    """Affine transformation. Transformations map from a space to another, e.g. from
    the space of voxel indices (with unit measure of voxels) to scanner coordinates
    (e.g. with unit measures of mm). Occiput gives names to spaces in order to verify
    consistency when applying a spatial transformation: e.g. a transformation that maps
    from voxel indexes to scanner coordinates cannot be used to transform a set of points
    from scanner coordinates to MNI space.
    'map_from' and 'map_to' store the names of the spaces.

    Attributes:
        data (ndarray): 4x4 affine transformation matrix.
        map_from (str): name of origin coordinate system.
        map_to (str): name of destination coordinate system. """

    def __init__(self, data=None, map_from="", map_to=""):
        self.set_map_from(map_from)
        self.set_map_to(map_to)
        self.set_data(data)

    def can_inverse_left_multiply(self, obj):
        return self.__space_of_obj(obj) == self.map_to

    def can_left_multiply(self, obj):
        return self.__space_of_obj(obj) == self.map_from

    def left_multiply(self, obj):
        # if obj is a nd-array (of size [4xN]):
        if isinstance(obj, np.ndarray):
            return np.dot(self.data, obj)
        # verify if object and affine are in the same space
        if not self.can_left_multiply(obj):
            if hasattr(obj, "map_from"):
                X = obj.map_from
            else:
                X = "X"
            print(("Affine is incompatible with the given object: composing [%s,%s] \
                    with [%s,%s] is not allowed. "
                    % (self.map_to, self.map_from,self.__space_of_obj(obj),X)))
        data = np.dot(self.data, obj.data)
        # return an affine transformation is the input was an affine transformation
        if isinstance(obj, Transform_Affine):
            return Transform_Affine(data=data, map_from=obj.map_from,
                                    map_to=self.map_to)
        else:
            return data

    def __space_of_obj(self, obj):
        if hasattr(obj, "space"):
            return obj.space
        elif hasattr(obj, "map_to"):
            return obj.map_to
        else:
            return ""

    def export_dictionary(self):
        """Export transformation as Python dictionary.

        Returns:
            dict: {'map_to':str, 'map_from':str, 'data':list}
        """
        return {
            "map_to": self.map_to,
            "map_from": self.map_from,
            "data": self.data.tolist(),
        }

    def export_json(self):
        """Export transformation as JSON string.

        Returns:
            dict: "{'map_to':str, 'map_from':str, 'data':list}"
        """
        return json.dumps(self.export_dictionary())

    def save_to_file(self, filename):
        with open(filename, "w") as fid:
            json.dump(self.export_json(), fid)

    def load_from_file(self, filename):
        with open(filename, "r") as fid:
            self.import_json(json.load(fid))

    def import_json(self, string):
        self.import_dictionary(json.loads(string))

    def import_dictionary(self, dict):
        self.map_to = dict["map_to"]
        self.map_from = dict["map_from"]
        self.data = np.asarray(dict["data"])

    def is_6DOF(self):
        # tra, rot, axis = self.to_translation_rotation()
        # if tra is None:
        #    return False
        # mat = np.dot(tr.translation_matrix(tra), tr.rotation_matrix(rot,axis))
        # return tr.is_same_transform(mat,self.data)
        return True  # FIXME

    # FIXME: implement the following functions (see is_6DOF() )
    def is_9DOF(self):
        tra, scale, rot, rot_axis = self.to_translation_scale_rotation()
        if tra is None:
            return False
        tra_mat = tr.translation_matrix(tra)
        scale_mat = np.diag([scale[0], scale[1], scale[2], 1.0])
        rot_mat = tr.rotation_matrix(rot, rot_axis)
        mat = np.dot(tra_mat, np.dot(scale_mat, rot_mat))
        return tr.is_same_transform(mat, self.data)

    def is_rigid(self):
        # return self.is_9DOF()
        return True

    def is_rotation(self):
        return False  # FIXME

    def is_translation(self):
        return False  # FIXME

    def is_scale(self):
        return False  # FIXME

    def determinant(self):
        try:
            det = np.linalg.det(self.data)
        except:
            det = None
        return det

    def to_translation_rotation(self):
        tra = tr.translation_from_matrix(self.data)
        mat = self.data.copy()
        mat[0:3, 3] = 0
        try:
            rot, axis, point = tr.rotation_from_matrix(mat)
        except ValueError:
            # print("Not a rotation matrix. ")
            return None, None, None
        return tra, rot, axis

    def from_translation_rotation(self, tra, rot, axis=[0, 0, 1]):
        return np.dot(tr.translation_matrix(tra),
                         tr.rotation_matrix(rot, axis))

    def to_translation_scale_rotation(self):
        # FIXME
        mat = self.data.copy()
        tra = tr.translation_from_matrix(mat)
        tra_mat = tr.translation_matrix(tra)
        mat = np.dot(np.linalg.inv(tra_mat), mat)
        factor, origin, direction = tr.scale_from_matrix(mat)
        scale_mat = tr.scale_matrix(factor, origin, direction)
        scale = np.diag(scale_mat)
        mat = np.dot(np.linalg.inv(scale_mat), mat)
        try:
            rot, rot_axis, point = tr.rotation_from_matrix(mat)
        except ValueError:
            # print("Not a rotation matrix. ")
            return None, None, None, None
        # print("Rotation axis: ",rot_axis)
        return tra, scale, rot, rot_axis

    def to_quaternion_translation(self):
        tra = tr.translation_from_matrix(self.data)
        qua = tr.quaternion_from_matrix(self.data)
        return qua, tra

    def from_quaternion_translation(self, quaternion, translation):
        rot = tr.quaternion_matrix(quaternion)
        return np.dot(translation, rot)

    def derivative_parameters(self, gradient_transformed_image):
        pass
        # FIXME implement (implemented in case of 6DOF in the derived class Transform_6DOF)

    def __get_inverse(self):
        inverse = Transform_Affine(
            data=self.__inverse, map_to=self.map_from,
            map_from=self.map_to
        )
        return inverse

    def __compute_inverse(self):
        self.__inverse = np.linalg.inv(self.data)

    def __get_data(self):
        return self.__data

    def set_data(self, data):
        if not isinstance(data, np.ndarray):
            if data is None:
                data = np.eye(4)
            else:
                # FIXME: raise exception
                return
        if not data.size == 16:
            # FIXME: raise exception
            return
        self.__data = data
        self.__compute_inverse()

    def __get_map_to(self):
        return self.__map_to

    def set_map_to(self, map_to):
        self.__map_to = map_to

    def __get_map_from(self):
        return self.__map_from

    def set_map_from(self, map_from):
        self.__map_from = map_from

    def __repr__(self):
        with printoptions(precision=4, suppress=True):
            s = (
                    "Transformation "
                    + "\n-from: "
                    + str(self.map_from)
                    + "\n-to:   "
                    + str(self.map_to)
                    + "\n-matrix: \n"
                    + self.__data.__repr__()
            )
        return s

    def _repr_html_(self):
        if not has_ipy_table:
            return "Please install ipy_table."
        s = "Transformation"
        table_data = [
            [s, "", "", "", "", "", ""],
            [
                "map_from",
                "map_to",
                "determinant",
                "is_rigid",
                "is_6DOF",
                "is_rotation",
                "is_translation",
            ],
            [
                self.map_from,
                self.map_to,
                str(self.determinant()),
                self.is_rigid(),
                self.is_6DOF(),
                self.is_rotation(),
                self.is_translation(),
            ],
        ]
        table = ipy_table.make_table(table_data)
        table = ipy_table.apply_theme("basic")
        table = ipy_table.set_cell_style(0, 0, column_span=7)
        table = ipy_table.set_cell_style(0, 0, align="center")
        table = ipy_table.set_row_style(1, color="#F7F7F7")
        table = ipy_table.set_row_style(2, color="#F7F7F7")
        table = ipy_table.set_row_style(0, color="#FFFF99")
        table = ipy_table.set_row_style(1, bold=True)
        table = ipy_table.set_row_style(1, align="center")
        table = ipy_table.set_row_style(1, height=30)
        table = ipy_table.set_global_style(float_format="%3.3f")
        table2 = ipy_table.make_table(self.data)
        table3 = ipy_table.make_table(["   "])
        table3 = ipy_table.set_row_style(0, no_border="all")
        s = (
                "<div><style>table {float: left; margin-right:10px;}</style> %s %s %s </div>"
                % (
                    table._repr_html_(), table3._repr_html_(),
                    table2._repr_html_())
        )
        return s

    data = property(__get_data, set_data)
    map_to = property(__get_map_to, set_map_to)
    map_from = property(__get_map_from, set_map_from)
    inverse = property(__get_inverse)


class Transform_Identity(Transform_Affine):
    def __init__(self, map_from="", map_to=""):
        Transform_Affine.__init__(self, np.eye(4), map_from, map_to)

# class Transform_Scale(Transform_Affine):
#    def __init__(self, factor, origin=None, direction=None, map_from="",map_to="" ):
#        mat = tr.scale_matrix(factor, origin, direction)
#        Transform_Affine.__init__(self, mat, map_from,map_to)
class Transform_Scale(Transform_Affine):
    # FIXME: figure out what transformations_operations.py means by scale matrix
    def __init__(self, scale_xyz, map_from="", map_to=""):
        mat = np.diag([scale_xyz[0], scale_xyz[1], scale_xyz[2], 1])
        Transform_Affine.__init__(self, mat, map_from, map_to)


class Transform_Rotation(Transform_Affine):
    def __init__(self, angle, direction, point=None, map_from="", map_to=""):
        mat = tr.rotation_matrix(angle, direction, point)
        Transform_Affine.__init__(self, mat, map_from, map_to)

    def derivative_parameters(self, gradient_transformed_image):
        pass


class Transform_Translation(Transform_Affine):
    def __init__(self, direction, map_from="", map_to=""):
        mat = tr.translation_matrix(direction)
        Transform_Affine.__init__(self, mat, map_from, map_to)

    def derivative_parameters(self, gradient_transformed_image):
        pass


class Transform_6DOF(Transform_Affine):
    def __init__(
            self,
            translation,
            rotation_angle,
            rotation_direction,
            rotation_point=None,
            map_from="",
            map_to="",
    ):
        rot = tr.rotation_matrix(rotation_angle, rotation_direction,
                                 rotation_point)
        tra = tr.translation_matrix(translation)
        mat = np.dot(tra, rot)
        Transform_Affine.__init__(self, mat, map_from, map_to)

    def derivative_parameters(self, gradient_transformed_image,
                              grid_transformed_image):
        dRx = []
        dRy = []
        dRz = []
        return [0, 0, 0, 0, 0, 0]