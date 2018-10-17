import numpy as np
from .parameter_set import ParameterSet, Parameter, ClassWithParameterSet, TYPE_EXTRINSIC1, TYPE_EXTRINSIC2
import json


class SpatialOrientation(ClassWithParameterSet):
    """
        CameraTransform class to calculate the position of objects from an image in 3D based
        on camera intrinsic parameters and observer position

        Parameters
        ----------
        focal_length: number
            focal length of the camera in mm
        sensor_size: tuple or number
            sensor size in mm, can be either a tuple (width, height) or just a the width, then the height
            is inferred from the aspect ratio of the image
        image_size: tuple or ndarray
            image size in pixel [width, height] or a numpy array representing the image
        observer_height: number
            observer elevation in m
        angel_to_horizon: number
            angle between the z-axis and the horizon
        """

    t = None
    R = None
    C = None
    C_inv = None

    R_earth = 6371e3

    def __init__(self, elevation_m=None, tilt_deg=None, roll_deg=None, heading_deg=None, pos_x_m=None, pos_y_m=None):
        self.parameters = ParameterSet(
            # the extrinsic parameters if the camera will not be compared to other cameras or maps
            elevation_m=Parameter(elevation_m, default=30, range=(0, None), type=TYPE_EXTRINSIC1),
            # the elevation of the camera above sea level in m
            tilt_deg=Parameter(tilt_deg, default=85, range=(-90, 90), type=TYPE_EXTRINSIC1),  # the tilt angle of the camera in degrees
            roll_deg=Parameter(roll_deg, default=0, range=(-180, 180), type=TYPE_EXTRINSIC1),  # the roll angle of the camera in degrees

            # the extrinsic parameters if the camera will be compared to other cameras or maps
            heading_deg=Parameter(heading_deg, default=0, type=TYPE_EXTRINSIC2),  # the heading angle of the camera in degrees
            pos_x_m=Parameter(pos_x_m, default=0, type=TYPE_EXTRINSIC2),  # the x position of the camera in m
            pos_y_m=Parameter(pos_y_m, default=0, type=TYPE_EXTRINSIC2),  # the y position of the camera in m
        )
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._initCameraMatrix
        self._initCameraMatrix()

    def __str__(self):
        string = ""
        string += "  position:\n"
        string += "    x:\t%f m\n    y:\t%f m\n    h:\t%f m\n" % (self.parameters.pos_x_m, self.parameters.pos_y_m, self.parameters.elevation_m)
        string += "  orientation:\n"
        string += "    tilt:\t\t%f°\n    roll:\t\t%f°\n    heading:\t%f°\n)" % (self.parameters.tilt_deg, self.parameters.roll_deg, self.parameters.heading_deg)
        return string

    def _initCameraMatrix(self, height=None, tilt_angle=None, roll_angle=None):
        if self.heading_deg < -360 or self.heading_deg > 360:
            self.heading_deg = self.heading_deg % 360
        # convert the angle to radians
        tilt = np.deg2rad(self.parameters.tilt_deg)
        roll = np.deg2rad(self.parameters.roll_deg)
        heading = np.deg2rad(self.parameters.heading_deg)

        # get the translation matrix and rotate it
        self.t = np.array([[self.parameters.pos_x_m, -self.parameters.pos_y_m, -self.parameters.elevation_m]]).T

        # construct the rotation matrices for tilt, roll and heading
        self.R_tilt = np.array([[1, 0, 0],
                                [0, np.cos(tilt), np.sin(tilt)],
                                [0, -np.sin(tilt), np.cos(tilt)]])
        self.R_roll = np.array([[+np.cos(roll), np.sin(roll), 0],
                                [-np.sin(roll), np.cos(roll), 0],
                                [0, 0, 1]])
        self.R_head = np.array([[+np.cos(heading), np.sin(heading), 0],
                                [-np.sin(heading), np.cos(heading), 0],
                                [0, 0, 1]])

        # rotate the translation around the tilt angle
        self.t = np.dot(self.R_tilt, np.dot(self.R_head, self.t))

        # get the rotation-translation matrix with the rotation composed with the translation
        self.R = np.vstack((np.hstack((np.dot(np.dot(self.R_roll, self.R_tilt), self.R_head), self.t)), [0, 0, 0, 1]))

        # compose the camera matrix with the rotation-translation matrix
        self.C = self.R
        # to get the x coordinate right, mirror the x direction
        self.C[:, 0] = -self.C[:, 0]
        self.C_inv = np.linalg.inv(self.C)

    def transformPoints(self, points):  # transform Space -> Camera
        points = np.array(points)
        points = np.hstack((points, points[..., 0:1]*0+1))
        return np.dot(points, self.C.T)[..., :-1]

    def transformInvertedPoints(self, points, direction=False):  # transform Space -> Camera
        points = np.array(points)
        points = np.hstack((points, points[..., 0:1]*0+(1-direction)))
        return np.dot(points, self.C_inv.T)[..., :-1]

    def cameraFromSpace(self, points):
        return self.transformPoints(points)

    def spaceFromCamera(self, points, direction=False):
        return self.transformInvertedPoints(points, direction)

    def save(self, filename):
        keys = self.parameters.parameters.keys()
        export_dict = {key: getattr(self, key) for key in keys}
        with open(filename, "w") as fp:
            fp.write(json.dumps(export_dict))

    def load(self, filename):
        with open(filename, "r") as fp:
            variables = json.loads(fp.read())
        for key in variables:
            setattr(self.parameters, key, variables[key])
