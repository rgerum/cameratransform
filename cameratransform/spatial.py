#!/usr/bin/env python
# -*- coding: utf-8 -*-
# spatial.py

# Copyright (c) 2017-2021, Richard Gerum
#
# This file is part of the cameratransform package.
#
# cameratransform is free software: you can redistribute it and/or modify
# it under the terms of the MIT licence.
#
# cameratransform is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
#
# You should have received a copy of the license
# along with cameratransform. If not, see <https://opensource.org/licenses/MIT>

import numpy as np
from .parameter_set import ParameterSet, Parameter, ClassWithParameterSet, TYPE_EXTRINSIC1, TYPE_EXTRINSIC2
import json


class SpatialOrientation(ClassWithParameterSet):
    r"""
    The orientation can be represented as a matrix multiplication in *projective coordinates*. First, we define rotation
    matrices around the three angles: *tilt*, *roll*, *heading*:

    .. math::
        R_{\mathrm{roll}} &=
        \begin{pmatrix}
        \cos(\alpha_\mathrm{roll}) & \sin(\alpha_\mathrm{roll}) & 0\\
        -\sin(\alpha_\mathrm{roll}) & \cos(\alpha_\mathrm{roll}) & 0\\
        0 & 0 & 1\\
         \end{pmatrix}\\
        R_{\mathrm{tilt}} &=
        \begin{pmatrix}
        1 & 0 & 0\\
        0 & \cos(\alpha_\mathrm{tilt}) & \sin(\alpha_\mathrm{tilt}) \\
        0 & -\sin(\alpha_\mathrm{tilt}) & \cos(\alpha_\mathrm{tilt}) \\
         \end{pmatrix}\\
         R_{\mathrm{heading}} &=
        \begin{pmatrix}
        \cos(\alpha_\mathrm{heading}) & -\sin(\alpha_\mathrm{heading}) & 0\\
        \sin(\alpha_\mathrm{heading}) & \cos(\alpha_\mathrm{heading}) & 0\\
        0 & 0 & 1\\
         \end{pmatrix}

    These angles correspond to ZXZ-Euler angles.

    And the position *x*, *y*, *z* (=elevation):

    .. math::
        t &=
        \begin{pmatrix}
        x\\
        y\\
        \mathrm{elevation}
         \end{pmatrix}

    We combine the rotation matrices to a single rotation matrix:

    .. math::
        R &=  R_{\mathrm{roll}} \cdot  R_{\mathrm{tilt}} \cdot  R_{\mathrm{heading}}\\

    and use this matrix to convert from the **camera coordinates** to the **space coordinates** and vice versa:

    .. math::
        x_\mathrm{camera} = R \cdot (x_\mathrm{space} - t)\\
        x_\mathrm{space} = R^{-1} \cdot x_\mathrm{space} + t\\

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
        string += "    tilt:\t\t%f°\n    roll:\t\t%f°\n    heading:\t%f°\n" % (self.parameters.tilt_deg, self.parameters.roll_deg, self.parameters.heading_deg)
        return string

    def _initCameraMatrix(self, height=None, tilt_angle=None, roll_angle=None):
        if self.heading_deg < -360 or self.heading_deg > 360:  # pragma: no cover
            self.heading_deg = self.heading_deg % 360
        # convert the angle to radians
        tilt = np.deg2rad(self.parameters.tilt_deg)
        roll = np.deg2rad(self.parameters.roll_deg)
        heading = np.deg2rad(self.parameters.heading_deg)

        # get the translation matrix and rotate it
        self.t = np.array([self.parameters.pos_x_m, self.parameters.pos_y_m, self.parameters.elevation_m])

        # construct the rotation matrices for tilt, roll and heading
        self.R_roll = np.array([[+np.cos(roll), np.sin(roll), 0],
                                [-np.sin(roll), np.cos(roll), 0],
                                [0, 0, 1]])
        self.R_tilt = np.array([[1, 0, 0],
                                [0, np.cos(tilt), np.sin(tilt)],
                                [0, -np.sin(tilt), np.cos(tilt)]])
        self.R_head = np.array([[np.cos(heading), -np.sin(heading), 0],
                                [np.sin(heading), np.cos(heading), 0],
                                [0, 0, 1]])

        self.R = np.dot(np.dot(self.R_roll, self.R_tilt), self.R_head)
        self.R_inv = np.linalg.inv(self.R)

    def cameraFromSpace(self, points):
        """
        Convert points (Nx3) from the **space** coordinate system to the **camera** coordinate system.

        Parameters
        ----------
        points : ndarray
            the points in **space** coordinates to transform, dimensions (3), (Nx3)

        Returns
        -------
        points : ndarray
            the points in the **camera** coordinate system, dimensions (3), (Nx3)

        Examples
        --------

        >>> import cameratransform as ct
        >>> orientation = ct.SpatialOrientation(elevation_m=15.4, tilt_deg=85)

        transform a single point from the space to the image:

        >>> orientation.spaceFromCamera([-0.09, -0.27, -1.00])
        [0.09 0.97 15.04]

        or multiple points in one go:

        >>> orientation.spaceFromCamera([[-0.09, -0.27, -1.00], [-0.18, -0.24, -1.00]])
        [[0.09 0.97 15.04]
         [0.18 0.98 15.07]]
        """
        points = np.array(points)
        return np.dot(points - self.t, self.R.T)

    def spaceFromCamera(self, points, direction=False):
        """
        Convert points (Nx3) from the **camera** coordinate system to the **space** coordinate system.

        Parameters
        ----------
        points : ndarray
            the points in **camera** coordinates to transform, dimensions (3), (Nx3)
        direction : bool, optional
            whether to transform a direction vector (used for the rays) which should just be rotated and not translated. Default False

        Returns
        -------
        points : ndarray
            the points in the **space** coordinate system, dimensions (3), (Nx3)

        Examples
        --------

        >>> import cameratransform as ct
        >>> orientation = ct.SpatialOrientation(elevation_m=15.4, tilt_deg=85)

        transform a single point from the space to the image:

        >>> orientation.spaceFromCamera([0.09 0.97 15.04])
        [-0.09 -0.27 -1.00]

        or multiple points in one go:

        >>> orientation.spaceFromCamera([[0.09, 0.97, 15.04], [0.18, 0.98, 15.07]])
        [[-0.09 -0.27 -1.00]
         [-0.18 -0.24 -1.00]]
        """
        points = np.array(points)
        if direction:
            return np.dot(points, self.R_inv.T)
        else:
            return np.dot(points, self.R_inv.T) + self.t

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
