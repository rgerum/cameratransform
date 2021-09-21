#!/usr/bin/env python
# -*- coding: utf-8 -*-
# projection.py

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

import json
import numpy as np
from .parameter_set import ParameterSet, ClassWithParameterSet, Parameter, TYPE_INTRINSIC


class CameraProjection(ClassWithParameterSet):
    """
    Defines a camera projection. The necessary parameters are:
    focalllength_x_px, focalllength_y_px, center_x_px, center_y_px, image_width_px, image_height_px. Depending on the
    information available different initialisation routines can be used.

    .. note::
        This is the base class for projections. it the should not be instantiated. Available projections are
        :py:class:`RectilinearProjection`, :py:class:`CylindricalProjection`, or :py:class:`EquirectangularProjection`.

    Examples
    --------

    This section provides some examples how the projections can be initialized.

    >>> import cameratransform as ct

    **Image Dimensions**:

    The image dimensions can be provided as two values:

    >>> projection = ct.RectilinearProjection(focallength_px=3863.64, image_width_px=4608, image_height_px=3456)

    or as a tuple:

    >>> projection = ct.RectilinearProjection(focallength_px=3863.64, image=(4608, 3456))

    or by providing a numpy array of an example image:

    >>> import matplotlib.pyplot as plt
    >>> im = plt.imread("test.jpg")
    >>> projection = ct.RectilinearProjection(focallength_px=3863.64, image=im)

    **Focal Length**:

    The focal length can be provided in mm, when also a sensor size is provided:

    >>> projection = ct.RectilinearProjection(focallength_mm=14, sensor=(17.3, 9.731), image=(4608, 3456))

    or directly in pixels without the sensor size:

    >>> projection = ct.RectilinearProjection(focallength_px=3863.64, image=(4608, 3456))

    or as a tuple to give different focal lengths in x and y direction, if the pixels on the sensor are not square:

    >>> projection = ct.RectilinearProjection(focallength_px=(3772, 3774), image=(4608, 3456))

    or the focal length is given by providing a field of view angle:

    >>> projection = ct.RectilinearProjection(view_x_deg=61.617, image=(4608, 3456))

    >>> projection = ct.RectilinearProjection(view_y_deg=48.192, image=(4608, 3456))

    **Central Point**:

    If the position of the optical axis or center of the image is not provided, it is assumed to be in the middle of the
    image. But it can be specifided, as two values or a tuple:

    >>> projection = ct.RectilinearProjection(focallength_px=3863.64, center=(2304, 1728), image=(4608, 3456))

    >>> projection = ct.RectilinearProjection(focallength_px=3863.64, center_x_px=2304, center_y_px=1728, image=(4608, 3456))

    """

    def __init__(self, focallength_px=None, focallength_x_px=None, focallength_y_px=None, center_x_px=None, center_y_px=None, center=None, focallength_mm=None, image_width_px=None, image_height_px=None,
                 sensor_width_mm=None, sensor_height_mm=None, image=None, sensor=None, view_x_deg=None, view_y_deg=None):

        # split image in width and height
        if image is not None:
            try:
                image_height_px, image_width_px = image.shape[:2]
            except AttributeError:
                image_width_px, image_height_px = image
        if center is not None:
            center_x_px, center_y_px = center
        if center_x_px is None and image_width_px is not None:
            center_x_px = image_width_px / 2
        if center_y_px is None and image_height_px is not None:
            center_y_px = image_height_px / 2

        # split sensor in width and height
        if sensor is not None:
            sensor_width_mm, sensor_height_mm = sensor
        if sensor_height_mm is None and sensor_width_mm is not None:
            sensor_height_mm = image_height_px / image_width_px * sensor_width_mm
        elif sensor_width_mm is None and sensor_height_mm is not None:
            sensor_width_mm = image_width_px / image_height_px * sensor_height_mm

        # get the focalllength
        focallength_x_px = focallength_x_px
        focallength_y_px = focallength_y_px
        if focallength_px is not None:
            try:
                focallength_x_px, focallength_y_px = focallength_px
            except TypeError:
                focallength_x_px = focallength_px
                focallength_y_px = focallength_px
        elif focallength_mm is not None and sensor_width_mm is not None:
            focallength_px = focallength_mm / sensor_width_mm * image_width_px
            focallength_x_px = focallength_px
            focallength_y_px = focallength_px

        self.parameters = ParameterSet(
            # the intrinsic parameters
            focallength_x_px=Parameter(focallength_x_px, default=3600, type=TYPE_INTRINSIC),  # the focal length in px
            focallength_y_px=Parameter(focallength_y_px, default=3600, type=TYPE_INTRINSIC),  # the focal length in px
            center_x_px=Parameter(center_x_px, default=0, type=TYPE_INTRINSIC),  # the focal length in mm
            center_y_px=Parameter(center_y_px, default=0, type=TYPE_INTRINSIC),  # the focal length in mm
            image_height_px=Parameter(image_height_px, default=3456, type=TYPE_INTRINSIC),  # the image height in px
            image_width_px=Parameter(image_width_px, default=4608, type=TYPE_INTRINSIC),  # the image width in px
            sensor_height_mm=Parameter(sensor_height_mm, default=13.0, type=TYPE_INTRINSIC),  # the sensor height in mm
            sensor_width_mm=Parameter(sensor_width_mm, default=17.3, type=TYPE_INTRINSIC),  # the sensor width in mm
        )
        # add parameter focallength_px that sets x and y simultaneously
        fx = self.parameters.parameters["focallength_x_px"]
        fy = self.parameters.parameters["focallength_y_px"]
        f = Parameter(focallength_x_px, default=3600, type=TYPE_INTRINSIC)
        def callback():
            fx.value = f.value
            if fx.callback is not None:
                fx.callback()
            fy.value = f.value
            if fy.callback is not None:
                fy.callback()
        f.callback = callback
        self.parameters.parameters["focallength_px"] = f

        if view_x_deg is not None or view_y_deg is not None:
            if sensor_width_mm is None:
                if view_x_deg is not None:
                    self.sensor_width_mm = self.imageFromFOV(view_x=view_x_deg)
                    self.sensor_height_mm = self.image_height_px / self.image_width_px * self.sensor_width_mm
                elif view_y_deg is not None:
                    self.sensor_height_mm = self.imageFromFOV(view_y=view_y_deg)
                    self.sensor_width_mm = self.image_width_px / self.image_height_px * self.sensor_height_mm
                if focallength_mm is not None:
                    self.focallength_px = focallength_mm / self.sensor_width_mm * self.image_width_px
                    self.focallength_x_px = focallength_px
                    self.focallength_y_px = focallength_px
            else:
                self.focallength_x_px = self.focallengthFromFOV(view_x_deg, view_y_deg)
                self.focallength_y_px = self.focallengthFromFOV(view_x_deg, view_y_deg)

    def __str__(self):
        string = ""
        string += "  intrinsic (%s):\n" % type(self).__name__
        #string += "    f:\t\t%.1f mm\n    sensor:\t%.2f×%.2f mm\n    image:\t%d×%d px\n" % (
        #    self.parameters.focallength_mm, self.parameters.sensor_width_mm, self.parameters.sensor_height_mm,
        #    self.parameters.image_width_px, self.parameters.image_height_px)
        string += "    f:\t\t%.1f px\n    sensor:\t%.2f×%.2f mm\n    image:\t%d×%d px\n" % (
            self.parameters.focallength_x_px, self.parameters.sensor_width_mm, self.parameters.sensor_height_mm,
            self.parameters.image_width_px, self.parameters.image_height_px)
        return string

    def save(self, filename):
        keys = self.parameters.parameters.keys()
        export_dict = {key: getattr(self, key) for key in keys if key != "focallength_px"}
        with open(filename, "w") as fp:
            fp.write(json.dumps(export_dict))

    def load(self, filename):
        with open(filename, "r") as fp:
            variables = json.loads(fp.read())
        for key in variables:
            setattr(self, key, variables[key])

    def imageFromCamera(self, points):  # pragma: no cover
        """
        Convert points (Nx3) from the **camera** coordinate system to the **image** coordinate system.

        Parameters
        ----------
        points : ndarray
            the points in **camera** coordinates to transform, dimensions (3), (Nx3)

        Returns
        -------
        points : ndarray
            the points in the **image** coordinate system, dimensions (2), (Nx2)

        Examples
        --------

        >>> import cameratransform as ct
        >>> proj = ct.RectilinearProjection(focallength_px=3729, image=(4608, 2592))

        transform a single point from the **camera** coordinates to the image:

        >>> proj.imageFromCamera([-0.09, -0.27, -1.00])
        [2639.61 2302.83]

        or multiple points in one go:

        >>> proj.imageFromCamera([[-0.09, -0.27, -1.00], [-0.18, -0.24, -1.00]])
        [[2639.61 2302.83]
         [2975.22 2190.96]]
        """
        # to be overloaded by the child class.
        return None

    def getRay(self, points, normed=False):  # pragma: no cover
        """
        As the transformation from the **image** coordinate system to the **camera** coordinate system is not unique,
        **image** points can only be uniquely mapped to a ray in **camera** coordinates.

        Parameters
        ----------
        points : ndarray
            the points in **image** coordinates for which to get the ray, dimensions (2), (Nx2)

        Returns
        -------
        rays : ndarray
            the rays in the **camera** coordinate system, dimensions (3), (Nx3)

        Examples
        --------

        >>> import cameratransform as ct
        >>> proj = ct.RectilinearProjection(focallength_px=3729, image=(4608, 2592))

        get the ray of a point in the image:

        >>> proj.getRay([1968, 2291])
        [0.09 -0.27 -1.00]

        or the rays of multiple points in the image:

        >>> proj.getRay([[1968, 2291], [1650, 2189]])
        [[0.09 -0.27 -1.00]
         [0.18 -0.24 -1.00]]
        """
        # to be overloaded by the child class.
        return None

    def getFieldOfView(self):  # pragma: no cover
        """
        The field of view of the projection in x (width, horizontal) and y (height, vertical) direction.

        Returns
        -------
        view_x_deg : float
            the horizontal field of view in degree.
        view_y_deg : float
            the vertical field of view in degree.
        """
        # to be overloaded by the child class.
        return 0, 0

    def focallengthFromFOV(self, view_x=None, view_y=None):  # pragma: no cover
        """
        The focal length (in x or y direction) based on the given field of view.

        Parameters
        ----------
        view_x : float
            the field of view in x direction in degrees. If not given only view_y is processed.
        view_y : float
            the field of view in y direction in degrees. If not given only view_y is processed.

        Returns
        -------
        focallength_px : float
            the focal length in pixels.
        """
        # to be overloaded by the child class.
        return 0

    def imageFromFOV(self, view_x=None, view_y=None):  # pragma: no cover
        """
        The image width or height in pixel based on the given field of view.

        Parameters
        ----------
        view_x : float
            the field of view in x direction in degrees. If not given only view_y is processed.
        view_y : float
            the field of view in y direction in degrees. If not given only view_y is processed.

        Returns
        -------
        width/height : float
            the width or height in pixels.
        """
        # to be overloaded by the child class.
        return 0


class RectilinearProjection(CameraProjection):
    r"""
    This projection is the standard "pin-hole", or frame camera model, which is the most common projection for single images. The angles
    :math:`\pm 180°` are projected to :math:`\pm \infty`. Therefore, the maximal possible field of view in this projection
    would be 180° for an infinitely large image.

    **Projection**:

    .. math::
        x_\mathrm{im} &= f_x \cdot \frac{x}{z} + c_x\\
        y_\mathrm{im} &= f_y \cdot \frac{y}{z} + c_y

    **Rays**:

    .. math::
        \vec{r} = \begin{pmatrix}
            (x_\mathrm{im} - c_x)/f_x\\
            (y_\mathrm{im} - c_y)/f_y\\
            1\\
        \end{pmatrix}

    **Matrix**:

    The rectilinear projection can also be represented in matrix notation:

    .. math::
        C_{\mathrm{intr.}} &=
        \begin{pmatrix}
         f_x & 0   & c_x \\
         0   & f_y & c_y \\
         0   & 0   &   1 \\
         \end{pmatrix}\\

    """

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # set z=focallenth and solve the other equations for x and y
        ray = np.array([-(points[..., 0] - self.center_x_px) / self.focallength_x_px,
                        (points[..., 1] - self.center_y_px) / self.focallength_y_px,
                        np.ones(points[..., 1].shape)]).T
        # norm the ray if desired
        if normed:
            ray /= np.linalg.norm(ray, axis=-1)[..., None]
        # return the ray
        return -ray

    def imageFromCamera(self, points, hide_backpoints=True):
        """
                          x                                y
            x_im = f_x * --- + offset_x      y_im = f_y * --- + offset_y
                          z                                z
        """
        points = np.array(points)
        # set small z distances to 0
        points[np.abs(points[..., 2]) < 1e-10] = 0
        # transform the points
        transformed_points = np.array([-points[..., 0] * self.focallength_x_px / points[..., 2] + self.center_x_px,
                                       points[..., 1] * self.focallength_y_px / points[..., 2] + self.center_y_px]).T
        if hide_backpoints:
            transformed_points[points[..., 2] > 0] = np.nan
        return transformed_points

    def getFieldOfView(self):
        return np.rad2deg(2 * np.arctan(self.image_width_px / (2 * self.focallength_x_px))), \
               np.rad2deg(2 * np.arctan(self.image_height_px / (2 * self.focallength_y_px)))

    def focallengthFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            return self.image_width_px / (2 * np.tan(np.deg2rad(view_x) / 2))
        else:
            return self.image_height_px / (2 * np.tan(np.deg2rad(view_y) / 2))
        
    def imageFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            # image_width_px
            return self.focallength_x_px * (2 * np.tan(np.deg2rad(view_x) / 2))
        else:
            # image_height_px
            return self.focallength_y_px * (2 * np.tan(np.deg2rad(view_y) / 2))


class CylindricalProjection(CameraProjection):
    r"""
    This projection is a common projection used for panoranic images. This projection is often used
    for wide panoramic images, as it can cover the full 360° range in the x-direction. The poles cannot
    be represented in this projection, as they would be projected to :math:`y = \pm\infty`.

    **Projection**:

    .. math::
        x_\mathrm{im} &= f_x \cdot \arctan{\left(\frac{x}{z}\right)} + c_x\\
        y_\mathrm{im} &= f_y \cdot \frac{y}{\sqrt{x^2+z^2}} + c_y

    **Rays**:

    .. math::
        \vec{r} = \begin{pmatrix}
            \sin\left(\frac{x_\mathrm{im} - c_x}{f_x}\right)\\
            \frac{y_\mathrm{im} - c_y}{f_y}\\
            \cos\left(\frac{x_\mathrm{im} - c_x}{f_x}\right)
        \end{pmatrix}
    """

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # set r=1 and solve the other equations for x and y
        r = 1
        alpha = (points[..., 0] - self.center_x_px) / self.focallength_x_px
        x = -np.sin(alpha) * r
        z = np.cos(alpha) * r
        y = r * (points[..., 1] - self.center_y_px) / self.focallength_y_px
        # compose the ray
        ray = np.array([x, y, z]).T
        # norm the ray if desired
        if normed:
            ray /= np.linalg.norm(ray, axis=-1)[..., None]
        # return the rey
        return -ray

    def imageFromCamera(self, points, hide_backpoints=True):
        """
                               ( x )                                     y
            x_im = f_x * arctan(---) + offset_x      y_im = f_y * --------------- + offset_y
                               ( z )                              sqrt(x**2+z**2)
        """
        # ensure that the points are provided as an array
        points = np.array(points)
        # set small z distances to 0
        points[np.abs(points[..., 2]) < 1e-10] = 0
        # transform the points
        transformed_points = np.array(
            [-self.focallength_x_px * np.arctan2(-points[..., 0], -points[..., 2]) + self.center_x_px,
             -self.focallength_y_px * points[..., 1] / np.linalg.norm(points[..., [0, 2]], axis=-1) + self.center_y_px]).T
        # ensure that points' x values are also nan when the y values are nan
        transformed_points[np.isnan(transformed_points[..., 1])] = np.nan
        # return the points
        return transformed_points

    def getFieldOfView(self):
        return np.rad2deg(self.image_width_px / self.focallength_x_px), \
               np.rad2deg(2 * np.arctan(self.image_height_px / (2 * self.focallength_y_px)))

    def focallengthFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            return self.image_width_px / np.deg2rad(view_x)
        else:
            return self.image_height_px / (2 * np.tan(np.deg2rad(view_y) / 2))

    def imageFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            # image_width_px
            return self.focallength_x_px * np.deg2rad(view_x)
        else:
            # image_height_px
            return self.focallength_y_px * (2 * np.tan(np.deg2rad(view_y) / 2))


class EquirectangularProjection(CameraProjection):
    r"""
    This projection is a common projection used for panoranic images. The projection can cover the
    full range of angles in both x and y direction.

    **Projection**:

    .. math::
        x_\mathrm{im} &= f_x \cdot \arctan{\left(\frac{x}{z}\right)} + c_x\\
        y_\mathrm{im} &= f_y \cdot \arctan{\left(\frac{y}{\sqrt{x^2+z^2}}\right)} + c_y

    **Rays**:

    .. math::
        \vec{r} = \begin{pmatrix}
            \sin\left(\frac{x_\mathrm{im} - c_x}{f_x}\right)\\
            \tan\left(\frac{y_\mathrm{im} - c_y}{f_y}\right)\\
            \cos\left(\frac{x_\mathrm{im} - c_x}{f_x}\right)
        \end{pmatrix}
    """

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # set r=1 and solve the other equations for x and y
        r = 1
        alpha = (points[..., 0] - self.center_x_px) / self.focallength_x_px
        x = -np.sin(alpha) * r
        z = np.cos(alpha) * r
        y = r * np.tan((points[..., 1] - self.center_y_px) / self.focallength_y_px)
        # compose the ray
        ray = np.array([x, y, z]).T
        # norm the ray if desired
        if normed:
            ray /= np.linalg.norm(ray, axis=-1)[..., None]
        # return the rey
        return -ray

    def imageFromCamera(self, points, hide_backpoints=True):
        """
                               ( x )                                    (       y       )
            x_im = f_x * arctan(---) + offset_x      y_im = f_y * arctan(---------------) + offset_y
                               ( z )                                    (sqrt(x**2+z**2))
        """
        # ensure that the points are provided as an array
        points = np.array(points)
        # set small z distances to 0
        points[np.abs(points[..., 2]) < 1e-10] = 0
        # transform the points
        transformed_points = np.array([-self.focallength_x_px * np.arctan(points[..., 0] / points[..., 2]) + self.center_x_px,
                                       -self.focallength_y_px * np.arctan(points[..., 1] / np.sqrt(
                                           points[..., 0] ** 2 + points[..., 2] ** 2)) + self.center_y_px]).T
        # return the points
        return transformed_points

    def getFieldOfView(self):
        return np.rad2deg(self.image_width_px / self.focallength_x_px),\
               np.rad2deg(self.image_height_px / self.focallength_y_px)

    def focallengthFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            return self.image_width_px / np.deg2rad(view_x)
        else:
            return self.image_height_px / np.deg2rad(view_y)

    def imageFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            # image_width_mm
            return self.focallength_x_px * np.deg2rad(view_x)
        else:
            # image_height_mm
            return self.focallength_y_px * np.deg2rad(view_y)
