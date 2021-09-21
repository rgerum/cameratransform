#!/usr/bin/env python
# -*- coding: utf-8 -*-
# lens_distortion.py

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
from .parameter_set import ClassWithParameterSet, ParameterSet, Parameter, TYPE_DISTORTION
import json

def invert_function(x, func):
    from scipy import interpolate
    from scipy.interpolate import dfitpack
    y = func(x)
    dy = np.concatenate(([0], np.diff(y)))
    y = y[dy>=0]
    x = x[dy>=0]
    try:
        inter = interpolate.InterpolatedUnivariateSpline(y, x)
    # dfitpack.error
    except Exception: # pragma: no cover
        inter = lambda x: x
    return inter


class LensDistortion(ClassWithParameterSet):  # pragma: no cover
    offset = np.array([0, 0])
    scale = 1

    def __init__(self):
        self.parameters = ParameterSet()

    def imageFromDistorted(self, points):
        # return the points as they are
        return points

    def distortedFromImage(self, points):
        # return the points as they are
        return points

    def __str__(self):
        string = ""
        string += "  lens (%s):\n" % type(self).__name__
        for name in self.parameters.parameters:
            string += "    %s:\t\t%.3f\n" % (name, getattr(self, name))
        return string

    def setProjection(self, projection):
        pass

    def save(self, filename):
        keys = self.parameters.parameters.keys()
        export_dict = {key: getattr(self, key) for key in keys}
        with open(filename, "w") as fp:
            fp.write(json.dumps(export_dict))

    def load(self, filename):
        with open(filename, "r") as fp:
            variables = json.loads(fp.read())
        for key in variables:
            setattr(self, key, variables[key])


class NoDistortion(LensDistortion):
    """
    The default model for the lens distortion which does nothing.
    """
    pass


class BrownLensDistortion(LensDistortion):
    r"""
    The most common distortion model is the Brown's distortion model. In CameraTransform, we only consider the radial part
    of the model, as this covers all common cases and the merit of tangential components is disputed. This model relies on
    transforming the radius with even polynomial powers in the coefficients :math:`k_1, k_2, k_3`. This distortion model is
    e.g. also used by OpenCV or AgiSoft PhotoScan.

    Adjust scale and offset of x and y to be relative to the center:

    .. math::
        x' &= \frac{x-c_x}{f_x}\\
        y' &= \frac{y-c_y}{f_y}

    Transform the radius from the center with the distortion:

    .. math::
        r &= \sqrt{x'^2 + y'^2}\\
        r' &= r \cdot (1 + k_1 \cdot r^2 + k_2 \cdot r^4 + k_3 \cdot r^6)\\
        x_\mathrm{distorted}' &= x' / r \cdot r'\\
        y_\mathrm{distorted}' &= y' / r \cdot r'

    Readjust scale and offset to obtain again pixel coordinates:

    .. math::
        x_\mathrm{distorted} &= x_\mathrm{distorted}' \cdot f_x + c_x\\
        y_\mathrm{distorted} &= y_\mathrm{distorted}' \cdot f_y + c_y
    """
    projection = None

    def __init__(self, k1=None, k2=None, k3=None, projection=None):
        self.parameters = ParameterSet(
            # the intrinsic parameters
            k1=Parameter(k1, default=0, range=(0, None), type=TYPE_DISTORTION),
            k2=Parameter(k2, default=0, range=(0, None), type=TYPE_DISTORTION),
            k3=Parameter(k3, default=0, range=(0, None), type=TYPE_DISTORTION),
        )
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._init_inverse
        self._init_inverse()

    def setProjection(self, projection):
        self.projection = projection
        self.parameters = ParameterSet(
            k1=self.parameters.parameters["k1"],
            k2=self.parameters.parameters["k2"],
            k3=self.parameters.parameters["k3"],
            image_width_px=self.projection.parameters.parameters["image_width_px"],
            image_height_px=self.projection.parameters.parameters["image_height_px"],
            focallength_x_px=self.projection.parameters.parameters["focallength_x_px"],
            focallength_y_px=self.projection.parameters.parameters["focallength_y_px"],
            center_x_px=self.projection.parameters.parameters["center_x_px"],
            center_y_px=self.projection.parameters.parameters["center_y_px"],
        )
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._init_inverse
        self._init_inverse()

    def _init_inverse(self):
        r = np.arange(0, 2, 0.1)
        self._convert_radius_inverse = invert_function(r, self._convert_radius)
        if self.projection is not None:
            self.scale = np.array([self.projection.focallength_x_px, self.projection.focallength_y_px])
            self.offset = np.array([self.projection.center_x_px, self.projection.center_y_px])

    def _convert_radius(self, r):
        return r*(1 + self.parameters.k1*r**2 + self.parameters.k2*r**4 + self.parameters.k3*r**6)

    def imageFromDistorted(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        points = points / r * self._convert_radius_inverse(r)
        # rescale back to the image
        return points * self.scale + self.offset

    def distortedFromImage(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        points = points / r * self._convert_radius(r)
        # rescale back to the image
        return points * self.scale + self.offset


class ABCDistortion(LensDistortion):
    r"""
    The ABC model is a less common distortion model, that just implements radial distortions. Here the radius is transformed
    using a polynomial of 4th order. It is used e.g. in PTGui.

    Adjust scale and offset of x and y to be relative to the center:

    .. math::
        s &= 0.5 \cdot \mathrm{min}(\mathrm{im}_\mathrm{width}, \mathrm{im}_\mathrm{height})\\
        x' &= \frac{x-c_x}{s}\\
        y' &= \frac{y-c_y}{s}

    Transform the radius from the center with the distortion:

    .. math::
        r &= \sqrt{x^2 + y^2}\\
        r' &= d \cdot r + c \cdot r^2 + b \cdot r^3 + a \cdot r^4\\
        d &= 1 - a - b - c

    Readjust scale and offset to obtain again pixel coordinates:

    .. math::
        x_\mathrm{distorted} &= x_\mathrm{distorted}' \cdot s + c_x\\
        y_\mathrm{distorted} &= y_\mathrm{distorted}' \cdot s + c_y


    """
    projection = None

    def __init__(self, a=None, b=None, c=None):
        self.parameters = ParameterSet(
            # the intrinsic parameters
            a=Parameter(a, default=0, type=TYPE_DISTORTION),
            b=Parameter(b, default=0, type=TYPE_DISTORTION),
            c=Parameter(c, default=0, type=TYPE_DISTORTION)
        )
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._init_inverse
        self._init_inverse()

    def setProjection(self, projection):
        self.projection = projection
        self.parameters = ParameterSet(
            a=self.parameters.parameters["a"],
            b=self.parameters.parameters["b"],
            c=self.parameters.parameters["c"],
            image_width_px=self.projection.parameters.parameters["image_width_px"],
            image_height_px=self.projection.parameters.parameters["image_height_px"],
            focallength_x_px=self.projection.parameters.parameters["focallength_x_px"],
            focallength_y_px=self.projection.parameters.parameters["focallength_y_px"],
            center_x_px=self.projection.parameters.parameters["center_x_px"],
            center_y_px=self.projection.parameters.parameters["center_y_px"],
        )
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._init_inverse
        self._init_inverse()

    def _init_inverse(self):
        self.d = 1 - self.a - self.b - self.c
        r = np.arange(0, 2, 0.1)
        self._convert_radius_inverse = invert_function(r, self._convert_radius)

        if self.projection is not None:
            self.scale = np.min([self.projection.image_width_px, self.projection.image_height_px]) / 2
            self.offset = np.array([self.projection.center_x_px, self.projection.center_y_px])

    def _convert_radius(self, r):
        return self.d * r + self.c * r**2 + self.b * r**3 + self.a * r**4

    def imageFromDistorted(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        points = points / r * self._convert_radius_inverse(r)
        # rescale back to the image
        return points * self.scale + self.offset

    def distortedFromImage(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=-1)[..., None]
        # transform the points
        points = points / r * self._convert_radius(r)
        # rescale back to the image
        return points * self.scale + self.offset
