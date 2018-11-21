import numpy as np
from .parameter_set import ClassWithParameterSet, ParameterSet, Parameter, TYPE_DISTORTION


def invert_function(x, func):
    from scipy import interpolate
    from scipy.interpolate import dfitpack
    y = func(x)
    dy = np.concatenate(([0], np.diff(y)))
    y = y[dy>=0]
    x = x[dy>=0]
    try:
        inter = interpolate.InterpolatedUnivariateSpline(y, x)
    except Exception:  # dfitpack.error
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


class NoDistortion(LensDistortion):
    pass


class BrownLensDistortion(LensDistortion):  # pragma: no cover
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
        r = np.linalg.norm(points, axis=1)[:, None]
        # transform the points
        points = points / r * self._convert_radius_inverse(r)
        # set nans to 0
        points[np.isnan(points)] = 0
        # rescale back to the image
        return points * self.scale + self.offset

    def distortedFromImage(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=1)[:, None]
        # transform the points
        points = points / r * self._convert_radius(r)
        # set nans to 0
        points[np.isnan(points)] = 0
        # rescale back to the image
        return points * self.scale + self.offset


class ABCDistortion(LensDistortion):  # pragma: no cover

    def __init__(self, a=None, b=None, c=None):
        self.parameters = ParameterSet(
            # the intrinsic parameters
            a=Parameter(a, default=0, type=TYPE_DISTORTION),
            b=Parameter(b, default=0, type=TYPE_DISTORTION),
            c=Parameter(c, default=0, type=TYPE_DISTORTION),
            image_width_px=self.projection.parameters["image_width_px"],
            image_height_px=self.projection.parameters["image_height_px"],
            focalllength_mm=self.projection.parameters["focalllength_mm"],
            sensor_width_mm=self.projection.parameters["sensor_width_mm"],
        )
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._init_inverse
        self._init_inverse()

    def _init_inverse(self):
        self.d = 1 - self.a - self.b - self.c
        r = np.arange(0, 2, 0.1)
        self._convert_radius_inverse = invert_function(r, self._convert_radius)

        self.scale = np.min([self.projection.image_width_px, self.projection.image_height_px]) / 2
        self.offset = np.array([self.projection.center_x_px, self.projection.center_y_px])

    def _convert_radius(self, r):
        return self.d * r + self.c * r**2 + self.b * r**3 + self.a * r**4

    def imageFromDistorted(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=1)[:, None]
        # transform the points
        points = points / r * self._convert_radius_inverse(r)
        # set nans to 0
        points[np.isnan(points)] = 0
        # rescale back to the image
        return points * self.scale + self.offset

    def distortedFromImage(self, points):
        # ensure that the points are provided as an array
        # and rescale the points to that the center is at 0 and the border at 1
        points = (np.array(points)-self.offset)/self.scale
        # calculate the radius form the center
        r = np.linalg.norm(points, axis=1)[:, None]
        # transform the points
        points = points / r * self._convert_radius(r)
        # set nans to 0
        points[np.isnan(points)] = 0
        # rescale back to the image
        return points * self.scale + self.offset
