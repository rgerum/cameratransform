import json
import numpy as np
from .parameter_set import ParameterSet, ClassWithParameterSet, Parameter, TYPE_INTRINSIC


class CameraProjection(ClassWithParameterSet):

    def __init__(self, focallength_px=None, center_x_px=None, center_y_px=None, center=None, focallength_mm=None, image_width_px=None, image_height_px=None,
                 sensor_width_mm=None, sensor_height_mm=None, image=None, sensor=None, view_x_deg=None, view_y_deg=None):

        # split image in width and height
        if image is not None:
            try:
                image_height_px, image_width_px = image.shape[:2]
            except AttributeError:
                image_width_px, image_height_px = image
        if center is not None:
            center_x_px, center_y_px = center
        if center_x_px is None:
            center_x_px = image_width_px / 2
        if center_y_px is None:
            center_y_px = image_height_px / 2

        # split sensor in width and height
        if sensor is not None:
            sensor_width_mm, sensor_height_mm = sensor
        if sensor_height_mm is None and sensor_width_mm is not None:
            sensor_height_mm = image_height_px / image_width_px * sensor_width_mm
        elif sensor_width_mm is None and sensor_height_mm is not None:
            sensor_width_mm = image_width_px / image_height_px * sensor_height_mm

        # get the focalllength
        focallength_x_px = None
        focallength_y_px = None
        if focallength_px is not None:
            try:
                focallength_x_px, focallength_y_px = focallength_px
            except IndexError:
                focallength_x_px = focallength_px
                focallength_y_px = focallength_px
        elif focallength_mm is not None:
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

        if view_x_deg is not None or view_y_deg is not None:
            if sensor_width_mm is None:
                if view_x_deg is not None:
                    self.sensor_width_mm = self.sensorFromFOV(view_x=view_x_deg)
                    self.sensor_height_mm = self.image_height_px / self.image_width_px * self.sensor_width_mm
                elif view_y_deg is not None:
                    self.sensor_height_mm = self.sensorFromFOV(view_y=view_y_deg)
                    self.sensor_width_mm = self.image_width_px / self.image_height_px * self.sensor_height_mm
            else:
                self.focallength_mm = self.focallengthFromFOV(view_x_deg, view_y_deg)

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
        export_dict = {key: getattr(self, key) for key in keys}
        with open(filename, "w") as fp:
            fp.write(json.dumps(export_dict))

    def load(self, filename):
        with open(filename, "r") as fp:
            variables = json.loads(fp.read())
        for key in variables:
            setattr(self, key, variables[key])


class RectilinearProjection(CameraProjection):

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # set z=focallenth and solve the other equations for x and y
        ray = np.array([(points[..., 0] - self.center_x_px) / self.focallength_x_px,
                        (points[..., 1] - self.center_y_px) / self.focallength_y_px,
                        np.ones(points[..., 1].shape)]).T
        # norm the ray if desired
        if normed:
            ray /= np.linalg.norm(ray, axis=-1)[..., None]
        # return the ray
        return -ray

    def imageFromCamera(self, points):
        """
                          x                                y
            x_im = f_x * --- + offset_x      y_im = f_y * --- + offset_y
                          z                                z
        """
        points = np.array(points)
        # set small z distances to 0
        points[np.abs(points[..., 2]) < 1e-10] = 0
        # transform the points
        transformed_points = np.array([points[..., 0] * self.focallength_px_x / points[..., 2] + self.center_x_px,
                                       points[..., 1] * self.focallength_px_y / points[..., 2] + self.center_y_px]).T
        transformed_points[points[..., 2] > 0] = np.nan
        return transformed_points

    def getFieldOfView(self):
        return np.rad2deg(2 * np.arctan(self.sensor_width_mm / (2 * self.focallength_mm))), \
               np.rad2deg(2 * np.arctan(self.sensor_height_mm / (2 * self.focallength_mm)))

    def focallengthFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            return self.sensor_width_mm / (2 * np.tan(np.deg2rad(view_x) / 2))
        else:
            return self.sensor_height_mm / (2 * np.tan(np.deg2rad(view_y) / 2))
        
    def sensorFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            # sensor_width_mm
            return self.focallength_mm * (2 * np.tan(np.deg2rad(view_x) / 2))
        else:
            # sensor_height_mm
            return self.focallength_mm * (2 * np.tan(np.deg2rad(view_y) / 2))


class CylindricalProjection(CameraProjection):

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # set r=1 and solve the other equations for x and y
        r = 1
        alpha = (points[..., 0] - self.center_x_px) / self.focallength_px_x
        x = np.sin(alpha) * r
        z = np.cos(alpha) * r
        y = r * (points[..., 1] - self.center_y_px) / self.focallength_px_y
        # compose the ray
        ray = np.array([x, y, z]).T
        # norm the ray if desired
        if normed:
            ray /= np.linalg.norm(ray, axis=-1)[..., None]
        # return the rey
        return -ray

    def imageFromCamera(self, points):
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
            [self.focallength_px_x * np.arctan2(-points[..., 0], -points[..., 2]) + self.center_x_px,
             -self.focallength_px_y * points[..., 1] / np.linalg.norm(points[..., [0, 2]], axis=-1) + self.center_y_px]).T
        # ensure that points' x values are also nan when the y values are nan
        transformed_points[np.isnan(transformed_points[..., 1])] = np.nan
        # return the points
        return transformed_points

    def getFieldOfView(self):
        return np.rad2deg(self.sensor_width_mm / self.focallength_mm), \
               np.rad2deg(2 * np.arctan(self.sensor_height_mm / (2 * self.focallength_mm)))

    def focallengthFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            return self.sensor_width_mm / np.deg2rad(view_x)
        else:
            return self.sensor_height_mm / (2 * np.tan(np.deg2rad(view_y) / 2))

    def sensorFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            # sensor_width_mm
            return self.focallength_mm * np.deg2rad(view_x)
        else:
            # sensor_height_mm
            return self.focallength_mm * (2 * np.tan(np.deg2rad(view_y) / 2))


class EquirectangularProjection(CameraProjection):

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # set r=1 and solve the other equations for x and y
        r = 1
        alpha = (points[..., 0] - self.center_x_px) / self.focallength_px_x
        x = np.sin(alpha) * r
        z = np.cos(alpha) * r
        y = r * np.tan((points[..., 1] - self.center_y_px) / self.focallength_px_y)
        # compose the ray
        ray = np.array([x, y, z]).T
        # norm the ray if desired
        if normed:
            ray /= np.linalg.norm(ray, axis=-1)[..., None]
        # return the rey
        return -ray

    def imageFromCamera(self, points):
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
        transformed_points = np.array([self.focallength_px_x * np.arctan(points[..., 0] / points[..., 2]) + self.center_x_px,
                                       -self.focallength_px_y * np.arctan(points[..., 1] / np.sqrt(
                                           points[..., 0] ** 2 + points[..., 2] ** 2)) + self.center_y_px]).T
        # return the points
        return transformed_points

    def getFieldOfView(self):
        return np.rad2deg(self.sensor_width_mm / self.focallength_mm),\
               np.rad2deg(self.sensor_height_mm / self.focallength_mm)

    def focallengthFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            return self.sensor_width_mm / np.deg2rad(view_x)
        else:
            return self.sensor_height_mm / np.deg2rad(view_y)

    def sensorFromFOV(self, view_x=None, view_y=None):
        if view_x is not None:
            # sensor_width_mm
            return self.focallength_mm * np.deg2rad(view_x)
        else:
            # sensor_height_mm
            return self.focallength_mm * np.deg2rad(view_y)
