import numpy as np
from .parameter_set import ParameterSet, ClassWithParameterSet, Parameter, TYPE_INTRINSIC
import json

class CameraProjection(ClassWithParameterSet):
    C1 = None
    C1_inv = None

    focallength_px = 0
    offset_x = 0
    offset_y = 0

    def __init__(self, focallength_mm=None, image_height_px=None, image_width_px=None, sensor_height_mm=None,
                 sensor_width_mm=None):
        self.parameters = ParameterSet(
            # the intrinsic parameters
            focallength_mm=Parameter(focallength_mm, default=14, type=TYPE_INTRINSIC),  # the focal length in mm
            image_height_px=Parameter(image_height_px, default=3456, type=TYPE_INTRINSIC),  # the image height in px
            image_width_px=Parameter(image_width_px, default=4608, type=TYPE_INTRINSIC),  # the image width in px
            sensor_height_mm=Parameter(sensor_height_mm, default=13.0, type=TYPE_INTRINSIC),  # the sensor height in mm
            sensor_width_mm=Parameter(sensor_width_mm, default=17.3, type=TYPE_INTRINSIC),  # the sensor width in mm
        )
        for name in self.parameters.parameters:
            self.parameters.parameters[name].callback = self._initIntrinsicMatrix
        self._initIntrinsicMatrix()

    def __str__(self):
        string = ""
        string += "  intrinsic:\n"
        string += "    f:\t\t%.1f mm\n    sensor:\t%.2f×%.2f mm\n    image:\t%d×%d px\n" % (
            self.parameters.focallength_mm, self.parameters.sensor_width_mm, self.parameters.sensor_height_mm, self.parameters.image_width_px,
            self.parameters.image_height_px)
        return string

    def _initIntrinsicMatrix(self):
        # normalize the focal length by the sensor width and the image_width
        self.mm_per_px = self.parameters.sensor_width_mm / self.parameters.image_width_px
        self.focallength_px = self.parameters.focallength_mm / self.mm_per_px
        self.offset_x = self.parameters.image_width_px / 2
        self.offset_y = self.parameters.image_height_px / 2

    def getTransformation(self):  # CameraWorld -> CameraImage
        pass

    def getInvertedTransformation(self):  # CameraImage -> CameraWorld
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



class RectilinearProjection(CameraProjection):
    def _initIntrinsicMatrix(self):
        CameraProjection._initIntrinsicMatrix(self)
        # compose the intrinsic camera matrix
        self.C1 = np.array([[self.focallength_px, 0, self.offset_x],
                            [0, self.focallength_px, self.offset_y],
                            [0, 0, 1]])
        self.C1_inv = np.linalg.inv(self.C1)

    def getTransformation(self):   # CameraWorld -> CameraImage
        if self.C1 is None:
            self._initIntrinsicMatrix()
        return self.C1

    def getInvertedTransformation(self):   # CameraImage -> CameraWorld
        if self.C1_inv is None:
            self._initIntrinsicMatrix()
        return self.C1_inv

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # set z=focallenth and solve the other equations for x and y
        ray = np.array([points[..., 0] - self.offset_x,
                        points[..., 1] - self.offset_y,
                        np.zeros(points[..., 1].shape)+self.focallength_px]).T
        # norm the ray if desired
        if normed:
            ray /= np.linalg.norm(ray, axis=-1)
        # return the ray
        return -ray

    def imageFromCamera(self, points):
        """
                        x                              y
            x_im = f * --- + offset_x      y_im = f * --- + offset_y
                        z                              z
        """
        points = np.array(points)
        transformed_points = np.array([points[..., 0] * self.focallength_px / points[..., 2] + self.offset_x,
                                       points[..., 1] * self.focallength_px / points[..., 2] + self.offset_y]).T
        transformed_points[points[..., 2] > 0] = np.nan
        return transformed_points


class CylindricalProjection(CameraProjection):

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # set z=1 and solve the other equations for x and y
        z = np.ones(points[..., 0].shape)
        x = z*np.tan((points[..., 0]-self.offset_x)/self.focallength_px)
        y = np.sqrt(x**2+z**2)*(points[..., 1] - self.offset_y)/self.focallength_px
        # compose the ray
        ray = np.array([x, y, z]).T
        # norm the ray if desired
        if normed:
            ray /= np.linalg.norm(ray, axis=-1)
        # return the rey
        return -ray

    def imageFromCamera(self, points):
        """
                             ( x )                                  y
            x_im = f * arctan(---) + offset_x      y_im = f * --------------- + offset_y
                             ( z )                            sqrt(x**2+z**2)
        """
        # ensure that the points are provided as an array
        points = np.array(points)
        # transform the points
        transformed_points = np.array([self.focallength_px * np.arctan2(-points[..., 0], -points[..., 2]) + self.offset_x,
                                       -self.focallength_px * points[..., 1] / np.linalg.norm(points[..., [0, 2]], axis=-1) + self.offset_y]).T
        # ignore points that are behind the camera
        transformed_points[points[..., 2] > 0] = np.nan
        # return the points
        return transformed_points


class EquirectangularProjection(CameraProjection):

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # set z=1 and solve the other equations for x and y
        z = np.ones(points[..., 0].shape)
        x = z*np.tan((points[..., 0]-self.offset_x)/self.focallength_px)
        y = np.sqrt(x**2+z**2)*np.tan((points[..., 1] - self.offset_y)/self.focallength_px)
        # compose the ray
        ray = np.array([x, y, z]).T
        # norm the ray if desired
        if normed:
            ray /= np.linalg.norm(ray, axis=-1)
        # return the rey
        return -ray

    def imageFromCamera(self, points):
        """
                             ( x )                                  (       y       )
            x_im = f * arctan(---) + offset_x      y_im = f * arctan(---------------) + offset_y
                             ( z )                                  (sqrt(x**2+z**2))
        """
        # ensure that the points are provided as an array
        points = np.array(points)
        # transform the points
        transformed_points = np.array([self.focallength_px * np.arctan(points[..., 0] / points[..., 2]) + self.offset_x,
                                       -self.focallength_px * np.arctan(points[..., 1] / np.sqrt(points[..., 0]**2 + points[..., 2]**2)) + self.offset_y]).T
        # ignore points that are behind the camera
        transformed_points[points[..., 2] > 0] = np.nan
        # return the points
        return transformed_points
