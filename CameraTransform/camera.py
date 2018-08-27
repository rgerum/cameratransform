import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize
from .parameter_set import ParameterSet, ClassWithParameterSet
from .projection import RectilinearProjection
from .spatial import SpatialOrientation
import json


class CameraGroup(ClassWithParameterSet):
    def __init__(self, projection, orientation_list=None):
        self.projection = projection
        self.orientation_list = orientation_list

        params = {}
        params.update(self.projection.parameters.parameters)
        for index, orientation in enumerate(orientation_list):
            for name in orientation.parameters.parameters:
                params["C%d_%s" % (index, name)] = name
        self.parameters = ParameterSet(**params)

        self.cameras = [Camera(projection, orientation) for orientation in self.orientation_list]

    def fit(self, cost_function):
        names = self.parameters.get_fit_parameters()
        ranges = self.parameters.get_parameter_ranges(names)
        estimates = self.parameters.get_parameter_defaults(names)

        def cost(p):
            self.parameters.set_fit_parameters(names, p)
            return cost_function()

        p = minimize(cost, estimates, bounds=ranges)
        self.parameters.set_fit_parameters(names, p["x"])
        return p


class Camera(ClassWithParameterSet):
    map = None
    last_extent = None
    last_scaling = None

    def __init__(self, projection, orientation):
        self.projection = projection
        self.orientation = orientation

        params = {}
        params.update(self.projection.parameters.parameters)
        params.update(self.orientation.parameters.parameters)
        self.parameters = ParameterSet(**params)

    def __str__(self):
        string = "CameraTransform(\n"
        string += str(self.projection)
        string += str(self.orientation)
        return string

    def fit(self, cost_function):
        names = self.parameters.get_fit_parameters()
        ranges = self.parameters.get_parameter_ranges(names)
        estimates = self.parameters.get_parameter_defaults(names)

        def cost(p):
            self.parameters.set_fit_parameters(names, p)
            return cost_function()

        p = minimize(cost, estimates, bounds=ranges)
        self.parameters.set_fit_parameters(names, p["x"])
        return p

    def getImageBorder(self, resolution=1):
        w, h = self.projection.parameters.image_width_px, self.projection.parameters.image_height_px
        border = []
        for y in np.arange(0, h, resolution):
            border.append([0, y])
        for x in np.arange(0, w, resolution):
            border.append([x, h])
        for y in np.arange(h, 0, -resolution):
            border.append([w, y])
        for x in np.arange(w, 0, -resolution):
            border.append([x, 0])
        return self.spaceFromImage(border, Z=0)

    def getCameraCone(self):
        w, h = self.projection.parameters.image_width_px, self.projection.parameters.image_height_px
        border = []
        corner_indices = [0]
        for y in range(h):
            border.append([0, y])
        corner_indices.append(len(border))
        for x in range(w):
            border.append([x, h])
        corner_indices.append(len(border))
        for y in np.arange(h, 0, -1):
            border.append([w, y])
        corner_indices.append(len(border))
        for x in np.arange(w, 0, -1):
            border.append([x, 0])
        corner_indices.append(len(border))
        border = list(self.spaceFromImage(border, Z=0))
        origin = self.orientation.spaceFromCamera([0, 0, 0])
        for corner_index in corner_indices:
            border.append([np.nan, np.nan, np.nan])
            border.append(origin)
            border.append(border[corner_index])
        return np.array(border)

    def imageFromSpace(self, points):
        # ensure that the points are provided as an array
        points = np.array(points)
        # project the points from the space to the camera and from the camera to the image
        return self.projection.imageFromCamera(self.orientation.cameraFromSpace(points))

    def getRay(self, points, normed=False):
        # ensure that the points are provided as an array
        points = np.array(points)
        # get the camera position in space (the origin of the camera coordinate system)
        offset = self.orientation.spaceFromCamera([0, 0, 0])
        # get the direction fo the ray from the points
        # the projection provides the ray in camera coordinates, which we convert to the space coordinates
        direction = self.orientation.spaceFromCamera(self.projection.getRay(points, normed=normed), direction=True)
        # return the offset point and the direction of the ray
        return offset, direction

    def spaceFromImage(self, points, X=None, Y=None, Z=0):
        # ensure that the points are provided as an array
        points = np.array(points)
        # get the index which coordinate to force to the given value
        given = np.array([X, Y, Z])
        index = np.argmax(given != None)
        # get the rays from the image points
        offset, direction = self.getRay(points)
        # solve the line equation for the factor (how many times the direction vector needs to be added to the origin point)
        factor = (given[index] - offset[..., index]) / direction[..., index]
        if not isinstance(factor, np.ndarray):
            factor = np.array([factor])
        # apply the factor to the direction vector plus the offset
        points = direction * factor[:, None] + offset[None, :]
        # ignore points that are behind the camera (e.g. trying to project points above the horizon to the ground)
        points[factor < 0] = np.nan
        return points

    def getMapXXX(self):
        if self.map is None:
            # get a mesh grid
            mesh = np.array(np.meshgrid(np.arange(int(self.projection.parameters.image_width_px)),
                                        np.arange(int(self.projection.parameters.image_height_px))))
            # convert it to a list of points Nx2
            mesh_points = mesh.reshape(2, mesh.shape[1] * mesh.shape[2]).T

            mesh_points_shape = self.spaceFromImage(mesh_points, Z=0)[..., :2]
            plt.plot(mesh_points_shape[:, 0], mesh_points_shape[:, 1], "ko")

            self.map = mesh_points_shape.T.reshape(mesh.shape).astype(np.float32)
        return self.map

    def getMap(self, extent=None, scaling=None):
        # if we have cached the map, use the cached map
        if self.map is not None and \
                all(self.last_extent == extent) and \
                self.last_scaling == scaling:
            return self.map

        # if no extent is given, take the maximum extent from the image border
        if extent is None:
            border = self.getImageBorder()
            extent = [np.nanmin(border[:, 0]), np.nanmax(border[:, 0]),
                      np.nanmin(border[:, 1]), np.nanmax(border[:, 1])]

        # if no scaling is given, scale so that the resulting image has an equal amount of pixels as the original image
        if scaling is None:
            scaling = np.sqrt((extent[1] - extent[0]) * (extent[3] - extent[2])) / \
                      np.sqrt((self.projection.parameters.image_width_px * self.projection.parameters.image_height_px))

        # get a mesh grid
        mesh = np.array(np.meshgrid(np.arange(extent[0], extent[1], scaling),
                                    np.arange(extent[2], extent[3], scaling)))

        # convert it to a list of points Nx2
        mesh_points = mesh.reshape(2, mesh.shape[1] * mesh.shape[2]).T
        mesh_points = np.hstack((mesh_points, np.zeros((mesh_points.shape[0], 1))))

        # transform the space points to the image
        mesh_points_shape = self.imageFromSpace(mesh_points)

        # reshape the map and cache it
        self.map = mesh_points_shape.T.reshape(mesh.shape).astype(np.float32)[:, ::-1, :]

        self.last_extent = extent
        self.last_scaling = scaling

        # return the calculated map
        return self.map

    def getTopViewOfImage(self, im, extent=None, scaling=None, do_plot=False):
        x, y = self.getMap(extent=extent, scaling=scaling)
        # ensure that the image has an alpha channel (to enable alpha for the points outside the image)
        if im.shape[2] == 3:
            im = np.dstack((im, np.ones(shape=(im.shape[0], im.shape[1], 1), dtype="uint8") * 255))
        image = cv2.remap(im, x, y,
                          interpolation=cv2.INTER_NEAREST,
                          borderValue=[0, 1, 0, 0])  # , borderMode=cv2.BORDER_TRANSPARENT)
        if do_plot:
            plt.imshow(image, extent=self.last_extent)  # , alpha=1)
        return image

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

def load_camera(filename):
    cam = Camera(RectilinearProjection(), SpatialOrientation())
    cam.load(filename)
    return cam
