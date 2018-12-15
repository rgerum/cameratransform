import numpy as np
import pandas as pd
import os
import json
import itertools
import matplotlib.pyplot as plt
import cv2
from scipy import stats
from .parameter_set import ParameterSet, ClassWithParameterSet, Parameter, TYPE_GPS
from .projection import RectilinearProjection, CameraProjection
from .spatial import SpatialOrientation
from .lens_distortion import NoDistortion, LensDistortion
from . import gps
from . import ray


def _getSensorFromDatabase(model):
    """
    Get the sensor size from the given model from the database at: https://github.com/openMVG/CameraSensorSizeDatabase

    Parameters
    ----------
    model: string
        the model name as received from the exif data

    Returns
    -------
    sensor_size: tuple
        (sensor_width, sensor_height) in mm or None
    """
    import requests

    url = "https://raw.githubusercontent.com/openMVG/CameraSensorSizeDatabase/master/sensor_database_detailed.csv"

    database_filename = "sensor_database_detailed.csv"
    # download the database if it is not there
    if not os.path.exists(database_filename):
        with open(database_filename, "w") as fp:
            print("Downloading database from:", url)
            r = requests.get(url)
            fp.write(r.text)
    # load the database
    with open(database_filename, "r") as fp:
        data = fp.readlines()

    # format the name
    model = model.replace(" ", ";", 1)
    name = model + ";"
    # try to find it
    for line in data:
        if line.startswith(name):
            # extract the sensor dimensions
            line = line.split(";")
            sensor_size = (float(line[3]), float(line[4]))
            return sensor_size
    # no sensor size found
    return None


def getCameraParametersFromExif(filename, verbose=False, sensor_from_database=True):
    """
    Try to extract the intrinsic camera parameters from the exif information.

    Parameters
    ----------
    filename: basestring
        the filename of the image to load.
    verbose: bool
         whether to print the output.
    sensor_from_database: bool
        whether to try to load the sensor size from a database at https://github.com/openMVG/CameraSensorSizeDatabase

    Returns
    -------
    focal_length: number
        the extracted focal length in mm
    sensor_size: tuple
        (width, height) of the camera sensor in mm
    image_size: tuple
        (width, height) of the image in pixel

    Examples
    --------

    >>> import CameraTransform as ct

    Supply the image filename to print the results:

    >>> ct.getCameraParametersFromExif("Image.jpg", verbose=True)
    Intrinsic parameters for 'Canon EOS 50D':
       focal length: 400.0 mm
       sensor size: 22.3 mm × 14.9 mm
       image size: 4752 × 3168 Pixels

    Or use the resulting parameters to initialize a CameraTransform instance:

    >>> focal_length, sensor_size, image_size = ct.getCameraParametersFromExif("Image.jpg")
    >>> cam = ct.Camera(focal_length, sensor=sensor_size, image=image_size)

    """
    from PIL import Image
    from PIL.ExifTags import TAGS

    def get_exif(fn):
        ret = {}
        i = Image.open(fn)
        info = i._getexif()
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            ret[decoded] = value
        return ret

    # read the exif information of the file
    exif = get_exif(filename)
    # get the focal length
    f = exif["FocalLength"][0] / exif["FocalLength"][1]
    # get the sensor size, either from a database
    if sensor_from_database:
        sensor_size = _getSensorFromDatabase(exif["Model"])
    # or from the exif information
    if not sensor_size or sensor_size is None:
        sensor_size = (
            exif["ExifImageWidth"] / (exif["FocalPlaneXResolution"][0] / exif["FocalPlaneXResolution"][1]) * 25.4,
            exif["ExifImageHeight"] / (exif["FocalPlaneYResolution"][0] / exif["FocalPlaneYResolution"][1]) * 25.4)
    # get the image size
    image_size = (exif["ExifImageWidth"], exif["ExifImageHeight"])
    # print the output if desired
    if verbose:
        print("Intrinsic parameters for '%s':" % exif["Model"])
        print("   focal length: %.1f mm" % f)
        print("   sensor size: %.1f mm × %.1f mm" % sensor_size)
        print("   image size: %d × %d Pixels" % image_size)
    return f, sensor_size, image_size


class CameraGroup(ClassWithParameterSet):
    projection_list = None
    orientation_list = None
    lens_list = None

    def __init__(self, projection, orientation=None, lens=None):
        ClassWithParameterSet.__init__(self)
        self.N = 1

        def checkCount(parameter, class_type, parameter_name, default):
            if parameter is None:
                setattr(self, parameter_name, [default()])
            elif isinstance(parameter, class_type):
                setattr(self, parameter_name, [parameter])
            else:
                setattr(self, parameter_name, list(parameter))
                self.N = len(getattr(self, parameter_name))

        checkCount(projection, CameraProjection, "projection_list", RectilinearProjection)
        checkCount(orientation, SpatialOrientation, "orientation_list", SpatialOrientation)
        checkCount(lens, LensDistortion, "lens_list", NoDistortion)

        params = {}
        def gatherParameters(parameter_list):
            if len(parameter_list) == 1:
                params.update(parameter_list[0].parameters.parameters)
            else:
                for index, proj in enumerate(parameter_list):
                    for name in proj.parameters.parameters:
                        params["C%d_%s" % (index, name)] = proj.parameters.parameters[name]

        gatherParameters(self.projection_list)
        gatherParameters(self.orientation_list)
        gatherParameters(self.lens_list)

        self.parameters = ParameterSet(**params)

        self.cameras = [Camera(projection, orientation, lens) for index, projection, orientation, lens in
                        zip(range(self.N), itertools.cycle(self.projection_list), itertools.cycle(self.orientation_list), itertools.cycle(self.lens_list))]

    def getBaseline(self):
        return np.sqrt((self[0].pos_x_m-self[1].pos_x_m)**2 + (self[0].pos_y_m-self[1].pos_y_m)**2)

    def spaceFromImages(self, points1, points2):
        p1, v1 = self.cameras[0].getRay(points1)
        p2, v2 = self.cameras[1].getRay(points2)
        return ray.intersectionOfTwoLines(p1, v1, p2, v2)

    def discanteBetweenRays(self, points1, points2):
        p1, v1 = self.cameras[0].getRay(points1, normed=True)
        p2, v2 = self.cameras[1].getRay(points2, normed=True)
        return ray.distanceOfTwoLines(p1, v1, p2, v2)

    def imagesFromSpace(self, points):
        return [cam.imageFromSpace(points) for cam in self.cameras]

    def __getitem__(self, item):
        return self.cameras[item]

    def __len__(self):
        return len(self.cameras)

    def __iter__(self):
        return iter(self.cameras)

    def addBaselineInformation(self, target_baseline, uncertainty=6):
        def baselineInformation(target_baseline=target_baseline, uncertainty=uncertainty):
            # baseline
            return np.sum(stats.norm(loc=target_baseline, scale=uncertainty).logpdf(self.getBaseline()))
        self.log_prob.append(baselineInformation)

    def addPointCorrespondenceInformation(self, corresponding1, corresponding2, uncertainty=1):
        def pointCorrespondenceInformation(corresponding1=corresponding1,
                                           corresponding2=corresponding2):
            distances = self.discanteBetweenRays(corresponding1, corresponding2)
            return np.sum(stats.norm(loc=0, scale=uncertainty).logpdf(distances))

        self.log_prob.append(pointCorrespondenceInformation)


class Camera(ClassWithParameterSet):
    """
    This class is the core of the CameraTransform package and represents a camera. Each camera has a projection
    (subclass of :py:class:`CameraProjection`), a spatial orientation (:py:class:`SpatialOrientation`) and optionally
    a lens distortion (subclass of :py:class:`LensDistortion`).
    """
    map = None
    last_extent = None
    last_scaling = None

    map_undistort = None
    last_extent_undistort = None
    last_scaling_undistort = None

    R_earth = 6371e3

    def __init__(self, projection, orientation=None, lens=None):
        ClassWithParameterSet.__init__(self)
        self.projection = projection
        if orientation is None:
            orientation = SpatialOrientation()
        self.orientation = orientation
        if lens is None:
            lens = NoDistortion()
        self.lens = lens
        self.lens.setProjection(projection)

        params = dict(gps_lat=Parameter(0, default=0, type=TYPE_GPS), gps_lon=Parameter(0, default=0, type=TYPE_GPS))
        params.update(self.projection.parameters.parameters)
        params.update(self.orientation.parameters.parameters)
        params.update(self.lens.parameters.parameters)
        self.parameters = ParameterSet(**params)

    def __str__(self):
        string = "CameraTransform(\n"
        string += str(self.lens)
        string += str(self.projection)
        string += str(self.orientation)
        string += ")"
        return string

    def setGPSpos(self, lat, lon=None, elevation=None):
        """
        Provide the earth position for the camera.

        Parameters
        ----------
        lat: number, string
            the latitude of the camera or the string representing the gps position.
        lon: number, optional
            the longitude of the camera.
        elevation: number, optional
            the elevation of the camera.

        Examples
        --------

        >>> import CameraTransform as ct
        >>> cam = ct.Camera()

        Supply the gps position of the camera as floats:

        >>> cam.setGPSpos(-66.66, 140.00, 19)

        or as a string:

        >>> cam.setGPSpos("66°39'53.4\"S	140°00'34.8\"")
        """
        # if it is a string
        if isinstance(lat, str):
            try:
                lat, lon, elevation = gps.gpsFromString(lat, height=elevation)
            except ValueError:
                lat, lon = gps.gpsFromString(lat, height=elevation)
        else:
            # if it is a tuple
            try:
                lat, lon, elevation = gps.splitGPS(lat, keep_deg=True)
            # or if it is just a single value
            except AttributeError:
                pass
        self.gps_lat = lat
        self.gps_lon = lon
        if elevation is not None:
            self.elevation_m = elevation

    def addObjectHeightInformation(self, points_feet, points_head, height, variation, only_plot=False, plot_color=None):
        """
        Add a term to the camera probability used for fitting. This term includes the probability to observe the objects
        with the given feet and head positions and a known height and height variation.

        Parameters
        ----------
        points_feet : ndarray
            the position of the objects feet, dimension (2) or (Nx2)
        points_head : ndarray
            the position of the objects head, dimension (2) or (Nx2)
        height : number, ndarray
            the mean height of the objects, dimensions scalar or (N)
        variation : number, ndarray
            the standard deviation of the heights of the objects, dimensions scalar or (N). If the variation is not known
            a pymc2 stochastic variable object can be used.
        only_plot : bool, optional
            when true, the information will be ignored for fitting and only be used to plot.
        """
        if not only_plot:
            if not isinstance(variation, (float, int)):
                self.additional_parameters += [variation]
                def heigthInformation(points_feet=points_feet, points_head=points_head, height=height,
                                      variation=variation):
                    height_distribution = stats.norm(loc=height, scale=variation.value)

                    # get the height of the penguins
                    heights = self.getObjectHeight(points_feet, points_head)

                    # the probability that the objects have this height
                    return np.sum(height_distribution.logpdf(heights))

            else:
                height_distribution = stats.norm(loc=height, scale=variation)
                def heigthInformation(points_feet=points_feet, points_head=points_head, height_distribution=height_distribution):
                    # get the height of the penguins
                    heights = self.getObjectHeight(points_feet, points_head)

                    # the probability that the objects have this height
                    return np.sum(height_distribution.logpdf(heights))

            self.log_prob.append(heigthInformation)

        def plotHeightPoints(points_feet=points_feet, points_head=points_head, color=plot_color):
            p, = plt.plot(points_feet[..., 0], points_feet[..., 1], "_", label="feet", color=color)

            # get the feet positions in the world
            point3D_feet = self.spaceFromImage(points_feet, Z=0)
            point3D_feet[..., 2] += height
            projected_head = self.imageFromSpace(point3D_feet)

            plt.scatter(projected_head[..., 0], projected_head[..., 1], label="heads", facecolors='none',
                        edgecolors=p.get_color())
            plt.plot(points_head[..., 0], points_head[..., 1], "+", label="heads fitted", color=p.get_color())

            data = np.concatenate(([points_head], [projected_head], [np.ones(points_head.shape)*np.nan]))
            if len(data.shape) == 3:
                data = data.transpose(1, 0, 2).reshape((-1, 2))
            else:
                data = data.reshape((-1, 2))

            plt.plot(data[..., 0], data[..., 1], "-", color=p.get_color())

        self.info_plot_functions.append(plotHeightPoints)

    def addLandmarkInformation(self, lm_points_image, lm_points_space, uncertainties, only_plot=False, plot_color=None):
        """
        Add a term to the camera probability used for fitting. This term includes the probability to observe the given
        landmarks and the specified positions in the image.

        Parameters
        ----------
        lm_points_image : ndarray
            the pixel positions of the landmarks in the image, dimension (2) or (Nx2)
        lm_points_space : ndarray
            the **space** positions of the landmarks, dimension (3) or (Nx3)
        uncertainties : number, ndarray
            the standard deviation uncertainty of the positions in the **space** coordinates. Typically for landmarks
            obtained by gps, it could be e.g. [3, 3, 5], dimensions scalar, (3) or (Nx3)
        only_plot : bool, optional
            when true, the information will be ignored for fitting and only be used to plot.
        """
        uncertainties = np.array(uncertainties)
        offset = np.max(uncertainties)
        sampled_offsets = np.linspace(-2*offset, +2*offset, 1000)
        if len(lm_points_image.shape) == 1:
            lm_points_image = lm_points_image[None, ...]
        if len(lm_points_space.shape) == 1:
            lm_points_space = lm_points_space[None, ...]
        if len(uncertainties.shape) == 1:
            uncertainties = uncertainties[None, ..., None]
        else:
            uncertainties = uncertainties[..., None]

        def landmarkInformation(lm_points_image=lm_points_image, lm_points_space=lm_points_space, uncertainties=uncertainties):
            origins, lm_rays = self.getRay(lm_points_image, normed=True)
            nearest_point = ray.getClosestPointFromLine(origins, lm_rays, lm_points_space)
            distance_from_camera = np.linalg.norm(nearest_point-np.array([self.pos_x_m, self.pos_y_m, self.elevation_m]), axis=-1)
            factor = distance_from_camera[..., None] + sampled_offsets

            distribution = stats.norm(lm_points_space[..., None], uncertainties)

            points_on_rays = origins[None, :, None] + lm_rays[:, :, None] * factor[:, None, :]

            return np.sum(distribution.logpdf(points_on_rays))

        if not only_plot:
            self.log_prob.append(landmarkInformation)

        def plotLandmarkPoints(lm_points_image=lm_points_image, lm_points_space=lm_points_space, color=plot_color):
            lm_projected_image = self.imageFromSpace(lm_points_space)

            p, = plt.plot(lm_points_image[..., 0], lm_points_image[..., 1], "+", label="landmarks fitted", color=color)
            plt.scatter(lm_projected_image[..., 0], lm_projected_image[..., 1], label="landmarks", facecolors='none', edgecolors=p.get_color())

            data = np.concatenate(([lm_points_image], [lm_projected_image], [np.ones(lm_points_image.shape) * np.nan]))
            if len(data.shape) == 3:
                data = data.transpose(1, 0, 2).reshape((-1, 2))
            else:
                data = data.reshape((-1, 2))

            plt.plot(data[..., 0], data[..., 1], "-", color=p.get_color())
        self.info_plot_functions.append(plotLandmarkPoints)

    def addHorizonInformation(self, horizon, uncertainty=1, only_plot=False, plot_color=None):
        """
        Add a term to the camera probability used for fitting. This term includes the probability to observe the horizon
        at the given pixel positions.

        Parameters
        ----------
        horizon : ndarray
            the pixel positions of points on the horizon in the image, dimension (2) or (Nx2)
        uncertainty : number, ndarray
            the pixels offset, how clear the horizon is visible in the image, dimensions () or (N)
        only_plot : bool, optional
            when true, the information will be ignored for fitting and only be used to plot.
        """
        # ensure that input is an numpy array
        horizon = np.array(horizon)

        def horizonInformation(horizon=horizon, uncertainty=uncertainty):
            # evaluate the horizon at the provided x coordinates
            image_horizon = self.getImageHorizon(horizon[..., 0])
            # calculate the difference of the provided to the estimated horizon in y pixels
            horizon_deviation = horizon[..., 1] - image_horizon[..., 1]
            # the distribution for the uncertainties
            distribution = stats.norm(loc=0, scale=uncertainty)
            # calculated the summed log probability
            return np.sum(distribution.logpdf(horizon_deviation))

        if not only_plot:
            self.log_prob.append(horizonInformation)

        def plotHorizonPoints(horizon=horizon, color=plot_color):
            image_horizon = self.getImageHorizon(horizon[..., 0])
            if 0:
                p, = plt.plot(image_horizon[..., 0], image_horizon[..., 1], "+", label="horizon fitted", color=color)

                plt.scatter(horizon[..., 0], horizon[..., 1], label="horizon", facecolors='none', edgecolors=p.get_color())
            else:
                p, = plt.plot(horizon[..., 0], horizon[..., 1], "+", label="horizon", color=color)

                plt.scatter(image_horizon[..., 0], image_horizon[..., 1], label="horizon fitted", facecolors='none', edgecolors=p.get_color())

            image_horizon_line = self.getImageHorizon(np.arange(self.image_width_px))
            plt.plot(image_horizon_line[..., 0], image_horizon_line[..., 1], "--", color=p.get_color())

            data = np.concatenate(([horizon], [image_horizon], [np.ones(horizon.shape) * np.nan]))
            if len(data.shape) == 3:
                data = data.transpose(1, 0, 2).reshape((-1, 2))
            else:
                data = data.reshape((-1, 2))

            plt.plot(data[..., 0], data[..., 1], "-", color=p.get_color())

        self.info_plot_functions.append(plotHorizonPoints)

    def distanceToHorizon(self):
        """
        Calculates the distance of the camera's position to the horizon of the earth. The horizon depends on the radius
        of the earth and the elevation of the camera.

        Returns
        -------
        distance : number
            the distance to the horizon.
        """
        return np.sqrt(2 * self.R_earth ** 2 * (1 - self.R_earth / (self.R_earth + self.elevation_m)))

    def getImageHorizon(self, pointsX=None):
        """
        This function calculates the position of the horizon in the image sampled at the points x=0, x=im_width/2,
        x=im_width.

        Parameters
        ----------
        pointsX : ndarray, optional
            the x positions of the horizon to determine, default is [0, image_width/2, image_width], dimensions () or (N)

        Returns
        -------
        horizon : ndarray
            the points im camera image coordinates of the horizon, dimensions (2), or (Nx2).
        """
        d = self.distanceToHorizon()
        if pointsX is None:
            pointsX = [0, self.image_width_px/2, self.image_width_px]
        pointsY = np.arange(0, self.image_height_px)

        if len(pointsX.shape) == 0:
            pointsX = np.array([pointsX])
            single_point = True
        else:
            single_point = False

        points = []
        # for every x-coordinate where we want to determine the horizon
        for x in pointsX:
            # test all y points of the image
            p = np.vstack((np.ones(len(pointsY))*x, pointsY)).T
            # transform them to the space with a fixed distance from the camera (the distance to the horizon)
            # and select the point with the z coordinate closest to 0
            try:
                y = np.nanargmin(np.abs(self.spaceFromImage(p, D=d)[:, 2]))
            except ValueError:
                y = np.nan
            # add the found point to the list
            points.append([x, y])
            if single_point:
                return np.array([x, y])
        return np.array(points)

    def getImageBorder(self, resolution=1):
        """
        Get the border of the image in a top view. Useful for drawing the field of view of the camera in a map.

        Parameters
        ----------
        resolution : number, optional
            the pixel distance between neighbouring points.

        Returns
        -------
        border : ndarray
            the border of the image in **space** coordinates, dimensions (Nx3)
        """
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
        """
        The cone of the camera's field of view. This includes the border of the image and lines to the origin of the
        camera.

        Returns
        -------
        cone: ndarray
            the cone of the camera in **space** coordinates, dimensions (Nx3)
        """
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
        """
        Convert points (Nx3) from the **space** coordinate system to the **image** coordinate system.

        Parameters
        ----------
        points : ndarray
            the points in **space** coordinates to transform, dimensions (3), (Nx3)

        Returns
        -------
        points : ndarray
            the points in the **image** coordinate system, dimensions (2), (Nx2)

        Examples
        --------

        >>> import CameraTransform as ct
        >>> cam = ct.Camera(ct.RectilinearProjection(focallength_px=3729, image=(4608, 2592)),
        >>>                    ct.SpatialOrientation(elevation_m=15.4, tilt_deg=85))

        transform a single point from the space to the image:

        >>> cam.imageFromSpace([-4.17, 45.32, 0.])
        [1969.52 2209.73]

        or multiple points in one go:

        >>> cam.imageFromSpace([[-4.03, 43.96,  0.], [-8.57, 47.91, 0.]]))
        [[1971.05 2246.95]
         [1652.73 2144.53]]
        """
        # ensure that the points are provided as an array
        points = np.array(points)
        # project the points from the space to the camera and from the camera to the image
        return self.lens.distortedFromImage(self.projection.imageFromCamera(self.orientation.cameraFromSpace(points)))

    def getRay(self, points, normed=False):
        """
        As the transformation from the **image** coordinate system to the **space** coordinate system is not unique,
        **image** points can only be uniquely mapped to a ray in **space** coordinates.

        Parameters
        ----------
        points : ndarray
            the points in **image** coordinates for which to get the ray, dimensions (2), (Nx2)

        Returns
        -------
        offset : ndarray
            the origin of the camera (= starting point of the rays) in **space** coordinates, dimensions (3)
        rays : ndarray
            the rays in the **space** coordinate system, dimensions (3), (Nx3)

        Examples
        --------

        >>> import CameraTransform as ct
        >>> cam = ct.Camera(ct.RectilinearProjection(focallength_px=3729, image=(4608, 2592)),
        >>>                    ct.SpatialOrientation(elevation_m=15.4, tilt_deg=85))

        get the ray of a point in the image:

        >>> offset, ray = cam.getRay([1968, 2291]))
        >>> offset
        [0.00 0.00 15.40]
        >>> ray
        [-0.09 0.97 -0.35]

        or the rays of multiple points in the image:

        >>> offset, ray, cam.getRay([[1968, 2291], [1650, 2189]])
        >>> offset
        [0.00 0.00 15.40]
        >>> ray
        [[-0.09 0.97 -0.35]
         [-0.18 0.98 -0.33]]
        """
        # ensure that the points are provided as an array
        points = np.array(points)
        # get the camera position in space (the origin of the camera coordinate system)
        offset = self.orientation.spaceFromCamera([0, 0, 0])
        # get the direction fo the ray from the points
        # the projection provides the ray in camera coordinates, which we convert to the space coordinates
        direction = self.orientation.spaceFromCamera(self.projection.getRay(self.lens.imageFromDistorted(points), normed=normed), direction=True)
        # return the offset point and the direction of the ray
        return offset, direction

    def spaceFromImage(self, points, X=None, Y=None, Z=0, D=None, mesh=None):
        """
        Convert points (Nx2) from the **image** coordinate system to the **space** coordinate system. This is not a unique
        transformation, therefore an additional constraint has to be provided. The X, Y, or Z coordinate(s) of the target
        points can be provided or the distance D from the camera.

        Parameters
        ----------
        points : ndarray
            the points in **image** coordinates to transform, dimensions (2), (Nx2)
        X : number, ndarray, optional
            the X coordinate in **space** coordinates of the target points, dimensions scalar, (N)
        Y : number, ndarray, optional
            the Y coordinate in **space** coordinates of the target points, dimensions scalar, (N)
        Z : number, ndarray, optional
            the Z coordinate in **space** coordinates of the target points, dimensions scalar, (N), default 0
        D : number, ndarray, optional
            the distance in **space** coordinates of the target points from the camera, dimensions scalar, (N)
        mesh : ndarray, optional
            project the image coordinates onto the mesh in **space** coordinates. The mesh is a list of M triangles,
            consisting of three 3D points each. Dimensions, (3x3), (Mx3x3)
        Returns
        -------
        points : ndarray
            the points in the **space** coordinate system, dimensions (3), (Nx3)

        Examples
        --------

        >>> import CameraTransform as ct
        >>> cam = ct.Camera(ct.RectilinearProjection(focallength_px=3729, image=(4608, 2592)),
        >>>                    ct.SpatialOrientation(elevation_m=15.4, tilt_deg=85))

        transform a single point (impliying the condition Z=0):

        >>> cam.spaceFromImage([1968 , 2291])
        [-3.93 42.45 0.00]

        transform multiple points:

        >>> cam.spaceFromImage([[1968 , 2291], [1650, 2189]])
        [[-3.93 42.45 0.00]
         [-8.29 46.11 -0.00]]

        points that cannot be projected on the image, because they are behind the camera (for the RectilinearProjection)
        are returned with nan entries:

        >>> cam.imageFromSpace([-4.17, -10.1, 0.])
        [nan nan]

        specify a y coordinate as for the back projection.

        >>> cam.spaceFromImage([[1968 , 2291], [1650, 2189]], Y=45)
        [[-4.17 45.00 -0.93]
         [-8.09 45.00 0.37]]

        or different y coordinates for each point:

        >>> cam.spaceFromImage([[1968 , 2291], [1650, 2189]], Y=[43, 45])
        [[-3.98 43.00 -0.20]
         [-8.09 45.00 0.37]]
        """
        # ensure that the points are provided as an array
        points = np.array(points)
        # get the index which coordinate to force to the given value
        given = np.array([X, Y, Z])
        if X is not None:
            index = 0
        elif Y is not None:
            index = 1
        elif Z is not None:
            index = 2

        # if a mesh is provided, intersect the rays with the mesh
        if mesh is not None:
            # get the rays from the image points
            offset, direction = self.getRay(points)
            return ray.ray_intersect_triangle(offset, direction, mesh)
        # transform to a given distance
        if D is not None:
            # get the rays from the image points (in this case it has to be normed)
            offset, direction = self.getRay(points, normed=True)
            # the factor is than simple the distance
            factor = D
        else:
            # get the rays from the image points
            offset, direction = self.getRay(points)
            # solve the line equation for the factor (how many times the direction vector needs to be added to the origin point)
            factor = (given[index] - offset[..., index]) / direction[..., index]

        if not isinstance(factor, np.ndarray):
            # if factor is not an array, we don't need to specify the broadcasting
            points = direction * factor + offset
        else:
            # apply the factor to the direction vector plus the offset
            points = direction * factor[:, None] + offset[None, :]
        # ignore points that are behind the camera (e.g. trying to project points above the horizon to the ground)
        points[factor < 0] = np.nan
        return points

    def gpsFromSpace(self, points):
        """
        Convert points (Nx3) from the **space** coordinate system to the **gps** coordinate system.

        Parameters
        ----------
        points : ndarray
            the points in **space** coordinates to transform, dimensions (3), (Nx3)

        Returns
        -------
        points : ndarray
            the points in the **gps** coordinate system, dimensions (3), (Nx3)
        """
        return gps.gpsFromSpace(points, np.array([self.gps_lat, self.gps_lon, self.elevation_m]))

    def spaceFromGPS(self, points):
        """
        Convert points (Nx3) from the **gps** coordinate system to the **space** coordinate system.

        Parameters
        ----------
        points : ndarray
            the points in **gps** coordinates to transform, dimensions (3), (Nx3)

        Returns
        -------
        points : ndarray
            the points in the **space** coordinate system, dimensions (3), (Nx3)
        """
        return gps.spaceFromGPS(points, np.array([self.gps_lat, self.gps_lon, self.elevation_m]))

    def gpsFromImage(self, points, X=None, Y=None, Z=0, D=None):
        """
        Convert points (Nx2) from the **image** coordinate system to the **gps** coordinate system.

        Parameters
        ----------
        points : ndarray
            the points in **image** coordinates to transform, dimensions (2), (Nx2)

        Returns
        -------
        points : ndarray
            the points in the **gps** coordinate system, dimensions (3), (Nx3)
        """
        return self.gpsFromSpace(self.spaceFromImage(points, X=X, Y=Y, Z=Z, D=D))

    def imageFromGPS(self, points):
        """
        Convert points (Nx3) from the **gps** coordinate system to the **image** coordinate system.

        Parameters
        ----------
        points : ndarray
            the points in **gps** coordinates to transform, dimensions (3), (Nx3)

        Returns
        -------
        points : ndarray
            the points in the **image** coordinate system, dimensions (2), (Nx2)
        """
        return self.imageFromSpace(self.spaceFromGPS(points))

    def getObjectHeight(self, point_feet, point_heads, Z=0):
        """
        Calculate the height of objects in the image, assuming the Z position of the objects is known, e.g. they are
        assumed to stand on the Z=0 plane.

        Parameters
        ----------
        point_feet : ndarray
            the positions of the feet, dimensions: (2) or (Nx2)
        point_heads : ndarray
            the positions of the heads, dimensions: (2) or (Nx2)
        Z : number, ndarray, optional
            the Z position of the objects, dimensions: scalar or (N), default 0

        Returns
        -------
        heights: ndarray
            the height of the objects in meters, dimensions: () or (N)
        """
        # get the feet positions in the world
        point3D_feet = self.spaceFromImage(point_feet, Z=Z)
        # get the head positions in the world
        point3D_head1 = self.spaceFromImage(point_heads, Y=point3D_feet[..., 1])
        point3D_head2 = self.spaceFromImage(point_heads, X=point3D_feet[..., 0])
        point3D_head = np.mean([point3D_head1, point3D_head2], axis=0)
        # the z difference between these two points
        return point3D_head[..., 2] - point3D_feet[..., 2]

    def _getUndistortMap(self, extent=None, scaling=None):
        # if no extent is given, take the maximum extent from the image border
        if extent is None:
            extent = [0, self.image_width_px, 0, self.image_height_px]

        # if we have cached the map, use the cached map
        if self.map_undistort is not None and \
                self.last_extent_undistort == extent and \
                self.last_scaling_undistort == scaling:
            return self.map_undistort

        # if no scaling is given, scale so that the resulting image has an equal amount of pixels as the original image
        if scaling is None:
            scaling = 1

        # get a mesh grid
        mesh = np.array(np.meshgrid(np.arange(extent[0], extent[1], scaling),
                                    np.arange(extent[2], extent[3], scaling)))

        # convert it to a list of points Nx2
        mesh_points = mesh.reshape(2, mesh.shape[1] * mesh.shape[2]).T

        # transform the space points to the image
        mesh_points_shape = self.lens.distortedFromImage(mesh_points)

        # reshape the map and cache it
        self.map_undistort = mesh_points_shape.T.reshape(mesh.shape).astype(np.float32)[:, ::-1, :]

        self.last_extent_undistort = extent
        self.last_scaling_undistort = scaling

        # return the calculated map
        return self.map_undistort

    def undistortImage(self, image, extent=None, scaling=None, do_plot=False, alpha=None, skip_size_check=False):
        """
        Applies the undistortion of the lens model to the image. The purpose of this function is mainly to check the
        sanity of a lens transformation. As CameraTransform includes the lens transformation in any calculations, it
        is not necessary to undistort images before using them.

        Parameters
        ----------
        image : ndarray
            the image to undistort.
        extent : list, optional
            the extent in pixels of the resulting image. This can be used to crop the resulting undistort image.
        scaling : number, optional
            the number of old pixels that are used to calculate a new pixel. A higher value results in a smaller target
            image.
        do_plot : bool, optional
            whether to plot the resulting image directly in a matplotlib plot.
        alpha : number, optional
            when plotting an alpha value can be specified, useful when comparing multiple images.
        skip_size_check : bool, optional
            if true, the size of the image is not checked to match the size of the cameras image.

        Returns
        -------
        image : ndarray
            the undistorted image
        """
        # check if the size of the image matches the size of the camera
        if not skip_size_check:
            assert image.shape[1] == self.image_width_px, "The with of the image (%d) does not match the image width of the camera (%d)" % (image.shape[1], self.image_width_px)
            assert image.shape[0] == self.image_height_px, "The height of the image (%d) does not match the image height of the camera (%d)." % (image.shape[0], self.image_height_px)

        x, y = self._getUndistortMap(extent=extent, scaling=scaling)
        # ensure that the image has an alpha channel (to enable alpha for the points outside the image)
        if len(image.shape) == 2:
            pass
        elif image.shape[2] == 3:
            image = np.dstack((image, np.ones(shape=(image.shape[0], image.shape[1], 1), dtype="uint8") * 255))
        image = cv2.remap(image, x, y,
                          interpolation=cv2.INTER_NEAREST,
                          borderValue=[0, 1, 0, 0])[::-1]  # , borderMode=cv2.BORDER_TRANSPARENT)
        if do_plot:
            extent = self.last_extent_undistort.copy()
            extent[2], extent[3] = extent[3]-1, extent[2]-1
            plt.imshow(image, extent=extent, alpha=alpha)
        return image

    def _getMap(self, extent=None, scaling=None, Z=0):
        # if no extent is given, take the maximum extent from the image border
        if extent is None:
            border = self.getImageBorder()
            extent = [np.nanmin(border[:, 0]), np.nanmax(border[:, 0]),
                      np.nanmin(border[:, 1]), np.nanmax(border[:, 1])]

        # if we have cached the map, use the cached map
        if self.map is not None and \
                self.last_extent == extent and \
                self.last_scaling == scaling:
            return self.map

        # if no scaling is given, scale so that the resulting image has an equal amount of pixels as the original image
        if scaling is None:
            scaling = np.sqrt((extent[1] - extent[0]) * (extent[3] - extent[2])) / \
                      np.sqrt((self.projection.parameters.image_width_px * self.projection.parameters.image_height_px))

        # get a mesh grid
        mesh = np.array(np.meshgrid(np.arange(extent[0], extent[1], scaling),
                                    np.arange(extent[2], extent[3], scaling)))

        # convert it to a list of points Nx2
        mesh_points = mesh.reshape(2, mesh.shape[1] * mesh.shape[2]).T
        mesh_points = np.hstack((mesh_points, Z*np.ones((mesh_points.shape[0], 1))))

        # transform the space points to the image
        mesh_points_shape = self.imageFromSpace(mesh_points)

        # reshape the map and cache it
        self.map = mesh_points_shape.T.reshape(mesh.shape).astype(np.float32)[:, ::-1, :]

        self.last_extent = extent
        self.last_scaling = scaling

        # return the calculated map
        return self.map

    def getTopViewOfImage(self, image, extent=None, scaling=None, do_plot=False, alpha=None, Z=0., skip_size_check=False):
        """
        Project an image to a top view projection. This will be done using a grid with the dimensions of the extent
        ([x_min, x_max, y_min, y_max]) in meters and the scaling, giving a resolution. For convenience, the image can
        be plotted directly. The projected grid is cached, so if the function is called a second time with the same
        parameters, the second call will be faster.

        Parameters
        ----------
        image : ndarray
            the image as a numpy array.
        extent : list, optional
            the extent of the resulting top view in meters: [x_min, x_max, y_min, y_max]. If no extent is given a suitable
            extent is guessed. If a horizon is visible in the image, the guessed extent will in most cases be too streched.
        scaling : number, optional
            the scaling factor, how many meters is the side length of each pixel in the top view. If no scaling factor is
            given, a good scaling factor is guessed, trying to get about the same number of pixels in the top view as in
            the original image.
        do_plot : bool, optional
            whether to directly plot the resulting image in a matplotlib figure.
        alpha : number, optional
            an alpha value used when plotting the image. Useful if multiple images should be overlaid.
        Z : number, optional
            the "height" of the plane on which to project.
        skip_size_check : bool, optional
            if true, the size of the image is not checked to match the size of the cameras image.

        Returns
        -------
        image : ndarray
            the top view projected image
        """
        # check if the size of the image matches the size of the camera
        if not skip_size_check:
            assert image.shape[1] == self.image_width_px, "The with of the image (%d) does not match the image width of the camera (%d)" % (image.shape[1], self.image_width_px)
            assert image.shape[0] == self.image_height_px, "The height of the image (%d) does not match the image height of the camera (%d)." % (image.shape[0], self.image_height_px)
        # get the mapping
        x, y = self._getMap(extent=extent, scaling=scaling, Z=Z)
        # ensure that the image has an alpha channel (to enable alpha for the points outside the image)
        if len(image.shape) == 2:
            pass
        elif image.shape[2] == 3:
            image = np.dstack((image, np.ones(shape=(image.shape[0], image.shape[1], 1), dtype="uint8") * 255))
        image = cv2.remap(image, x, y,
                          interpolation=cv2.INTER_NEAREST,
                          borderValue=[0, 1, 0, 0])  # , borderMode=cv2.BORDER_TRANSPARENT)
        if do_plot:
            plt.imshow(image, extent=self.last_extent, alpha=alpha)
        return image

    def generateLUT(self, undef_value=0):
        """
        Generate LUT to calculate area covered by one pixel in the image dependent on y position in the image

        Parameters
        ----------
        undef_value: number, optional
            what values undefined positions should have, default=0

        Returns
        -------
        LUT: ndarray
            same length as image height
        """

        def get_square(x, y):
            p0 = [x - 0.5, y - 0.5]
            p1 = [x + 0.5, y - 0.5]
            p2 = [x + 0.5, y + 0.5]
            p3 = [x - 0.5, y + 0.5]
            return np.array([p0, p1, p2, p3])

        def calc_quadrilateral_size(rect):
            A, B, C, D = rect
            return 0.5 * abs((A[1] - C[1]) * (D[0] - B[0]) + (B[1] - D[1]) * (A[0] - C[0]))

        x = self.image_width_px / 2

        horizon = self.getImageHorizon([x])
        y_stop = max([0, int(horizon[0, 1])])
        y_start = self.image_height_px

        y_lookup = np.zeros(self.image_height_px) + undef_value

        for y in range(y_stop, y_start):
            rect = get_square(x, y)
            rect = self.spaceFromImage(rect, Z=0)
            A = calc_quadrilateral_size(rect)
            y_lookup[y] = A

        return y_lookup

    def rotateSpace(self, delta_heading):
        """
        Rotates the whole camera setup, this will turn the heading and rotate the camera position (pos_x_m, pos_y_m)
        around the origin.

        Parameters
        ----------
        delta_heading : number
            the number of degrees to rotate the camera clockwise.
        """
        self.heading_deg += delta_heading
        delta_heading_rad = np.deg2rad(delta_heading)
        pos = np.array([self.pos_x_m, self.pos_y_m])
        s, c = np.sin(delta_heading_rad), np.cos(delta_heading_rad)
        self.pos_x_m, self.pos_y_m = np.dot(np.array([[c, s], [-s, c]]), pos)

    def save(self, filename):
        """
        Saves the camera parameters to a json file.

        Parameters
        ----------
        filename : str
            the filename where to store the parameters.
        """
        keys = self.parameters.parameters.keys()
        export_dict = {key: getattr(self, key) for key in keys}
        with open(filename, "w") as fp:
            fp.write(json.dumps(export_dict, indent=4))

    def load(self, filename):
        """
        Load the camera parameters from a json file.

        Parameters
        ----------
        filename : str
            the filename of the file to load.
        """
        with open(filename, "r") as fp:
            variables = json.loads(fp.read())
        for key in variables:
            setattr(self, key, variables[key])


def load_camera(filename):
    """
    Create a :py:class:`Camera` instance with the parameters from the file.

    Parameters
    ----------
    filename : str
        the filename of the file to load.

    Returns
    -------
    camera : :py:class:`Camera`
        the camera with the given parameters.
    """
    cam = Camera(RectilinearProjection(), SpatialOrientation())
    cam.load(filename)
    return cam
