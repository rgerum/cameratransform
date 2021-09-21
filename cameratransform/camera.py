#!/usr/bin/env python
# -*- coding: utf-8 -*-
# camera.py

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
import os
import json
import itertools
from scipy import stats
from .parameter_set import ParameterSet, ClassWithParameterSet, Parameter, TYPE_GPS
from .projection import RectilinearProjection, EquirectangularProjection, CylindricalProjection, CameraProjection
from .spatial import SpatialOrientation
from .lens_distortion import NoDistortion, LensDistortion, ABCDistortion, BrownLensDistortion
from . import gps
from . import ray

RECTILINEAR = 0
CYLINDRICAL = 1
EQUIRECTANGULAR = 2

NODISTORTION = 0
ABCDDISTORTION = 1
BROWNLENSDISTORTION = 2

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

    >>> import cameratransform as ct

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
        def pointCorrespondenceInformation(corresponding1=corresponding1, corresponding2=corresponding2):
            sum = 0
            corresponding = [corresponding1, corresponding2]
            # iterate over cam1 -> cam2 and cam2 -> cam1
            for i in [0, 1]:
                # get the ray from the correspondences in the first camera's image
                world_epipole, world_ray = self[i].getRay(corresponding[i])
                # project them to the image of the second camera
                p1 = self[1 - i].imageFromSpace(world_epipole + world_ray * 1)
                p2 = self[1 - i].imageFromSpace(world_epipole + world_ray * 2)
                # find the perpendicular point from the epipolar lines to the correspondes point
                perpendicular_point = ray.getClosestPointFromLine(p1, p2 - p1, corresponding[1 - i])
                # calculate the distances
                distances = np.linalg.norm(perpendicular_point - corresponding[1 - i], axis=-1)
                # sum the logprob of these distances
                sum += np.sum(stats.norm(loc=0, scale=uncertainty).logpdf(distances))
            # return the sum of the logprobs
            return sum

        self.log_prob.append(pointCorrespondenceInformation)

    def pointCorrespondenceError(self, corresponding1, corresponding2):
        sum = 0
        corresponding = [corresponding1, corresponding2]
        distances_list = []
        # iterate over cam1 -> cam2 and cam2 -> cam1
        for i in [0, 1]:
            # get the ray from the correspondences in the first camera's image
            world_epipole, world_ray = self[i].getRay(corresponding[i])
            # project them to the image of the second camera
            p1 = self[1 - i].imageFromSpace(world_epipole + world_ray * 1, hide_backpoints=False)
            p2 = self[1 - i].imageFromSpace(world_epipole + world_ray * 2, hide_backpoints=False)
            # find the perpendicular point from the epipolar lines to the correspondes point
            perpendicular_point = ray.getClosestPointFromLine(p1, p2 - p1, corresponding[1 - i])
            # calculate the distances
            distances = np.linalg.norm(perpendicular_point - corresponding[1 - i], axis=-1)
            # sum the logprob of these distances
            distances_list.append(distances)
        # return the sum of the logprobs
        return distances_list

    def getLogProbability(self):
        """
        Gives the sum of all terms of the log probability. This function is used for sampling and fitting.
        """
        prob = np.sum([logProb() for logProb in self.log_prob]) + np.sum([logProb() for cam in self for logProb in cam.log_prob])
        return prob if not np.isnan(prob) else -np.inf

    def setCameraParametersByPointCorrespondence(self, corresponding1, corresponding2, baseline):
        import cv2
        cam1 = self[0]
        cam2 = self[1]
        f, cx, cy = cam1.focallength_x_px, cam1.center_x_px, cam1.center_y_px
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        mat, mask = cv2.findEssentialMat(corresponding1, corresponding2, cam1.focallength_x_px,
                                         (cam1.center_x_px, cam1.center_y_px))
        n, rot, t, mask = cv2.recoverPose(mat, corresponding1, corresponding2, K)
        cam1.heading_deg = 0
        cam1.tilt_deg = 0
        cam1.roll_deg = 0

        def rotationToEuler(R):
            alpha = np.rad2deg(np.arctan2(R[0, 2], -R[1, 2]))
            beta = np.rad2deg(np.arccos(R[2, 2]))
            gamma = np.rad2deg(np.arctan2(R[2, 0], R[2, 1]))
            return np.array([180 + alpha, beta, 180 + gamma])

        roll, tilt, heading = rotationToEuler(rot)
        data = dict(roll_deg=roll,
                    tilt_deg=tilt,
                    heading_deg=heading,
                    pos_x_m=cam1.pos_x_m + t[0, 0]*baseline,
                    pos_y_m=cam1.pos_y_m + t[1, 0]*baseline,
                    elevation_m=cam1.elevation_m + t[2, 0]*baseline)
        print(data)
        cam2.parameters.set_fit_parameters(data.keys(), data.values())

    def plotEpilines(self, corresponding1, corresponding2, im1, im2):
        import cv2
        import matplotlib.pyplot as plt
        cam1 = self[0]
        cam2 = self[1]
        F, mask = cv2.findFundamentalMat(corresponding1, corresponding2)#, method=cv2.FM_8POINT)

        lines1 = cv2.computeCorrespondEpilines(corresponding2, 2, F)[:, 0, :]
        lines2 = cv2.computeCorrespondEpilines(corresponding1, 1, F)[:, 0, :]

        def drawLine(line, x_min, x_max, y_min, y_max):
            a, b, c = line
            points = []
            y_x_min = -(a * x_min + c) / b
            if y_min < y_x_min < y_max:
                points.append([x_min, y_x_min])

            y_x_max = -(a * x_max + c) / b
            if y_min < y_x_max < y_max:
                points.append([x_max, y_x_max])

            x_y_min = -(b * y_min + c) / a
            if x_min < x_y_min < x_max:
                points.append([x_y_min, y_min])

            x_y_max = -(b * y_max + c) / a
            if x_min < x_y_max < x_max:
                points.append([x_y_max, y_max])

            if len(points) == 0:
                return
            points = np.array(points)
            p, = plt.plot(points[:, 0], points[:, 1], "-")
            return p

        def drawEpilines(camA, camB, lines, points):
            border = camA.getImageBorder()
            plt.plot(border[:, 0], border[:, 1], "r-")

            for point, line in zip(points, lines):
                line = drawLine(line, 0, camA.image_width_px, 0, camA.image_height_px)
                plt.plot(point[0], point[1], "o", color=line.get_color())

            p = camA.imageFromSpace(camB.getPos())
            print("p", p)
            plt.plot(p[0], p[1], "r+", ms=5)

            plt.axis("equal")

        plt.subplot(121)
        drawEpilines(cam1, cam2, lines1, corresponding1)
        plt.imshow(im1)

        plt.subplot(122)
        drawEpilines(cam2, cam1, lines2, corresponding2)
        plt.imshow(im2)

        plt.show()

    def plotMyEpiploarLines(self, corresponding1, corresponding2, im1=None, im2=None):
        import matplotlib.pyplot as plt
        cam1 = self[0]
        cam2 = self[1]

        def drawEpilines(camA, camB, pointsA, pointsB):
            for pointA, pointB in zip(pointsA, pointsB):

                origin, world_ray = camB.getRay(pointB, normed=True)
                pixel_points = []
                for i in np.arange(-10000, 10000, 100):
                    pixel_points.append(camA.imageFromSpace(origin + world_ray*i, hide_backpoints=False))
                pixel_points = np.array(pixel_points)
                p, = plt.plot(pixel_points[:, 0], pixel_points[:, 1], "-")
                plt.plot(pointA[0], pointA[1], "o", color=p.get_color())

                # find the perpendicular point from the epipolar lines to the correspondes point
                perpendicular_point = ray.getClosestPointFromLine(pixel_points[0], pixel_points[1] - pixel_points[0], pointA)

                plt.plot(perpendicular_point[0], perpendicular_point[1], "+", color=p.get_color())

                plt.plot([pointA[0], perpendicular_point[0]], [pointA[1], perpendicular_point[1]], "--", color=p.get_color())

                # calculate the distances
                distances = np.linalg.norm(perpendicular_point - pointA, axis=-1)
                plt.text(pointA[0], pointA[1], "%.1f" % distances, color=p.get_color())

        plt.subplot(121)
        drawEpilines(cam1, cam2, corresponding1, corresponding2)
        if im1 is not None:
            plt.imshow(im1)

        plt.subplot(122)
        drawEpilines(cam2, cam1, corresponding2, corresponding1)
        if im2 is not None:
            plt.imshow(im2)

    def scaleSpace(self, scale):
        for cam in self:
            cam.pos_x_m, cam.pos_y_m, cam.elevation_m = np.array([cam.pos_x_m, cam.pos_y_m, cam.elevation_m]) * scale


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

        >>> import cameratransform as ct
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
            except (AttributeError, Exception):
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
            import matplotlib.pyplot as plt
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

    def addObjectLengthInformation(self, points_front, points_back, length, variation, Z=0, only_plot=False,
                                   plot_color=None):
        """
        Add a term to the camera probability used for fitting. This term includes the probability to observe the objects
        with a given length lying flat on the surface. The objects are assumed to be like flat rods lying on the z=0 surface.

        Parameters
        ----------
        points_front : ndarray
            the position of the objects front, dimension (2) or (Nx2)
        points_back : ndarray
            the position of the objects back, dimension (2) or (Nx2)
        length : number, ndarray
            the mean length of the objects, dimensions scalar or (N)
        variation : number, ndarray
            the standard deviation of the lengths of the objects, dimensions scalar or (N). If the variation is not known
            a pymc2 stochastic variable object can be used.
        only_plot : bool, optional
            when true, the information will be ignored for fitting and only be used to plot.
        """
        if not only_plot:
            if not isinstance(variation, (float, int)):
                self.additional_parameters += [variation]

                def lengthInformation(points_front=points_front, points_back=points_back, length=length,
                                      variation=variation, Z=Z):
                    length_distribution = stats.norm(loc=length, scale=variation.value)

                    # get the length of the objects
                    heights = self.getObjectLength(points_front, points_back, Z)

                    # the probability that the objects have this height
                    return np.sum(length_distribution.logpdf(heights))

            else:
                length_distribution = stats.norm(loc=length, scale=variation)

                def lengthInformation(points_front=points_front, points_back=points_back,
                                      length_distribution=length_distribution, Z=Z):
                    # get the length of the objects
                    heights = self.getObjectLength(points_front, points_back, Z)

                    # the probability that the objects have this height
                    return np.sum(length_distribution.logpdf(heights))

            self.log_prob.append(lengthInformation)

        def plotHeightPoints(points_front=points_front, points_back=points_back, Z=Z, color=plot_color):
            import matplotlib.pyplot as plt
            p, = plt.plot(points_front[..., 0], points_front[..., 1], "_", label="front", color=color)

            # get the back positions in the world
            point3D_front = self.spaceFromImage(points_front, Z=Z)
            point3D_back = self.spaceFromImage(points_back, Z=Z)
            difference = point3D_back - point3D_front
            difference /= np.linalg.norm(difference, axis=-1)[..., None]
            predicted_back = point3D_front + difference * length
            projected_back = self.imageFromSpace(predicted_back)

            plt.scatter(projected_back[..., 0], projected_back[..., 1], label="back", facecolors='none',
                        edgecolors=p.get_color())
            plt.plot(points_back[..., 0], points_back[..., 1], "+", label="back fitted", color=p.get_color())

            data = np.concatenate(([points_front], [projected_back], [np.ones(points_front.shape) * np.nan]))
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
            import matplotlib.pyplot as plt
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
            import matplotlib.pyplot as plt
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
        pointsX = np.array(pointsX)
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

    def getPos(self):
        return np.array([self.pos_x_m, self.pos_y_m, self.elevation_m])

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
        return np.array(border)

    def getCameraCone(self, project_to_ground=False, D=1):
        """
        The cone of the camera's field of view. This includes the border of the image and lines to the origin of the
        camera.

        Returns
        -------
        cone: ndarray
            the cone of the camera in **space** coordinates, dimensions (Nx3)
        """
        w, h = self.projection.parameters.image_width_px, self.projection.parameters.image_height_px
        if project_to_ground:
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
        else:
            border = []
            corner_indices = [0]
            border.append([0, h])
            corner_indices.append(len(border))
            border.append([w, h])
            corner_indices.append(len(border))
            border.append([w, 0])
            corner_indices.append(len(border))
            border.append([0, 0])
            corner_indices.append(len(border))
            border.append([0, h])
            corner_indices.append(len(border))
            border = list(self.spaceFromImage(border, D=D))

        origin = self.orientation.spaceFromCamera([0, 0, 0])
        for corner_index in corner_indices:
            border.append([np.nan, np.nan, np.nan])
            border.append(origin)
            border.append(border[corner_index])
        return np.array(border)

    def imageFromSpace(self, points, hide_backpoints=True):
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

        >>> import cameratransform as ct
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
        return self.lens.distortedFromImage(self.projection.imageFromCamera(self.orientation.cameraFromSpace(points), hide_backpoints=hide_backpoints))

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

        >>> import cameratransform as ct
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

        >>> import cameratransform as ct
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
        given = np.array([X, Y, Z], dtype=object)
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

    def getObjectLength(self, point_front, point_back, Z=0):
        """
        Calculate the length of objects in the image, assuming the Z position of the objects is known, e.g. they are
        assumed to lie flat on the Z=0 plane.

        Parameters
        ----------
        point_front : ndarray
            the positions of the front end, dimensions: (2) or (Nx2)
        point_back : ndarray
            the positions of the back end, dimensions: (2) or (Nx2)
        Z : number, ndarray, optional
            the Z position of the objects, dimensions: scalar or (N), default 0

        Returns
        -------
        lengths: ndarray
            the lengths of the objects in meters, dimensions: () or (N)
        """
        # get the front positions in the world
        point3D_front = self.spaceFromImage(point_front, Z=Z)
        # get the back positions in the world
        point3D_back = self.spaceFromImage(point_back, Z=Z)
        # the z difference between these two points
        return np.linalg.norm(point3D_front - point3D_back, axis=-1)

    def _getUndistortMap(self, extent=None, scaling=None):
        # if no extent is given, take the maximum extent from the image border
        if extent is None:
            extent = [0, self.image_width_px, 0, self.image_height_px]

        # if no scaling is given, scale so that the resulting image has an equal amount of pixels as the original image
        if scaling is None:
            scaling = 1

        # if we have cached the map, use the cached map
        if self.map_undistort is not None and \
                self.last_extent_undistort == extent and \
                self.last_scaling_undistort == scaling:
            return self.map_undistort

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
        import cv2

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
            import matplotlib.pyplot as plt
            extent = self.last_extent_undistort.copy()
            extent[2], extent[3] = extent[3]-1, extent[2]-1
            plt.imshow(image, extent=extent, alpha=alpha)
        return image

    def _getMap(self, extent=None, scaling=None, Z=0, hide_backpoints=True):
        # if no extent is given, take the maximum extent from the image border
        if extent is None:
            border = self.getImageBorder()
            extent = [np.nanmin(border[:, 0]), np.nanmax(border[:, 0]),
                      np.nanmin(border[:, 1]), np.nanmax(border[:, 1])]

        # if we have cached the map, use the cached map
        if self.map is not None and \
                all(self.last_extent == np.array(extent)) and \
                (self.last_scaling == scaling):
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
        mesh_points_shape = self.imageFromSpace(mesh_points, hide_backpoints=hide_backpoints)

        # reshape the map and cache it
        self.map = mesh_points_shape.T.reshape(mesh.shape).astype(np.float32)[:, ::-1, :]

        self.last_extent = extent
        self.last_scaling = scaling

        # return the calculated map
        return self.map

    def getTopViewOfImage(self, image, extent=None, scaling=None, do_plot=False, alpha=None, Z=0., skip_size_check=False, hide_backpoints=True):
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
        import cv2

        # check if the size of the image matches the size of the camera
        if not skip_size_check:
            assert image.shape[1] == self.image_width_px, "The with of the image (%d) does not match the image width of the camera (%d)" % (image.shape[1], self.image_width_px)
            assert image.shape[0] == self.image_height_px, "The height of the image (%d) does not match the image height of the camera (%d)." % (image.shape[0], self.image_height_px)
        # get the mapping
        x, y = self._getMap(extent=extent, scaling=scaling, Z=Z, hide_backpoints=hide_backpoints)
        # ensure that the image has an alpha channel (to enable alpha for the points outside the image)
        if len(image.shape) == 2:
            pass
        elif image.shape[2] == 3:
            image = np.dstack((image, np.ones(shape=(image.shape[0], image.shape[1], 1), dtype="uint8") * 255))
        image = cv2.remap(image, x, y,
                          interpolation=cv2.INTER_NEAREST,
                          borderValue=[0, 1, 0, 0])  # , borderMode=cv2.BORDER_TRANSPARENT)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.imshow(image, extent=self.last_extent, alpha=alpha)
        return image

    def generateLUT(self, undef_value=0, whole_image=False):
        """
        Generate LUT to calculate area covered by one pixel in the image dependent on y position in the image

        Parameters
        ----------
        undef_value : number, optional
            what values undefined positions should have, default=0
        whole_image : bool, optional
            whether to generate the look up table for the whole image or just for a y slice

        Returns
        -------
        LUT: ndarray
            same length as image height
        """

        def get_square(points):
            p0 = points + np.array([-0.5, -0.5])
            p1 = points + np.array([+0.5, -0.5])
            p2 = points + np.array([+0.5, +0.5])
            p3 = points + np.array([-0.5, +0.5])
            squares = np.array([p0, p1, p2, p3])

            if len(squares.shape) == 3:
                return squares.transpose(1, 0, 2)
            return squares

        if whole_image:
            x = np.arange(0, self.image_width_px)
            y = np.arange(0, self.image_height_px)
            xv, yv = np.meshgrid(x, y)
            points = np.array([xv.flatten(), yv.flatten()]).T
        else:
            y = np.arange(self.image_height_px)
            x = self.image_width_px / 2 * np.ones(len(y))
            points = np.array([x, y]).T

        squares = get_square(points).reshape(-1, 2)
        squares_space = self.spaceFromImage(squares, Z=0).reshape(-1, 4, 3)
        A = ray.areaOfQuadrilateral(squares_space)

        if whole_image:
            A = A.reshape(self.image_height_px, self.image_width_px)
        A[np.isnan(A)] = undef_value
        return A

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
        export_dict = {key: getattr(self, key) for key in keys if key != "focallength_px"}

        # check projections and save
        if isinstance(self.projection, RectilinearProjection):
            export_dict["projection"] = RECTILINEAR
        elif isinstance(self.projection, CylindricalProjection):
            export_dict["projection"] = CYLINDRICAL
        elif isinstance(self.projection, EquirectangularProjection):
            export_dict["projection"] = EQUIRECTANGULAR

        # check lens distortions and save
        if isinstance(self.lens, NoDistortion):
            export_dict["lens"] = NODISTORTION
        elif isinstance(self.lens, ABCDistortion):
            export_dict["lens"] = ABCDDISTORTION
        elif isinstance(self.lens, BrownLensDistortion):
            export_dict["lens"] = BROWNLENSDISTORTION

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

        if "projection" in variables.keys():
            if variables["projection"] == RECTILINEAR:
                projection = RectilinearProjection()
            elif variables["projection"] == CYLINDRICAL:
                projection = CylindricalProjection()
            elif variables["projection"] == EQUIRECTANGULAR:
                projection = EquirectangularProjection()
            variables.pop("projection")
        else:
            projection = RectilinearProjection()

        if "lens" in variables.keys():
            if variables["lens"] == NODISTORTION:
                lens = NoDistortion()
            elif variables["lens"] == ABCDDISTORTION:
                lens = ABCDistortion()#variables.get("a", None), variables.get("b", None),variables.get("c", None))
            elif variables["lens"] == BROWNLENSDISTORTION:
                lens = BrownLensDistortion()#variables.get("k1", None), variables.get("k2", None),variables.get("k3", None))
            variables.pop("lens")
        else:
            lens = None
        self.__init__(projection=projection, lens=lens, orientation=SpatialOrientation())
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
    cam = Camera(RectilinearProjection(), SpatialOrientation(), NoDistortion())
    cam.load(filename)
    return cam
