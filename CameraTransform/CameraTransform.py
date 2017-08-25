# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numbers
import numpy as np
from scipy.optimize import minimize
import json

try:
    import cv2
    cv2_installed = True
except ImportError:
    cv2_installed = False
try:
    from matplotlib import pyplot as plt
    plt_installed = True
except ImportError:
    plt_installed = False


def formatGPS(lat, lon, format=None, asLatex=False):
    """ Formats a latiture, longitude pair in degress according to the format string.
        The format string can contain a %s, to denote the letter symbol (N, S, W, E) and up to three number formaters (%d or %f), to
        denote the degrees, minutes and seconds. To not lose precision, the last one can be float number.
        
        common formats are e.g.:
        
           +--------------------------------+--------------------------------------+
           | format                         | output                               |
           +================================+===================+==================+
           | %2d° %2d' %6.3f" %s (default)  | 70° 37'  4.980" S | 8°  9' 26.280" W |
           +--------------------------------+-------------------+------------------+
           | %2d° %2d.3f %s                 | 70° 37.083 S      | 8°  9.438 W      |
           +--------------------------------+-------------------+------------------+
           | %2d°                           | -70.618050°       | -8.157300°       |
           +--------------------------------+-------------------+------------------+
            
        :param number lat: the latitude in degrees
        :param number lon: the longitude in degrees
        :param string format: the format string
        :param bool asLatex: whether to encode the degree symbol
        :return: a tuple of the formatted values
    """
    import re
    # default format
    if format is None:
        format = "%2d° %2d' %6.3f\" %s"
    # try to split the format string into it's place holders
    match = re.findall(r"(%[.\d]*[sdf])", format)
    if len(match) == 0:
        raise ValueError("no valid format place holder specified")

    # try to find a %s to see if we have to use a letter symbol or a negative sign
    use_letter = False
    counter = 0
    for entry in match:
        if entry[-1] == "s":
            use_letter = True
        else:
            counter += 1
    if counter > 3:
        raise ValueError("too many format strings, only 3 numbers are allowed")

    result = []
    for degs, letters in zip([lat, lon], ["NS", "EW"]):
        # split sign
        neg = degs < 0
        degs = abs(degs)
        # get minutes
        mins = (degs * 60) % 60
        # get seconds
        secs = (mins * 60) % 60

        # if no letter symbol is used, keep the sign
        if not use_letter and neg:
            degs = -degs

        # array of values and empty array of fills
        values = [degs, mins, secs]
        fills = []
        # gather the values to fill
        for entry in match:
            # the letter symbol
            if entry[-1] == "s":
                fills.append(letters[neg])
            # one of the values
            else:
                fills.append(values.pop(0))
        # format the string
        string = format % tuple(fills)
        # replace, if desired the degree sign
        if asLatex:
            string = string.replace("°", "\N{DEGREE SIGN}")
        # append to the results
        result.append(string)

    # return the results
    return result


def _getSensorFromDatabase(model):
    """
    Get the sensor size from the given model from the database at: https://github.com/openMVG/CameraSensorSizeDatabase

    :param string model: the model name as received from the exif data
    :return: a tuple (sensor_width, sensor_height) or None
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

    :param string filename: the filename of the image to load. 
    :param bool verbose: whether to print the output.
    :param bool sensor_from_database: whether to try to load the sensor size from a database at https://github.com/openMVG/CameraSensorSizeDatabase
    :return: focal_length, sensor_size, image_size
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


def MapTransform(image_size, scale=1, rotation=0, offset=None):
    """
    Create a top view :py:class:`~.CameraTransform`  object with the provided affine transformations.
    
    :param image_size: a tuple (im_width, im_height) or a numpy array representing the image
    :type image_size: tuple, array
    :param number scale:  the scale to apply to the image, e.g. how many pixels are one meter
    :param number rotation: a rotation of the image in degrees
    :param tuple offset: a tuple (x, y) as an offset for the map  
    :return: a :py:class:`~.CameraTransform` object with the transformation
    """
    try:
        shape, im = image_size.shape, image_size
        image_size = (shape[1], shape[0])
    except AttributeError:
        im = None

    cam = CameraTransform(focal_length=1, sensor_size=image_size,
                           image_size=image_size)
    cam.height = scale
    cam.tilt = 0
    cam.roll = rotation
    cam.pos_x = image_size[0] / 2 * scale
    cam.pos_y = -image_size[1] / 2 * scale
    if offset is not None:
        cam.pos_x += offset[0]
        cam.pos_y += offset[1]
    cam._initCameraMatrix()
    return cam


def LoadTransform(filename):
    """
    Create a :py:class:`~.CameraTransform` object from the parameters stored in a file.
    
    :param string filename: the filename to load.
    :return: a :py:class:`~.CameraTransform` object with the parameters from the file.
    """
    # initialize empty camera
    cam = CameraTransform()
    # load the parameters from file
    cam.load(filename)
    # return the camera
    return cam


class CameraTransform:
    """
    CameraTransform class to calculate the position of objects from an image in 3D based
    on camera intrinsic parameters and observer position
    
    :param number focal_length:    focal length of the camera in mm
    :param sensor_size:     sensor size in mm, can be either a tuple (width, height) or just a the width, then the height
                            is inferred from the aspect ratio of the image                        
    :type sensor_size: tuple or number
    :param image_size:      image size in mm [width, height] or a numpy array representing the image
    :type image_size: tuple or ndarray
    :param float observer_height: observer elevation in m
    :param float angel_to_horizon: angle between the z-axis and the horizon
    """
    t = None
    R = None
    C = None

    R_earth = 6371e3

    lat = None
    lon = None
    gps_heading = None

    cam_location = None
    cam_heading = None
    cam_heading_rotation_matrix = None

    height = None
    roll = None
    heading = 0
    tilt = None
    tan_tilt = None

    pos_x = 0
    pos_y = 0

    fixed_horizon = None

    estimated_height = 30
    estimated_tilt = 85
    estimated_heading = 0
    estimated_roll = 0

    f = None
    sensor_width = None
    sensor_height = None
    fov_h_angle = None
    fov_v_angle = None
    im_width = None
    im_height = None

    use_fit_bounds = None

    def __init__(self, focal_length=None, sensor_size=None, image_size=None, observer_height=None,
                 angel_to_horizon=None):

        # store and convert arguments
        if focal_length:
            # convert focal length to meters
            self.f = focal_length * 1e-3  # in m
            # splice image size
            try:
                shape, im = image_size.shape, image_size
                image_size = (shape[1], shape[0])
            except AttributeError:
                im = None
            self.im_width, self.im_height = image_size
            # if only the sensor width is given, calculate its height from the aspect ratio of the image
            if isinstance(sensor_size, numbers.Number):
                sensor_size = (sensor_size, sensor_size * self.im_height / self.im_width)
            # split sensor size
            self.sensor_width, self.sensor_height = np.array(sensor_size) * 1e-3  # in m
            # and calculate field of view (fov) for both dimensions
            self.fov_h_angle = 2 * np.arctan(self.sensor_width / (2 * self.f))
            self.fov_v_angle = 2 * np.arctan(self.sensor_height / (2 * self.f))

            self._initIntrinsicMatrix()

        if observer_height is not None:
            self.height = observer_height
            self.tilt = angel_to_horizon
            self._initCameraMatrix()

    def _initIntrinsicMatrix(self):
        # normalize the focal length by the sensor width and the image_width
        self.f_normed = self.f / self.sensor_width * self.im_width
        # compose the intrinsic camera matrix
        self.C1 = np.array([[self.f_normed,             0,  self.im_width / 2, 0],
                            [            0, self.f_normed, self.im_height / 2, 0],
                            [            0,             0,                  1, 0]])

    def _initCameraMatrix(self, height=None, tilt_angle=None, roll_angle=None):
        # convert the angle to radians
        if tilt_angle is None:
            if self.tilt is not None:
                tilt_angle = self.tilt
            else:
                self.tilt = np.arctan(self.tan_tilt) * 180 / np.pi + 90
                tilt_angle = self.tilt
        else:
            self.tilt = tilt_angle
        if height is None:
            height = self.height
        else:
            self.height = height
        angle = tilt_angle * np.pi / 180
        if roll_angle is None:
            if self.roll:
                roll = self.roll * np.pi / 180
            else:
                roll = 0
        else:
            roll = roll_angle * np.pi / 180

        if self.heading:
            heading = self.heading * np.pi / 180
        else:
            heading = 0

        # get the translation matrix and rotate it
        self.t = np.array([[self.pos_x, self.pos_y, -height]]).T

        # construct the rotation matrices for tilt, roll and heading
        self.R_tilt = np.array([[1, 0, 0],
                                [0, np.cos(angle), np.sin(angle)],
                                [0, -np.sin(angle), np.cos(angle)]])
        self.R_roll = np.array([[+np.cos(roll), np.sin(roll), 0],
                                [-np.sin(roll), np.cos(roll), 0],
                                [0, 0, 1]])
        self.R_head = np.array([[+np.cos(heading), np.sin(heading), 0],
                                [-np.sin(heading), np.cos(heading), 0],
                                [0, 0, 1]])

        # rotate the translation around the tilt angle
        self.t = np.dot(self.R_tilt, self.t)

        # get the rotation-translation matrix with the rotation composed with the translation
        self.R = np.vstack((np.hstack((np.dot(np.dot(self.R_roll, self.R_tilt), self.R_head), self.t)), [0, 0, 0, 1]))

        # compose the camera matrix with the rotation-translation matrix
        self.C = np.dot(self.C1, self.R)
        # to get the x coordinate right, mirror the x direction
        self.C[:, 0] = -self.C[:, 0]

    def _ensurePointFormat(self, x, dimensions=2):
        # also accept clickpoints markers
        try:
            x = np.array([[m.x, m.y] for m in x]).T
        except (AttributeError, TypeError):
            pass

        # make a copy so that we don't change the original data
        x = x.copy()

        # reshape input x array to two dimensions
        try:
            if len(x.shape) == 1:
                x = x[:, None]
        except AttributeError:
            x = np.array([x]).T

        # fill third dimension with zeros if none is present
        if dimensions == 3 and x.shape[0] == 2:
            x = np.vstack((x, np.zeros(x.shape[1], dtype=x.dtype)))

        # test if the first dimension of the array is long enough
        assert x.shape[0] == dimensions, "Input array has to be of the shape [%dxN] or [%d]" % (dimensions, dimensions)

        # return the process array
        return x

    def transWorldToCam(self, x):
        """
        Transform from 3D world coordinates to 2D camera coordinates.

        :param ndarray x: a point in world coordinates [x, y, z], or an array of world points in the shape of [3xN]
        :return: a list of projected points
        """

        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=3)

        # add a 1 as a 3rd dimension for the projective coordinates
        x = np.vstack((x, np.ones(x.shape[1])))

        # multiply it with the camera matrix
        X = np.dot(self.C, x)
        # rescale it so the lowest component is again 1
        X = X[:2] / X[2]
        return X

    def _transCamToWorldFixedDimension(self, x, fixed, dimension):
        # add a 1 as a 3rd dimension for the projective coordinates
        x = np.vstack((x, np.ones(x.shape[1])))

        # make a copy of the projection matrix
        P = self.C.copy()
        # create a reduced matrix, that collapses two columns. Namely the desired fixed dimension with the
        # projective dimension e.g. if z is fixed
        # ( a b c d )   ( x*s )   ( a b c*z+d )   ( x*s )
        # ( e f g h ) * ( y*s ) = ( e f g*z+h ) * ( y*s )
        # ( i j k l )   ( z*s )   ( i j k*z+h )   (  s  )
        #               (  s  )
        P[:, dimension] = P[:, dimension] * fixed + P[:, 3]
        P = P[:, :3]
        # this results then in a 3x3 matrix corresponding to a system of linear equations
        X = np.linalg.solve(P, x)

        # the new vector has then to be rescaled from projective coordinates to normal coordinates
        # scaling by the value of the projective dimension entry
        X = X / X[dimension]
        # and adding the fixed value
        # (as we had used the entry of the fixed dimension for the projective entry, we can use this entry and overwrite
        #  it with the desired fixed value)
        X[dimension] = fixed

        return X

    def transCamToWorld(self, x, X=None, Y=None, Z=None):
        """
        Transform from 2D camera coordinates to 3D world coordinates. One of the 3D values has to be fixed. 
        This can be specified by supplying one of the three X, Y, Z.
        
        :param x: a point in camera coordinates [x, y], or an array of camera points in the shape of [2xN]
        :param X: when given project the camera points to world coordinates with their X value set to this parameter. 
                  Can be a single value or a list.
        :param Y: when given project the camera points to world coordinates with their Y value set to this parameter. 
                  Can be a single value or a list.
        :param Z: when given project the camera points to world coordinates with their Z value set to this parameter.
                  Can be a single value or a list.
        :return: a list of projected points
        """

        # test whether the input is good
        if (X is None) + (Y is None) + (Z is None) != 2:
            raise ValueError("Exactly one of X, Y, Z has to be given.")

        # process the input
        if X is not None:
            fixed = X
            dimension = 0
        elif Y is not None:
            fixed = Y
            dimension = 1
        else:
            fixed = Z
            dimension = 2

        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=2)

        # if the fixed value is a list, we have to transform each coordinate separately
        if not isinstance(fixed, int) and not isinstance(fixed, float):
            return np.array(
                [self._transCamToWorldFixedDimension(x[:, i:i + 1], fixed=fixed[i], dimension=dimension) for i in
                 range(x.shape[1])])[:, :, 0].T
        # else transform everything in one go
        return self._transCamToWorldFixedDimension(x, fixed, dimension)

    def transCamToEarth(self, x, H=None, max_iter=100, max_distance=0.01):
        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=2)

        if H is None:
            H = 0
        r = self.R_earth + H
        result = []
        new_point = None
        alpha = 0
        for index in range(x.shape[1]):
            point = x[:, index]
            last_point = None
            next_z = H
            for i in range(max_iter):
                new_point = self._transCamToWorldFixedDimension(point[:, None], fixed=next_z, dimension=2)[:, 0]
                alpha = np.arctan(new_point[1] / r)
                if last_point is not None and np.linalg.norm(new_point - last_point) < max_distance:
                    result.append([new_point[0], r * alpha, H])
                    break
                next_z = -(r / np.cos(alpha) - r)
                last_point = new_point
            else:
                result.append([new_point[0], r * alpha, H])
        return np.array(result).T

    def transEarthToCam(self, x):
        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=3)

        return self.transWorldToCam(self.transEarthToWorld(x))

    def transWorldToEarth(self, x):
        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=3)

        x = x.copy()
        earth_center = np.array([0, 0, -self.R_earth])
        r_eff = np.linalg.norm(x - earth_center, axis=0)
        x[1] = np.acos(x[1] / r_eff) * r_eff
        x[2] = r_eff
        return x

    def transEarthToWorld(self, x):
        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=3)

        x = x.copy()
        radius = x[2] + self.R_earth
        alpha = x[1] / radius
        x[1] = np.tan(alpha) * radius
        x[2] = radius - radius * np.cos(alpha)
        return x

    def transGPSToEarth(self, x):
        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=3)

        x = x.copy()
        # latitude, longitude, height
        diff = np.array(self.cam_location - x[:2])
        diff = np.dot(self.cam_heading_rotation_matrix, diff)
        x[:2] = diff * np.pi / 180 * self.R_earth
        return x

    def transGPSToWorld(self, x):
        """
        Transform from (lat, lon, height) coordinates to 3D world coordinates.

        :param x: a point in gps coordinates [lat, lon, height], or an array of gps points in the shape of [3xN]
        :return: a list of projected points
        """
        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=3)

        lat, lon, h = x
        phi2 = lat * np.pi / 180
        phi1 = self.lat * np.pi / 180
        delta_lambda = (self.lon - lon) * np.pi / 180
        # spherical law of cosines
        distance = np.arccos(
            np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda)) * self.R_earth
        bearing = np.arctan2(np.sin(delta_lambda) * np.cos(phi2),
                             np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda))
        bearing = bearing + self.cam_heading
        # distance and height correction
        #r = self.R_earth + h
        #alpha = distance / r
        #d_world = r * np.sin(alpha)
        #z_world = r * np.cos(alpha) - self.R_earth
        #print("distances", distance, d_world)
        #
        x = distance * np.sin(bearing)
        y = distance * np.cos(bearing)
        return np.array([x, y, h])

    def transGPSToCam(self, x):
        """
        Transform 2D camera coordinates to (lat, lon, height) coordinates to 2D camera coordinates.

        :param x: a point in gps coordinates [lat, lon, height], or an array of gps points in the shape of [3xN]
        :return: a list of projected points
        """
        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=3)

        x = self.transGPSToWorld(x)
        return self.transWorldToCam(x)

    def transWorldToGPS(self, x):
        """
        Transform 3D world coordinates to (lat, lon, height) coordinates.

        :param x: a point in world coordinates [x, y, z], or an array of world points in the shape of [3xN]
        :return: a list of projected points
        """
        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=3)

        h = x[2]
        dist, bearing = self.distanceBearing(x)
        # correction for round earth
        #alpha = np.arctan(dist/(self.R_earth + z))
        #r = dist/np.sin(alpha)
        #dist_earth = r * alpha
        #h = r-self.R_earth
        #print("Dist_erath", dist, dist_earth)
        #print("Z", z, h)
        #
        lat, lon = self.moveGPS(self.cam_location[0], self.cam_location[1], dist,
                                self.cam_heading * 180 / np.pi + bearing)
        return np.array([lat, lon, h])

    def transCamToGPS(self, x, H=0):
        """
        Transform 2D camera coordinates to (lat, lon, height) coordinates.

        :param x: a point in camera coordinates [x, y], or an array of camera points in the shape of [2xN]
        :param H: specify the height the gps points should have. (default=0)
        :return: a list of projected points
        """
        # reshape input x array to two dimensions
        x = self._ensurePointFormat(x, dimensions=2)

        x = self.transCamToWorld(x, Z=H)
        return self.transWorldToGPS(x)

    def distanceBearing(self, pos):
        """ distance and bearing from a point to the camera 
        
        :param array pos: point(s) in the world coordinates. In the shape of [3xN]
        :return: distance (meters), bearing (degree)
        """
        # relative to camera foot point
        x = pos[0] - self.pos_x
        y = pos[1] - self.pos_y
        # distance according to Pythagoras
        dist = np.sqrt(x ** 2 + y ** 2)
        # arcustangens for the angle
        bearing = -np.arctan(x / y) * 180 / np.pi
        return dist, bearing

    def moveGPS(self, lat, lon, distance, bearing):
        """
        Move a gps point the given distance in the given direction.

        :param number lat: in degree
        :param number lon: in degree
        :param number distance: in meter
        :param number bearing: in degree
        :return: the tuple of lat and lon
        :rtype: tuple
        """

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        bearing_rad = np.deg2rad(bearing)

        lat2 = np.arcsin(
            np.sin(lat_rad) * np.cos(distance / self.R_earth) +
            np.cos(lat_rad) * np.sin(distance / self.R_earth) * np.cos(bearing_rad))
        lon2 = lon_rad + np.arctan2(np.sin(bearing_rad) * np.sin(distance / self.R_earth) * np.cos(lat_rad),
                                    np.cos(distance / self.R_earth) - np.sin(lat_rad) * np.sin(lat2))

        return np.rad2deg(lat2), np.rad2deg(lon2)

    def setCamHeading(self, angle):
        angle = angle * np.pi / 180
        self.cam_heading = angle
        self.cam_heading_rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                                     [-np.sin(angle), np.cos(angle)]])

    def setCamGPS(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.cam_location = np.array([lat, lon])

    def generateLUT(self, undef_value=0):
        """
        Generate LUT to calculate area covered by one pixel in the image dependent on y position in the image

        :return: LUT, same length as image height
        """

        def get_square(x, y):
            p0 = [x - 0.5, y - 0.5, 0]
            p1 = [x + 0.5, y - 0.5, 0]
            p2 = [x + 0.5, y + 0.5, 0]
            p3 = [x - 0.5, y + 0.5, 0]
            return np.array([p0, p1, p2, p3]).T

        def calc_quadrilateral_size(rect):
            A, B, C, D = rect.T
            return 0.5 * abs((A[1] - C[1]) * (D[0] - B[0]) + (B[1] - D[1]) * (A[0] - C[0]))

        horizon = self.getImageHorizon()
        print("horizon", horizon[1])
        y_stop = max([0, int(horizon[1][1])])
        y_start = self.im_height
        print(y_start, y_stop)

        y_lookup = np.zeros(self.im_height) + undef_value

        x = self.im_width / 2

        for y in range(y_stop, y_start):
            rect = get_square(x, y)[:2, :]
            rect = self.transCamToWorld(rect, Z=0)
            A = calc_quadrilateral_size(rect)
            y_lookup[y] = A

        return y_lookup

    def fitCamParametersFromObjects(self, points_foot=None, points_head=None, lines=None, object_height=1,
                                    object_elevation=0):
        """
        Fit the camera parameters for given objects of equal heights. The foot positions are given in points_foot and 
        the heads are given in points_head. As an alternative the positions can be given as ClickPoints line objects in 
        lines. The height of each objects is given in object_height, and if the objects are not at sea level, an 
        object_elevation can be given.
        
        :param points_foot: The pixel positions of the feet of the objects in the image. 
        :param points_head:  The pixel positions of the heads of the objects in the image. 
        :param lines: An alternative for the points_foot and points_head arguments, ClickPoints lines can be directly 
                      given.
        :param object_height: The height of the objects.
        :param object_elevation: The elevation of the feet ot the objects.
        :return: the fitted parameters.
        """
        if lines is not None:
            y1 = [np.max([l.y1, l.y2]) for l in lines]
            y2 = [np.min([l.y1, l.y2]) for l in lines]
            x = [np.mean([l.x1, l.x2]) for l in lines]
            points_foot = np.vstack((x, y1))
            points_head = np.vstack((x, y2))

        def cost():
            estimated_foot_3D = self.transCamToWorld(points_foot.copy(), Z=object_elevation)
            estimated_foot_3D[2, :] = object_elevation + object_height
            estimated_head = self.transWorldToCam(estimated_foot_3D)
            pixels = np.linalg.norm(points_head - estimated_head, axis=0)
            return np.mean(pixels ** 2)

        return self._fit(cost)

    def _getAngleFromHorizonAndHeight(self, horizon=None, height=None):
        if horizon is None:
            horizon = self.fixed_horizon
        if height is None:
            height = self.height
        angle = np.arccos(height / np.sqrt(height ** 2 + 2 * height * self.R_earth))
        angle = angle + (horizon - self.im_height / 2) / self.im_height * self.fov_v_angle
        return angle * 180 / np.pi

    def fixRoll(self, roll):
        """
        Set the roll parameter of the camera to a given value and hold it there in subsequent fitting functions.
        
        :param roll: The roll of the camera in degree. 
        """
        self.roll = roll

    def fixHeight(self, height):
        """
        Set the height parameter of the camera to a given value and hold it there in subsequent fitting functions.

        :param height: The height of the camera in meters.
        """
        self.height = height
        if self.fixed_horizon:
            self._fit(lambda: 0)

    def fixHorizon(self, horizon):
        """
        Fix the horizon to go through the points given. This will adjust in subsequent fitting functions the tilt angle
        to always match the horizon with these points. Also if no roll angle has been specified before, the roll angle 
        is fitted from the horizon.
        
        :param horizon: Pixel coordinates of points at the horizon in the shape of [2xN] 
        """
        # reshape input x array to two dimensions
        horizon = self._ensurePointFormat(horizon, dimensions=2)
        # fit a line through the points
        m, t = np.polyfit(horizon[0, :], horizon[1, :], deg=1)
        # calculate the center of the line
        self.fixed_horizon = self.im_width / 2 * m + t
        # set the roll if it is not fixed yet
        if self.roll is None:
            self.roll = -np.arctan(m) * 180 / np.pi
        # update the camera matrix if we already have a height
        if self.height is not None:
            self.tilt = self._getAngleFromHorizonAndHeight(self.im_width / 2 * m + t, self.height)
            self._initCameraMatrix()

    def fitCamParametersFromLandmarks(self, marks, distances, heading=None):
        """
        Fit the camera parameters form objects of known distance to the camera.
        
        :param marks: The pixel positions of the objects in the image. In the shape of [2xN] 
        :param distances: The distances of the mark points to the camera.
        :param heading: Optional a heading angle in degrees of the objects. When given the heading of the camera will be
                        fitted, too.
        :return: the fitted parameters.
        """

        def cost():
            estimated_pos_3D = self.transCamToWorld(marks, Z=0)
            return np.mean((distances - estimated_pos_3D[1, :]) ** 2)

        if heading is not None:
            self.heading = None
            marks_3D = []
            for dist, head in zip(distances, heading):
                marks_3D.append(np.array([np.sin(head * np.pi / 180) * dist, np.cos(head * np.pi / 180) * dist, 0]))
            marks_3D = np.array(marks_3D).T

            def cost():
                estimated_pos_3D = self.transCamToWorld(marks.copy(), Z=0)
                return np.mean(np.linalg.norm(estimated_pos_3D - marks_3D, axis=0) ** 2)

        return self._fit(cost)

    def fitCamParametersFromPointCorrespondences(self, points2D, points3D):
        """
        Fit the camera parameters form points known in both coordinate systems, the camera and the world.

        :param points2D: The points in the camera image.
        :param points3D: The corresponding points in the world coordinates. This list has to have the same order than 
                         the points 2D list.
        :return: the fitted parameters.
        """
        # if the points are given in ClickPoints markers, split them in x and y component
        try:
            points2D = np.array([[m.x, m.y] for m in points2D]).T
        except AttributeError:
            pass
        try:
            points3D = np.array([[-m.x, m.y, 0] for m in points3D]).T
        except AttributeError:
            pass

        # fit the position of the camera, too
        self.pos_x = None
        self.pos_y = None
        self.heading = None

        # define a cost function
        def cost():
            # project the 3D points to the camera
            points3D_proj = self.transWorldToCam(points3D)
            # and calculate the distances to their corresponding points in the camera image
            return np.mean(np.linalg.norm(points3D_proj - points2D, axis=0))

        # fit the camera matrix to the cost function
        return self._fit(cost)

    def fitCamParametersFromLengths(self, points, distances):
        """
        Fit the camera parameters form objects of known distance to the camera.

        :param points: TODO 
        :param distances: The distances of the mark points to the camera.
        :return: the fitted parameters.
        """
        # if the horizon is given in ClickPoints markers, split them in x and y component
        try:
            points1 = np.array([[m.x1, m.y1] for m in points]).T
            points2 = np.array([[m.x2, m.y2] for m in points]).T
        except AttributeError:
            points1, points2 = points

        def cost():
            p1 = self.transCamToWorld(points1, Z=0)
            p2 = self.transCamToWorld(points2, Z=0)
            calculated_dist = np.linalg.norm(p2 - p1, axis=0)
            return np.mean((distances - calculated_dist) ** 2)

        return self._fit(cost)

    do_grid = False

    def _fit(self, cost):
        # define the fit parameters and their estimates
        #estimates = {"height": self.estimated_height, "tan_tilt": np.tan((90 + self.estimated_tilt) * np.pi / 180),
        #             "roll": self.estimated_roll, "heading": self.estimated_heading, "pos_x": 0, "pos_y": 0}
        estimates = {"height": self.estimated_height, "tilt": self.estimated_tilt, "roll": self.estimated_roll,
                     "heading": self.estimated_heading, "pos_x": 0, "pos_y": 0}
        bounds = {"height": (1e-6, None), "tilt": (0, 180)}

        fit_parameters = list(estimates.keys())

        # remove known parameters from list
        if self.roll is not None:
            fit_parameters.remove("roll")
        if self.heading is not None:
            fit_parameters.remove("heading")
        if self.height is not None:
            fit_parameters.remove('height')
        if self.pos_x is not None:
            fit_parameters.remove("pos_x")
        if self.pos_y is not None:
            fit_parameters.remove("pos_y")

        # if use_fit_bounds is undefined, use bounds only for low dimension fits
        if self.use_fit_bounds is None:
            fit_bounds = len(fit_parameters) <= 2
        else:
            fit_bounds = self.use_fit_bounds
        # if we want to fit with bounds prepare the bounds list
        if fit_bounds:
            bounds.update({key: (None, None) for key in fit_parameters if key not in bounds})
            bounds = [bounds[key] for key in fit_parameters]
        # if not, set it to None
        else:
            bounds = None

        self.horizon_error = 0

        # define error function as a wrap around the cost function
        def error(p):
            # set the fit parameters
            for key, value in zip(fit_parameters, p):
                setattr(self, key, value)
            # calculate the camera matrix
            self._initCameraMatrix()

            if self.fixed_horizon:
                horizon = self.getImageHorizon()
                if np.isnan(horizon[1, :]).any():
                    self.horizon_error = 99999999
                else:
                    m, t = np.polyfit(horizon[0, :], horizon[1, :], deg=1)
                    # calculate the center of the line
                    fixed_horizon2 = self.im_width / 2 * m + t
                    self.horizon_error = abs(self.fixed_horizon - fixed_horizon2) * 0.01

            # calculate the cost function
            return cost() + self.horizon_error

        # minimize the unknown parameters with the given cost function
        p = minimize(error, [estimates[key] for key in fit_parameters], bounds=bounds)
        # call a last time the error function to ensure that the camera matrix has been set properly
        error(p["x"])
        # print the results and return them
        print({key: value for key, value in zip(fit_parameters, p["x"])})
        if "tan_tilt" in fit_parameters:
            print("tilt", self.tilt)
        # display a parameter grid, if desired
        if self.do_grid:
            self.do_grid = False
            self._grid(cost)
        return p

    def _grid(self, cost):
        fx = 1
        fy = 1
        height = self.height
        tilt = self.tilt

        from matplotlib import pyplot as plt
        rangeH = np.arange(10, 200, 1)
        rangeA = np.arange(0, 90, 1)
        results = np.zeros((len(rangeH), len(rangeA)))
        for i, h in enumerate(rangeH):
            for j, a in enumerate(rangeA):
                self.height = h
                self.tilt = a
                self._initCameraMatrix()
                c = cost() ** 0.01
                results[i, j] = c

        # plt.figure(4)
        plt.imshow(results.T[::-1, :],
                   extent=[rangeH[0] * fx, rangeH[-1] * fx, rangeA[0] * fy, rangeA[-1] * fy])
        plt.yticks(np.array([0, 15, 30, 45, 60, 75, 89]) * fy, [0, 15, 30, 45, 60, 75, 90])
        plt.plot(height * fx, tilt * fy, 'r+')
        plt.xlabel("Height (m)")
        plt.ylabel("Tilt angle (deg)")
        cb = plt.colorbar()
        cb.set_label("Cost (a.u.)")
        plt.tight_layout()
        plt.savefig("Grid.png")
        plt.show()

        self.height = height
        self.tilt = tilt

    def distanceToHorizon(self):
        return np.sqrt(2 * self.R_earth ** 2 * (1 - self.R_earth / (self.R_earth + self.height)))

    def getImageHorizon(self):
        """
        This function calculates the position of the horizon in the image sampled at the points x=0, x=im_width/2, 
        x=im_width.
        
        :return: The points im camera image coordinates of the horizon in the format of [2xN]. 
        """
        # calculate the distance to the horizon and make a copy of the camera matrix
        distance = self.distanceToHorizon()
        P = self.C.copy()
        # compose a mixed transformation, where we fix 3D_Y to distance, 3D_Z to 0
        P[:, 1] = P[:, 1] * distance + P[:, 2] * 0 + P[:, 3]
        # and bring 2D_Y to the other side to search for it, too
        P[:, 2] = [0, -1, 0]
        # to the unknown values are 3D_X, 2D_Y and 3D_Scale
        P = P[:, :3]
        # this means vectors in the left side of the equation have the shape of [2D_X, 0, 1]
        x = np.array([[0, 0, 1], [self.im_width / 2, 0, 1], [self.im_width, 0, 1]]).T
        # solve
        X = np.linalg.solve(P, x)
        # enter the found 2D_Y values into the vector
        x[1, :] = X[2, :]
        x = x[:2, :]
        # return the results
        return x

    def getImageExtend(self):
        # get the horizon
        horizon = self.getImageHorizon()
        # test if it is in the image
        if ((0 < horizon[1, :])*(horizon[1, :] < self.im_height)).any():
            # scale it a bit down to only see meaningful image data (pixels near the horizon are sstretchedvery big)
            horizon[1, :] = self.im_height-(self.im_height-horizon[1, :])*0.95
            # add the lower edge of the image
            points = np.array([[0, 0], [self.im_width, 0]]).T
            # stack the points together
            points = np.hstack((horizon, points))
            # and project the points to the world
            points = self.transCamToWorld(points, Z=0)
            return points
        else:
            # get the corners of the image
            points = np.array([[0, 0], [0, self.im_height], [self.im_width, self.im_height], [self.im_width, 0]]).T
            # and project them to the world
            points = self.transCamToWorld(points, Z=0)
            return points

    def getTopViewOfImage(self, im, extent=None, scaling=None, do_plot=False, border_value=0):
        """
        Transform the given image of the camera to a top view, e.g. project it on the 3D plane and display a birds view.
        
        :param im: The image of the camera. 
        :param extent: The part of the 3D plane to show: [x_min, x_max, y_min, y_max]. The same format as the extent 
                       parameter in in plt.imshow.
        :param scaling: How many pixels to use per meter. A smaller value gives a more detailed image, but takes more 
                        time to calculate.
        :param do_plot: Whether to plot the image directly, with the according extent settings.
        :return: the transformed image
        """
        # check the dependencies
        if not cv2_installed:
            raise ImportError("package cv2 has to be installed to use this function.")
        if do_plot and not plt_installed:
            raise ImportError("package matplotlib has to be installed to be able to plot the image.")

        # guess the extend for the image
        if extent is None:
            x, y, z = self.getImageExtend()
            extent = [min(x), max(x), min(y), max(y)]
            print("extent", extent)

        # check if we want a transparent background
        if border_value == "transparent":
            border_value = 0
            # if we have an RGB image, add an alpha channel
            if im.shape[2] == 3:
                im = np.dstack((im, np.ones(im.shape[:2], dtype=im.dtype) * 255))

        # split the extent
        x_lim, y_lim = extent[:2], extent[2:]
        width = x_lim[1] - x_lim[0]
        distance = y_lim[1] - y_lim[0]
        # if no scaling is given, scale so that the resulting image has an equal amount of pixels as the original image
        if scaling is None:
            scaling = np.sqrt((width * distance)) / np.sqrt((self.im_width * self.im_height))
        # copy the camera matrix
        P = self.C.copy()
        # set scaling and offset
        f = scaling
        x_off = x_lim[0]
        y_off = y_lim[0]
        # offset and scale the camera matrix
        P = np.dot(P, np.array([[f, 0, 0, x_off], [0, f, 0, y_off], [0, 0, f, 0], [0, 0, 0, 1]]))
        # transform the camera matrix so that it projects on the z=0 plane (for details see transCamToWorld)
        P[:, 2] = P[:, 2] * 0 + P[:, 3]
        P = P[:, :3]
        # invert the matrix
        P = np.linalg.inv(P)
        # transform the image using OpenCV
        im = cv2.warpPerspective(im, P, dsize=(int(width / f), int(distance / f)), borderValue=border_value, borderMode=cv2.BORDER_TRANSPARENT)[::-1, :]
        # and plot the image if desired
        if do_plot:
            plt.imshow(im, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
        # return the image
        return im

    def save(self, filename):
        """
        Save the camera parameters to a file that can be loaded with :py:meth:`~.CameraTransform.load`. 
        
        :param filename: The filename where to save the parameters 
        """
        keys = ["height", "roll", "heading", "tilt", "pos_x", "pos_y",
                "f", "sensor_width", "sensor_height", "fov_h_angle", "fov_v_angle", "im_width", "im_height"]
        export_dict = {key: getattr(self, key) for key in keys}
        with open(filename, "w") as fp:
            fp.write(json.dumps(export_dict))

    def load(self, filename):
        """
        Load the camera parameters from a file that was saved before with :py:meth:`~.CameraTransform.save`. 

         filename: The filename from which to load the parameters 
        """
        with open(filename, "r") as fp:
            variables = json.loads(fp.read())
        for key in variables:
            setattr(self, key, variables[key])
        self._initIntrinsicMatrix()
        self._initCameraMatrix()

    def plotPointCorrespondence(self, im_cam, im_map, cam_map, points_cam, points_map):
        points_cam = self._ensurePointFormat(points_cam, dimensions=2)
        points_map = self._ensurePointFormat(points_map, dimensions=3)

        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133, sharex=ax2, sharey=ax2)

        # plot the camera image points
        for i, p in enumerate(points_cam.T):
            # into the camera image
            plt.sca(ax1)
            plt.plot(p[0], p[1], 'bo')
            plt.text(p[0], p[1], i)
            # and the map image
            plt.sca(ax2)
            p = self.transCamToWorld(p, Z=0)
            plt.plot(p[0], p[1], 'bo')
            plt.text(p[0], p[1], i)
        # plot the map points
        for i, p0 in enumerate(points_map.T):
            # into the camera image
            plt.sca(ax1)
            p = self.transWorldToCam([p0[0], p0[1], 0])
            plt.plot(p[0], p[1], 'r+')
            plt.text(p[0], p[1], i)
            # and into the map image
            plt.sca(ax2)
            plt.plot(p0[0], p0[1], 'r+')
            plt.text(p0[0], p0[1], i)
            # and into the map image
            plt.sca(ax3)
            plt.plot(p0[0], p0[1], 'r+')
            plt.text(p0[0], p0[1], i)

        # plot the camera image
        plt.sca(ax1)
        plt.imshow(im_cam)

        # and the map image
        plt.sca(ax2)
        cam_map.getTopViewOfImage(im_map, do_plot=True)
        self.getTopViewOfImage(im_cam, do_plot=True, border_value="transparent")

        # and the map image
        plt.sca(ax3)
        cam_map.getTopViewOfImage(im_map, do_plot=True)
