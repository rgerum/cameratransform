import numpy as np
from scipy.optimize import fmin


""" some helper functions """

def fillNans(im2):
    valid_mask = ~np.isnan(im2)
    coords = np.arange(len(im2))[valid_mask]
    values = im2[valid_mask]

    return np.interp(np.arange(len(im2)), coords, values)

def getSquare(x, y):
    p0 = [x-0.5, y-0.5, 0]
    p1 = [x+0.5, y-0.5, 0]
    p2 = [x+0.5, y+0.5, 0]
    p3 = [x-0.5, y+0.5, 0]
    return np.array([p0, p1, p2, p3]).T

def getRectangle(w, d):
    p0 = [-w, d-w, 0]
    p1 = [+w, d-w, 0]
    p2 = [+w, d+w, 0]
    p3 = [-w, d+w, 0]
    return np.array([p0, p1, p2, p3]).T

def calcTrapezSize(rect):
    p0, p1, p2, p3 = rect.T
    a = abs(p0[0] - p1[0])
    c = abs(p2[0] - p3[0])
    h = np.mean([abs(p0[1] - p3[1]), abs(p1[1] - p2[1])])
    return (a + c) / 2 * h

def calcQuadrilateralSize(rect):
    A, B, C, D = rect.T
    return 0.5 * abs((A[1]-C[1])*(D[0]-B[0]) + (B[1]-D[1])*(A[0]-C[0]))

class CameraTransform():
    """
    CameraTransform class to calculate the position of objects from an image in 3D based
    on camera intrinsic parameters and observer position
    """
    t = None
    R = None
    C = None

    def __init__(self, focal_length, sensor_size, image_size, observer_height=None, angel_to_horizon=None):
        """
        Init routine to setup calculation basics

        :param focal_length:    focal length of the camera in mm
        :param sensor_size:     sensor size in mm [width, height]
        :param image_size:      image size in mm [width, height]
        :param observer_height: observer elevation in m
        :param angel_to_horizon: angle between the z-axis and the horizon
        """

        # store and convert arguments
        self.f = focal_length * 1e-3 # in m
        self.sensor_width, self.sensor_height = np.array(sensor_size) * 1e-3 # in m
        self.im_width, self.im_height = image_size

        # normalize the focal length by the sensor width and the image_width
        self.f = self.f / self.sensor_width * self.im_width

        # compose the intrinsic camera matrix
        self.C1 = np.array([[self.f, 0, self.im_width / 2, 0], [0, self.f, self.im_height / 2, 0], [0, 0, 1, 0]])

        if observer_height is not None:
            self.initCameraMatrix(observer_height, angel_to_horizon)

    def initCameraMatrix(self, height, tilt_angle, roll=0):
        # convert the angle to radians
        angle = tilt_angle * np.pi / 180

        # get the translation matrix and rotate it
        self.t = [0, 0, -height]
        self.t = np.dot(np.array([[1, 0, 0],
                                  [0,  np.cos(angle), np.sin(angle)],
                                  [0, -np.sin(angle), np.cos(angle)]]),
                        np.array([self.t]).T)

        # get the rotation-translation matrix with the rotation composed with the translation
        self.R = np.array(
            [[1,              0,             0, self.t[0]],
             [0,  np.cos(angle), np.sin(angle), self.t[1]],
             [0, -np.sin(angle), np.cos(angle), self.t[2]],
             [0,              0,             0,         1]])

        # compose the camera matrix with the rotation-translation matrix
        self.C = np.dot(self.C1, self.R)

    def transWorldToCam(self, x):
        """
        Transform from 3D world coordinates to 2D camera coordinates.

        :param x: a point in world coordinates [x, y, z], or an array of world points in the shape of [3xN]
        :return: a list of projected points
        """

        # reshape input x array to two dimensions
        try:
            if len(x.shape) == 1:
                x = x[:, None]
        except AttributeError:
            x = np.array([x]).T
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
        :param X: when given project the camera points to world coordinates with their X value set to this parameter. Can be a single value or a list.
        :param Y: when given project the camera points to world coordinates with their Y value set to this parameter. Can be a single value or a list.
        :param Z: when given project the camera points to world coordinates with their Z value set to this parameter. Can be a single value or a list.
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
        try:
            if len(x.shape) == 1:
                x = x[:, None]
        except AttributeError:
            x = np.array([x]).T

        # if the fixed value is a list, we have to transform each coordinate separately
        if not isinstance(fixed, int) and not isinstance(fixed, float):
            return np.array(
                [self._transCamToWorldFixedDimension(x[:, i:i + 1], fixed=fixed[i], dimension=dimension) for i in
                 range(x.shape[1])])[:, :, 0].T
        # else transform everything in one go
        return self._transCamToWorldFixedDimension(x, fixed, dimension)

    def generateLUT(self, distance_start, distance_stop):
        """
        Generate LUT to calculate area covered by one pixel in the image dependent on y position in the image

        :param distance_start: LUT start position in meter
        :param distance_stop:  LUT stio position in meter
        :return: LUT, same length as image height
        """
        self.y_lookup = np.zeros(self.im_height) + np.nan

        for i in range(distance_start, distance_stop):
            rect = getRectangle(10, i)
            rect = self.transWorldToCam(rect)
            y = np.mean(np.array(rect)[1, :])
            A = calcTrapezSize(rect)
            if 0 < y < self.im_height:
                self.y_lookup[int(y)] = A

        self.y_lookup = fillNans(self.y_lookup)
        return self.y_lookup

    def fitCamParametersFromObjects(self, points_foot=None, points_head=None, lines=None, estimated_height=30, estimated_angle=85, object_height=1):
        if lines is not None:
            y1 = [np.max([l.y1, l.y2]) for l in lines]
            y2 = [np.min([l.y1, l.y2]) for l in lines]
            x = [np.mean([l.x1, l.x2]) for l in lines]
            points_foot = np.vstack((x, y1))
            points_head = np.vstack((x, y2))

        def error(p):
            self.initCameraMatrix(p[0], p[1])
            estimated_foot_3D = self.transCamToWorld(points_foot.copy(), Z=0)
            estimated_foot_3D[2, :] = object_height
            estimated_head = self.transWorldToCam(estimated_foot_3D)
            pixels = np.linalg.norm(points_head - estimated_head, axis=0)
            return np.mean(pixels ** 2)

        def error2(p):
            self.initCameraMatrix(p[0], p[1])
            estimated_foot_3D = self.transCamToWorld(points_foot.copy(), Z=0)
            estimated_head_3D = self.transCamToWorld(points_head.copy(), Y=estimated_foot_3D[1, :])
            heights = estimated_foot_3D[2, :] - estimated_head_3D[2, :]
            return np.std((heights - object_height) ** 2)

        print([estimated_height, estimated_angle])
        p = fmin(error2, [estimated_height * 0.5, estimated_angle])
        print("Optimisation2", p)
        p = fmin(error, [estimated_height * 0.5, estimated_angle])
        print("Optimisation1", p)
        return p
