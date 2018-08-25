import numpy as np
from .parameter_set import ParameterSet, Parameter, TYPE_DISTORTION


class BrownLensDistortion:
    offset_x = 0
    offset_y = 0

    def __init__(self, k1, k2, k3):
        self.parameters = ParameterSet(
            # the intrinsic parameters
            k1=Parameter(k1, type=TYPE_DISTORTION),
            k2=Parameter(k2, type=TYPE_DISTORTION),
            k3=Parameter(k3, type=TYPE_DISTORTION),
        )

    def imageFromDistorted(self, points):
        # ensure that the points are provided as an array
        points = np.array(points)
        r = np.sqrt(points[..., 0]**2 + points[..., 1]**2)
        return points*(1 + self.parameters.k1*r**2 + self.parameters.k2*r**4 + self.parameters.k3*r**6)

    def distortedFromImage(self, points):
        # ensure that the points are provided as an array
        points = np.array(points)
