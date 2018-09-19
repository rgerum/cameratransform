import matplotlib
matplotlib.use('agg')
import unittest
import numpy as np
from hypothesis import given, assume, note, strategies as st
from hypothesis.extra import numpy as st_np

import sys
import mock

while True:
    # try to import CameraTransform
    try:
        import CameraTransform as ct
    # if an import error occurs
    except ImportError as err:
        # get the module name from the error message
        name = str(err).split("'")[1]
        print("Mock:", name)
        # and mock it
        sys.modules.update((mod_name, mock.MagicMock()) for mod_name in [name])
        # then try again to import it
        continue
    else:
        break

points = st_np.arrays(dtype="float", shape=st.tuples(st.integers(2, 2), st.integers(0, 100)), elements=st.floats(0, 10000))

class TestFits(unittest.TestCase):

    def test_fitCamParametersFromObjects(self):
        # setup the camera
        # setup the camera
        cam = ct.Camera(ct.RectilinearProjection(6.2, 2448, 3264, 4.55, 6.17), ct.SpatialOrientation())
        # set the parameters
        cam.elevation_m = 34.025954
        cam.roll_deg = -1.933622
        cam.tilt_deg = 83.307121
        cam.heading_deg = 0
        # get horizon
        horizon = np.array([[185.708, 1689.906, 2709.230, 3171.416, 938.221],
                            [795.467, 847.153, 880.665, 896.025, 820.386]]).T

        # get feet and heads
        feet = np.array([[2444.983, 2648.486, 2843.791, 2820.373, 3193.688, 1902.720, 1543.973, 1490.770, 1660.660,
                              1787.174, 2020.442, 2074.914, 2555.684, 2904.453, 3084.393, 1595.931, 1562.091, 1851.034,
                              2215.969, 2756.997, 1969.461, 1870.150],
                             [1569.205, 1556.647, 1545.462, 1445.683, 1554.840, 1412.185, 1497.422, 1407.026, 1563.188,
                              1588.567, 1606.385, 1606.847, 1607.042, 1462.714, 1597.782, 1520.728, 1481.460, 1441.808,
                              1419.911, 1535.729, 1597.528, 1449.718]]).T

        feet_space = cam.spaceFromImage(feet)
        heads_space = feet_space.copy()
        heads_space[:, 2] += 1
        heads = cam.imageFromSpace(heads_space)

        def cost():
            estimated_feet_space = cam2.spaceFromImage(feet.copy(), Z=0)
            estimated_heads_space = estimated_feet_space.copy()
            estimated_heads_space[:, 2] += 1
            estimated_heads = cam2.imageFromSpace(estimated_heads_space)
            pixels = np.linalg.norm(heads - estimated_heads, axis=0)
            return np.mean(pixels ** 2)

        # setup the camera
        cam2 = ct.Camera(cam.projection, ct.SpatialOrientation())
        cam2.pos_x_m = 0
        cam2.pos_y_m = 0
        cam2.heading_deg = 0
        # fit
        cam2.fit(cost)
        # check the fitted parameters
        for name in ["elevation_m", "tilt_deg"]:
            self.assertAlmostEqual(getattr(cam2, name), getattr(cam, name), 1, "Parameter %s is not fitted correctly" % name)


if __name__ == '__main__':
    unittest.main()


