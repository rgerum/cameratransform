import unittest
import CameraTransform as ct
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as st_np

points = st_np.arrays(dtype="float", shape=st.tuples(st.integers(2, 2), st.integers(0, 100)), elements=st.floats(0, 10000))

class TestFits(unittest.TestCase):

    def test_fitCamParametersFromObjects(self):
        # setup the camera
        cam = ct.CameraTransform(6.2, (6.17, 4.55), (3264, 2448))
        # set the parameters
        cam.fixHeight(34.025954)
        cam.fixRoll(-1.933622)
        cam.fixTilt(83.307121)
        cam.heading = 0
        # get horizon
        horizon = np.array([[185.708, 1689.906, 2709.230, 3171.416, 938.221],
                            [795.467, 847.153, 880.665, 896.025, 820.386]])

        # get feet and heads
        feet = np.array([[2444.983, 2648.486, 2843.791, 2820.373, 3193.688, 1902.720, 1543.973, 1490.770, 1660.660,
                              1787.174, 2020.442, 2074.914, 2555.684, 2904.453, 3084.393, 1595.931, 1562.091, 1851.034,
                              2215.969, 2756.997, 1969.461, 1870.150],
                             [1569.205, 1556.647, 1545.462, 1445.683, 1554.840, 1412.185, 1497.422, 1407.026, 1563.188,
                              1588.567, 1606.385, 1606.847, 1607.042, 1462.714, 1597.782, 1520.728, 1481.460, 1441.808,
                              1419.911, 1535.729, 1597.528, 1449.718]])
        heads = np.array([[2444.983, 2648.486, 2843.791, 2820.373, 3193.688, 1902.720, 1543.973, 1490.770, 1660.660,
                               1787.174, 2020.442, 2074.914, 2555.684, 2904.453, 3084.393, 1595.931, 1562.091, 1851.034,
                               2215.969, 2756.997, 1969.461, 1870.150],
                              [1548.691, 1537.182, 1526.287, 1429.232, 1535.311, 1395.279, 1478.906, 1390.332, 1542.916,
                               1566.117, 1584.135, 1585.859, 1588.504, 1445.887, 1576.425, 1501.253, 1463.102, 1424.282,
                               1403.555, 1516.264, 1576.358, 1432.114]])

        # setup the camera
        cam2 = ct.CameraTransform(6.2, (6.17, 4.55), (3264, 2448))
        # fit
        cam2.fixHorizon(horizon)
        cam2.fitCamParametersFromObjects(feet, heads)
        for name in ["height", "tilt"]:
            self.assertAlmostEqual(getattr(cam2, name), getattr(cam, name), 1, "Parameter %s is not fitted correctly" % name)


if __name__ == '__main__':
    unittest.main()


