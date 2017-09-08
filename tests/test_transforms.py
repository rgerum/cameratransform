import unittest
import CameraTransform as ct
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as st_np

points = st_np.arrays(dtype="float", shape=st.tuples(st.integers(2, 2), st.integers(0, 100)), elements=st.floats(0, 10000))

class TestTransforms(unittest.TestCase):

    def test_init_cam(self):
        # intrinsic camera parameters
        f = 6.2
        sensor_size = (6.17, 4.55)
        image_size = (3264, 2448)

        # initialize the camera
        cam = ct.CameraTransform(f, sensor_size, image_size)

    @given(points, st.floats(-3, 3), st.floats(0.1, 10), st.floats(0.01, 90-0.01))
    def test_transWorldToCam(self, p, Z, height, tilt):
        # setup the camera
        cam = ct.CameraTransform(6.2, (6.17, 4.55), (3264, 2448))
        # set the parameters
        cam.fixHeight(height)
        cam.fixTilt(tilt)
        # transform point
        p1 = cam.transCamToWorld(p, Z=Z)
        p2 = cam.transWorldToCam(p1)
        np.testing.assert_almost_equal(p, p2, 1, err_msg="Transforming from camera to world and back doesn't return "
                                                         "the original point.")

    @given(points, st.floats(-3, 3), st.floats(0.1, 10), st.floats(0.01, 90 - 0.01))
    def test_transGPSToCam(self, p, Z, height, tilt):
        # setup the camera
        cam = ct.CameraTransform(43.0, (42.55, 27.20), (3072, 2048))
        # set the parameters
        cam.fixHeight(26.400)
        cam.fixTilt(87.788)
        cam.fixRoll(-1.10577)
        cam.setCamHeading(0)
        cam.setCamGPS(-70.61805, -8.1573)
        # transform point
        p1 = cam.transCamToGPS(p)
        p2 = cam.transGPSToCam(p1)
        # iterate over points
        for i in range(p.shape[1]):
            # test if they are not none
            if not np.isnan(p2[:, i]).any():
                # then they should be valid
                self.assertTrue(np.allclose(p[:, i], p2[:, i], rtol=0.1, atol=0.1),
                                "Transforming from camera to gps and back doesn't return the original point. %s %s" %
                                (str(p[:, i]), str(p2[:, i])))

if __name__ == '__main__':
    unittest.main()
