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


if __name__ == '__main__':
    unittest.main()


