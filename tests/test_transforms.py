import unittest
import CameraTransform as ct
import numpy as np
from hypothesis import given, reproduce_failure, strategies as st
from hypothesis.extra import numpy as st_np

points = st_np.arrays(dtype="float", shape=st.tuples(st.integers(1, 100), st.integers(2, 2)), elements=st.floats(0, 2448))


class TestTransforms(unittest.TestCase):

    def test_init_cam(self):
        # intrinsic camera parameters
        f = 6.2
        sensor_size = (6.17, 4.55)
        image_size = (3264, 2448)

        # initialize the camera
        cam = ct.Camera(ct.RectilinearProjection(f, image_size[1], image_size[0], sensor_size[1], sensor_size[0]),
                        ct.SpatialOrientation())

    @given(points, st.floats(-3, 3), st.floats(0.1, 10), st.floats(0.01, 90-0.01), st.one_of(st.just(ct.RectilinearProjection),
                                                                                             st.just(ct.CylindricalProjection),
                                                                                             st.just(ct.EquirectangularProjection)))
    def test_transWorldToCam(self, p, Z, height, tilt, projection):
        # setup the camera
        cam = ct.Camera(projection(6.2, 2448, 3264, 4.55, 6.17), ct.SpatialOrientation())
        # set the parameters
        cam.elevation_m = height
        cam.tilt_deg = tilt
        # transform point
        p1 = cam.spaceFromImage(p, Z=Z)
        p2 = cam.imageFromSpace(p1)
        # points behind the camera are allowed to be nan
        p[np.isnan(p2)] = np.nan
        np.testing.assert_almost_equal(p, p2, 1, err_msg="Transforming from camera to world and back doesn't return "
                                                         "the original point.")


if __name__ == '__main__':
    unittest.main()
