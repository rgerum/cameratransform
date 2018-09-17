import unittest
import numpy as np
import sys
import os

from hypothesis import given, reproduce_failure, assume, strategies as st
from hypothesis.extra import numpy as st_np
import uuid

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

#print("path", os.path.dirname(__file__))
#print("path", os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import strategies as ct_st

points = st_np.arrays(dtype="float", shape=st.tuples(st.integers(1, 100), st.integers(2, 2)), elements=st.floats(0, 2448))

class TempFile:
    def __enter__(self):
        self.filename = str(uuid.uuid4())
        return self.filename

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.filename):
            os.remove(self.filename)


class TestTransforms(unittest.TestCase):

    def test_init_cam(self):
        # intrinsic camera parameters
        f = 6.2
        sensor_size = (6.17, 4.55)
        image_size = (3264, 2448)

        # initialize the camera
        cam = ct.Camera(ct.RectilinearProjection(f, image_size[1], image_size[0], sensor_size[1], sensor_size[0]),
                        ct.SpatialOrientation())

    @given(ct_st.camera())
    def test_transFieldOfView(self, cam):
        viewX, viewY = cam.projection.getFieldOfView()
        focalX = cam.projection.fieldOfViewToFocallength(viewX)
        focalY = cam.projection.fieldOfViewToFocallength(view_y=viewY)
        np.testing.assert_almost_equal(cam.projection.focallength_mm, focalX, 2,
                                       err_msg="Converting focallength to view and back failed.")
        np.testing.assert_almost_equal(cam.projection.focallength_mm, focalY, 2,
                                       err_msg="Converting focallength to view and back failed.")

    @given(ct_st.camera())
    def test_saveLoad(self, cam):
        with TempFile() as filename:
            cam.save(filename)
            cam2 = ct.load_camera(filename)
            for key in cam.parameters.parameters:
                assert(getattr(cam, key) == getattr(cam2, key))

    @given(ct_st.camera_image_points(), st.floats(0, 100))
    def test_transWorldToCam(self, params, Z):
        cam, p = params
        # when the camera is exactly on the desired plane, the output will not work
        assume(Z != cam.elevation_m)
        p = np.array(p)
        # transform point
        p1 = cam.spaceFromImage(p, Z=Z)
        p2 = cam.imageFromSpace(p1)
        # points behind the camera are allowed to be nan
        p[np.isnan(p2)] = np.nan
        np.testing.assert_almost_equal(p, p2, 1, err_msg="Transforming from camera to world and back doesn't return "
                                                         "the original point.")

    @given(ct_st.camera_down_with_world_points())
    def test_pointBehindCamera(self, params):
        cam, p = params
        cam.tilt_deg = 0
        cam.roll_deg = 0
        cam.heading_deg = 0
        # transform point
        p1 = cam.imageFromSpace(p)
        # points behind the camera are allowed to be nan
        p[p[:, 2] >= cam.elevation_m] = np.nan
        np.testing.assert_equal(np.isnan(np.sum(p, axis=1)), np.isnan(np.sum(p1, axis=1)),
                                err_msg="Points behind the camera do not produce a nan value.")


if __name__ == '__main__':
    unittest.main()
