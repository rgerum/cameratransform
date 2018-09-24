# to prevent matplotlib from trying to import tkinter
import matplotlib
matplotlib.use('agg')

import unittest
import numpy as np
import sys
import os

from hypothesis import given, reproduce_failure, assume, note, strategies as st
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

sys.path.insert(0, os.path.dirname(__file__))
import strategies as ct_st


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
                self.assertAlmostEqual(getattr(cam, key), getattr(cam2, key), 3)

            cam.projection.save(filename)
            cam2.focallength = 999
            cam2.projection.load(filename)
            for key in cam.parameters.parameters:
                self.assertAlmostEqual(getattr(cam, key), getattr(cam2, key), 3)

            cam.orientation.save(filename)
            cam2.elevation_m = 9
            cam2.heading_deg = 9
            cam2.tilt_deg = 9
            cam2.roll_deg = 9
            cam2.orientation.load(filename)
            for key in cam.orientation.parameters.parameters:
                self.assertAlmostEqual(getattr(cam, key), getattr(cam2, key), 3)

    @given(ct_st.camera())
    def test_print(self, cam):
        str(cam)

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
        p = np.array(p)
        # transform point
        p1 = cam.imageFromSpace(p)
        # points behind the camera are allowed to be nan
        p[p[..., 2] > cam.elevation_m] = np.nan
        np.testing.assert_equal(np.isnan(p[..., :2]), np.isnan(p1),
                                err_msg="Points behind the camera do not produce a nan value.")

    @given(ct_st.camera_image_points(), st.floats(1, 100))
    def test_rays(self, params, factor):
        cam, p = params
        note(cam)
        offset, rays = cam.getRay(p, normed=True)
        np.testing.assert_almost_equal(np.linalg.norm(rays, axis=-1), np.ones(rays.shape[0]), 2)
        p2 = cam.imageFromSpace(offset + rays*factor)
        np.testing.assert_almost_equal(p, p2, 1, err_msg="Transforming from camera to world and back doesn't return "
                                                         "the original point.")

    @given(ct_st.projection(), ct_st.projection(), ct_st.orientation(), ct_st.orientation())
    def test_cameraGroup(self, proj1, proj2, orientation1, orientation2):
        def length(obj):
            try:
                return len(obj)
            except TypeError:
                return 1

        for projections in [proj1, (proj1, proj2)]:
            for orientations in [orientation1, (orientation1, orientation2)]:
                camGroup = ct.CameraGroup(projections, orientations)
                assert len(camGroup) == max(length(projections), length(orientations))
                for index, cam in enumerate(camGroup):
                    if isinstance(projections, tuple):
                        assert cam.projection == projections[index]
                    else:
                        assert cam.projection == projections
                    if isinstance(orientations, tuple):
                        assert cam.orientation == orientations[index]
                    else:
                        assert cam.orientation == orientations

    @given(ct_st.camera_down_with_world_points())
    def test_stereoCamera(self, params):
        cam, p = params
        camGroup = ct.CameraGroup(cam.projection, (cam.orientation, ct.SpatialOrientation()))
        cam1 = camGroup[0]
        cam2 = camGroup[1]

        cam1.tilt_deg = 0
        cam1.roll_deg = 0
        cam1.heading_deg = 0

        cam2.tilt_deg = 0
        cam2.roll_deg = 0
        cam2.heading_deg = 0
        cam2.pos_x_m = 10

        p = np.array(p)
        # transform point
        p1, p2 = camGroup.imagesFromSpace(p)
        p3 = camGroup.spaceFromImages(p1, p2)
        # points behind the camera are allowed to be nan
        p[p[..., 2] > cam.elevation_m] = np.nan
        np.testing.assert_almost_equal(p, p3, 4,
                                err_msg="Points behind the camera do not produce a nan value.")


if __name__ == '__main__':
    unittest.main()
