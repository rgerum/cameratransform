#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_transforms.py

# Copyright (c) 2017-2019, Richard Gerum
#
# This file is part of the cameratransform package.
#
# cameratransform is free software: you can redistribute it and/or modify
# it under the terms of the MIT licence.
#
# cameratransform is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
#
# You should have received a copy of the license
# along with cameratransform. If not, see <https://opensource.org/licenses/MIT>

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
        import cameratransform as ct
    # if an import error occurs
    except ImportError as err:
        # get the module name from the error message
        name = str(err).split("'")[1]
        print("Mock:", name, file=sys.stderr)
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
        cam = ct.Camera(ct.RectilinearProjection(focallength_mm=f, image_width_px=image_size[1], image_height_px=image_size[0],
                                                 sensor_width_mm=sensor_size[1], sensor_height_mm=sensor_size[0]),
                        ct.SpatialOrientation())

    @given(ct_st.projection())
    def test_initProjection(self, proj):
        proj.__class__(image=[proj.image_width_px, proj.image_height_px], center=[proj.center_x_px, proj.center_y_px],
                       focallength_px=proj.focallength_x_px)

        proj.__class__(image=np.zeros([proj.image_height_px, proj.image_width_px]),
                       center_x_px=proj.center_x_px, center_y_px=proj.center_y_px,
                       focallength_px=proj.focallength_x_px, focallength_y_px=proj.focallength_y_px)

        proj.__class__(image=np.zeros([proj.image_height_px, proj.image_width_px, 3]),
                       center=[proj.center_x_px, proj.center_y_px],
                       focallength_px=proj.focallength_x_px)

        proj.__class__(image=[proj.image_width_px, proj.image_height_px],
                       sensor=[proj.sensor_width_mm, proj.sensor_height_mm],
                       focallength_mm=14)

        proj.__class__(image=[proj.image_width_px, proj.image_height_px],
                       sensor_width_mm=proj.sensor_width_mm,
                       focallength_mm=14)

        proj.__class__(image=[proj.image_width_px, proj.image_height_px],
                       sensor_height_mm=proj.sensor_height_mm,
                       focallength_mm=14)

        proj.__class__(image=[proj.image_width_px, proj.image_height_px],
                       view_x_deg=proj.getFieldOfView()[0],
                       focallength_mm=14)

        proj.__class__(image=[proj.image_width_px, proj.image_height_px],
                       view_y_deg=proj.getFieldOfView()[0],
                       focallength_mm=14)

        proj.__class__(image=[proj.image_width_px, proj.image_height_px],
                       view_x_deg=proj.getFieldOfView()[0],
                       sensor=[proj.sensor_width_mm, proj.sensor_height_mm])

    @given(ct_st.camera())
    def test_transFieldOfView(self, cam):
        viewX, viewY = cam.projection.getFieldOfView()
        focalX = cam.projection.focallengthFromFOV(viewX)
        focalY = cam.projection.focallengthFromFOV(view_y=viewY)
        np.testing.assert_almost_equal(cam.projection.focallength_x_px, focalX, 2,
                                       err_msg="Converting focallength to view and back failed.")
        np.testing.assert_almost_equal(cam.projection.focallength_y_px, focalY, 2,
                                       err_msg="Converting focallength to view and back failed.")

        np.testing.assert_almost_equal(cam.projection.imageFromFOV(viewX), cam.projection.image_width_px, err_msg="imageFromFOV failed for view_x")
        np.testing.assert_almost_equal(cam.projection.imageFromFOV(view_y=viewY), cam.projection.image_height_px, err_msg="imageFromFOV failed for view_y")

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

    @given(ct_st.lens(), ct_st.projection(), st.floats(0, 0.01))
    def test_lens(self, lens, proj, k):
        if lens == ct.NoDistortion:
            lens = lens()
        else:
            lens = lens(k)
        cam = ct.Camera(projection=proj, lens=lens)
        y = [proj.image_height_px*0.5]*100
        x = np.linspace(0, 1, 100)*proj.image_width_px
        pos0 = np.round(np.array([x, y]).T).astype(int)
        pos1 = cam.lens.distortedFromImage(pos0)
        pos2 = np.round(cam.lens.imageFromDistorted(pos1)).astype(int)
        np.testing.assert_almost_equal(pos2, pos0, 0, err_msg="Transforming from distorted to undistorted image fails.")

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

    @given(ct_st.camera_down_with_world_points(projection=ct_st.projection(projection_type=st.just(ct.RectilinearProjection))))
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
        p2 = cam.imageFromSpace(offset + rays * factor)
        print(p)
        print(offset + rays * factor)
        print(p2)
        if not np.sum(np.isnan(p2)):
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
        return
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
        p1_single = camGroup[0].spaceFromImage(camGroup[0].imageFromSpace(p))
        p2_single = camGroup[1].spaceFromImage(camGroup[1].imageFromSpace(p))

        p[np.isnan(p1_single)] = np.nan
        p[np.isnan(p2_single)] = np.nan
        np.testing.assert_almost_equal(p, p3, 4, err_msg="Points from two cameras cannot be projected to the space and back")


if __name__ == '__main__':
    unittest.main()
