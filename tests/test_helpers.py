#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_helpers.py

# Copyright (c) 2017-2021, Richard Gerum
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
import sys
import os
import unittest
import numpy as np
from hypothesis import given, reproduce_failure, strategies as st
from hypothesis.extra import numpy as st_np
from cameratransform import RectilinearProjection

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


class TestParameterSet(unittest.TestCase):

    def test_parameterSet(self):
        cam = ct.Camera(ct.RectilinearProjection(image=(100, 50), focallength_px=100), ct.SpatialOrientation())
        cam.defaults.elevation_m = 99
        assert cam.elevation_m == 99
        assert cam.defaults.elevation_m == 99
        cam.elevation_m = 10
        assert cam.elevation_m == 10
        assert cam.defaults.elevation_m == 99
        cam.elevation_m = None
        assert cam.elevation_m == 99
        assert cam.defaults.elevation_m == 99
        # test if the parameter sets are liked properly
        cam.orientation.elevation_m = 900
        assert cam.elevation_m == 900
        cam.elevation_m = 800
        assert cam.orientation.elevation_m == 800
        # test non parameter
        cam.foo = 99
        assert cam.foo == 99
        cam.defaults.foo = 99
        assert cam.defaults.foo == 99
        self.assertRaises(AttributeError, lambda: cam.bla)
        self.assertRaises(AttributeError, lambda: cam.parameters.bla)
        self.assertRaises(AttributeError, lambda: cam.projection.bla)
        self.assertRaises(AttributeError, lambda: cam.orientation.bla)
        self.assertRaises(AttributeError, lambda: cam.defaults.bla)

    @given(ct_st.camera())
    def test_cameraCone(self, cam):
        cone = cam.getCameraCone()
        cone_image = np.round(cam.imageFromSpace(cone))
        cone_image[cone_image[:, 0] == 0] = np.nan
        cone_image[cone_image[:, 0] == cam.projection.image_width_px] = np.nan
        cone_image[cone_image[:, 1] == cam.projection.image_height_px] = np.nan
        cone_image[cone_image[:, 1] == 0] = np.nan
        if isinstance(cam.projection, RectilinearProjection):
            assert np.all(np.isnan(cone_image))

        cone = cam.getImageBorder()
        cone_image = np.round(cone).astype("float")
        cone_image[cone_image[:, 0] == 0] = np.nan
        cone_image[cone_image[:, 0] == cam.projection.image_width_px] = np.nan
        cone_image[cone_image[:, 1] == cam.projection.image_height_px] = np.nan
        cone_image[cone_image[:, 1] == 0] = np.nan
        if isinstance(cam.projection, RectilinearProjection):
            assert np.all(np.isnan(cone_image))

    @given(ct_st.camera())
    def test_cameraOrigin(self, cam):
        origin, ray = cam.getRay([0, 0])
        image_point = cam.imageFromSpace(origin)
        if isinstance(cam.projection, RectilinearProjection):
            assert np.all(np.isnan(image_point))


if __name__ == '__main__':
    unittest.main()


