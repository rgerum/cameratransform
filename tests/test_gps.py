#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_gps.py

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
# GNU General Public License for more details.
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

    @given(st.floats(-90, 90), st.floats(-90, 90), st.sampled_from(["%2d° %2d' %6.3f\" %s", "%2d° %2.3f' %s", "%2.3f°"]), st.floats(0, 100))
    def test_gpsStringBackForth(self, lat, lon, format, height):
        gps_string = [" ".join(ct.formatGPS(lat, lon, format=format))]
        gps_tuple = ct.gpsFromString(gps_string)[0]
        np.testing.assert_almost_equal(gps_tuple, [lat, lon], 2)

        gps_string = ct.formatGPS(lat, lon, format=format)
        gps_tuple = ct.gpsFromString(gps_string)
        np.testing.assert_almost_equal(gps_tuple, [lat, lon], 2)

        gps_string = " ".join(ct.formatGPS(lat, lon, format=format))
        gps_tuple = ct.gpsFromString(gps_string, height=height)
        np.testing.assert_almost_equal(gps_tuple, [lat, lon, height], 2)

        gps_string = ct.formatGPS(lat, lon, format=format)
        gps_tuple = ct.gpsFromString(gps_string, height=height)
        np.testing.assert_almost_equal(gps_tuple, [lat, lon, height], 2)

        # test to transform with default format
        ct.formatGPS(lat, lon)

        # test to raise an error if the format string is not valid
        self.assertRaises(ValueError, lambda: ct.formatGPS(lat, lon, ""))
        self.assertRaises(ValueError, lambda: ct.formatGPS(lat, lon, "%2d %2d %2d %2d %2d"))
        # for latex it should replace the degree symbol
        ct.formatGPS(lat, lon, asLatex=True)

        # try split gps
        ct.splitGPS(gps_tuple, keep_deg=True)

    @given(st.floats(-89, 89), st.floats(-89, 89), st.floats(1, 1000), st.floats(-180, 180), st.floats(0, 100))
    def test_gpsSpaceBackForth(self, lat, lon, distance, bearing, height):
        # without height
        gps0 = [lat, lon]
        gps = ct.moveDistance(gps0, distance, bearing)
        space = ct.spaceFromGPS(gps, gps0)

        gps_2 = ct.gpsFromSpace(space[..., :2], gps0)
        # this has to be only exact to 1 decimal, as it neglects the curvature of the earth and therefore is not exact
        np.testing.assert_almost_equal(min([ct.getDistance(gps, gps_2)/distance, ct.getDistance(gps, gps_2)]), 0, 0)

        np.testing.assert_almost_equal(distance, ct.getDistance(gps0, gps), 0)

        difference_angle = bearing - ct.getBearing(gps0, gps)
        while difference_angle > 180:
            difference_angle -= 360
        while difference_angle < -180:
            difference_angle += 360
        np.testing.assert_almost_equal(difference_angle, 0, 0)

        # with height
        gps0 = [lat, lon, 1]
        gps = ct.moveDistance(gps0, distance, bearing)
        space = ct.spaceFromGPS(gps, gps0)
        gps_2 = ct.gpsFromSpace(space, gps0)

        # this has to be only exact to 1 decimal, as it neglects the curvature of the earth and therefore is not exact
        np.testing.assert_almost_equal(min([ct.getDistance(gps, gps_2) / distance, ct.getDistance(gps, gps_2)]), 0, 0)

        np.testing.assert_almost_equal(distance, ct.getDistance(gps0, gps), 0)

        difference_angle = bearing - ct.getBearing(gps0, gps)
        while difference_angle > 180:
            difference_angle -= 360
        while difference_angle < -180:
            difference_angle += 360
        np.testing.assert_almost_equal(difference_angle, 0, 0)

    def test_gpsDifferentFormats(self):
        ct.gpsFromString("060° 37′ 36″")
        ct.gpsFromString("85° 19′ 14″ N, 000° 02′ 43″ E")
        gps_tuple = ct.gpsFromString("66°39'56.12862''S,  140°01'20.39562''")
        gps_tuple = ct.gpsFromString(["66°39'56.12862''S", "140°01'20.39562''"])
        gps_tuple = ct.gpsFromString("66°39'56.12862''S  140°01'20.39562''", 13.769)

        gps_tuple = ct.gpsFromString(["66°39'56.12862''S  140°01'20.39562''", "66°39'58.73922''S  140°01'09.55709''"])

        gps_tuple = ct.gpsFromString(
            [["66°39'56.12862''S", "140°01'20.39562''", 13.769], ["66°39'58.73922''S", "140°01'09.55709''", 13.769]])

        gps_tuple = ct.gpsFromString([["66°39'56.12862''S  140°01'20.39562''", 13.769], ["66°39'58.73922''S  140°01'09.55709''", 13.769]])


if __name__ == '__main__':
    unittest.main()


