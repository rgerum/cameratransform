#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_ray.py

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
from hypothesis import given, settings, reproduce_failure, strategies as st, HealthCheck
from hypothesis.extra import numpy as st_np
from hypothesis.control import reject

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


def getPointInTriangle(tri, origin, p1, p2):
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        if np.allclose(tri[i], tri[j], 0.001):
            return reject()

    d1 = tri[1] - tri[0]
    d1 /= np.linalg.norm(d1)
    d2 = tri[2] - tri[0]
    d2 /= np.linalg.norm(d2)
    if np.allclose(d1, d2, 0.001):
        return reject()

    target_point = tri[0] + p1 * (tri[1] - tri[0]) + p2 * (1 - p1) * (tri[2] - tri[0])
    ray = target_point - origin
    ray /= np.linalg.norm(ray)
    if np.any(np.isnan(ray)):
        return reject()
    return target_point, ray


class TestParameterSet(unittest.TestCase):

    @given(st_np.arrays(dtype="float", shape=(3, ), elements=st.floats(0, 5)),
           st_np.arrays(dtype="float", shape=st.tuples(st.integers(1, 3), st.just(3), st.just(3)), elements=st.floats(10, 100)),
           st.floats(0.1, 0.9), st.floats(0.1, 0.9))
    def test_rayIntersectTriangle(self, origin, points, p1, p2):
        # add some offsets to make sure that the triangle is a real one (with an area > 0)
        points[..., 1, :] += np.array([1.7, 0.5, 0.3])
        points[..., 2, :] += np.array([2.7, 4.5, 3.3])

        point_count = points.shape[0]

        if points.shape[0] == 1:
            points = points[0, ...]
            tri = points
        else:
            tri = points[0]

        if point_count == 2:
            target_point1, ray1 = getPointInTriangle(points[0], origin, p1, p2)
            target_point2, ray2 = getPointInTriangle(points[1], origin, p1, p2)
            target_point = np.array([target_point1, target_point2])
            ray = np.array([ray1, ray2])
        else:
            target_point, ray = getPointInTriangle(tri, origin, p1, p2)

        intersection = ct.ray.ray_intersect_triangle(origin, ray, points)
        # if there are multiple triangles, just check if there is an intersection
        if point_count > 1:
            self.assertFalse(np.any(np.isnan(intersection)))
        # if there is just a single triangle the intersection has to be the constructed point
        else:
            try:
                np.testing.assert_almost_equal(target_point, intersection)
            except Exception:
                np.testing.assert_almost_equal(origin, intersection)

    @settings(suppress_health_check=(HealthCheck.filter_too_much,))
    @given(ct_st.lines())
    def test_lineDistance(self, line):
        p1, v1, p2, v2, center, distance, c1, c2 = line
        if len(p1.shape) == 2:
            p1 = p1[0]
            p2 = p2[0]
            center_list = []
            distance_list = []
            for i in range(v1.shape[0]):
                dist = ct.ray.distanceOfTwoLines(p1, v1[i], p2, v2[i])
                distance_list.append(dist)
                center = ct.ray.intersectionOfTwoLines(p1, v1[i], p2, v2[i])
                center_list.append(center)
            center = np.array(center_list)
            distance = np.array(distance_list)
        else:
            c1_new = ct.ray.getClosestPointFromLine(p1, v1, c2)
            np.testing.assert_almost_equal(c1, c1_new, 1)
            c1_new = ct.ray.getClosestPointFromLine(p1, np.array([v1]), np.array([c2]))
            np.testing.assert_almost_equal(np.array([c1]), c1_new, 1)
        dist = ct.ray.distanceOfTwoLines(p1, v1, p2, v2)
        np.testing.assert_almost_equal(dist, distance, 1)
        intersection = ct.ray.intersectionOfTwoLines(p1, v1, p2, v2)
        if np.all(np.isnan(intersection)) is False:
            np.testing.assert_almost_equal(intersection, center, 1)

    @given(st_np.arrays(dtype="float", shape=(4, 2), elements=st.floats(-100, 100)))
    def test_extrudeLine(self, line):
        mesh = ct.ray.extrudeLine(line, 0, 1)
        distance = 0
        last_point = None
        for point in line:
            if last_point is not None:
                distance += np.linalg.norm(point-last_point)
            last_point = point
        area = 0
        for part in mesh:
            area += ct.ray.areaOfTriangle(part)
        self.assertAlmostEqual(distance, area, 2)

if __name__ == '__main__':
    unittest.main()


