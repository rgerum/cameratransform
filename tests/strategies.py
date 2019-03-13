#!/usr/bin/env python
# -*- coding: utf-8 -*-
# strategies.py

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

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np
from hypothesis.control import reject

import cameratransform as ct


def sensor_sizes():
    return st.one_of(
        st.just([1.28, 0.96]),
        st.just([1.60, 1.20]),
        st.just([2.40, 1.80]),
        st.just([3.60, 2.70]),
        st.just([4.00, 3.00]),
        st.just([4.54, 3.42]),
        st.just([4.8, 3.5]),
        st.just([4.80, 3.60]),
        st.just([5.37, 4.04]),
        st.just([5.79, 4.01]),
        st.just([5.76, 4.29]),
        st.just([6.17, 4.55]),
        st.just([6.30, 4.72]),
        st.just([6.40, 4.80]),
        st.just([7.18, 5.32]),
        st.just([7.60, 5.70]),
        st.just([8.08, 6.01]),
        st.just([8.80, 6.60]),
        st.just([10.26, 7.49]),
        st.just([10.67, 8.00]),
        st.just([12.48, 7.02]),
        st.just([12.52, 7.41]),
        st.just([13.20, 8.80]),
        st.just([12.80, 9.60]),
        st.just([15.81, 8.88]),
        st.just([17.30, 13]),
        st.just([21.12, 11.88]),
        st.just([18.70, 14]),
        st.just([21.95, 9.35]),
        st.just([20.70, 13.80]),
        st.just([23.00, 10.80]),
        st.just([24.89, 9.35]),
        st.just([22.30, 14.90]),
        st.just([22.0, 16.0]),
        st.just([25.34, 14.25]),
        st.just([23.6, 15.60]),
        st.just([24.89, 13.86]),
        st.just([25.6, 13.5]),
        st.just([24.89, 18.66]),
        st.just([27.90, 18.60]),
        st.just([29.90, 15.77]),
        st.just([30.7, 15.8]),
        st.just([35.8, 23.9]),
        st.just([36.70, 25.54]),
        st.just([40.96, 21.60]),
        st.just([45, 30]),
        st.just([44, 33]),
        st.just([52.48, 23.01]),
        st.just([54.12, 25.58]),
        st.just([49, 36.80]),
        st.just([56, 36]),
        st.just([53.7, 40.2]),
        st.just([53.90, 40.40]),
        st.just([42, 56]),
        st.just([56, 56]),
        st.just([70, 56]),
        st.just([70.41, 52.63]),
        st.just([84, 56]),
        st.just([121, 97]),
        st.just([178, 127]),
        st.just([254, 203]),
    )


def projection_type():
    return st.one_of(st.just(ct.RectilinearProjection), st.just(ct.CylindricalProjection),
                     st.just(ct.EquirectangularProjection))


@st.composite
def projection(draw, view_x_deg=st.floats(1, 80), width=st.integers(10, 1000), sensor=sensor_sizes(),
               projection_type=projection_type()):
    view_x_deg = draw(view_x_deg)
    width = draw(width)
    sensor_width, sensor_height = draw(sensor)
    height = int(width / sensor_width * sensor_height)
    projection_type = draw(projection_type)
    return projection_type(None, image_height_px=height, image_width_px=width, sensor_height_mm=sensor_height, sensor_width_mm=sensor_width, view_x_deg=view_x_deg)


@st.composite
def orientation(draw, elevation=st.floats(0, 1000), tilt_deg=st.floats(0, 90), roll_deg=st.floats(-180, 180),
                heading_deg=st.floats(-360, 360), x_m=st.floats(-100, 100), y_m=st.floats(-100, 100)):
    return ct.SpatialOrientation(draw(elevation), draw(tilt_deg), draw(roll_deg), draw(heading_deg), draw(x_m),
                                 draw(y_m))


def lens():
    return st.one_of(st.just(ct.NoDistortion), st.just(ct.BrownLensDistortion), st.just(ct.ABCDistortion))


@st.composite
def simple_orientation(draw, elevation=st.floats(0, 1000), tilt_deg=st.floats(0, 90)):
    return ct.SpatialOrientation(draw(elevation), draw(tilt_deg))


@st.composite
def camera(draw, projection=projection(), orientation=orientation()):
    return ct.Camera(projection=draw(projection), orientation=draw(orientation))


@st.composite
def camera_image_points(draw, camera=camera(), n=st.one_of(st.integers(1, 1000), st.just(1))):
    camera = draw(camera)
    width = camera.projection.image_width_px
    height = camera.projection.image_height_px
    n = draw(n)
    # the points can either be
    if n == 1:
        points = draw(st.tuples(st.floats(0, height), st.floats(0, width)))
    else:
        pointsX = draw(
            st_np.arrays(dtype="float", shape=st.tuples(st.just(n), st.just(1)), elements=st.floats(0, width)))
        pointsY = draw(
            st_np.arrays(dtype="float", shape=st.tuples(st.just(n), st.just(1)), elements=st.floats(0, height)))
        points = np.hstack((pointsX, pointsY))
    return camera, points

@st.composite
def camera_down_with_world_points(draw, projection=projection(),  n=st.one_of(st.integers(1, 1000), st.just(1))):
    camera = ct.Camera(projection=draw(projection), orientation=ct.SpatialOrientation())
    n = draw(n)
    # the points can either be
    if n == 1:
        points = draw(st.tuples(st.floats(-100, 100), st.floats(-100, 100), st.floats(-100, 100)))
    else:
        points = draw(
            st_np.arrays(dtype="float", shape=st.tuples(st.just(n), st.just(3)), elements=st.floats(-100, 100)))
    return camera, points

@st.composite
def line(draw):
    origin = draw(st_np.arrays(dtype="float", shape=(3, ), elements=st.floats(-100, 100)))
    anglesTheta = draw(st_np.arrays(dtype="float", shape=(3, ), elements=st.floats(-90, 90)))
    anglesPhi = draw(st_np.arrays(dtype="float", shape=(3, ), elements=st.floats(0, 360)))

    distance = draw(st.floats(0, 10))

    sPhi, cPhi = np.sin(np.deg2rad(anglesPhi)), np.cos(np.deg2rad(anglesPhi))
    sTheta, cTheta = np.sin(np.deg2rad(anglesTheta)), np.cos(np.deg2rad(anglesTheta))
    direction = np.array([sTheta*cPhi, sTheta*sPhi, cTheta]).T
    x, y, z = direction[0]
    if x == 0:
        reject()
    c = sPhi
    b = cPhi
    a = (-z*c-y*b)/x
    direction[1] = np.array([a[1], b[1], c[1]])
    direction[2] = np.array([a[2], b[2], c[2]])
    direction[1] /= np.linalg.norm(direction[1])
    direction[2] /= np.linalg.norm(direction[2])

    if np.all(direction[0] - direction[1] < 1e-2) or \
       np.all(direction[0] - direction[2] < 1e-2) or \
       np.all(direction[1] - direction[2] < 1e-2):
        reject()

    # origin will the the closes point between the two lines
    # p1, p2 will be the closes points of the line to each other
    p1 = origin - 0.5*direction[0]*distance
    p2 = origin + 0.5*direction[0]*distance

    o1 = p1 + direction[1]*2
    o2 = p2 + direction[2]*2

    return o1, direction[1], o2, direction[2], origin, distance, p1, p2


@st.composite
def lines(draw, n=st.one_of(st.integers(1, 3), st.just(1))):
    n = draw(n)
    # the points can either be
    if n == 1:
        data = draw(line())
        return data
    else:
        data = None
        for i in range(n):
            data_new = draw(line())
            if data is None:
                data = []
                for value in data_new:
                    data.append([value])
            else:
                for j, value in enumerate(data_new):
                    data[j].append(value)
        for i in range(len(data)):
            data[i] = np.array(data[i])
        return data
