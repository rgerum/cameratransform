import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import CameraTransform as ct


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