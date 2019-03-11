import matplotlib
matplotlib.use('agg')
import sys
import os
import unittest
import numpy as np
from hypothesis import given, reproduce_failure, strategies as st
from hypothesis.extra import numpy as st_np
from hypothesis.control import reject

import mock

while True:
    # try to import CameraTransform
    try:
        import CameraTransform as ct
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

    #@reproduce_failure('4.8.0', b'AXicY2DAB9JFPAABxgDE')
    #@reproduce_failure('4.8.0', b'AXicY2DAChhhNCMzqgQTAACgAAk=')
    #@reproduce_failure('4.8.0', b'AXicY2DABpgYGRgZQIiBkRkqxMgJY/DCGMIwhihMHyOEAgAJQABN')
    #@reproduce_failure('4.8.0', b'AXicY2DABhjBZNGpJ4zMaBIAKHoCJw==')
    #@reproduce_failure('4.8.0', b'AXicY2DABhiZYAwOEMHIqA/mFTM+BwAFFwGa')
    #@reproduce_failure('4.8.0', b'AXicY2CAAMb/IPCdD8zhZGRkYGQGCgIAb4kGEg==')
    #@reproduce_failure('4.8.0', b'AXicY2DABhgZWRiZGRg5gCwAAJ0AFQ==')
    #@reproduce_failure('4.8.0', b'AXicY2SAAEYGrICRgZmRGSgHAACbAAw=')
    #@reproduce_failure('4.8.0', b'AXicY2RkAAMohQ4YGRkY2RkZWAEAywAU')
    #@reproduce_failure('4.8.0', b'AXicY2DABwAAHgAB')
    @given(st_np.arrays(dtype="float", shape=(3, ), elements=st.floats(0, 5)),
           st_np.arrays(dtype="float", shape=st.tuples(st.integers(1, 2), st.just(3), st.just(3)), elements=st.floats(10, 100)),
           st.floats(0.1, 0.9), st.floats(0.1, 0.9))
    def test_rayIntersectTriangle(self, origin, points, p1, p2):
        # add some offsets to make sure that the triangle is a real one (with an area > 0)
        points[..., 1, :] += np.array([1.7, 0.5, 0.3])
        points[..., 2, :] += np.array([2.7, 4.5, 3.3])

        if points.shape[0] == 1:
            points = points[0, ...]
            tri = points
        else:
            points = points[:1, ...]
            tri = points[0]

        for i, j in [(0, 1), (0, 2), (1, 2)]:
            if np.allclose(tri[i], tri[j], 0.001):
                return reject()

        d1 = tri[1]-tri[0]
        d1 /= np.linalg.norm(d1)
        d2 = tri[2]-tri[0]
        d2 /= np.linalg.norm(d2)
        if np.allclose(d1, d2, 0.001):
            return reject()

        target_point = tri[0] + p1 * (tri[1]-tri[0]) + p2 * (1-p1) * (tri[2]-tri[0])
        ray = target_point-origin
        ray /= np.linalg.norm(ray)
        if np.any(np.isnan(ray)):
            return reject()
        intersection = ct.ray.ray_intersect_triangle(origin, ray, points)
        try:
            np.testing.assert_almost_equal(target_point, intersection)
        except Exception:
            np.testing.assert_almost_equal(origin, intersection)

if __name__ == '__main__':
    unittest.main()


