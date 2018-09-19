import matplotlib
matplotlib.use('agg')
import unittest
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as st_np

import sys
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

points = st_np.arrays(dtype="float", shape=st.tuples(st.integers(2, 2), st.integers(0, 100)), elements=st.floats(0, 10000))

class TestParameterSet(unittest.TestCase):

    def test_parameterSet(self):
        cam = ct.Camera(ct.RectilinearProjection(), ct.SpatialOrientation())
        cam.defaults.elevation_m = 99
        assert cam.elevation_m == 99
        assert cam.defaults.elevation_m == 99
        cam.elevation_m = 10
        assert cam.elevation_m == 10
        assert cam.defaults.elevation_m == 99
        cam.elevation_m = None
        assert cam.elevation_m == 99
        assert cam.defaults.elevation_m == 99
        # test non parameter
        cam.foo = 99
        assert cam.foo == 99
        cam.defaults.foo = 99
        assert cam.defaults.foo == 99



if __name__ == '__main__':
    unittest.main()


