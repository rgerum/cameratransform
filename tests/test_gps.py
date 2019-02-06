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

    @given(st.floats(-90, 90), st.floats(-90, 90), st.sampled_from(["%2d° %2d' %6.3f\" %s", "%2d° %2.3f' %s", "%2.3f°"]))
    def test_gpsBackForth(self, lat, lon, format):
        gps_string = " ".join(ct.formatGPS(lat, lon, format=format))
        gps_tuple = ct.gpsFromString(gps_string)
        np.testing.assert_almost_equal(gps_tuple, [lat, lon], 2)


if __name__ == '__main__':
    unittest.main()


