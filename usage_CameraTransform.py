import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
from natsort import natsorted

import CameraTransform as ct

# camera parameters
f = 14
observer_height = 25
sensor_size = [17.3, 13.0]
image_size = [4608, 2592]
angle = 83.052

# initialize class
ctrans = ct.CameraTransform(f, sensor_size, image_size, observer_height, angle)
# generate LUT for desired range
y_lookup = ctrans.generateLUT(50, 500)


def extractAreas(data):
    area = []
    area2 = []
    ds = []
    images = natsorted(glob.glob(data))

    for name in images:
        print(name)
        im = imageio.imread(name)

        fname = os.path.split(name)[1]
        i = int(fname[7:-4])

        mask = im[:, :, 0] < 128
        mask[:855, :] = False
        print("mask shape", mask.shape)

        area.append(np.sum(mask))
        area2.append(np.sum(np.sum(mask, 1) / y_lookup))
        ds.append(i)

    return ds, area, area2


ds, area_v, area2_v = extractAreas(os.path.join("CameraTransform", "test_data_volume", "Blender*.png"))
ds, area_plane, area2_plane = extractAreas(os.path.join("CameraTransform", "test_data_plane", "Blender*.png"))

fig = plt.figure()
ax = plt.subplot(111, xlabel="Distance in m", ylabel="Area / mean(area)")
plt.plot(ds, area_plane / np.mean(area_plane), label="raw area plane")
plt.plot(ds, area2_plane / np.mean(area2_plane), label="corrected area plane")
plt.plot(ds, area_v / np.mean(area_v), label="raw area volume")
plt.plot(ds, area2_v / np.mean(area2_v), label="corrected area volume")
plt.ylim([0.4, 2.0])
plt.xlim([90, 150])
plt.legend()
plt.tight_layout()

##
""" run the test for varing height"""
observer_height = 20
# initialize class
ctrans = ct.CameraTransform(f, sensor_size, image_size, observer_height, angle)
# generate LUT for desired range
y_lookup = ctrans.generateLUT(50, 500)

ds, area_v_me, area2_v_me = extractAreas(os.path.join("CameraTransform", "test_data_volume", "Blender*.png"))
print("ds", len(ds))
print("ds", len(area_v_me))
print("ds", len(area2_v_me))

fig = plt.figure()
ax = plt.subplot(111, xlabel="Distance in m", ylabel="Area / mean(area)")
# plt.plot(ds, area / np.mean(area), label="raw area volume")
# plt.plot(ds, area2 / np.mean(area2), label="corrected area volume")
# # plt.plot(ds, area_me / np.mean(area_me), label="raw area volume - 5m")
# plt.plot(ds, area2_me / np.mean(area2_me), label="corrected area volume -5m")

plt.plot(ds, area2_v, label="corrected area volume")
plt.plot(ds, area2_v_me, label="corrected area volume -5m")

plt.plot(ds, area2_v / np.mean(area2_v), label="corrected area volume")
plt.plot(ds, area2_v_me / np.mean(area2_v_me), label="corrected area volume -5m")

plt.ylim([0.4, 2.0])
plt.xlim([90, 150])
plt.legend()
plt.tight_layout()
plt.show()
##
# '''
