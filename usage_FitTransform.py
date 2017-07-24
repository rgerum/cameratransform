import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
from natsort import natsorted
import clickpoints

import CameraTransform as ct

f = 14
sensor_width = 17.3
height = 20
angle = 83.052
im_width = 4608
im_height = 2592

camera = ct.CameraTransform(f, (sensor_width, 1), (im_width, im_height))

db = clickpoints.DataFile(os.path.join("CameraTransform", "test_data_volume", "lines.cdb"))
lines = db.getLines()
p = camera.fitCamParametersFromObjects(lines=lines, estimated_height=height, estimated_angle=angle)

y1 = [np.max([l.y1, l.y2]) for l in lines]
y2 = [np.min([l.y1, l.y2]) for l in lines]
x = [np.mean([l.x1, l.x2]) for l in lines]
points1 = np.vstack((x, y1))
points2 = np.vstack((x, y2))


p1 = camera.transCamToWorld(points1.copy(), Z=0)
print(p1.shape)

p1[2, :] = 1
p2 = camera.transWorldToCam(p1)
print(p2.shape)

images = natsorted(glob.glob(os.path.join("CameraTransform", "test_data_volume", "Blender*.png")))
print(images)
plotted = False
name = images[0]
print(name)
im = imageio.imread(name)
plt.plot(points1[0, :], points1[1, :], 'o')
plt.plot(p2[0, :], p2[1, :], 'o')
plt.plot(points2[0, :], points2[1, :], 'o')
plt.imshow(im)

plt.show()
