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

def getErrorOnHeighVariation(height2, angle2):
    cam = ct.CameraTransform(f, (sensor_width, sensor_width * im_height / im_width), (im_width, im_height), height,
                             angle)

    y = np.arange(50, 300, 10)
    p1_t = cam.transWorldToCam(np.array([[0] * len(y), y, [0] * len(y)])).astype("int")
    p2_t = cam.transWorldToCam(np.array([[0] * len(y), y, [1] * len(y)])).astype("int")

    cam.tilt = angle2
    cam.height = height2
    cam._initCameraMatrix()

    p1 = cam.transCamToWorld(p1_t.copy(), Z=0)
    p2 = cam.transCamToWorld(p2_t.copy(), Y=p1[1, :])
    heights = p2[2, :] - p1[2, :]
    return heights

if 1:
    plt.figure(0)
    name = natsorted(glob.glob(os.path.join("CameraTransform", "test_data_volume", "Blender*.png")))[0]
    im = imageio.imread(name)

    cam = ct.CameraTransform(f, (sensor_width, sensor_width*im_height/im_width), (im_width, im_height), height, angle)

    db = clickpoints.DataFile(os.path.join("CameraTransform", "test_data_volume", "lines.cdb"))
    lines = db.getLines()
    horizon = cam.getImageHorizon()
    plt.plot(horizon[0, :], horizon[1, :], '-m')

    if 1:
        if 0:
            horizon = db.getMarkers(type="horizon")
            horizon = np.array([[m.x, m.y] for m in horizon]).T
        else:
            horizon = cam.getImageHorizon()
        #horizon = horizon+np.random.rand(*horizon.shape)*100-50
        print("horizon", horizon)
        cam.fixHorizon(horizon)
        cam.fixRoll(0)
        plt.plot(horizon[0, :], horizon[1, :], 'c+')

    if 1:
        plt.figure(1)
        lines = np.array(list(lines))

        heights_points = []
        heights_mean = []
        angle_points = []
        angle_means = []
        for i in range(1, len(lines)):
            heights = []
            angles = []
            for j in range(10):
                indices = np.arange(len(lines))
                np.random.shuffle(indices)
                p = cam.fitCamParametersFromObjects(lines=np.array(lines)[indices[:i].astype("int")])
                heights_points.append([i, cam.height])
                heights.append(cam.height)
                angle_points.append([i, cam.tilt])
                angles.append(cam.tilt)
            heights_mean.append([i, np.mean(heights)])
            angle_means.append([i, np.mean(angles)])
        heights_points = np.array(heights_points)
        heights_mean = np.array(heights_mean)
        angle_points = np.array(angle_points)
        angle_means = np.array(angle_means)
        plt.subplot(131)
        plt.plot(heights_points[:, 0], heights_points[:, 1], 'bo')
        plt.plot(heights_mean[:, 0], heights_mean[:, 1], 'r+')
        plt.axhline(height, color='k', ls="--")
        plt.ylim(height*0.9, height*1.1)
        plt.xlabel("Number of used objects")
        plt.ylabel("Fitted height")
        plt.subplot(132)
        plt.plot(angle_points[:, 0], angle_points[:, 1], 'bo')
        plt.plot(angle_means[:, 0], angle_means[:, 1], 'r+')
        plt.axhline(angle, color='k', ls="--")
        plt.ylim(angle*0.9, angle*1.1)
        plt.xlabel("Number of used objects")
        plt.ylabel("Fitted tilt")

        plt.subplot(133)
        for i in range(heights_mean.shape[0]):
            heights = getErrorOnHeighVariation(heights_mean[i, 1], angle_means[i, 1])
            plt.errorbar(heights_mean[i, 0], np.mean(heights), yerr=np.std(heights), color="b", marker="o")
            plt.text(heights_mean[i, 0], np.mean(heights), "%.2f" % np.mean(heights), rotation=90)
        plt.axhline(1, color="k", ls='--')
        heights = getErrorOnHeighVariation(height, angle)
        plt.errorbar(heights_mean[i, 0]+1, np.mean(heights), yerr=np.std(heights), color="r", marker="o")

        plt.xlabel("Number of used objects")
        plt.ylabel("Reconstructed heights of objects")

        plt.tight_layout()

    plt.figure(0)
    #horizon = cam.getImageHorizon()
    #cam.fixHorizon(horizon)
    cam.fixRoll(0)
    p = cam.fitCamParametersFromObjects(lines=lines)

    horizon = cam.getImageHorizon()
    plt.plot(horizon[0, :], horizon[1, :], '-b')

    cam = ct.CameraTransform(f, (sensor_width, sensor_width * im_height / im_width), (im_width, im_height), height,
                             angle)
    cam.fixHorizon(horizon)

    p = cam.fitCamParametersFromObjects(lines=lines)

    y1 = [np.max([l.y1, l.y2]) for l in lines]
    y2 = [np.min([l.y1, l.y2]) for l in lines]
    x = [np.mean([l.x1, l.x2]) for l in lines]
    points1 = np.vstack((x, y1))
    points2 = np.vstack((x, y2))


    p1 = cam.transCamToWorld(points1.copy(), Z=0)
    print(p1.shape)

    p1[2, :] = 1
    p2 = cam.transWorldToCam(p1)
    print(p2.shape)


    plt.plot(points1[0, :], points1[1, :], 'o')
    plt.plot(p2[0, :], p2[1, :], 'o')
    plt.plot(points2[0, :], points2[1, :], 'o')
    plt.imshow(im)

    horizon = cam.getImageHorizon()
    plt.plot(horizon[0, :], horizon[1, :], '--k')

    plt.show()
elif 0:
    camera = ct.CameraTransform(f, (sensor_width, 1), (im_width, im_height), height, angle)

    images = natsorted(glob.glob(os.path.join("CameraTransform", "test_data_volume", "Blender*.png")))
    name = images[0]
    print(name)
    im = imageio.imread(name)

    horizon = camera.getImageHorizon()
    plt.plot(horizon[0, :], horizon[1, :], '-k')

    rect = ct.getRectangle(10, 100)
    rect = np.hstack((rect, rect[:, 0:1]))
    rect = camera.transWorldToCam(rect)

    plt.plot(rect[0, :], rect[1, :])

    plt.imshow(im)

    plt.show()
else:
    camera = ct.CameraTransform(f, (sensor_width, 1), (im_width, im_height), height, angle)
    plt.subplot(122)
    print(camera.distanceToHorizon())
    rect0 = np.array([[0, 0], [im_width, 0]])
    rect0 = np.concatenate((rect0, camera.getImageHorizon().T, [[0, 0]])).T
    rect0 = camera.transCamToWorld(rect0, Z=0)
    print(rect0)
    #plt.plot(rect0[0, :], rect0[1, :], '-')

    rect = ct.getRectangle(10, 100)
    print(rect)
    rect = np.hstack((rect, rect[:, 0:1]))

    images = natsorted(glob.glob(os.path.join("CameraTransform", "test_data_volume", "Blender*.png")))
    name = images[0]
    print(name)
    im = imageio.imread(name)

    im2 = camera.getTopViewOfImage(im, [-200, 100, 0, 300], scaling=10, doplot=True)
    plt.plot(rect[0, :], rect[1, :], '-')
    #plt.imshow(im2)
    plt.plot()

    plt.subplot(121)
    horizon = camera.getImageHorizon()
    plt.plot(horizon[0, :], horizon[1, :], '-k')

    rect = camera.transWorldToCam(rect)

    plt.plot(rect[0, :], rect[1, :])

    plt.imshow(im)

    plt.show()
