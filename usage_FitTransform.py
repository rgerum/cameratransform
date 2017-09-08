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

def getArea(height2, angle2):
    cam = ct.CameraTransform(f, (sensor_width, sensor_width * im_height / im_width), (im_width, im_height), height,
                             angle)
    cam.tilt = angle2
    cam.height = height2
    cam._initCameraMatrix()

    im2 = cam.getTopViewOfImage(mask, [-200, 200, 0, 1000], scaling=0.1, do_plot=True)
    A = np.sum(im2 > 0.5) * (0.1 * 0.1)
    print("Area", A)
    return A

if 1:
    db = clickpoints.DataFile(os.path.join("CameraTransform", "test_data_volume", "lines2.cdb"))
    mask = db.getMasks()[0]
    mask = mask.data

    plt.figure(1)
    ax0b = plt.subplot(311)

    ax0a = plt.subplot(311)
    plt.figure(0)
    ax1a = plt.subplot(334)
    ax2a = plt.subplot(335)
    ax3a = plt.subplot(336)
    ax1b = plt.subplot(337)
    ax2b = plt.subplot(338)
    ax3b = plt.subplot(339)



    for plot_index in range(2):
        if plot_index == 0:
            ax1, ax2, ax3 = ax1a, ax2a, ax3a
            ax0 = ax0a
        else:
            ax1, ax2, ax3 = ax1b, ax2b, ax3b
            ax0 = ax0b

        plt.sca(ax0)
        name = natsorted(glob.glob(os.path.join("CameraTransform", "test_data_volume", "Blender*.png")))[0]
        im = imageio.imread(name)

        cam = ct.CameraTransform(f, (sensor_width, sensor_width*im_height/im_width), (im_width, im_height), height, angle)

        im2 = cam.getTopViewOfImage(mask, [-200, 200, 0, 1000], scaling=0.1, do_plot=True)
        A0 = np.sum(im2 > 0.5) * (0.1 * 0.1)

        db = clickpoints.DataFile(os.path.join("CameraTransform", "test_data_volume", "lines.cdb"))
        lines = db.getLines()
        horizon = cam.getImageHorizon()
        plt.plot(horizon[0, :], horizon[1, :], '-m')

        if plot_index:
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

            lines = np.array(list(lines))

            heights_points = []
            heights_mean = []
            angle_points = []
            angle_means = []
            if 0:
                for i in range(1, len(lines)):
                    heights = []
                    angles = []
                    for j in range(100):
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
                np.savez("height_angle%d.npz" % plot_index, heights_points=heights_points, heights_mean=heights_mean, angle_points=angle_points, angle_means=angle_means)
            else:
                data = np.load("height_angle%d.npz" % plot_index)
                globals().update(data)
            plt.sca(ax1)
            plt.plot(heights_points[:, 0], heights_points[:, 1], 'C0o', ms=2, alpha=0.5)
            plt.plot(heights_mean[:, 0], heights_mean[:, 1], 'C3+')
            plt.axhline(height, color='k', ls="--")
            plt.ylim(height*0.9, height*1.1)
            if plot_index == 1:
                plt.xlabel("Number of used objects")
            plt.ylabel("Fitted height (m)")
            plt.sca(ax2)
            plt.plot(angle_points[:, 0], angle_points[:, 1], 'C0o', ms=2, alpha=0.5)
            plt.plot(angle_means[:, 0], angle_means[:, 1], 'C3+')
            plt.axhline(angle, color='k', ls="--")
            plt.ylim(angle*0.9, angle*1.1)
            if plot_index == 1:
                plt.xlabel("Number of used objects")
            plt.ylabel("Fitted tilt (deg)")

            plt.sca(ax3)
            for i in range(heights_mean.shape[0]):
                heights = getErrorOnHeighVariation(heights_mean[i, 1], angle_means[i, 1])
                plt.errorbar(heights_mean[i, 0], np.mean(heights), yerr=np.std(heights), color="C0", marker="o", ms=2)

            if 0:
                areas = []
                for i in range(heights_mean.shape[0]):
                    A = getArea(heights_mean[i, 1], angle_means[i, 1])
                    areas.append(A)
                np.savez("height_angle_area%d.npz" % plot_index, areas=areas)
            else:
                data = np.load("height_angle_area%d.npz" % plot_index)
                globals().update(data)
            plt.plot(heights_mean[:, 0], areas/A0, color="C2", marker="o", ms=2)
            #plt.text(heights_mean[i, 0], np.mean(heights), "%.2f" % np.mean(heights), rotation=45, ha="left", va="bottom")
            plt.ylim(0.8, 1.4)
            plt.axhline(1, color="k", ls='--')
            heights = getErrorOnHeighVariation(height, angle)
            #plt.errorbar(heights_mean[i, 0]+1, np.mean(heights), yerr=np.std(heights), color="C3", marker="o")

            if plot_index == 1:
                plt.xlabel("Number of used objects")
            plt.ylabel("Reconstructed\nheights of obj. (m)")

            plt.tight_layout()

        plt.sca(ax0)
        #horizon = cam.getImageHorizon()
        #cam.fixHorizon(horizon)
        cam.fixRoll(0)
        p = cam.fitCamParametersFromObjects(lines=lines)

        horizon = cam.getImageHorizon()
        plt.plot(horizon[0, :], horizon[1, :], '-C1')

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

    import pylustration
    plt.figure(0)

    pylustration.fig_text(1, 1, "A", fontsize=14)
    pylustration.fig_text(1, 1, "B", fontsize=14)
    pylustration.fig_text(1, 1, "C", fontsize=14)
    pylustration.fig_text(1, 1, "D", fontsize=14)
    pylustration.fig_text(1, 1, "E", fontsize=14)
    pylustration.fig_text(1, 1, "F", fontsize=14)

    #% start: automatic generated code from pylustration
    plt.gcf().set_size_inches(16.256000/2.54, 8.000000/2.54, forward=True)
    plt.gcf().axes[0].set_position([0.096918, 0.581996, 0.201901, 0.332067])
    plt.gcf().axes[1].set_position([0.412843, 0.581996, 0.201901, 0.332067])
    plt.gcf().axes[2].set_position([0.769392, 0.581996, 0.201901, 0.332067])
    plt.gcf().axes[3].set_position([0.096918, 0.153157, 0.201901, 0.332067])
    plt.gcf().axes[4].set_position([0.412843, 0.153157, 0.201901, 0.332067])
    plt.gcf().axes[5].set_position([0.769392, 0.153157, 0.201901, 0.332067])
    plt.gcf().texts[0].set_position([0.039062, 0.945860])
    plt.gcf().texts[1].set_position([0.357813, 0.945860])
    plt.gcf().texts[2].set_position([0.632812, 0.945860])
    plt.gcf().texts[3].set_position([0.039062, 0.512739])
    plt.gcf().texts[4].set_position([0.357813, 0.512739])
    plt.gcf().texts[5].set_position([0.632812, 0.512739])
    #% end: automatic generated code from pylustration
    plt.savefig("ObjectNumber.png")
    plt.savefig("ObjectNumber.pdf")
    pylustration.StartPylustration()

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

    im2 = camera.getTopViewOfImage(im, [-200, 100, 0, 300], scaling=10, do_plot=True)
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
