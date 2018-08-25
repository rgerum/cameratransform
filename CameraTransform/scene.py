import numpy as np
import matplotlib.pyplot as plt


class Scene:
    objects = []
    camera = None

    def __init__(self):
        self.objects = []

    def setCamera(self, camera):
        self.camera = camera

    def add(self, object):
        self.objects.append(object)

    def plotSceneViews(self):
        plt.subplot(221)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.subplot(222)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.axis("equal")
        plt.subplot(223)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.axis("equal")
        if self.camera is not None:
            plt.subplot(224)
            plt.xlabel("image x")
            plt.ylabel("image y")
            im = np.zeros(
                [self.camera.projection.parameters.image_height_px, self.camera.projection.parameters.image_width_px])
            plt.imshow(im)
        for object in self.objects:
            plt.subplot(221)
            plt.plot(object[:, 0], object[:, 1], "-")
            plt.subplot(222)
            plt.plot(object[:, 0], object[:, 2], "-")
            plt.subplot(223)
            plt.plot(object[:, 1], object[:, 2], "-")
            if self.camera is not None:
                plt.subplot(224)
                object_im = self.camera.imageFromSpace(object)
                plt.plot(object_im[:, 0], object_im[:, 1], "-")

    def renderImage(self, filename):
        fig = plt.figure(0, (self.camera.projection.parameters.image_width_px / 100,
                             self.camera.projection.parameters.image_height_px / 100))
        ax = plt.axes([0, 0, 1, 1])
        im = np.zeros(
            [self.camera.projection.parameters.image_height_px, self.camera.projection.parameters.image_width_px])
        plt.imshow(im)
        for object in self.objects:
            object_im = self.camera.imageFromSpace(object)
            plt.plot(object_im[:, 0], object_im[:, 1], "-")
        plt.xlim(0, self.camera.projection.parameters.image_width_px)
        plt.ylim(self.camera.projection.parameters.image_height_px, 0)
        plt.savefig(filename, dpi=100)
        fig.clear()
