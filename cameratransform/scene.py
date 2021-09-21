#!/usr/bin/env python
# -*- coding: utf-8 -*-
# scene.py

# Copyright (c) 2017-2021, Richard Gerum
#
# This file is part of the cameratransform package.
#
# cameratransform is free software: you can redistribute it and/or modify
# it under the terms of the MIT licence.
#
# cameratransform is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
#
# You should have received a copy of the license
# along with cameratransform. If not, see <https://opensource.org/licenses/MIT>

import numpy as np


class Scene:  # pragma: no cover
    objects = []
    camera = None

    def __init__(self):
        self.objects = []

    def setCamera(self, camera):
        self.camera = camera

    def add(self, object):
        self.objects.append(object)

    def plotSceneViews(self):
        import matplotlib.pyplot as plt
        cone = self.camera.getCameraCone()

        plt.subplot(221)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.plot(cone[:, 0], cone[:, 1], "r-")

        plt.subplot(222)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.axis("equal")
        plt.plot(cone[:, 0], cone[:, 2], "r-")

        plt.subplot(223)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.axis("equal")
        plt.plot(cone[:, 1], cone[:, 2], "r-")

        if self.camera is not None:
            plt.subplot(224)
            plt.xlabel("image x")
            plt.ylabel("image y")
            im = np.zeros([self.camera.projection.parameters.image_height_px,
                           self.camera.projection.parameters.image_width_px])
            plt.imshow(im)
            horizon = self.camera.getImageHorizon(np.linspace(0, self.camera.projection.parameters.image_width_px, 100))
            plt.plot(horizon[:, 0], horizon[:, 1], "c-")
            plt.xlim(0, im.shape[1])
            plt.ylim(im.shape[0], 0)

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
        import matplotlib.pyplot as plt
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

    def addCube(self, pos, width):
        points = []
        x, y, z = pos
        r = width/2
        points.append([x + r, y + r, z - r])
        points.append([x + r, y - r, z - r])
        points.append([x - r, y - r, z - r])
        points.append([x - r, y + r, z - r])
        points.append([x + r, y + r, z - r])
        points.append([np.nan, np.nan, np.nan])
        points.append([x + r, y + r, z - r])
        points.append([x + r, y + r, z + r])
        points.append([np.nan, np.nan, np.nan])
        points.append([x + r, y - r, z - r])
        points.append([x + r, y - r, z + r])
        points.append([np.nan, np.nan, np.nan])
        points.append([x - r, y - r, z - r])
        points.append([x - r, y - r, z + r])
        points.append([np.nan, np.nan, np.nan])
        points.append([x - r, y + r, z - r])
        points.append([x - r, y + r, z + r])
        points.append([np.nan, np.nan, np.nan])
        points.append([x + r, y + r, z + r])
        points.append([x + r, y - r, z + r])
        points.append([x - r, y - r, z + r])
        points.append([x - r, y + r, z + r])
        points.append([x + r, y + r, z + r])
        points.append([np.nan, np.nan, np.nan])

        self.add(np.array(points))
        return np.array(points)

    def addRect(self, pos, width):
        points = []
        x, y, z = pos
        r = width / 2
        points.append([x + r, y, z - r])
        points.append([x + r, y, z + r])
        points.append([x - r, y, z + r])
        points.append([x - r, y, z - r])
        points.append([x + r, y, z - r])
        points.append([np.nan, np.nan, np.nan])

        self.add(np.array(points))
        return np.array(points)
