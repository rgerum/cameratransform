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

    def plotSceneViews(self, axes=None):
        import matplotlib.pyplot as plt
        cone = self.camera.getCameraCone()

        if axes is None:
            axes = [plt.subplot(221), plt.subplot(222), plt.subplot(223), plt.subplot(224)]

        plt.sca(axes[0])
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.plot(cone[:, 0], cone[:, 1], "r-")

        plt.sca(axes[1])
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.axis("equal")
        plt.plot(cone[:, 0], cone[:, 2], "r-")

        plt.sca(axes[2])
        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        plt.axis("equal")
        plt.plot(cone[:, 1], cone[:, 2], "r-")

        if self.camera is not None:
            plt.sca(axes[3])
            plt.xlabel("image x (px)")
            plt.ylabel("image y (px)")
            im = np.zeros([2, 2])
            plt.imshow(im, extent=[0, self.camera.projection.parameters.image_width_px, self.camera.projection.parameters.image_height_px, 0], vmin=0, vmax=255)
            horizon = self.camera.getImageHorizon(np.linspace(0, self.camera.projection.parameters.image_width_px, 100))
            plt.plot(horizon[:, 0], horizon[:, 1], "c-")
            plt.xlim(0, self.camera.projection.parameters.image_width_px)
            plt.ylim(self.camera.projection.parameters.image_height_px, 0)

        for object in self.objects:
            plt.sca(axes[0])
            plt.plot(object[:, 0], object[:, 1], "-")
            plt.sca(axes[1])
            plt.plot(object[:, 0], object[:, 2], "-")
            plt.sca(axes[2])
            plt.plot(object[:, 1], object[:, 2], "-")
            if self.camera is not None:
                plt.sca(axes[3])
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

    def addGrid(self, x_values, y_values):
        points = []
        min_x = min(x_values)
        max_x = max(x_values)
        for y in y_values:
            points.append([min_x, y, 0])
            points.append([max_x, y, 0])
            points.append([np.nan, np.nan, np.nan])
        min_y = min(y_values)
        max_y = max(y_values)
        for x in x_values:
            points.append([x, min_y, 0])
            points.append([x, max_y, 0])
            points.append([np.nan, np.nan, np.nan])

        self.add(np.array(points))
        return np.array(points)