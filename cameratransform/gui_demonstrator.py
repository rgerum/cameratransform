#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gui_demonstrator.py

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

import sys
import os
import numpy as np
import cameratransform as ct

from qtpy import QtGui, QtCore, QtWidgets

sys.path.insert(0, os.path.dirname(__file__))
import QtShortCuts


def getClassDefinitions(module, baseclass):
    class_definitions = []
    for name in dir(module):
        current_class_definition = getattr(module, name)
        try:
            if issubclass(current_class_definition, baseclass) and current_class_definition != baseclass:
                class_definitions.append(current_class_definition)
        except TypeError:
            pass
    return class_definitions

def getClassDefinitionsDict(module, baseclass):
    class_definitions = []
    for name in module:
        current_class_definition = module[name]
        try:
            if issubclass(current_class_definition, baseclass) and current_class_definition != baseclass:
                class_definitions.append(current_class_definition)
        except TypeError:
            pass
    return class_definitions


class Window(QtWidgets.QWidget):
    def __init__(self, cam, scene):
        QtWidgets.QWidget.__init__(self)

        # widget layout and elements
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setWindowTitle("Camera Tester")

        self.scene = scene
        self.cam = cam

        main_layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(main_layout)

        layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(layout)

        self.plot_widget = QtShortCuts.MatplotlibWidget(self)
        layout.addWidget(self.plot_widget)

        layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(layout)

        self.scenes = getClassDefinitionsDict(globals(), ct.Scene)
        self.scenes = {p.__name__: p for p in self.scenes}
        print(self.scenes)
        self.select_scene = QtShortCuts.QInputChoice(layout, "Scene", self.scene.__class__.__name__,
                                                              values=list(self.scenes.keys()))

        self.projections = getClassDefinitions(ct, ct.CameraProjection)
        self.projections = {p.__name__: p for p in self.projections}
        self.select_transformation = QtShortCuts.QInputChoice(layout, "Projection", "RectilinearProjection", values=list(self.projections.keys()))

        parameters = [
            dict(name="focallength_x_px", value=cam.focallength_x_px, min=0, max=5000),
            dict(name="focallength_y_px", value=cam.focallength_y_px, min=0, max=5000),
            dict(name="loc_aspect_ratio", value=True, focalloc=True),
            None,
            dict(name="center_x_px", value=cam.center_x_px, min=0, max=5000),
            dict(name="center_y_px", value=cam.center_y_px, min=0, max=5000),
            dict(name="center_image", value=True, centerimage=True),
            None,
            dict(name="image_width_px", value=cam.image_width_px, min=0, max=5000),
            dict(name="image_height_px", value=cam.image_height_px, min=0, max=5000),
            None,
            dict(name="heading_deg", value=0., min=-180, max=180),
            dict(name="tilt_deg", value=0., min=-180, max=180),
            dict(name="roll_deg", value=0., min=-180, max=180),
            None,
            dict(name="pos_x_m", value=0., min=-10, max=10),
            dict(name="pos_y_m", value=0., min=-10, max=10),
            dict(name="elevation_m", value=5., min=-10, max=10),

        ]
        self.sliders = {}
        for param in parameters:
            if param is None:
                QtShortCuts.QHLine(layout)
            elif "focalloc" in param:
                self.focalloc = QtShortCuts.QInputBool(layout, param["name"], value=param["value"])
                self.focalloc.valueChanged.connect(self.updatePlot)
            elif "centerimage" in param:
                self.center_image = QtShortCuts.QInputBool(layout, param["name"], value=param["value"])
                self.center_image.valueChanged.connect(self.updatePlot)
            else:
                slider = QtShortCuts.QInputNumber(layout, param["name"], param["value"], min=param["min"], max=param["max"], use_slider=True, float=isinstance(param["value"], float))
                slider.valueChanged.connect(self.updatePlot)
                self.sliders[param["name"]] = slider
        layout.addStretch()

        self.select_transformation.valueChanged.connect(self.updatePlot)
        self.select_scene.valueChanged.connect(self.updatePlot)

        self.updatePlot()

    def updatePlot(self):
        import matplotlib.pyplot as plt
        if self.select_scene.value() != self.scene.__class__.__name__:
            self.scene = self.scenes[self.select_scene.value()](self.cam)

        if self.focalloc.value() is True:
            self.sliders["focallength_y_px"].setDisabled(True)
            self.sliders["focallength_y_px"].setValue(self.sliders["focallength_x_px"].value())
        else:
            self.sliders["focallength_y_px"].setDisabled(False)

        self.sliders["center_x_px"].setDisabled(self.center_image.value())
        self.sliders["center_y_px"].setDisabled(self.center_image.value())
        if self.center_image.value() is True:
            self.sliders["center_x_px"].setValue(self.sliders["image_width_px"].value()/2)
            self.sliders["center_y_px"].setValue(self.sliders["image_height_px"].value()/2)

        if self.select_transformation.value() != self.cam.projection.__class__.__name__:
            self.cam.projection.__class__ = self.projections[self.select_transformation.value()]
        self.cam.parameters.set_fit_parameters(self.sliders.keys(), [s.value() for s in self.sliders.values()])
        print(self.cam)
        self.scene.camera = self.cam
        plt.clf()

        self.scene.plotSceneViews()
        plt.draw()


class Scene9Cubes(ct.Scene):

    def __init__(self, camera):
        ct.Scene.__init__(self)
        self.camera = camera
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                self.addCube(np.array([i * 2, j * 2, -5]) + np.array([0, 0, 0.5]), 1)


class SceneObjectsOnPlane(ct.Scene):

    def __init__(self, camera):
        ct.Scene.__init__(self)
        self.camera = camera
        for i in range(100):
            self.addCube(np.array([np.random.normal(0, 100), np.random.randint(1, 1000), 0]) + np.array([0, 0, 0.5]), 1)


def startDemonstratorGUI():
    cam = ct.Camera(ct.RectilinearProjection(focallength_px=3860, image=[4608, 2592]))

    app = QtWidgets.QApplication(sys.argv)

    window = Window(cam, Scene9Cubes(cam))
    window.show()
    app.exec_()


if __name__ == '__main__':
    startDemonstratorGUI()
