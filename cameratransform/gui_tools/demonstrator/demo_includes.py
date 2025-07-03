import sys
import os
import numpy as np
import cameratransform as ct


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


class Scene9Cubes(ct.Scene):

    def __init__(self, camera):
        ct.Scene.__init__(self)
        self.camera = camera
        self.addGrid(np.arange(-10, 10, 1), np.arange(-10, 10, 1))
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                self.addCube(np.array([i * 2, j * 2, 0]) + np.array([0, 0, 0.5]), 1)


class SceneObjectsOnPlane(ct.Scene):

    def __init__(self, camera):
        ct.Scene.__init__(self)
        self.camera = camera
        for i in range(100):
            self.addCube(np.array([np.random.normal(0, 100), np.random.randint(1, 1000), 0]) + np.array([0, 0, 0.5]), 1)
