#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py

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

from setuptools import setup

setup(name='cameratransform',
      version="1.2",
      description='Projects coordinates from 2D to 3D and can fit camera parameters',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      license='MIT',
      packages=['cameratransform'],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'tqdm',
      ],
      extras_require={
        'projecting_top_view':  ["cv2", "matplotlib"],
        'exif_extraction':  ["pillow", "requests"]
      }
      )
