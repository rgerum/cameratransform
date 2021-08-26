#!/usr/bin/env python
# -*- coding: utf-8 -*-
# upload_pypi.py

# Copyright (c) 2017-2019, Richard Gerum
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

from __future__ import print_function, division

import os
import sys

current_folder = os.getcwd()

try:
    # go to parent folder
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cameratransform"))
    import cameratransform

    current_version = cameratransform.__version__

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-u", "--username", action="store", dest="username")
    parser.add_option("-p", "--password", action="store", dest="password")
    (options, args) = parser.parse_args()

    # upgrade twine (used for uploading)
    os.system("pip install twine --upgrade")

    # pack cameratransform
    os.system("python setup.py sdist")

    # the command
    command_string = "twine upload dist/cameratransform-%s.tar.gz" % current_version
    # optionally add the username
    if options.username:
        command_string += " --username %s" % options.username
    # optionally add the password
    if options.password:
        command_string += " --password %s" % options.password
    # print the command string
    print(command_string)
    # and execute it
    os.system(command_string)
finally:
    os.chdir(current_folder)
