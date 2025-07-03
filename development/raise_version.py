#!/usr/bin/env python
# -*- coding: utf-8 -*-
# raise_version.py

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

import os, sys
import shutil
import glob
import fnmatch
import re
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cameratransform"))
import cameratransform
current_version = cameratransform.__version__

def CheckForUncommitedChanges(directory):
    old_dir = os.getcwd()
    os.chdir(directory)
    uncommited = os.popen("hg status -m").read().strip()
    if uncommited != "":
        print("ERROR: uncommited changes in repository", directory)
        sys.exit(1)
    os.system("hg pull -u")
    os.chdir(old_dir)

def RelaceVersion(file, version_old, version_new):
    print("change in file", file)
    with open(file, "r") as fp:
        data = fp.readlines()
    with open(file, "w") as fp:
        for line in data:
            fp.write(line.replace(version_old, version_new))

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-v", "--version", action="store", type="string", dest="version")
parser.add_option("-t", "--test", action="store_false", dest="release", default=False)
parser.add_option("-r", "--release", action="store_true", dest="release")
parser.add_option("-u", "--username", action="store", dest="username")
parser.add_option("-p", "--password", action="store", dest="password")
(options, args) = parser.parse_args()
if options.version is None and len(args):
    options.version = args[0]

print("raise version started ... (current version is %s)" % current_version)
# go to parent directory
os.chdir("..")

# check for new version name as command line argument
new_version = None
new_version = options.version

if new_version is None:
    print("ERROR: no version number supplied. Use 'raise_version.py 0.9' to release as version 0.9")
    sys.exit(1)

# check if new version name differs
if options.release and current_version == new_version:
    print("ERROR: new version is the same as old version")
    sys.exit(1)

print("Setting version number to", new_version)

# check for uncommited changes
#if options.release:
#    for path in paths:
#        CheckForUncommitedChanges(path)
#    CheckForUncommitedChanges(path_to_website)

""" Let's go """
RelaceVersion("pyproject.toml", current_version, new_version)
#RelaceVersion("meta.yaml", current_version, new_version)
RelaceVersion("docs/source/conf.py", current_version, new_version)
RelaceVersion("cameratransform/__init__.py", current_version, new_version)

if options.release:
    # commit changes
    os.system("git add setup.py docs/conf.py cameratransform/__init__.py")
    os.system("git commit -m \"set version to v%s\"" % new_version)
    os.system("git tag \"v%s\"" % new_version)

print("version raise completed!")
