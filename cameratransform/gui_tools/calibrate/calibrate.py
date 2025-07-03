#!/usr/bin/env python
# -*- coding: utf-8 -*-
# calibrate.py

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

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''


import os
import sys
import getopt
import glob

import cv2 as cv
import numpy as np


def processImage(fn, pattern_size, w=None, h=None, pattern_points=None, output_directory=None):
    print('processing %s... ' % fn)
    name = os.path.splitext(os.path.split(fn)[1])[0]
    corners_file = os.path.join(output_directory, name + '_corners.txt')
    output_image = os.path.join(output_directory, name + '_chess.png')

    # if an output directory is given
    if output_directory:
        # if there already exists a file with the corner data stored, load it
        if os.path.exists(corners_file):
            # load the corners
            corners = np.loadtxt(corners_file).astype("float32")
            print("loaded cached corners", corners_file, corners.shape)
            # check if the number of corners equals the desired count
            if pattern_points is not None and corners.shape[0] != pattern_points.shape[0]:
                return None
            # return the corners
            return corners, pattern_points

    # read the image file
    img = cv.imread(fn, 0)
    if img is None:
        print("Failed to load", fn)
        return None

    # ensure that the image is oriented in landscape format
    if img.shape[1] == h and img.shape[0] == w:
        img = img.transpose(1, 0)

    # ensure that the image has the same size as the rest of the batch
    if w is not None and not (w == img.shape[1] and h == img.shape[0]):
        print("image size not", [w, h], "but instead", [img.shape[1], img.shape[0]], "skipping image")
        return None

    # find the chessboard corners in the images
    found, corners = cv.findChessboardCorners(img, pattern_size)
    if not found:
        print('chessboard not found')
        return None

    # and refine their positions
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
    cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    # flatten the array
    corners = corners.reshape(-1, 2)

    # when an output folder is given
    if output_directory:
        # draw a debug image with the found chessboard corners
        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(vis, pattern_size, corners, found)
        cv.imwrite(output_image, vis)
        # and save the corner positions
        np.savetxt(os.path.join(output_directory, name + '_corners.txt'), corners)

    # return the corners
    print('           %s... OK' % fn)
    return corners, pattern_points


if __name__ == '__main__':
    # load the arguments
    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)
    args.setdefault('--threads', 4)
    if not img_mask:
        print("ERROR: no image filename provided.", file=sys.stderr)
        exit()
    else:
        img_mask = img_mask[0]

    # load all image filenames
    img_names = glob.glob(img_mask)
    if len(img_names) == 0:
        print("ERROR: no images found that match %s." % img_mask, file=sys.stderr)
        exit()
    # optionally create an output directory
    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    # get the size of the chess board (normally not needed)
    square_size = float(args.get('--square_size'))
    print(square_size)

    # create the chess board pattern
    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # get the size of the first image (which should remain the same over the whole batch)
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]

    # process all images (with threads or not)
    threads_num = int(args.get('--threads'))
    if threads_num <= 1:
        chessboards = [processImage(fn, pattern_size, w, h, pattern_points, debug_dir) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool

        pool = ThreadPool(threads_num)
        chessboards = pool.map(lambda x: processImage(x, pattern_size, w, h, pattern_points, debug_dir), img_names)

    # initialize lists of objects and image points
    obj_points = []
    img_points = []

    # split the obtained data in image points and object points
    for data in chessboards:
        if data is None:
            continue
        corners, pattern_points = data
        img_points.append(corners)
        obj_points.append(pattern_points)

    # calculate camera distortion
    print("fit calibration...")
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None,
                                                                      flags=cv.CALIB_ZERO_TANGENT_DIST)

    # split the fitted components
    k1, k2, t1, t2, k3 = dist_coefs.ravel()
    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix.astype("int"))
    print("distortion coefficients: ", dist_coefs.ravel())
    print("focallength_x_px=%f, focallength_y_px=%f, center_x_px=%d, center_y_px=%d, k1=%f, k2=%f, k3=%f"
          % (camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2], k1, k2, k3))

    # undistort the image with the calibration
    print('')
    for fn in img_names if debug_dir else []:
        name = os.path.splitext(os.path.split(fn)[1])[0]
        img_found = os.path.join(debug_dir, name + '_chess.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        # newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        newcameramtx = camera_matrix

        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)

    cv.destroyAllWindows()
