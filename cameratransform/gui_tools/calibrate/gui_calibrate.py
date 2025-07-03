#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gui_calibrate.py

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

import glob
import json
import os
import re
import sys
import threading

import cv2
import numpy as np
from qimage2ndarray import array2qimage
from qtpy import QtGui, QtCore, QtWidgets

import cameratransform as ct
from cameratransform.gui_tools.demonstrator import QtShortCuts
from cameratransform.gui_tools.calibrate.QExtendedGraphicsView import QExtendedGraphicsView

sys.path.insert(0, os.path.dirname(__file__))
from calibrate import processImage

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


class MyQTable(QtWidgets.QTableWidget):
    def __init__(self):
        QtWidgets.QTableWidget.__init__(self)

    def setList(self, list):
        self.setRowCount(len(list))
        for row_id, row in enumerate(list):
            self.setColumnCount(len(row))
            for column_id, item in enumerate(row):
                widget = self.item(row_id, column_id)
                if widget is None:
                    widget = QtWidgets.QTableWidgetItem()
                    widget.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    self.setItem(row_id, column_id, widget)
                widget.setText(str(item))


class MyImage(QtWidgets.QWidget):
    status_changed = QtCore.Signal()
    processing_finished = QtCore.Signal()
    loading_image_finished = QtCore.Signal()
    corner_data = None
    status = "False"
    use = False

    image_data = None
    image_data_undistorted = None

    def __init__(self, filename, output_folder):
        QtWidgets.QWidget.__init__(self)
        self.filename = filename
        self.name = os.path.split(filename)[1]

        basename = os.path.splitext(self.name)[0]
        self.corner_path = os.path.join(output_folder, basename + "_corners.txt")
        self.output_image_corners = os.path.join(output_folder, basename + "_chess.png")
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        if self.getProcessed():
            self.status = "True"
            self.use = True

        self.status_changed.connect(self.setTableRow)

        self.icon = self.createIcon()

    def createIcon(self):
        pixmap = QtGui.QPixmap(self.filename)
        return QtGui.QIcon(pixmap)

    def setTableRow(self):
        self.table.item(self.table_row, 0).setIcon(self.icon)
        for column_id, item in enumerate(self.getData()):
            self.table.item(self.table_row, column_id).setText(str(item))

    def getProcessed(self):
        if os.path.exists(self.corner_path):
            self.corner_data = np.loadtxt(self.corner_path).astype("float32")
            pattern_size, pattern_points = self.getPatternPoints()
            self.pattern_points = pattern_points
            return True
        return False

    def getData(self):
        return [self.name, self.status, "X" if self.use else ""]

    def getPatternPoints(self):
        square_size = 1

        # create the chess board pattern
        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        return pattern_size, pattern_points

    def processImage(self):
        self.status = "in progress"
        self.status_changed.emit()

        pattern_size, pattern_points = self.getPatternPoints()

        result = processImage(self.filename, pattern_size, None, None, pattern_points,
                              output_directory=self.output_folder)
        if result is None:
            self.status = "Failed"
        else:
            corners, pattern_points = result
            self.corner_data = corners
            self.pattern_points = pattern_points
            self.status = "True"
            self.use = True
        self.status_changed.emit()
        self.processing_finished.emit()

    def getImage(self):
        import matplotlib.pyplot as plt
        if self.image_data is not None:
            return self.image_data
        if os.path.exists(self.output_image_corners):
            self.image_data = (plt.imread(self.output_image_corners) * 255).astype(np.uint8)
        else:
            self.image_data = plt.imread(self.filename)
        return self.image_data

    def getUndistortedImage(self, camera):
        if self.image_data_undistorted is None:
            if camera is None:
                self.image_data_undistorted = self.image_data
            else:
                self.image_data_undistorted = camera.undistortImage(self.image_data)
        return self.image_data_undistorted

    def resetCalibration(self):
        self.image_data_undistorted = None

    def loadImage(self, camera):
        self.getImage()
        self.getUndistortedImage(camera)
        self.loading_image_finished.emit()


class Window(QtWidgets.QWidget):
    calibration = None

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # the camera
        self.camera = ct.Camera(ct.RectilinearProjection(image=[1920, 1080], focallength_px=1200), lens=ct.BrownLensDistortion())
        # the list of loaded images
        self.images = []

        # widget layout and elements
        self.setMinimumWidth(1300)
        self.setMinimumHeight(400)
        self.setWindowTitle("Camera Calibrator")

        # add the main layout
        main_layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(main_layout)

        """ the left pane with the image list """
        layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(layout)

        # the table
        self.tableWidget = MyQTable()
        layout.addWidget(self.tableWidget)

        self.tableWidget.itemSelectionChanged.connect(self.imageSelected)
        self.tableWidget.cellDoubleClicked.connect(self.imageDoubleClicked)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(["File", "Processed", "Use"])
        self.tableWidget.setMinimumWidth(350)

        # the button layout
        self.button_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.button_layout)

        # the batch loader
        self.input_filename = QtShortCuts.QInputFilename(self.button_layout, None, os.getcwd(), button_text="Add Images",
                                                         file_type="Images (*.jpg *.png)", existing=True, just_button=True)
        self.input_filename.valueChanged.connect(self.loadBatch)

        # clear button
        self.button_clear = QtWidgets.QPushButton("Clear Images")
        self.button_clear.clicked.connect(self.clearImages)
        self.button_layout.addWidget(self.button_clear)

        # process button
        self.button_start = QtWidgets.QPushButton("Process")
        self.button_start.clicked.connect(self.processImages)
        self.button_layout.addWidget(self.button_start)

        """ the middle pane with the image display """

        layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(layout)

        # view/scene setup
        self.view = QExtendedGraphicsView(dropTarget=self)
        layout.addWidget(self.view)
        self.origin = self.view.origin

        self.preview_pixMapItem = QtWidgets.QGraphicsPixmapItem(self.origin)

        self.view2 = QExtendedGraphicsView(dropTarget=self)
        layout.addWidget(self.view2)
        self.origin2 = self.view2.origin

        self.preview_pixMapItem2 = QtWidgets.QGraphicsPixmapItem(self.origin2)

        self.input_display_undistorted = QtShortCuts.QInputBool(layout, "Display Corrected (D)", True)
        self.input_display_undistorted.valueChanged.connect(self.displayImage)

        self.progessBar = QtWidgets.QProgressBar()
        layout.addWidget(self.progessBar)

        """ the right pane with the fit data """

        layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(layout)

        # calibration flags
        self.input_fix_center = QtShortCuts.QInputBool(layout, "Fit Center")
        self.input_fix_aspect = QtShortCuts.QInputBool(layout, "Fit Aspect Ratio")
        self.input_fix_k1 = QtShortCuts.QInputBool(layout, "Fit K1", True)
        self.input_fix_k2 = QtShortCuts.QInputBool(layout, "Fit K2", True)
        self.input_fix_k3 = QtShortCuts.QInputBool(layout, "Fit K3", True)

        # start calibration button
        self.button_fit = QtWidgets.QPushButton("Fit Calibration")
        self.button_fit.clicked.connect(self.fitCalibration)
        layout.addWidget(self.button_fit)

        # a label to display the results
        self.calibration_result = QtWidgets.QLabel()
        layout.addWidget(self.calibration_result)

        # a strech
        layout.addStretch()

        # the plot of the result
        self.plot = QtShortCuts.MatplotlibWidget()
        layout.addWidget(self.plot)

        # the button layout
        self.button_layout2 = QtWidgets.QHBoxLayout()
        layout.addLayout(self.button_layout2)

        # save button
        self.button_save = QtShortCuts.QInputFilename(self.button_layout2, None, os.getcwd(),
                                                      button_text="Save Calibration",
                                                      file_type="Text File (*.txt)", existing=False,
                                                      just_button=True)
        self.button_save.valueChanged.connect(self.saveCalibration)
        self.button_save.setDisabled(True)

        # load button
        self.button_load = QtShortCuts.QInputFilename(self.button_layout2, None, os.getcwd(),
                                                      button_text="Load Calibration",
                                                      file_type="Text File (*.txt)", existing=True,
                                                      just_button=True)
        self.button_load.valueChanged.connect(self.loadCalibration)

    def clearImages(self):
        # remove all images from the list
        self.images = []
        self.updateTable()

    def fitCalibration(self):
        # initialize lists of objects and image points
        obj_points = []
        img_points = []

        # split the obtained data in image points and object points
        for image in self.images:
            if image.corner_data is None or image.use is False:
                continue
            img_points.append(image.corner_data)
            obj_points.append(image.pattern_points)

        # calculate camera distortion
        flags = cv2.CALIB_ZERO_TANGENT_DIST
        if self.input_fix_aspect.value() is False:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        if self.input_fix_center.value() is False:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if self.input_fix_k1.value() is False:
            flags |= cv2.CALIB_FIX_K1
        if self.input_fix_k2.value() is False:
            flags |= cv2.CALIB_FIX_K2
        if self.input_fix_k3.value() is False:
            flags |= cv2.CALIB_FIX_K3
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (self.w, self.h),
                                                                           None, None,
                                                                           flags=flags)

        # split the fitted components
        k1, k2, t1, t2, k3 = dist_coefs.ravel()

        self.calibration = dict(rms=rms,
                                image_width_px=self.w,
                                image_height_px=self.h,
                                focallength_x_px=camera_matrix[0, 0],
                                focallength_y_px=camera_matrix[1, 1],
                                center_x_px=camera_matrix[0, 2],
                                center_y_px=camera_matrix[1, 2],
                                k1=k1,
                                k2=k2,
                                k3=k3)
        self.newCalibration()

    def newCalibration(self):
        import matplotlib.pyplot as plt
        # a new calibration has been fitted or loaded

        # display the calibration data
        self.calibration_result.setText(
            "\n".join(str(key) + ": " + str(value) for key, value in self.calibration.items()))
        # update the camera
        self.camera.parameters.set_fit_parameters(self.calibration.keys(), self.calibration.values())
        # remove the cached undistort map
        self.camera.map_undistort = None
        # enable the save button
        self.button_save.setDisabled(False)

        self.w = self.calibration["image_width_px"]
        self.h = self.calibration["image_height_px"]

        # plot the calibration curve
        x = np.arange(0, self.w)
        points = np.vstack((x, np.zeros_like(x))).T
        transformed = self.camera.lens.distortedFromImage(points)
        plt.cla()
        plt.plot(x, x, "-k")
        plt.plot(x, transformed[:, 0])
        plt.ylabel("raw pixel x")
        plt.xlabel("corrected pixel x position")
        plt.ylim(0, self.w)
        plt.axis("equal")
        plt.tight_layout()
        plt.draw()

        # notify all images that the calibration has been changed
        for image in self.images:
            image.resetCalibration()
        # update the currently displayed image
        self.imageSelected()

    def saveCalibration(self, filename):
        # save the calibration as a json file
        with open(filename, "w") as fp:
            json.dump(self.calibration, fp)

        # inform the user that the data was saved
        QtWidgets.QMessageBox.information(self, 'Saved', "The calibration has been saved to %s" % filename,
                                          QtWidgets.QMessageBox.Ok)

    def loadCalibration(self, filename):
        # load the data
        with open(filename, "r") as fp:
            self.calibration = json.load(fp)

        # update the display with the new calibration data
        self.newCalibration()

    def processImages(self):
        # aysnchronously process all images, start with the first
        self.process_image_index = 0
        self.processNextImage()

    def processNextImage(self):
        # if no images are left, abort
        if self.process_image_index >= len(self.images):
            return
        # get the next image
        image = self.images[self.process_image_index]
        # and schedule the next image
        self.process_image_index += 1
        image.processing_finished.connect(self.processNextImage)
        # start the processing in a second thread
        self._run_thread = threading.Thread(target=image.processImage, args=(), daemon=True)
        self._run_thread.start()

    def updateTable(self):
        # update the image list table
        self.tableWidget.setList([image.getData() for image in self.images])
        for image in self.images:
            image.setTableRow()

    def loadBatch(self, filename):
        # load a batch of images from a folder
        # split the filename from the directory
        directory, filename = os.path.split(filename)
        # replace numbers with * (for glob)
        filename = os.path.join(directory, re.sub("\\d+", "*", filename))
        # define the output directory (as a subdirectory "output")
        output_directory = os.path.join(directory, "output")

        # create MyImage objects with the filenames
        self.images += [MyImage(file, output_directory) for file in glob.glob(filename)]
        # and set their table and table_row index
        for index, image in enumerate(self.images):
            image.table_row = index
            image.table = self.tableWidget

        # update the table
        self.updateTable()

        # load the first image
        im = self.images[0].getImage()
        # store the dimensions of the image
        self.w = im.shape[1]
        self.h = im.shape[0]
        # set the ranges of the view1
        self.view.setExtend(self.w, self.h)
        self.view.fitInView()
        # set the ranges of the view2
        self.view2.setExtend(self.w, self.h)
        self.view2.fitInView()
        # display the images
        self.displayImage()

    def imageDoubleClicked(self, row, col):
        # double clicking on the use row toggles the state
        if col == 2:
            im = self.images[row]
            # if the image can be used
            if im.corner_data is not None:
                # toggle the state and update the table row
                im.use = not im.use
                im.setTableRow()

    def imageSelected(self):
        # get the Image object
        image = self.getSelectedImage()
        if image is None:
            return
        # schedule loading the data
        image.loading_image_finished.connect(self.displayImage)
        # get the camera if there is a calibration
        camera = self.camera
        if self.calibration is None:
            camera = None

        # start loading the image in a second thread
        self._run_thread = threading.Thread(target=image.loadImage, args=(camera,), daemon=True)
        self._run_thread.start()
        self.progessBar.setRange(0, 0)

    def getSelectedImage(self):
        # get the selected row
        try:
            return self.images[self.tableWidget.selectedIndexes()[0].row()]
        # default to 0
        except IndexError:
            if len(self.images):
                return self.images[0]
            else:
                return None

    def displayImage(self):
        # get the current Image
        image = self.getSelectedImage()
        if image is None:
            return
        # if the image data is there
        if image.image_data is not None:
            # set it to the display
            self.preview_pixMapItem.setPixmap(QtGui.QPixmap(array2qimage(image.image_data)))

        # if there is an undistorted image and the bock is ticked, display the undistorted in the second view
        if self.input_display_undistorted.value() is True and image.image_data_undistorted is not None:
            self.preview_pixMapItem2.setPixmap(QtGui.QPixmap(array2qimage(image.image_data_undistorted)))
        # if not, display the normal image in the second view, too
        else:
            self.preview_pixMapItem2.setPixmap(QtGui.QPixmap(array2qimage(image.image_data)))
        # stop the progress bar
        self.progessBar.setRange(0, 1)
        self.progessBar.setValue(1)

    def keyPressEvent(self, event):
        # @key ---- General ----
        if event.key() == QtCore.Qt.Key_F:
            # @key F: fit image to view
            self.view.fitInView()
            self.view2.fitInView()

        if event.key() == QtCore.Qt.Key_D:
            # @key D: switch between display of distorted and undistorted image
            self.input_display_undistorted.setValue(not self.input_display_undistorted.value())
            self.displayImage()


def startCalibrationGUI():
    app = QtWidgets.QApplication(sys.argv)

    window = Window()
    window.show()
    app.exec_()


if __name__ == '__main__':
    startCalibrationGUI()
