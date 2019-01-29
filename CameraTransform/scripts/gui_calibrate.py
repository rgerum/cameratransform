from __future__ import division, print_function
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import CameraTransform as ct
import glob
import re
import cv2

from qtpy import QtGui, QtCore, QtWidgets

from CameraTransform import QtShortCuts
from CameraTransform.includes.qextendedgraphicsview.QExtendedGraphicsView import QExtendedGraphicsView
from qimage2ndarray import array2qimage, rgb_view
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
                    widget.setFlags(QtCore.Qt.ItemIsSelectable |QtCore.Qt.ItemIsEnabled)
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
        self.corner_path = os.path.join(output_folder, basename+"_corners.txt")
        self.output_image_corners = os.path.join(output_folder, basename+"_chess.png")
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

        result = processImage(self.filename, pattern_size, None, None, pattern_points, output_directory=self.output_folder)
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
        if self.image_data is not None:
            return self.image_data
        if os.path.exists(self.output_image_corners):
            self.image_data = (plt.imread(self.output_image_corners)*255).astype(np.uint8)
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

        # widget layout and elements
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setWindowTitle("Camera Calibrator")

        main_layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(main_layout)

        layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(layout)

        self.input_filename = QtShortCuts.QInputFilename(layout, "Filebatch", os.getcwd(), file_type="Images (*.jpg *.png)", existing=True)
        self.input_filename.valueChanged.connect(self.loadBatch)

        self.tableWidget = MyQTable()
        layout.addWidget(self.tableWidget)

        self.tableWidget.itemSelectionChanged.connect(self.imageSelected)
        self.tableWidget.cellDoubleClicked.connect(self.imageDoubleClicked)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(["File", "Processed", "Use"])
        self.tableWidget.setMinimumWidth(350)

        self.button_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.button_layout)

        self.button_clear = QtWidgets.QPushButton("Clear")
        self.button_clear.clicked.connect(self.clearImages)
        self.button_layout.addWidget(self.button_clear)

        self.button_start = QtWidgets.QPushButton("Process")
        self.button_start.clicked.connect(self.processImages)
        self.button_layout.addWidget(self.button_start)

        layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(layout)

        # view/scene setup
        self.view = QExtendedGraphicsView(dropTarget=self)
        layout.addWidget(self.view)
        # self.view.zoomEvent = self.zoomEvent
        self.local_scene = self.view.scene
        self.origin = self.view.origin

        self.preview_pixMapItem = QtWidgets.QGraphicsPixmapItem(self.origin)

        self.view2 = QExtendedGraphicsView(dropTarget=self)
        layout.addWidget(self.view2)
        # self.view.zoomEvent = self.zoomEvent
        self.local_scene2 = self.view2.scene
        self.origin2 = self.view2.origin

        self.preview_pixMapItem2 = QtWidgets.QGraphicsPixmapItem(self.origin2)

        self.input_display_undistorted = QtShortCuts.QInputBool(layout, "Display Corrected (D)", True)
        self.input_display_undistorted.valueChanged.connect(self.displayImage)

        self.progessBar = QtWidgets.QProgressBar()
        layout.addWidget(self.progessBar)

        layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(layout)

        self.input_fix_center = QtShortCuts.QInputBool(layout, "Fit Center")
        self.input_fix_aspect = QtShortCuts.QInputBool(layout, "Fit Aspect Ratio")
        self.input_fix_k1 = QtShortCuts.QInputBool(layout, "Fit K1", True)
        self.input_fix_k2 = QtShortCuts.QInputBool(layout, "Fit K2", True)
        self.input_fix_k3 = QtShortCuts.QInputBool(layout, "Fit K3", True)

        self.button_fit = QtWidgets.QPushButton("Fit Calibration")
        self.button_fit.clicked.connect(self.fitCalibration)
        layout.addWidget(self.button_fit)

        self.calibration_result = QtWidgets.QLabel()
        layout.addWidget(self.calibration_result)

        layout.addStretch()

        self.plot = QtShortCuts.MatplotlibWidget()
        layout.addWidget(self.plot)

        self.button_save = QtWidgets.QPushButton("Save Calibration")
        self.button_save.clicked.connect(self.saveCalibration)
        self.button_save.setDisabled(True)
        layout.addWidget(self.button_save)
        self.button_load = QtWidgets.QPushButton("Load Calibration")
        self.button_load.clicked.connect(self.loadCalibration)
        layout.addWidget(self.button_load)

        self.camera = ct.Camera(ct.RectilinearProjection(), lens=ct.BrownLensDistortion())

        self.images = []

        self.loadBatch("D:/Repositories/CameraTransform/CameraTransform/scripts/example_data\P*.JPG")

    def clearImages(self):
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
        print("fit calibration...")
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
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (self.w, self.h), None, None,
                                                                          flags=flags)

        # split the fitted components
        k1, k2, t1, t2, k3 = dist_coefs.ravel()
        print("\nRMS:", rms)
        print("camera matrix:\n", camera_matrix.astype("int"))
        print("distortion coefficients: ", dist_coefs.ravel())
        print("focallength_x_px=%f, focallength_y_px=%f, center_x_px=%d, center_y_px=%d, k1=%f, k2=%f, k3=%f"
              % (camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2], k1, k2, k3))
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
        self.calibration_result.setText("\n".join(str(key)+": "+str(value) for key, value in self.calibration.items()))
        self.camera.parameters.set_fit_parameters(self.calibration.keys(), self.calibration.values())
        self.camera.map_undistort = None
        self.button_save.setDisabled(False)

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

        for image in self.images:
            image.resetCalibration()
        self.imageSelected()

    def saveCalibration(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(None, "Save Calibration", os.getcwd(), "Text File (*.txt)")

        # get the string
        if isinstance(filename, tuple):  # Qt5
            filename = filename[0]
        else:  # Qt4
            filename = str(filename)

        import json
        with open(filename, "w") as fp:
            json.dump(self.calibration, fp)

        QtWidgets.QMessageBox.information(self, 'Saved',
                                       "The calibration has been saved to %s" % filename,
                                       QtWidgets.QMessageBox.Ok)

    def loadCalibration(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(None, "Load Calibration", os.getcwd(), "Text File (*.txt)")

        # get the string
        if isinstance(filename, tuple):  # Qt5
            filename = filename[0]
        else:  # Qt4
            filename = str(filename)

        import json
        with open(filename, "r") as fp:
            self.calibration = json.load(fp)
        self.newCalibration()

    def processImages(self):
        self.process_image_index = 0
        self.processNextImage()

    def processNextImage(self):
        import threading
        if self.process_image_index >= len(self.images):
            return
        image = self.images[self.process_image_index]
        self.process_image_index += 1
        image.processing_finished.connect(self.processNextImage)
        self._run_thread = threading.Thread(target=image.processImage, args=())
        self._run_thread.daemon = True
        self._run_thread.start()

    def updateTable(self):
        self.tableWidget.setList([image.getData() for image in self.images])

    def loadBatch(self, filename):
        directory, filename = os.path.split(filename)
        filename = os.path.join(directory, re.sub("\d+", "*", filename))
        output_directory = os.path.join(directory, "output")

        self.images += [MyImage(file, output_directory) for file in glob.glob(filename)]
        for index, image in enumerate(self.images):
            image.table_row = index
            image.table = self.tableWidget

        self.updateTable()

        for image in self.images:
            image.setTableRow()

        im = self.images[0].getImage()
        self.preview_pixMapItem.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.w = im.shape[1]
        self.h = im.shape[0]
        self.view.setExtend(self.w, self.h)
        self.view.fitInView()

        self.preview_pixMapItem2.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view2.setExtend(self.w, self.h)
        self.view2.fitInView()

    def imageDoubleClicked(self, row, col):
        im = self.images[row]
        if col == 2:
            if im.corner_data is not None:
                im.use = not im.use
                im.setTableRow()

    def imageSelected(self):
        try:
            index = self.tableWidget.selectedIndexes()[0].row()
        except IndexError:
            index = 0
        im = self.images[index]
        im.loading_image_finished.connect(self.displayImage)
        camera = self.camera
        if self.calibration is None:
            camera = None
        import threading
        self._run_thread = threading.Thread(target=im.loadImage, args=(camera,))
        self._run_thread.daemon = True
        self._run_thread.start()
        self.progessBar.setRange(0, 0)

    def displayImage(self):
        try:
            index = self.tableWidget.selectedIndexes()[0].row()
        except IndexError:
            index = 0
        image = self.images[index]
        if image.image_data is not None:
            self.preview_pixMapItem.setPixmap(QtGui.QPixmap(array2qimage(image.image_data)))
        if self.input_display_undistorted.value() is True:
            if image.image_data_undistorted is not None:
                self.preview_pixMapItem2.setPixmap(QtGui.QPixmap(array2qimage(image.image_data_undistorted)))
            else:
                self.preview_pixMapItem2.setPixmap(QtGui.QPixmap(array2qimage(image.image_data)))
        else:
            self.preview_pixMapItem2.setPixmap(QtGui.QPixmap(array2qimage(image.image_data)))
            self.view.setDisabled(True)
        self.progessBar.setRange(0, 1)
        self.progessBar.setValue(1)

    def keyPressEvent(self, event):
        # @key ---- General ----
        if event.key() == QtCore.Qt.Key_F:
            # @key F: fit image to view
            self.view.fitInView()
            self.view2.fitInView()

        if event.key() == QtCore.Qt.Key_D:
            self.input_display_undistorted.setValue(not self.input_display_undistorted.value())
            self.displayImage()

def startDemonstratorGUI():
    cam = ct.Camera(ct.RectilinearProjection(focallength_px=3860, image=[4608, 2592]))

    app = QtWidgets.QApplication(sys.argv)

    window = Window()
    window.show()
    app.exec_()


if __name__ == '__main__':
    startDemonstratorGUI()
