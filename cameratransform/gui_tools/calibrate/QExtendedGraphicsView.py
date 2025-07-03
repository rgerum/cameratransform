#!/usr/bin/env python
# -*- coding: utf-8 -*-
# QExtendedGraphicsView.py

# Copyright (c) 2015-2022, Richard Gerum, Sebastian Richter, Alexander Winterl
#
# This file is part of ClickPoints.
#
# ClickPoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ClickPoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ClickPoints. If not, see <http://www.gnu.org/licenses/>

import sys

from qtpy import QtGui, QtCore, QtWidgets

import numpy as np


def PosToArray(pos):
    return np.array([pos.x(), pos.y()])


class MyScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent, dropTarget=None):
        QtWidgets.QGraphicsScene.__init__(self, parent)
        if dropTarget is None:
            self.setAcceptDrops(False)
        self.dropTarget = dropTarget

    def dragEnterEvent(self, e):
        if self.dropTarget:
            return self.dropTarget.dragEnterEvent(e)

        e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if self.dropTarget:
            return self.dropTarget.dragMoveEvent(e)
        e.acceptProposedAction()

    def dropEvent(self, e):
        if self.dropTarget:
            return self.dropTarget.dropEvent(e)
        self.parent()
        e.accept()

class QExtendedGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, dropTarget=None):
        QtWidgets.QGraphicsView.__init__(self)

        if dropTarget:
            self.scene = MyScene(self, dropTarget)
        else:
            self.scene = QtWidgets.QGraphicsScene(self)
        self.scene_pan = np.array([250, 250])
        self.scene_panning = False
        self.last_pos = [0, 0]
        self.scene_zoom = 1.

        self.setScene(self.scene)
        self.scene.setBackgroundBrush(QtCore.Qt.black)

        self.scaler = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap())
        self.scene.addItem(self.scaler)
        self.translater = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(), self.scaler)

        self.rotater1 = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(), self.translater)
        self.rotater2 = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(), self.rotater1)
        self.offset = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(), self.rotater2)
        self.origin = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(), self.offset)
        self.origin.angle = 0

        self.hud = QtWidgets.QGraphicsPathItem()
        self.scene.addItem(self.hud)
        self.hud_lowerRight = QtWidgets.QGraphicsPathItem()
        self.scene.addItem(self.hud_lowerRight)
        self.hud_lowerRight.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width(), self.size().height()))

        self.hud_upperRight = QtWidgets.QGraphicsPathItem()
        self.scene.addItem(self.hud_upperRight)
        self.hud_upperRight.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width(), 0))

        self.hud_lowerLeft = QtWidgets.QGraphicsPathItem()
        self.scene.addItem(self.hud_lowerLeft)
        self.hud_lowerLeft.setTransform(QtGui.QTransform(1, 0, 0, 1, 0, self.size().height()))

        self.hud_lowerCenter = QtWidgets.QGraphicsPathItem()
        self.scene.addItem(self.hud_lowerCenter)
        self.hud_lowerCenter.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width() * 0.5, self.size().height()))

        self.hud_upperCenter = QtWidgets.QGraphicsPathItem()
        self.scene.addItem(self.hud_upperCenter)
        self.hud_upperCenter.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width() * 0.5, 0))

        self.hud_leftCenter = QtWidgets.QGraphicsPathItem()
        self.scene.addItem(self.hud_leftCenter)
        self.hud_leftCenter.setTransform(QtGui.QTransform(1, 0, 0, 1, 0, self.size().height() * 0.5))

        self.hud_rightCenter = QtWidgets.QGraphicsPathItem()
        self.scene.addItem(self.hud_rightCenter)
        self.hud_rightCenter.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width(), self.size().height() * 0.5))

        self.hud_center = QtWidgets.QGraphicsPathItem()
        self.scene.addItem(self.hud_center)
        self.hud_center.setTransform(
            QtGui.QTransform(1, 0, 0, 1, self.size().width() * 0.5, self.size().height() * 0.5))

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.setTransform(QtGui.QTransform())
        self.initialized = False
        self.painted = False
        self.view_rect = [1, 1]
        self.fitted = 1
        self.rotation = 0
        self.setStyleSheet("border-width: 0px; border-style: outset;")
        # self.setContentsMargins(0, 0, 0, 0)

    def setExtend(self, width, height):
        do_fit_to_view = (self.fitted and self.view_rect != [width, height])
        self.rotater1.resetTransform()
        self.origin.resetTransform()
        self.view_rect = [width, height]
        self.rotater1.setTransform(QtGui.QTransform(1, 0, 0, 1, width / 2, height / 2))
        self.origin.setTransform(QtGui.QTransform(1, 0, 0, 1, -width / 2, -height / 2))
        if do_fit_to_view:
            self.fitInView()

    def GetExtend(self, with_transform=False):
        # get the cosine and sine for the rotation
        c = np.cos(self.origin.angle * np.pi / 180)
        s = np.sin(self.origin.angle * np.pi / 180)
        
        if with_transform:
            # compose the transformation matrix
            t = self.origin.transform() * QtGui.QTransform(c, s, -s, c, 0, 0) * self.rotater1.transform() * self.translater.transform() * self.scaler.transform()
        else:
            # leave out the rotation
            t = self.origin.transform() * self.rotater1.transform() * self.translater.transform() * self.scaler.transform()
        # transfrom upper left and lower right pixel
        start = t.inverted()[0].map(QtCore.QPoint(0, 0))
        end = t.inverted()[0].map(QtCore.QPoint(self.size().width(), self.size().height()))
        # split in x and y
        start_x, start_y = start.x(), start.y()
        end_x, end_y = end.x(), end.y()
        # sort vales
        start_x, end_x = sorted([start_x, end_x])
        start_y, end_y = sorted([start_y, end_y])
        # and there we are!
        return [start_x, start_y, end_x, end_y]

    def paintEvent(self, QPaintEvent):
        if not self.initialized:
            self.initialized = True
            self.fitInView()
        super().paintEvent(QPaintEvent)
        self.painted = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fitted:
            self.fitInView()
        self.hud_lowerRight.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width(), self.size().height()))
        self.hud_upperRight.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width(), 0))
        self.hud_lowerLeft.setTransform(QtGui.QTransform(1, 0, 0, 1, 0, self.size().height()))

        self.hud_lowerCenter.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width() * 0.5, self.size().height()))
        self.hud_upperCenter.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width() * 0.5, 0))
        self.hud_leftCenter.setTransform(QtGui.QTransform(1, 0, 0, 1, 0, self.size().height() * 0.5))
        self.hud_rightCenter.setTransform(QtGui.QTransform(1, 0, 0, 1, self.size().width(), self.size().height() * 0.5))

        self.hud_center.setTransform(
            QtGui.QTransform(1, 0, 0, 1, self.size().width() * 0.5, self.size().height() * 0.5))
        self.setSceneRect(0, 0, self.size().width(), self.size().height())

    def rotate(self, angle):
        self.rotation = (self.rotation + angle) % 360
        self.origin.angle += angle
        self.rotater2.setRotation(self.origin.angle)
        if self.fitted:
            self.fitInView()

    def fitInView(self):
        # Reset View
        width, height = self.view_rect
        scale = min((self.size().width() / width, self.size().height() / height))
        if self.rotation == 90 or self.rotation == 270:
            scale = min((self.size().width() / height, self.size().height() / width))
        self.scaler.setTransform(QtGui.QTransform(scale, 0, 0, scale, 0, 0))
        xoff = self.size().width() - width * scale
        yoff = self.size().height() - height * scale
        self.translater.setTransform(QtGui.QTransform(1, 0, 0, 1, xoff * 0.5 / scale, yoff * 0.5 / scale))
        self.panEvent(xoff, yoff)
        self.zoomEvent(scale, QtCore.QPoint(0, 0))
        self.fitted = 1

    def fitSmallDimensionInView(self):
        """
        Adjust the view so that the smallest dimension is fitted to the view and the pan of the other dimension is kept,
        i.e., covers the whole view with image data.
        """
        # get the position of the current view
        start_x, start_y, end_x, end_y = self.GetExtend()

        # fit to view with max
        width, height = self.view_rect
        scale = max((self.size().width() / width, self.size().height() / height))
        if self.rotation == 90 or self.rotation == 270:
            scale = max((self.size().width() / height, self.size().height() / width))
        self.scaler.setTransform(QtGui.QTransform(scale, 0, 0, scale, 0, 0))
        xoff = self.size().width() - width * scale
        yoff = self.size().height() - height * scale
        self.translater.setTransform(QtGui.QTransform(1, 0, 0, 1, xoff * 0.5 / scale, yoff * 0.5 / scale))

        # re-pan to the old x-center or y-center
        if self.size().width() / width < self.size().height() / height:
            self.centerOn(np.mean([start_x, end_x]), height/2)
        else:
            self.centerOn(width/2, np.mean([start_y, end_y]))

    def centerOn(self, x, y):
        """ center the view on pos(x,y)"""
        # get the size of the dispalyed image in px
        width, height = self.view_rect
        # print("image dimensions:", width,height)

        # check for rotation
        if self.rotation == 180:
            x = width - x
            y = height - y
        elif self.rotation == 270 or self.rotation == 90:
            pass
            # TODO: implement CenterOn for rotation of 90 or 270 degrees

        # get the current scale factor
        scale = self.scaler.transform().m11()

        # calculate center of view position
        # as it depends on the current window size and the pixmap size
        # calculation in image pixels
        xoff = self.size().width()/scale - width
        yoff = self.size().height()/scale - height

        # move the target coordinates to the center of the view
        # part 1 -> move point to the center of image (yes image not view!)
        # part 2 -> add offset between image and view
        self.translater.setTransform(QtGui.QTransform(1, 0, 0, 1, (width/2 - x) + xoff/2 , (height/2 - y) + yoff/2),
                                     combine=False)

    def translateOrigin(self, x, y):
        # TODO do we still use this function or can we remove it?
        self.translater.setTransform(QtGui.QTransform(1, 0, 0, 1, x, y))
        self.panEvent(x, y)

    def scaleOrigin(self, scale, pos):
        pos = self.mapToScene(pos)
        x, y = (pos.x(), pos.y())
        s0 = self.scaler.transform().m11()
        self.translater.setTransform(QtGui.QTransform(1, 0, 0, 1, -x / s0, -y / s0), combine=True)
        self.scaler.setTransform(QtGui.QTransform(scale, 0, 0, scale, 0, 0), combine=True)
        s0 = self.scaler.transform().m11()
        self.translater.setTransform(QtGui.QTransform(1, 0, 0, 1, +x / s0, +y / s0), combine=True)
        self.zoomEvent(self.scaler.transform().m11(), pos)
        self.fitted = 0

    def getOriginScale(self):
        return self.scaler.transform().m11()

    def mapSceneToOrigin(self, pos):
        pos = self.mapToScene(pos)
        return QtCore.QPoint(pos.x() / self.scaler.transform().m11() - self.translater.transform().dx(),
                             pos.y() / self.scaler.transform().m22() - self.translater.transform().dy())

    def mapToOrigin(self, pos):
        pos = self.mapToScene(pos)
        return QtCore.QPoint(pos.x() / self.scaler.transform().m11() - self.translater.transform().dx(),
                             pos.y() / self.scaler.transform().m22() - self.translater.transform().dy())

    def mapFromOrigin(self, x, y=None):
        try:
            x = x.x()
            y = y.y()
        except:
            pass
        # pos = self.mapToScene(pos)
        pos = QtCore.QPoint((x + self.translater.transform().dx()) * self.scaler.transform().m11(),
                            (y + self.translater.transform().dy()) * self.scaler.transform().m22())
        return self.mapFromScene(pos)

    def mousePressEvent(self, event):
        if event.button() == 2:
            self.last_pos = PosToArray(self.mapToScene(event.pos()))
            self.scene_panning = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.scene_panning:
            new_pos = PosToArray(self.mapToScene(event.pos()))
            delta = new_pos - self.last_pos
            self.translater.setTransform(QtGui.QTransform(1, 0, 0, 1, *delta / self.scaler.transform().m11()),
                                         combine=True)
            self.last_pos = new_pos
            self.fitted = 0
            self.panEvent(*delta)
        super().mouseMoveEvent(event)

    def DoTranslateOrigin(self, delta):
        self.offset.setTransform(QtGui.QTransform(1, 0, 0, 1, *delta), combine=True)

    def mouseReleaseEvent(self, event):
        if event.button() == 2:
            self.scene_panning = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        event.ignore()
        super().wheelEvent(event)
        if event.isAccepted():
            return

        # if qt_version == '5':
        try:  # PyQt 5
            angle = event.angleDelta().y()
        except AttributeError:  # PyQt 4
            angle = event.delta()
        if angle > 0:
            self.scaleOrigin(1.1, event.pos())
        else:
            self.scaleOrigin(0.9, event.pos())
        event.accept()

    def zoomEvent(self, scale, pos):
        pass

    def panEvent(self, x, y):
        pass

    def keyPressEvent(self, event):
        event.setAccepted(False)
        return


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    view = QExtendedGraphicsView()
    view.show()
    sys.exit(app.exec_())
