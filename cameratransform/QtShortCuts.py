#!/usr/bin/env python
# -*- coding: utf-8 -*-
# QtShortCuts.py

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

import colorsys
import os
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
#from . import HTMLColorToRGB

from qtpy import API_NAME as QT_API_NAME

if QT_API_NAME.startswith("PyQt4"):
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt4agg import FigureManager
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt5agg import FigureManager
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib import _pylab_helpers
from matplotlib.figure import Figure


class MatplotlibWidget(Canvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.patch.set_facecolor([0, 1, 0, 0])
        self.axes = self.figure.add_subplot(111)

        Canvas.__init__(self, self.figure)
        self.setParent(parent)

        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

        self.manager = FigureManager(self, 1)
        self.manager._cidgcf = self.figure
        _pylab_helpers.Gcf.set_active(self.manager)

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)


class QHLine:
    def __init__(self, layout):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)

class QInput(QtWidgets.QWidget):
    """
    A base class for input widgets with a text label and a unified API.

    - The valueChanged signal is emitted when the user has changed the input.

    - The value of the input element get be set with setValue(value) and queried by value()

    """
    # the signal when the user has changed the value
    valueChanged = QtCore.Signal('PyQt_PyObject')

    no_signal = False

    last_emited_value = None

    def __init__(self, layout=None, name=None, tooltip=None, stretch=False):
        # initialize the super widget
        super(QInput, self).__init__()

        # initialize the layout of this widget
        QtWidgets.QHBoxLayout(self)
        self.layout().setContentsMargins(0, 0, 0, 0)

        # add me to a parent layout
        if layout is not None:
            if stretch is True:
                self.wrapper_layout = QtWidgets.QHBoxLayout()
                self.wrapper_layout.setContentsMargins(0, 0, 0, 0)
                self.wrapper_layout.addWidget(self)
                self.wrapper_layout.addStretch()
                layout.addLayout(self.wrapper_layout)
            else:
                layout.addWidget(self)

        # add a label to this layout
        self.label = QtWidgets.QLabel(name)
        self.layout().addWidget(self.label)

        if tooltip is not None:
            self.setToolTip(tooltip)

    def setLabel(self, text):
        # update the label
        self.label.setText(text)

    def _emitSignal(self):
        if self.value() != self.last_emited_value:
            self.valueChanged.emit(self.value())
            self.last_emited_value = self.value()

    def _valueChangedEvent(self, value):
        if self.no_signal:
            return
        self.setValue(value)
        self._emitSignal()

    def setValue(self, value):
        self.no_signal = True
        try:
            self._doSetValue(value)
        finally:
            self.no_signal = False

    def _doSetValue(self, value):
        # dummy method to be overloaded by child classes
        pass

    def value(self):
        # dummy method to be overloaded by child classes
        pass


class QInputNumber(QInput):
    slider_dragged = False

    def __init__(self, layout=None, name=None, value=0, min=None, max=None, use_slider=False, float=True, decimals=2,
                 unit=None, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        if float is False:
            self.decimals = 0
        else:
            if decimals is None:
                decimals = 2
            self.decimals = decimals
        self.decimal_factor = 10**self.decimals

        if use_slider and min is not None and max is not None:
            # slider
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.layout().addWidget(self.slider)
            self.slider.setRange(min * self.decimal_factor, max * self.decimal_factor)
            self.slider.valueChanged.connect(lambda x: self._valueChangedEvent(x / self.decimal_factor))
            self.slider.sliderPressed.connect(lambda: self._setSliderDragged(True))
            self.slider.sliderReleased.connect(lambda: self._setSliderDragged(False))
        else:
            self.slider = None

        # add spin box
        if float:
            self.spin_box = QtWidgets.QDoubleSpinBox()
            self.spin_box.setDecimals(decimals)
        else:
            self.spin_box = QtWidgets.QSpinBox()
        if unit is not None:
            self.spin_box.setSuffix(" " + unit)
        self.layout().addWidget(self.spin_box)
        self.spin_box.valueChanged.connect(self._valueChangedEvent)

        if min is not None:
            self.spin_box.setMinimum(min)
        if max is not None:
            self.spin_box.setMaximum(max)

        self.setValue(value)

    def _setSliderDragged(self, value):
        self.slider_dragged = value
        if value is False:
            self._emitSignal()

    def _valueChangedEvent(self, value):
        if self.no_signal:
            return
        self.setValue(value)
        if not self.slider_dragged:
            self._emitSignal()

    def _doSetValue(self, value):
        self.spin_box.setValue(value)
        if self.slider is not None:
            self.slider.setValue(value * self.decimal_factor)

    def value(self):
        return self.spin_box.value()


class QInputString(QInput):

    def __init__(self, layout=None, name=None, value="", **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        self.line_edit = QtWidgets.QLineEdit()
        self.layout().addWidget(self.line_edit)
        self.line_edit.editingFinished.connect(lambda: self._valueChangedEvent(self.value()))

        self.setValue(value)

    def _doSetValue(self, value):
        self.line_edit.setText(str(value))

    def value(self):
        return self.line_edit.text()


class QInputBool(QInput):

    def __init__(self, layout=None, name=None, value=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        self.checkbox = QtWidgets.QCheckBox()
        self.layout().addWidget(self.checkbox)
        self.checkbox.stateChanged.connect(lambda: self._valueChangedEvent(self.value()))

        self.setValue(value)

    def _doSetValue(self, value):
        self.checkbox.setChecked(value)

    def value(self):
        return self.checkbox.isChecked()


class QInputChoice(QInput):

    def __init__(self, layout=None, name=None, value=None, values=None, value_names=None, reference_by_index=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        self.reference_by_index = reference_by_index
        self.values = values
        self.value_names = value_names if value_names is not None else values

        self.combobox = QtWidgets.QComboBox()
        self.layout().addWidget(self.combobox)

        self.combobox.addItems(self.value_names)
        self.combobox.currentIndexChanged.connect(lambda: self._valueChangedEvent(self.value()))

        if value is not None:
            self.setValue(value)

    def _doSetValue(self, value):
        if self.reference_by_index is True:
            self.combobox.setCurrentIndex(value)
        else:
            try:
                self.combobox.setCurrentIndex(self.values.index(value))
            except ValueError:
                self.combobox.setCurrentIndex(self.value_names.index(value))

    def value(self):
        if self.reference_by_index is True:
            return self.combobox.currentIndex()
        else:
            return self.values[self.combobox.currentIndex()]


class QInputColor(QInput):

    def __init__(self, layout=None, name=None, value=None, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        self.button = QtWidgets.QPushButton()
        self.button.setMaximumWidth(40)
        self.layout().addWidget(self.button)
        self.button.clicked.connect(self._openDialog)

        # set the color
        self.setValue(value)

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.EnabledChange:
            if not self.isEnabled():
                self.button.setStyleSheet("background-color: #f0f0f0;")
            else:
                self.setValue(self.color)

    def _openDialog(self):
        import matplotlib as mpl
        # get new color from color picker
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*tuple(mpl.colors.to_rgba_array(self.value())[0] * 255)),
                                                self.parent(), self.label.text() + " choose color")
        # if a color is set, apply it
        if color.isValid():
            color = mpl.colors.to_hex(color.getRgbF())
            self.setValue(color)
            self._emitSignal()

    def _doSetValue(self, value):
        # display and save the new color
        if value is None:
            value = "#FF0000FF"
        self.button.setStyleSheet("background-color: %s;" % value)
        self.color = value

    def value(self):
        # return the color
        return self.color


class QInputFilename(QInput):
    last_folder = None

    def __init__(self, layout=None, name=None, value=None, dialog_title="Choose File", file_type="All", button_text="choose file", filename_checker=None, existing=False, just_button=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        self.dialog_title = dialog_title
        self.file_type = file_type
        self.filename_checker = filename_checker
        self.existing = existing

        self.line = QtWidgets.QLineEdit()
        self.layout().addWidget(self.line)
        self.line.setEnabled(False)

        self.button = QtWidgets.QPushButton(button_text)
        self.layout().addWidget(self.button)
        self.button.clicked.connect(self._openDialog)

        if just_button:
            self.label.setHidden(True)
            self.line.setHidden(True)

        # set the color
        self.setValue(value)
        if value is None:
            self.last_folder = os.getcwd()

    def _openDialog(self):
        # open an new files
        if not self.existing:
            filename = QtWidgets.QFileDialog.getSaveFileName(None, self.dialog_title, self.last_folder, self.file_type)
        # or choose an existing file
        else:
            filename = QtWidgets.QFileDialog.getOpenFileName(None, self.dialog_title, self.last_folder, self.file_type)

        # get the string
        if isinstance(filename, tuple):  # Qt5
            filename = filename[0]
        else:  # Qt4
            filename = str(filename)

        # optical check the filename
        if self.filename_checker and filename:
            filename = self.filename_checker(filename)

        # set the filename
        if filename:
            self.setValue(filename)
            self._emitSignal()

    def _doSetValue(self, value):
        self.last_folder = os.path.dirname(value)
        self.line.setText(value)

    def value(self):
        # return the color
        return self.line.text()


def AddQSpinBox(layout, text, value=0, float=True, strech=False):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    if float:
        spinBox = QtWidgets.QDoubleSpinBox()
    else:
        spinBox = QtWidgets.QSpinBox()
    spinBox.label = text
    spinBox.setRange(-99999, 99999)
    spinBox.setValue(value)
    spinBox.setHidden_ = spinBox.setHidden

    def setHidden(hidden):
        spinBox.setHidden_(hidden)
        text.setHidden(hidden)

    spinBox.setHidden = setHidden
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(spinBox)
    spinBox.managingLayout = horizontal_layout
    if strech:
        horizontal_layout.addStretch()
    return spinBox


def AddQLineEdit(layout, text, value=None, strech=False, editwidth=None):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    lineEdit = QtWidgets.QLineEdit()
    if editwidth:
        lineEdit.setFixedWidth(editwidth)
    if value:
        lineEdit.setText(value)
    lineEdit.label = text
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(lineEdit)
    lineEdit.managingLayout = horizontal_layout
    if strech:
        horizontal_layout.addStretch()
    return lineEdit


def AddQSaveFileChoose(layout, text, value=None, dialog_title="Choose File", file_type="All", filename_checker=None,
                       strech=False):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    lineEdit = QtWidgets.QLineEdit()
    if value:
        lineEdit.setText(value)
    lineEdit.label = text
    lineEdit.setEnabled(False)

    def OpenDialog():
        srcpath = QtWidgets.QFileDialog.getSaveFileName(None, dialog_title, os.getcwd(), file_type)
        if isinstance(srcpath, tuple):
            srcpath = srcpath[0]
        else:
            srcpath = str(srcpath)
        if filename_checker and srcpath:
            srcpath = filename_checker(srcpath)
        if srcpath:
            lineEdit.setText(srcpath)

    button = QtWidgets.QPushButton("Choose File")
    button.pressed.connect(OpenDialog)
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(lineEdit)
    horizontal_layout.addWidget(button)
    lineEdit.managingLayout = horizontal_layout
    if strech:
        horizontal_layout.addStretch()
    return lineEdit


def AddQOpenFileChoose(layout, text, value=None, dialog_title="Choose File", file_type="All", filename_checker=None,
                       strech=False):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    lineEdit = QtWidgets.QLineEdit()
    if value:
        lineEdit.setText(value)
    lineEdit.label = text
    lineEdit.setEnabled(False)

    def OpenDialog():
        srcpath = QtWidgets.QFileDialog.getOpenFileName(None, dialog_title, os.getcwd(), file_type)
        if isinstance(srcpath, tuple):
            srcpath = srcpath[0]
        else:
            srcpath = str(srcpath)
        if filename_checker and srcpath:
            srcpath = filename_checker(srcpath)
        if srcpath:
            lineEdit.setText(srcpath)

    button = QtWidgets.QPushButton("Choose File")
    button.pressed.connect(OpenDialog)
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(lineEdit)
    horizontal_layout.addWidget(button)
    lineEdit.managingLayout = horizontal_layout
    if strech:
        horizontal_layout.addStretch()
    return lineEdit


def AddQColorChoose(layout, text, value=None, strech=False):
    # add a layout
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    # add a text
    text = QtWidgets.QLabel(text)
    button = QtWidgets.QPushButton("")
    button.label = text

    def OpenDialog():
        # get new color from color picker
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*HTMLColorToRGB(button.getColor())))
        # if a color is set, apply it
        if color.isValid():
            color = "#%02x%02x%02x" % color.getRgb()[:3]
            button.setColor(color)

    def setColor(value):
        # display and save the new color
        button.setStyleSheet("background-color: %s;" % value)
        button.color = value

    def getColor():
        # return the color
        return button.color

    # default value for the color
    if value is None:
        value = "#FF0000"
    # add functions to button
    button.pressed.connect(OpenDialog)
    button.setColor = setColor
    button.getColor = getColor
    # set the color
    button.setColor(value)
    # add widgets to the layout
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(button)
    # add a strech if requested
    if strech:
        horizontal_layout.addStretch()
    return button


def AddQComboBox(layout, text, values=None, selectedValue=None):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    comboBox = QtWidgets.QComboBox()
    comboBox.label = text
    for value in values:
        comboBox.addItem(value)
    if selectedValue:
        comboBox.setCurrentIndex(values.index(selectedValue))
    comboBox.values = values

    def setValues(new_values):
        for i in range(len(comboBox.values)):
            comboBox.removeItem(0)
        for value in new_values:
            comboBox.addItem(value)
        comboBox.values = new_values

    comboBox.setValues = setValues
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(comboBox)
    comboBox.managingLayout = horizontal_layout
    return comboBox


def AddQCheckBox(layout, text, checked=False, strech=False):
    horizontal_layout = QtWidgets.QHBoxLayout()
    layout.addLayout(horizontal_layout)
    text = QtWidgets.QLabel(text)
    checkBox = QtWidgets.QCheckBox()
    checkBox.label = text
    checkBox.setChecked(checked)
    horizontal_layout.addWidget(text)
    horizontal_layout.addWidget(checkBox)
    if strech:
        horizontal_layout.addStretch()
    return checkBox


def AddQLabel(layout, text=None):
    text = QtWidgets.QLabel(text)
    if text:
        layout.addWidget(text)
    return text


def AddQHLine(layout):
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    layout.addWidget(line)
    return line


def GetColorByIndex(index):
    colors = np.linspace(0, 1, 16, endpoint=False).tolist() * 3  # 16 different hues
    saturations = [1] * 16 + [0.5] * 16 + [1] * 16  # in two different saturations
    value = [1] * 16 + [1] * 16 + [0.5] * 16  # and two different values
    return "#%02x%02x%02x" % tuple((np.array(
        colorsys.hsv_to_rgb((np.array(colors[index]) * 3) % 1, saturations[index], value[index])) * 255).astype(int))


# set the standard colors for the color picker dialog
colors = np.linspace(0, 1, 16, endpoint=False).tolist() * 3  # 16 different hues
saturations = [1] * 16 + [0.5] * 16 + [1] * 16  # in two different saturations
value = [1] * 16 + [1] * 16 + [0.5] * 16  # and two different values
for index, (color, sat, val) in enumerate(zip(colors, saturations, value)):
    # deform the index, as the dialog fills them column wise and we want to fill them row wise
    index = index % 8 * 6 + index // 8
    # convert color from hsv to rgb, to an array, to an tuple, to a hex string then to an integer
    color_integer = int("%02x%02x%02x" % tuple((np.array(colorsys.hsv_to_rgb(color, sat, val)) * 255).astype(int)), 16)
    try:
        QtWidgets.QColorDialog.setStandardColor(index, QtGui.QColor(color_integer))  # for Qt5
    except TypeError:
        QtWidgets.QColorDialog.setStandardColor(index, color_integer)  # for Qt4
