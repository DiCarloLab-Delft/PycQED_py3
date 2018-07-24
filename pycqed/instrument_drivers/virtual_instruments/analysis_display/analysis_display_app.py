import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pycqed.instrument_drivers.virtual_instruments.\
       analysis_display.image_display_widget as idisp
import pycqed.analysis.analysis_toolbox as a_tools

class AnalysisDisplayApp:

    def __init__(self, filename=''):

        self.app = pg.mkQApp()
        self.window = QtGui.QWidget()

        self.autoloadCheckbox = QtGui.QCheckBox()
        self.autoloadCheckbox.setText('Autoload')

        folderDown = QtGui.QPushButton()
        folderDown.setIcon(
            self.window.style().standardIcon(QtGui.QStyle.SP_ArrowDown))
        folderDown.setMinimumWidth(folderDown.sizeHint().height())
        folderDown.setMaximumWidth(folderDown.sizeHint().height())
        folderDown.clicked.connect(self._prev_folder_clicked)
        folderLabel = QtGui.QLabel()
        folderLabel.setText('Measurement')
        folderUp = QtGui.QPushButton()
        folderUp.setIcon(
            self.window.style().standardIcon(QtGui.QStyle.SP_ArrowUp))
        folderUp.setMinimumWidth(folderDown.sizeHint().height())
        folderUp.setMaximumWidth(folderDown.sizeHint().height())
        folderUp.clicked.connect(self._next_folder_clicked)

        plotLeft = QtGui.QPushButton()
        plotLeft.setIcon(
            self.window.style().standardIcon(QtGui.QStyle.SP_ArrowLeft))
        plotLeft.setMinimumWidth(folderDown.sizeHint().height())
        plotLeft.setMaximumWidth(folderDown.sizeHint().height())
        plotLeft.clicked.connect(self._prev_plot_clicked)
        plotLabel = QtGui.QLabel()
        plotLabel.setText('Plot')
        plotRight = QtGui.QPushButton()
        plotRight.setIcon(
            self.window.style().standardIcon(QtGui.QStyle.SP_ArrowRight))
        plotRight.setMinimumWidth(folderDown.sizeHint().height())
        plotRight.setMaximumWidth(folderDown.sizeHint().height())
        plotRight.clicked.connect(self._next_plot_clicked)

        self.text = QtGui.QLabel()
        self.text.setSizePolicy(QtGui.QSizePolicy.Ignored,
                                QtGui.QSizePolicy.Fixed)
        self.text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.image = idisp.ImageDisplay()

        hlayout = QtGui.QHBoxLayout()
        hlayout.addWidget(self.autoloadCheckbox)
        hlayout.addStretch(1)
        hlayout.addWidget(folderDown)
        hlayout.addWidget(folderLabel)
        hlayout.addWidget(folderUp)
        hlayout.addStretch(1)
        hlayout.addWidget(plotLeft)
        hlayout.addWidget(plotLabel)
        hlayout.addWidget(plotRight)

        vlayout = QtGui.QVBoxLayout()
        vlayout.addLayout(hlayout)
        vlayout.addWidget(self.text)
        vlayout.addWidget(self.image)

        self.window.setLayout(vlayout)
        self.window.setWindowTitle('Analysis Display')

        self.window.show()

        self.filename(filename)

    def filename(self, val=None):
        if val is None:
            return self._filename
        else:
            self._filename = val
            self.text.setText(val)
            self.image.setPixmap(QtGui.QPixmap(val))

    def autoload(self, val=None):
        if val is None:
            return self.autoloadCheckbox.isChecked()
        else:
            self.autoloadCheckbox.setChecked(val)

    def _current_folder(self):
        fn = self.filename()
        if os.path.isdir(fn):
            return fn.rstrip('/\\')
        else:
            return os.path.split(fn)[0]

    def _current_filename(self):
        fn = self.filename()
        if os.path.isdir(fn):
            return ''
        else:
            return os.path.split(fn)[1]

    def _current_timestamp(self):
        try:
            sp = splitpath(self._current_folder())
            return sp[-2] + '_' + sp[-1][:6]
        except IndexError:
            return None

    def _prev_folder(self):
        ts = self._current_timestamp()
        if ts is not None:
            return a_tools.latest_data(older_than=self._current_timestamp())
        else:
            return None

    def _next_folder(self):
        ts = self._current_timestamp()
        if ts is None:
            return None
        else:
            return a_tools.get_folder(
                a_tools.get_timestamps_in_range(ts, label='')[1])

    def _get_images_in_current_folder(self):
        return list(sorted([fn for fn in os.listdir(self._current_folder())
                            if fn[-4:] == '.png']))

    def _next_file(self):
        images = self._get_images_in_current_folder()
        if len(images) == 0:
            return self.filename()
        try:
            idx = images.index(self._current_filename())
        except ValueError:
            idx = -1
        return os.path.join(self._current_folder(), images[(idx+1)%len(images)])

    def _prev_file(self):
        images = self._get_images_in_current_folder()
        if len(images) == 0:
            return self.filename()
        try:
            idx = images.index(self._current_filename())
        except ValueError:
            idx = 1
        return os.path.join(self._current_folder(), images[(idx-1)%len(images)])

    def _next_folder_clicked(self):
        self.filename(self._next_folder())
        self.filename(self._next_file())

    def _prev_folder_clicked(self):
        self.filename(self._prev_folder())
        self.filename(self._next_file())

    def _next_plot_clicked(self):
        self.filename(self._next_file())

    def _prev_plot_clicked(self):
        self.filename(self._prev_file())

    def set_data_path(self, data_path):
        a_tools.datadir = data_path

    def update(self):
        if self.autoloadCheckbox.isChecked():
            self.filename(a_tools.latest_data())
            self.filename(self._next_file())

def splitpath(path):
    els = []
    lastpath = path
    path, el = os.path.split(path)
    while path != lastpath:
        els.append(el)
        lastpath=path
        path, el = os.path.split(path)
    els.append(path)
    return list(reversed(els))