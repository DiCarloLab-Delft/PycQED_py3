# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui, QtCore

class ImageDisplay(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent=parent)
        self.p = None
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

    def setPixmap(self, p):
        self.p = p
        self.repaint()

    def aspectRatio(self):
        if self.p and self.p.height() != 0:
            return self.p.width()/self.p.height()
        else:
            return 1

    def centeredViewport(self, width, height):

        heightFromWidth = int(width / self.aspectRatio())
        widthFromHeight = int(height * self.aspectRatio())

        if heightFromWidth <= height:
            return QtCore.QRect(0, (height - heightFromWidth) / 2,
                                width, heightFromWidth)
        else:
            return QtCore.QRect((width - widthFromHeight) / 2, 0,
                                widthFromHeight, height)

    def paintEvent(self, event):
        if self.p:
            painter = QtGui.QPainter(self)
            painter.setViewport(self.centeredViewport(self.width(),
                                                      self.height()))
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            rect = QtCore.QRect(QtCore.QPoint(0, 0), self.size())
            painter.drawPixmap(rect, self.p)

    def sizeHint(self):
        return QtCore.QSize(self.width(), self.width()/self.aspectRatio())