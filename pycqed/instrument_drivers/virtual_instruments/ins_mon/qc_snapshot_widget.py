# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui
from pycqed.analysis.tools.plotting import SI_UNITS, SI_PREFIXES
import types
import traceback
import numpy as np

try:
    import metaarray
    HAVE_METAARRAY = True
except Exception:
    HAVE_METAARRAY = False


class QcSnaphotWidget(QtGui.QTreeWidget):

    """
    Widget for displaying QcoDes instrument snapshots.
    Heavily inspired by the DataTreeWidget.
    """

    def __init__(self, parent=None, data=None):
        QtGui.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(4)
        self.setHeaderLabels(['Name', 'Value', 'Unit', 'Last update'])

    def setData(self, data):
        """data should be a QCoDes snapshot of a station."""
        self.clear()
        self.buildTreeSnapshot(snapshot=data)
        self.resizeColumnToContents(0)

    def buildTreeSnapshot(self, snapshot):
        # exists so that function can be called with no data in construction
        if snapshot is None:
            return
        parent = self.invisibleRootItem()

        for ins in sorted(snapshot.keys()):
            ins_snapshot = snapshot[ins]
            node = QtGui.QTreeWidgetItem([ins, "", ""])
            parent.addChild(node)
            for par_name in sorted(ins_snapshot['parameters'].keys()):
                par_snap = ins_snapshot['parameters'][par_name]
                # Depending on the type of data stored in value do different
                # things.
                if not isinstance(par_snap['value'], dict):
                    value_str, unit = par_val_to_msg(par_snap['value'],
                                                     par_snap['unit'])

                    # Omits printing of the date to make it more readable
                    if par_snap['ts'] is not None:
                        latest_str = par_snap['ts'][11:]
                    else:
                        latest_str = ''
                    par_node = QtGui.QTreeWidgetItem(
                        [par_name, value_str, unit, latest_str])
                    node.addChild(par_node)

    def buildTree(self, data, parent, name='', hideRoot=False):
        if hideRoot:
            node = parent
        else:
            typeStr = type(data).__name__
            if typeStr == 'instance':
                typeStr += ": " + data.__class__.__name__
            node = QtGui.QTreeWidgetItem([name, typeStr, ""])
            parent.addChild(node)

        # convert traceback to a list of strings
        if isinstance(data, types.TracebackType):
            data = list(
                map(str.strip, traceback.format_list(
                    traceback.extract_tb(data))))
        elif HAVE_METAARRAY and (hasattr(data, 'implements')
                                 and data.implements('MetaArray')):
            data = {
                'data': data.view(np.ndarray),
                'meta': data.infoCopy()
            }

        if isinstance(data, dict):
            for k in data.keys():
                self.buildTree(data[k], node, str(k))
        elif isinstance(data, list) or isinstance(data, tuple):
            for i, d in enumerate(data):
                self.buildTree(d, node, str(i))
        else:
            node.setText(2, str(data))


def par_val_to_msg(val, unit=None):
    validtypes = (float, int, np.integer, np.floating)
    if unit in SI_UNITS and isinstance(val, validtypes):
        if val == 0:
            prefix_power = 0
        else:
            # The defined prefixes go down to -24 but this is below
            # the numerical precision of python
            prefix_power = np.clip(-15, (np.log10(abs(val))//3 * 3), 24)
        # Determine SI prefix, number 8 corresponds to no prefix
        SI_prefix_idx = int(prefix_power/3 + 8)
        prefix = SI_PREFIXES[SI_prefix_idx]
        # Convert the unit
        val = val*10**-prefix_power
        unit = prefix+unit

    value_str = str(val)
    return value_str, unit
