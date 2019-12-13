# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui
from pycqed.analysis.tools.plotting import SI_val_to_msg_str


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
        self.nodes = {}

    def setData(self, data):
        """
        data should be a QCoDes snapshot of a station.
        """
        self.buildTreeSnapshot(snapshot=data)
        self.resizeColumnToContents(0)

    def buildTreeSnapshot(self, snapshot):
        # exists so that function can be called with no data in construction
        if snapshot is None:
            return
        parent = self.invisibleRootItem()

        for ins in sorted(snapshot.keys()):
            ins_snapshot = snapshot[ins]
            if ins not in self.nodes:
                self.nodes[ins] = QtGui.QTreeWidgetItem([ins, "", ""])
                parent.addChild(self.nodes[ins])

            node = self.nodes[ins]
            for par_name in sorted(ins_snapshot['parameters'].keys()):
                par_snap = ins_snapshot['parameters'][par_name]
                # Depending on the type of data stored in value do different
                # things, currently only blocks non-dicts
                if 'value' in par_snap.keys():
                    # Some parameters do not have a value, these are not shown
                    # in the instrument monitor.
                    if not isinstance(par_snap['value'], dict):
                        value_str, unit = SI_val_to_msg_str(par_snap['value'],
                                                            par_snap['unit'])

                        # Omits printing of the date to make it more readable
                        if par_snap['ts'] is not None:
                            latest_str = par_snap['ts'][11:]
                        else:
                            latest_str = ''

                        # Name of the node in the self.nodes dictionary
                        param_node_name = '{}.{}'.format(ins, par_name)
                        # If node does not yet exist, create a node
                        if param_node_name not in self.nodes:
                            param_node = QtGui.QTreeWidgetItem(
                                [par_name, value_str, unit, latest_str])
                            node.addChild(param_node)
                            self.nodes[param_node_name] = param_node
                        else:  # else update existing node
                            param_node = self.nodes[param_node_name]
                            param_node.setData(1, 0, value_str)
                            param_node.setData(2, 0, unit)
                            param_node.setData(3, 0, latest_str)
