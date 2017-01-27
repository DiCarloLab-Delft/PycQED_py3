import logging
import numpy as np
from qcodes.utils.helpers import make_unique, DelegateAttributes

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


class DeviceObject(Instrument):
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.msmt_suffix = '_' + name  # used to append to measuremnet labels
        self._qubits = {}
        self.add_parameter('qubits',
                           get_cmd=self._get_qubits,
                           vals=vals.Anything())
        self.delegate_attr_dicts += ['_qubits']

    def _get_qubits(self):
        return self._qubits

    def add_qubits(self, qubits):
        """
        Add one or more qubit objects to the device

        Args:
            component (Any): components to add to the Station.
            name (str): name of the qubit

        Returns:
            str: the name assigned this qubit, which may have been changed to
             make it unique among previously added qubits.

        """

        if type(qubits) == list:
            for q in qubits:
                self.add_qubits(q)
        else:
            name = qubits.name
            self._qubits[name] = qubits
            return name

    def get_operation_dict(self):
        operation_dict = {}
        for name, q in self.qubits().items():
            q.get_operation_dict(operation_dict)
        return operation_dict
