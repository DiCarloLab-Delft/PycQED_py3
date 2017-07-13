from unittest import TestCase
import pycqed as pq
import os

from pycqed.instrument_drivers.virtual_instruments.pyqx.qasm_loader import qasm_loader


class Test_qasm_loader(TestCase):

    def test_replace(self):
        """
        test here if replacing lines works correctly
        """
        # empty file is used to initialize this test, all is overwritten
        # in the test
        qasm_file = os.path.join(
            pq.__path__[0], 'tests', 'qasm_files', "empty.qasm")
        loader = qasm_loader(qasm_file, 2)
        lines = ["qubits", "init_all", ".c", "x180",
                 "invalid command", "init_all"]
        results = []

        for l in lines:
            results.append(loader.replaceLine(l))

        self.assertFalse(results[0])
        self.assertEquals(results[1], [".c1", "prepz q0", "prepz q1"])
        self.assertFalse(results[2])
        self.assertFalse(results[3])
        self.assertFalse(results[4])
        self.assertEquals(results[5], [".c3", "prepz q0", "prepz q1"])

    def test_qasm_file(self):
        qasm_file = os.path.join(
            pq.__path__[0], 'tests', 'qasm_files', "qasm_loader_test.qasm")
        loader = qasm_loader(qasm_file, 2)
        expected_result = [
            ".c1",
            "prepz q0",
            "prepz q1",
            "rx180 q0",
            '.c',  # this manual circuit remains in the code
            ".c3",
            "prepz q0",
            "prepz q1",
            'x q0',
            'measure q0',
            'measure q1',
            ".c4",
            "prepz q0",
            "prepz q1",
            'measure q0',
            'measure q1',
            'ry90 q0',
            'measure q0',
            'measure q1'
        ]
        self.assertEquals(loader.lines, expected_result)
