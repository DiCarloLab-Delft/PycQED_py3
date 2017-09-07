import os
import numpy as np
import unittest
from openql import Kernel, Program
from openql import openql as ql
# from test_QISA_assembler_present import assemble

rootDir = os.path.dirname(os.path.realpath(__file__))

curdir = os.path.dirname(__file__)
# config_fn = os.path.join(curdir, 'test_cfg_cbox.json')
# platf = ql.Platform("starmon", config_fn)
config_fn = os.path.join(curdir, 'test_data/test_cfg_CCL.json')
platf = ql.Platform('seven_qubits_chip', config_fn)

output_dir = os.path.join(curdir, 'test_output')
ql.set_output_dir(output_dir)


class Test_single_qubit_seqs_CCL(unittest.TestCase):
    def test_allxy(self):
        p = Program(pname="AllXY", nqubits=1, p=platf)
        # uppercase lowercase problems

        allXY = [['i', 'i'], ['rx180', 'rx180'], ['ry180', 'ry180'],
                 ['rx180', 'ry180'], ['ry180', 'rx180'],
                 ['rx90', 'i'], ['ry90', 'i'], ['rx90', 'ry90'],
                 ['ry90', 'rx90'], ['rx90', 'ry180'], ['ry90', 'rx180'],
                 ['rx180', 'ry90'], ['ry180', 'rx90'], ['rx90', 'rx180'],
                 ['rx180', 'rx90'], ['ry90', 'ry180'], ['ry180', 'ry90'],
                 ['rx180', 'i'], ['ry180', 'i'], ['rx90', 'rx90'],
                 ['ry90', 'ry90']]

        # this should be implicit
        p.set_sweep_points(np.arange(len(allXY), dtype=float), len(allXY))

        for i, xy in enumerate(allXY):
            k = Kernel("allXY"+str(i), p=platf)
            k.prepz(0)
            k.gate(xy[0], 0)
            k.gate(xy[1], 0)
            k.measure(0)
            p.add_kernel(k)

        p.compile()

        # Test that the generated code is valid
        # QISA_fn = os.path.join(output_dir, p.name+'.qisa')
        # assemble(QISA_fn)
