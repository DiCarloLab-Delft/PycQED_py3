import unittest
import os
import pycqed as pq
import pycqed.measurement.gate_set_tomography.gate_set_tomography_CC as gstCC

class Test_GST_CC(unittest.TestCase):

    def test_parser(self):
        testfile = os.path.join(pq.__path__[0], 'tests', 'gst_files',
                                'GST_template_short.txt')
        gateDict = {
        'Gi' : 'I 20',
        'Gx' : 'X90 QR',
        'Gy' : 'Y90 QR',
        'RO' : 'RO QR'
        }

        exp_list = gstCC.get_experiments_from_file(testfile, gateDict,
                                                   use_pygsti_parser=True)

        solShort = solution = [
            ['\ninit_all', 'RO QR'],
            ['\ninit_all', 'X90 QR', 'RO QR'],
            ['\ninit_all', 'Y90 QR', 'X90 QR', 'I 20', 'RO QR'],
            ['\ninit_all', 'I 20', 'RO QR'],
            ['\ninit_all', 'I 20', 'Y90 QR', 'RO QR'],
            ['\ninit_all', 'X90 QR', 'Y90 QR', 'X90 QR', 'Y90 QR', 'X90 QR',
                'Y90 QR', 'X90 QR', 'I 20', 'I 20', 'X90 QR', 'Y90 QR',
                'X90 QR', 'Y90 QR', 'RO QR']
            ]

        # self.assertEqual(exp_list, solution)

        exp_list = gstCC.get_experiments_from_file(testfile, gateDict,
                                                   use_pygsti_parser=False)

        solShort = solution = [
            ['\ninit_all', 'RO QR'],
            ['\ninit_all', 'X90 QR', 'RO QR'],
            ['\ninit_all', 'Y90 QR', 'X90 QR', 'I 20', 'RO QR'],
            ['\ninit_all', 'I 20', 'RO QR'],
            ['\ninit_all', 'I 20', 'Y90 QR', 'RO QR'],
            ['\ninit_all', 'X90 QR', 'Y90 QR', 'X90 QR', 'Y90 QR', 'X90 QR',
                'Y90 QR', 'X90 QR', 'I 20', 'I 20', 'X90 QR', 'Y90 QR',
                'X90 QR', 'Y90 QR', 'RO QR']
            ]

        self.assertEqual(exp_list, solution)
