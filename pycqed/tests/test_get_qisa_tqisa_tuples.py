import unittest
import os
import pycqed as pq
from pycqed.measurement.openql_experiments.get_qisa_tqisa_timing_tuples import (
        get_qisa_tqisa_timing_tuples
    )

class Test_get_tqisa_tuples(unittest.TestCase):
    
    def test_main(self):

        file_paths_root = os.path.join(pq.__path__[0], 'tests', 'get_qisa_tqisa_timing_tuples_files')
        
        result_fp       = file_paths_root + '/sample_main.result'
        sample_qisa_fp  = file_paths_root + '/sample_main.qisa'
        sample_tqisa_fp = file_paths_root + '/sample_main.tqisa'
        ccl_config_json = file_paths_root + '/ccl_config_new.json'
        output_fp       = file_paths_root + '/sample_main.qisa.out'

        try:
            with open(result_fp,'r') as result_file:
                result_str = result_file.read()
        except :
            raise Exception('Unable to read result file')

        tuple_result = self.get_tuple_result()

        result =  get_qisa_tqisa_timing_tuples( 
                                                sample_qisa_fp,
                                                sample_tqisa_fp,
                                                ccl_config_json,
                                                output_fp
                                              )

        self.assertEqual(tuple_result, result)

        try:
            with open(output_fp,'r') as output_file:
                output_str = output_file.read()
        except :
            raise Exception('Unable to read output file')

        self.assertEqual(result_str, output_str)



    def get_tuple_result(self):
        tuple_result = [
                         (0, 'prepz s0'), 
                         (10000, 'prepz s10'), 
                         (20000, 'cw_03 s0'), 
                         (20001, ['fl_cw_01 t0', 'cw_00 s1']),
                         (20016, 'cw_03 s0'),
                         (20017, 'cw_00 s1'),
                         (20018, ['prepz s1', 'cw_00 s0']),
                         (30018, 'cw_03 s1'),
                         (30019, 'cw_00 s0'), 
                         (30020, ['fl_cw_01 t0', 'cw_03 s1']),
                         (30035, 'measz s0'), 
                         (30075, ['measz s1', 'prepz s0']),
                         (40075, 'prepz s10'), 
                         (50075, 'cw_03 s0'), 
                         (50076, ['fl_cw_01 t0', 'cw_00 s1']), 
                         (50091, 'cw_03 s0'), 
                         (50092, 'cw_00 s1'), 
                         (50093, ['prepz s1', 'cw_00 s0']), 
                         (60093, 'cw_04 s1'), 
                         (60094, 'cw_00 s0'), 
                         (60095, ['fl_cw_01 t0', 'cw_04 s1']), 
                         (60110, 'measz s0'), 
                         (60150, ['measz s1', 'prepz s0']), 
                         (70150, 'prepz s10'), 
                         (80150, 'cw_04 s0'), 
                         (80151, ['fl_cw_01 t0', 'cw_00 s1']), 
                         (80166, 'cw_04 s0'), 
                         (80167, 'cw_00 s1'), 
                         (80168, ['prepz s1', 'cw_00 s0']), 
                         (90168, 'cw_03 s1'),
                         (90169, 'cw_00 s0'), 
                         (90170, ['fl_cw_01 t0', 'cw_03 s1']), 
                         (90185, 'measz s0'), 
                         (90225, ['measz s1', 'prepz s0']), 
                         (100225, 'prepz s10'), 
                         (110225, 'cw_04 s0'), 
                         (110226, ['fl_cw_01 t0', 'cw_00 s1']), 
                         (110241, 'cw_04 s0'), 
                         (110242, 'cw_00 s1'), 
                         (110243, ['prepz s1', 'cw_00 s0']), 
                         (120243, 'cw_04 s1'), 
                         (120244, 'cw_00 s0'), 
                         (120245, ['fl_cw_01 t0', 'cw_04 s1']), 
                         (120260, 'measz s11') 
                       ]

        return tuple_result
