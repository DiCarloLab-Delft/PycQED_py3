import unittest
import os
import pycqed as pq
from pycqed.measurement.openql_experiments.get_qisa_tqisa_timing_tuples import get_qisa_tqisa_timing_tuples as gqt

class Test_get_tqisa_tuples(unittest.TestCase):
    
    def test_main(self):
		# Get the result file
        file_paths_root = os.path.join(pq.__path__[0], 'tests', 'get_qisa_tqisa_timing_tuples_files')
        
        result_fp       = file_paths_root + 'sample_main.result'
        sample_qisa_fp  = file_paths_root + 'sample_main.qisa'
        sample_tqisa_fp = file_paths_root + 'sample_main.tqisa'
        output_fp       = file_paths_root + 'sample_main.qisa.out'

        try:
            with open(result_fp,'r') as result_file:
                result_str = result_file.read()
        except :
            raise Exception('Unable to read result file')

        
