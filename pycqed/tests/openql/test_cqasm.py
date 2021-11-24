# FIXME: based on OpenQL test, cleanup

import os
import unittest
import pathlib
import inspect
#from utils import file_compare

import openql as ql

import pycqed.measurement.openql_experiments.generate_CC_cfg_modular as gen
import pycqed.measurement.openql_experiments.cqasm.special_cq as spcq


curdir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(curdir, 'test_output')


class Test_cQASM(unittest.TestCase):

    def run_test_case(self, name):
        old_wd = os.getcwd()
        try:
            os.chdir(curdir)

            in_fn = 'test_' + name + '.cq'
            out_fn = 'test_output/' + name + '_out.cq'
            gold_fn = 'golden/' + name + '_out.cq'

            ql.initialize()
            ql.set_option('log_level', 'LOG_INFO')
            # ql.set_option('log_level', 'LOG_DEBUG')
            # ql.set_option('log_level', 'LOG_WARNING')

            if 1:
                # use pass manager
                pl = ql.Platform("cc", "config_cc_s17_direct_iq_openql_0_10.json")
                c = pl.get_compiler()

                if 1:
                    # insert decomposer for legacy decompositions
                    # See; see https://openql.readthedocs.io/en/latest/gen/reference_passes.html#instruction-decomposer
                    c.prefix_pass(
                        'dec.Instructions',
                        'legacy',  # sets predicate key to use legacy decompositions (FIXME: TBC)
                        {
                            'output_prefix': 'test_output/%N.%P',
                            'debug': 'yes'
                        }
                    )

                # insert cQASM reader (as very first step)
                c.prefix_pass(
                    'io.cqasm.Read',
                    'reader',
                    {
                        'cqasm_file': in_fn,
                        'output_prefix': 'test_output/%N.%P',
                        'debug': 'yes'
                    }
                )

                c.print_strategy()
                c.compile_with_frontend(pl)

        finally:
            os.chdir(old_wd)

    def run_test_case_string(self, name: str, src: str):
        pathlib.Path(curdir+"/test_"+name+".cq").write_text(inspect.cleandoc(src))
        self.run_test_case(name)

    def test_nested_rus_angle_0(self):
        gen.generate_config_modular(curdir + '/config_cc_s17_direct_iq_openql_0_10.json') # FIXME: naming, Path

        ancilla1_idx = 10
        ancilla2_idx = 8
        data_idx = 11
        angle = 0

        src = spcq.nested_rus(
            "config_cc_s17_direct_iq_openql_0_10.json",
            ancilla1_idx,
            ancilla2_idx,
            data_idx,
            angle,
        )
        self.run_test_case_string('nested_rus_angle_0', src)
