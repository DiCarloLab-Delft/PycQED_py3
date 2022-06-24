"""
Usage:
pytest -v pycqed/tests/openql/test_cqasm.py
pytest -v pycqed/tests/openql/test_cqasm.py --log-level=DEBUG --capture=no
"""

import unittest
import pathlib

#from utils import file_compare

import openql as ql

import pycqed.measurement.openql_experiments.generate_CC_cfg_modular as gen
import pycqed.measurement.openql_experiments.cqasm.special_cq as spcq
import pycqed.measurement.openql_experiments.openql_helpers as oqh
from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram


this_path = pathlib.Path(__file__).parent
output_path = pathlib.Path(this_path) / 'test_output_cc'
platf_cfg_path = output_path / 'config_cc_s17_direct_iq_openql_0_10.json'


class Test_cQASM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        gen.generate_config_modular(platf_cfg_path)
        OqlProgram.output_dir = str(output_path)

    if oqh.is_compatible_openql_version_cc():  # we require unreleased version not yet available for CI
        def test_nested_rus_angle_0(self):
            ancilla1_idx = 10
            ancilla2_idx = 8
            data_idx = 11
            angle = 0

            p = spcq.nested_rus(
                str(platf_cfg_path),
                ancilla1_idx,
                ancilla2_idx,
                data_idx,
                angle
            )

            # check that a file with the expected name has been generated
            assert pathlib.Path(p.filename).is_file()

        @unittest.skip('CC backend cannot yet handle decomposition into if statements')
        def test_parameterized_gate_decomposition(self):
            name = f'test_parametrized_gate_decomp'
            src = f"""
                # Note:         file generated by {__file__}::test_parameterized_gate_decomposition
                # File:         {name}.cq
                # Purpose:      test parameterized gate decomposition

                version 1.2

                pragma @ql.name("{name}")  # set the name of generated files

                _test_rotate q[0],3                
            """

            p = OqlProgram(name, str(platf_cfg_path))  # NB: name must be identical to name set by "pragma @ql.name" above
            p.compile_cqasm(src)

        def test_hierarchical_gate_decomposition(self):
            name = f'test_hierarchical_gate_decomp'
            src = f"""
                # Note:         file generated by {__file__}::test_hierarchical_gate_decomposition
                # File:         {name}.cq
                # Purpose:      test hierarchical gate decomposition

                version 1.2

                pragma @ql.name("{name}")  # set the name of generated files

                _fluxdance_1                
            """

            p = OqlProgram(name, str(platf_cfg_path))  # NB: name must be identical to name set by "pragma @ql.name" above
            p.compile_cqasm(src)

        @unittest.skipIf(ql.get_version() < '0.10.5', "test requires later OpenQL version")
        def test_experimental_functions(self):
            name = f'test_experimental_functions'
            src = f"""
                # Note:         file generated by {__file__}::test_experimental_functions
                # File:         {name}.cq
                # Purpose:      test experimental functions

                version 1.2

                pragma @ql.name("{name}")  # set the name of generated files

                map i = creg(0)
                map foo = creg(31) 
                map b = breg(0) # FIXME: assign on PL state, not DSM

                # Configure Random Number Generators. NB: on all CCIO:  
                set foo = rnd_seed(0, 12345678)                
                set foo = rnd_threshold(0, 0.5)                
                # set b = rnd_bit(0)

                cond (rnd_bit(0)) rx180 q[0]              
            """

            p = OqlProgram(name, str(platf_cfg_path))  # NB: name must be identical to name set by "pragma @ql.name" above
            p.compile_cqasm(src)

        @unittest.skipIf(ql.get_version() < '0.10.5', "test requires later OpenQL version")
        # FIXME: fails with "Inconsistency detected in bundle contents: time travel not yet possible in this version"
        def test_rb_2q(self):
            name = f'test_rb_2q'
            qa = 'q[0]'
            qb = 'q[1]'
            src = f"""
                # Note:         file generated by {__file__}::test_rb_2q
                # File:         {name}.cq
                # Purpose:      test experimental functions

                version 1.2

                pragma @ql.name("{name}")  # set the name of generated files

                map i = creg(0)
                map r = creg(1)
                map r1 = creg(2)
                map r2 = creg(3)
                map r3 = creg(4)
                map r4 = creg(5)
                map foo = creg(31) 

                # Configure Random Number Generators. NB: on all modules:  
                # Quoting PycQED:
                #     The two qubit clifford group (C2) consists of 11520 two-qubit cliffords
                #     These gates can be subdivided into four classes.
                #         1. The Single-qubit like class  | 576 elements  (24^2)
                #         2. The CNOT-like class          | 5184 elements (24^2 * 3^2)
                #         3. The iSWAP-like class         | 5184 elements (24^2 * 3^2)
                #         4. The SWAP-like class          | 576  elements (24^2)
                #         --------------------------------|------------- +
                #         Two-qubit Clifford group C2     | 11520 elements

                .config_RND
                # NB: we are currently required by cQASM to assign the function result to a variable
                set foo = rnd_seed(0, 12345678)                
                set foo = rnd_range(0, 20)                
                set foo = rnd_seed(1, 12345678)                
                set foo = rnd_range(1, 24)                
                set foo = rnd_seed(2, 12345678)                
                set foo = rnd_range(2, 3)                

                .rb_2q
                for (i=0; i<10000; i=i+1) {{
                    set r = rnd(0)
                    if(r < 1) {{                # single-qubit-like, probability 576/11520 = 1/20
                        set r1 = rnd(1)
                        set r2 = rnd(1)
                        __test_single_qubit_like_gates {qa}, {qb}, r1, r2
                    }} else if(r < 1+9) {{      # CNOT-like, probability 5184/11520 = 9/20             
                        set r1 = rnd(1)
                        set r2 = rnd(1)
                        set r3 = rnd(2)
                        set r4 = rnd(2)
                        __test_cnot_like_gates {qa}, {qb}, r1, r2, r3, r4
                    }} else if(r < 1+9+9) {{    # iSWAP-like, probability 5184/11520 = 9/20   
                    }} else {{                  # SWAP-like, probability 576/11520 = 1/20
                    }}
                }}
            """

            p = OqlProgram(name, str(platf_cfg_path))  # NB: name must be identical to name set by "pragma @ql.name" above
            p.compile_cqasm(src)

        def test_decompose_measure_specialized(self):
            # NB: OpenQL < 0.10.3 reverses order of decompositions
            name = f'test_decompose_measure_specialized'
            src = f"""
                # Note:         file generated by {__file__}::test_decompose_measure_specialized
                # File:         {name}.cq
                # Purpose:      test specialized decomposition of measure

                version 1.2

                pragma @ql.name("{name}")  # set the name of generated files

                measure q[0]  
                measure q[6]  # specialized: prepend with rx12             
            """

            p = OqlProgram(name, str(platf_cfg_path))  # NB: name must be identical to name set by "pragma @ql.name" above
            p.compile_cqasm(src)

        # FIXME: not cqasm, move
        def test_decompose_measure_specialized_api(self):
            p = OqlProgram("test_decompose_measure_specialized_api", str(platf_cfg_path))

            k = p.create_kernel("kernel")
            k.measure(0)
            k.measure(6)
            p.add_kernel(k)
            p.compile()

        # FIXME: not cqasm, move
        @unittest.skip("FIXME: disabled, call does not match prototype")
        def test_decompose_fluxdance_api(self):
            p = OqlProgram("test_decompose_fluxdance_api", str(platf_cfg_path))

            k = p.create_kernel("kernel")
            k.gate("_flux_dance_1", 0)
            p.add_kernel(k)
            p.compile()

        def test_measure_map_output(self):
            name = f'test_measure_map_output'
            src = f"""
                # Note:         file generated by {__file__}::test_measure_map_output
                # File:         {name}.cq
                # Purpose:      test map output file for measurements

                version 1.2

                pragma @ql.name("{name}")  # set the name of generated files

                measure q[0:16]  
                measure q[6]
                barrier
                measure q[0:8]
            """

            p = OqlProgram(name, str(platf_cfg_path))  # NB: name must be identical to name set by "pragma @ql.name" above
            p.compile_cqasm(src)

        def test_qi_barrier(self):
            name = f'test_qi_barrier'
            src = f"""
                # Note:         file generated by {__file__}::test_qi_barrier
                # File:         {name}.cq
                # Purpose:      test qi barrier

                version 1.0
                qubits 5

                pragma @ql.name("{name}")  # set the name of generated files

                x q[0]
                barrier q[0,1]
                x q[1]  
            """

            p = OqlProgram(name, str(platf_cfg_path))  # NB: name must be identical to name set by "pragma @ql.name" above
            p.compile_cqasm(src)

        def test_qi_curly_brackets(self):
            name = f'test_qi_curly_brackets'
            src = f"""
                # Note:         file generated by {__file__}::test_qi_curly_brackets
                # File:         {name}.cq
                # Purpose:      test qi curly brackets

                version 1.0
                qubits 5

                pragma @ql.name("{name}")  # set the name of generated files

                {{ x q[0,2] }}
            """

            p = OqlProgram(name, str(platf_cfg_path))  # NB: name must be identical to name set by "pragma @ql.name" above
            p.compile_cqasm(src)

        def test_qi_wait(self):
            name = f'test_qi_wait'
            src = f"""
                # Note:         file generated by {__file__}::test_qi_wait
                # File:         {name}.cq
                # Purpose:      test qi wait

                version 1.0
                qubits 5

                pragma @ql.name("{name}")  # set the name of generated files

                prepz q[0]
                wait q[0], 100
                measure q[0]
            """

            p = OqlProgram(name, str(platf_cfg_path))  # NB: name must be identical to name set by "pragma @ql.name" above
            p.compile_cqasm(src)


    else:
        @unittest.skip('OpenQL version does not support CC')
        def test_fail(self):
            pass
