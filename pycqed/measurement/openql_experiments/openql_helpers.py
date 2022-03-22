import re
import logging
import json
import pathlib
import inspect
import numpy as np
from os import remove
from os.path import join, dirname, isfile
from typing import List, Tuple
from deprecated import deprecated

import openql as ql

from pycqed.utilities.general import suppress_stdout
from pycqed.utilities.general import is_more_recent
from pycqed.utilities.general import get_file_sha256_hash


log = logging.getLogger(__name__)

"""
FIXME:
concept should support:
- programs with 'runtime' parameters
- retrieval of return (measurement) data
- running a set of programs
- make-like 'update only if needed'
- multiple HW-platforms
- other compilers than OpenQL

"""

class OqlProgram:
    # we use a class global variable 'output_dir' to replace the former OpenQL option of the same name. Since that is a
    # global OpenQL option, it no longer has direct effect now we no longer use the OpenQL generated list of passes (see
    # self._configure_compiler). Additionally, OpenQL options are reset upon ql.initialize, so the value does not persist.
    output_dir = join(dirname(__file__), 'output')

    def __init__(
            self,
            name: str,
            platf_cfg: str,
            nregisters: int = 32
    ):
        """
        create OpenQL Program (and Platform)

        Args:
            name:
                name of the program

            platf_cfg:
                path of the platform configuration file used by OpenQL

            nregisters:
                the number of classical registers required in the program
        """


        # setup OpenQL
        ql.initialize()  # reset options, may initialize more functionality in the future

        # set OpenQL log level before anything else
        ql.set_option('log_level', 'LOG_WARNING')

        # store/initialize some parameters
        self.name = name
        self._platf_cfg = platf_cfg
        self.nregisters = nregisters  # NB: not available via platform
        self.filename = ""
        self.sweep_points = None

        # create Platform and Program
        self.platform = ql.Platform('OpenQL_Platform', platf_cfg)
        self.nqubits = self.platform.get_qubit_number()
        self.program = ql.Program(
            name,
            self.platform,
            self.nqubits,
            self.nregisters
        )  # NB: unused if we use compile_cqasm()

        # detect OpenQL backend ('eqasm_compiler') used by inspecting platf_cfg
        eqasm_compiler = ''
        with open(self._platf_cfg) as f:
            for line in f:
                if 'eqasm_compiler' in line:
                    m = re.search('"eqasm_compiler" *: *"(.*?)"', line)
                    eqasm_compiler = m.group(1)
                    break
        if eqasm_compiler == '':
            log.error(f"key 'eqasm_compiler' not found in file '{self._platf_cfg}'")

        # determine architecture and extension of generated file
        if eqasm_compiler == 'cc_light_compiler':
            # NB: OpenQL>=0.9.0 no longer has a backend for CC-light
            self._arch = 'CCL'
            self._ext = '.qisa'  # CC-light, QCC
        else:
            self._arch = 'CC'
            self._ext = '.vq1asm'  # CC

        # save name of file that OpenQL will generate on compilation to allow uploading
        # NB: for cQasm, the actual name is determined by 'pragma @ql.name' in the source, not by self.name,
        # so users must maintain consistency
        self.filename = join(OqlProgram.output_dir, self.name + self._ext)

        # map file for OpenQL>=0.10.3
        self._map_filename = join(OqlProgram.output_dir, self.name + ".map")


    def add_kernel(self, k: ql.Kernel) -> None:
        self.program.add_kernel(k)


    def create_kernel(
            self,
            kname: str
    ) -> ql.Kernel:
        """
        Wrapper around constructor of openQL "Kernel" class.
        """
        kname = kname.translate(
            {ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+ "})

        k = ql.Kernel(kname, self.platform, self.nqubits, self.nregisters)
        return k


    def compile(
            self,
            quiet: bool = False,
            extra_pass_options: List[Tuple[str, str]] = None
    ) -> None:
        """
        Compile an OpenQL Program created using the legacy Program/Kernel API.

        Args:
            quiet:
                suppress all output (not recommended, because warnings are hidden)

            extra_pass_options:
                extra pass options for OpenQL. These consist of a tuple 'path, value' where path is structured as
                "<passName>.<passOption>" and value is the option value, see
                https://openql.readthedocs.io/en/latest/reference/python.html#openql.Compiler.set_option

                See https://openql.readthedocs.io/en/latest/gen/reference_passes.html for passes and their options
        """

        if quiet:
            with suppress_stdout():
                self.program.compile()
        else:  # show warnings
            c = self._configure_compiler("", extra_pass_options)
            c.compile(self.program)


    def compile_cqasm(
            self,
            src: str,
            extra_pass_options: List[Tuple[str, str]] = None
    ) -> None:
        """
        Compile a string with cQasm source code.

        Note that, contrary to the behaviour of compile(), the program runs just once by default, since looping can be
        easily and more subtly performed in cQasm if desired.

        Args:
            src:
                the cQasm source code string

            extra_pass_options:
                extra pass options for OpenQL. These consist of a tuple 'path, value' where path is structured as
                "<passName>.<passOption>" and value is the option value, see
                https://openql.readthedocs.io/en/latest/reference/python.html#openql.Compiler.set_option

                See https://openql.readthedocs.io/en/latest/gen/reference_passes.html for passes and their options
        """

        # save src to file (as needed by pass 'io.cqasm.Read')
        src_filename = OqlProgram.output_dir+"/"+self.name+".cq"
        pathlib.Path(src_filename).write_text(inspect.cleandoc(src))

        c = self._configure_compiler(src_filename, extra_pass_options)
        c.compile_with_frontend(self.platform)


    def get_map(self) -> dict:
        """
        get map data produced by OpenQL>=0.10.3
        """

        return json.load(self._map_filename)


    def get_measurement_map(self) -> dict:
        """
        get measurements from map data produced by OpenQL>=0.10.3
        """

        data = self.get_map()
        return data["measurements"]


    # NB: used in clifford_rb_oql.py to skip both generation of RB sequences, and OpenQL compilation if
    # contents of platf_cfg or clifford_rb_oql (i.e. the Python file that generates the RB sequence) have changed
    def check_recompilation_needed_hash_based(
            self,
            clifford_rb_oql: str,
            recompile: bool = True,
    ) -> dict:
        """
        Similar functionality to the deprecated `check_recompilation_needed` but
        based on a file that is generated alongside with the program file
        containing hashes of the files that are relevant to the generation of the
        RB sequences and that might be modified somewhat often

        NB: Not intended for stand alone use!
        The code invoking this function should later invoke:
            `os.rename(recompile_dict["tmp_file"], recompile_dict["file"])` # FIXME: create member function for that

        The behavior of this function depends on the recompile argument.
        recompile:
            True -> True, the program should be compiled

            'as needed' -> compares filename to timestamp of config
                and checks if the file exists, if required recompile.
            False -> checks if the file exists, if it doesn't
                compilation is required and raises a ValueError.
                Use carefully, only if you know what you are doing!
                Use 'as needed' to stay safe!
        """

        hashes_ext = ".hashes"
        tmp_ext = ".tmp"
        rb_system_hashes_fn = self.filename + hashes_ext
        tmp_fn = rb_system_hashes_fn + tmp_ext

        platf_cfg_hash = get_file_sha256_hash(self._platf_cfg, return_hexdigest=True)
        this_file_hash = get_file_sha256_hash(clifford_rb_oql, return_hexdigest=True)
        file_hashes = {self._platf_cfg: platf_cfg_hash, clifford_rb_oql: this_file_hash}

        _recompile = False
        if not isfile(self.filename):
            if recompile is False:
                raise ValueError('No file:\n{}'.format(self.filename))
            else:
                # Force recompile, there is no program file
                _recompile |= True  # FIXME: why "|="?

        # Determine if compilation is needed based on the hashed files
        if not isfile(rb_system_hashes_fn):
            # There is no file with the hashes, we must compile to be safe
            _recompile |= True
        else:
            # Hashes exist, we use them to determine if recompilations is needed
            with open(rb_system_hashes_fn) as json_file:
                hashes_dict = json.load(json_file)
            # Remove file to signal a compilation in progress
            remove(rb_system_hashes_fn)

            for fn in file_hashes.keys():
                # Recompile becomes true if any of the hashed files has a different
                # hash now
                _recompile |= hashes_dict.get(fn, "") != file_hashes[fn]

        # Write the updated hashes
        # We use a temporary file such that for parallel compilations, if the
        # process is interrupted before the end there will be no hash and
        # recompilation will be forced
        pathlib.Path(tmp_fn).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(tmp_fn).write_text(json.dumps(file_hashes))

        res_dict = {
            "file": rb_system_hashes_fn,
            "tmp_file": tmp_fn
        }

        if recompile is False:
            if _recompile is True:
                log.warning(
                    "`{}` or\n`{}`\n might have been modified! Are you sure you didn't"
                    " want to compile?".format(self._platf_cfg, clifford_rb_oql)
                )
            res_dict["recompile"] = False
        elif recompile is True:
            # Enforce recompilation
            res_dict["recompile"] = True
        elif recompile == "as needed":
            res_dict["recompile"] = _recompile

        return res_dict

    #############################################################################
    # Calibration points
    #############################################################################

    """
    FIXME: while refactoring these from separate functions to class methods, it
     was found that most functions returned the program (that was provided as a
     parameter, which makes no sense), and that the return parameter was mostly
     ignored (which makes no difference). The function documentation was
     inconsistent with the actual code in this respect, probably as a result of
     earlier refactoring.
     Function 'add_multi_q_cal_points' would return different types dependent
     on a boolean parameter 'return_comb', but no cases were found where this
     parameter was set to True, so this behaviour was removed
    """

    def add_single_qubit_cal_points(
            self,
            qubit_idx: int,
            f_state_cal_pts: bool = False,
            measured_qubits=None
    ) -> None:
        """
        Adds single qubit calibration points to an OpenQL program

        Args:
            qubit_idx:
                index of qubit

            f_state_cal_pts:
                if True, add calibration points for the 2nd exc. state

            measured_qubits:
                selects which qubits to perform readout on. If measured_qubits == None, it will default
                to measuring the qubit for which there are cal points.

        Returns:

        """

        if measured_qubits == None:
            measured_qubits = [qubit_idx]

        for i in np.arange(2):
            k = self.create_kernel("cal_gr_" + str(i))
            k.prepz(qubit_idx)
            k.gate('wait', measured_qubits, 0)
            for measured_qubit in measured_qubits:
                k.measure(measured_qubit)
            k.gate('wait', measured_qubits, 0)
            self.add_kernel(k)

        for i in np.arange(2):
            k = self.create_kernel("cal_ex_" + str(i))
            k.prepz(qubit_idx)
            k.gate('rx180', [qubit_idx])
            k.gate('wait', measured_qubits, 0)
            for measured_qubit in measured_qubits:
                k.measure(measured_qubit)
            k.gate('wait', measured_qubits, 0)
            self.add_kernel(k)
        if f_state_cal_pts:
            for i in np.arange(2):
                k = self.create_kernel("cal_f_" + str(i))
                k.prepz(qubit_idx)
                k.gate('rx180', [qubit_idx])
                k.gate('rx12', [qubit_idx])
                k.gate('wait', measured_qubits, 0)
                for measured_qubit in measured_qubits:
                    k.measure(measured_qubit)
                k.gate('wait', measured_qubits, 0)
                self.add_kernel(k)


    def add_two_q_cal_points(
            self,
            q0: int,
            q1: int,
            reps_per_cal_pt: int = 1,
            f_state_cal_pts: bool = False,
            #        f_state_cal_pt_cw: int = 31,
            measured_qubits=None,
            interleaved_measured_qubits=None,
            interleaved_delay=None,
            nr_of_interleaves=1
    ) -> None:
        """
        Adds two qubit calibration points to an OpenQL program

        Args:
            q0:
                index of first qubit

            q1:
                index of second qubit

            reps_per_cal_pt:
                number of times to repeat each cal point

            f_state_cal_pts:
                if True, add calibration points for the 2nd exc. state

            measured_qubits:
                selects which qubits to perform readout on. If measured_qubits == None, it will default
                to measuring the qubit for which there are cal points

            interleaved_measured_qubits:
            interleaved_delay:
            nr_of_interleaves:

        """

        kernel_list = [] # FIXME: not really used (anymore?)
        combinations = (["00"] * reps_per_cal_pt +
                        ["01"] * reps_per_cal_pt +
                        ["10"] * reps_per_cal_pt +
                        ["11"] * reps_per_cal_pt)
        if f_state_cal_pts:
            extra_combs = (['02'] * reps_per_cal_pt + ['20'] * reps_per_cal_pt +
                           ['22'] * reps_per_cal_pt)
            combinations += extra_combs

        if measured_qubits == None:
            measured_qubits = [q0, q1]

        for i, comb in enumerate(combinations):
            k = self.create_kernel('cal{}_{}'.format(i, comb))
            k.prepz(q0)
            k.prepz(q1)
            if interleaved_measured_qubits:
                for j in range(nr_of_interleaves):
                    for q in interleaved_measured_qubits:
                        k.measure(q)
                    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0)
                    if interleaved_delay:
                        k.gate('wait', [0, 1, 2, 3, 4, 5, 6],
                               int(interleaved_delay * 1e9))

            if comb[0] == '0':
                k.gate('i', [q0])
            elif comb[0] == '1':
                k.gate('rx180', [q0])
            elif comb[0] == '2':
                k.gate('rx180', [q0])
                # FIXME: this is a workaround
                # k.gate('rx12', [q0])
                k.gate('cw_31', [q0])

            if comb[1] == '0':
                k.gate('i', [q1])
            elif comb[1] == '1':
                k.gate('rx180', [q1])
            elif comb[1] == '2':
                k.gate('rx180', [q1])
                # FIXME: this is a workaround
                # k.gate('rx12', [q1])
                k.gate('cw_31', [q1])

            # Used to ensure timing is aligned
            k.gate('wait', measured_qubits, 0)
            for q in measured_qubits:
                k.measure(q)
            k.gate('wait', measured_qubits, 0)
            kernel_list.append(k)
            self.add_kernel(k)


    def add_multi_q_cal_points(
            self,
            qubits: List[int],
            combinations: List[str] = ["00", "01", "10", "11"],
            reps_per_cal_pnt: int = 1,
            f_state_cal_pt_cw: int = 9,  # 9 is the one listed as rX12 in `mw_lutman`
            nr_flux_dance: int = None,
            flux_cw_list: List[str] = None
    ) -> None:
        """

        Args:
            qubits:
                list of qubit indices

            combinations:
                list with the target multi-qubit state
                e.g. ["00", "01", "10", "11"] or
                ["00", "01", "10", "11", "02", "20", "22"] or
                ["000", "010", "101", "111"]

            reps_per_cal_pnt:
                number of times to repeat each cal point

            f_state_cal_pt_cw:
                the cw_idx for the pulse to the ef transition.

            nr_flux_dance:
            flux_cw_list:
        """

        comb_repeated = []
        for state in combinations:
            comb_repeated += [state] * reps_per_cal_pnt

        state_to_gates = {
            "0": ["i"],
            "1": ["rx180"],
            "2": ["rx180", "cw_{:02}".format(f_state_cal_pt_cw)],
        }

        for i, comb in enumerate(comb_repeated):
            k = self.create_kernel('cal{}_{}'.format(i, comb))

            # NOTE: for debugging purposes of the effect of fluxing on readout,
            #       prepend flux dance before calibration points
            for q_state, q in zip(comb, qubits):
                k.prepz(q)
            k.gate("wait", [], 0)  # alignment

            if nr_flux_dance and flux_cw_list:
                for i in range(int(nr_flux_dance)):
                    for flux_cw in flux_cw_list:
                        k.gate(flux_cw, [0])
                    k.gate("wait", [], 0)
                # k.gate("wait", [], 20) # prevent overlap of flux with mw gates

            for q_state, q in zip(comb, qubits):
                for gate in state_to_gates[q_state]:
                    k.gate(gate, [q])
            k.gate("wait", [], 0)  # alignment
            # k.gate("wait", [], 20)  # alignment

            # for q_state, q in zip(comb, qubits):
            #     k.prepz(q)
            #     for gate in state_to_gates[q_state]:
            #         k.gate(gate, [q])
            # k.gate("wait", [], 0)  # alignment

            for q in qubits:
                k.measure(q)
            k.gate('wait', [], 0)  # alignment
            self.add_kernel(k)


    def add_two_q_cal_points_special_cond_osc(
            self,
            q0: int,
            q1: int,
            q2=None,
            reps_per_cal_pt: int = 1,
            f_state_cal_pts: bool = False,
            #        f_state_cal_pt_cw: int = 31,
            measured_qubits=None,
            interleaved_measured_qubits=None,
            interleaved_delay=None,
            nr_of_interleaves=1
    ) -> None:
        """

        Args:
            q0:
            q1:
            q2:
            reps_per_cal_pt:
                number of times to repeat each cal point

            f_state_cal_pts:
                 if True, add calibration points for the 2nd exc. state

            measured_qubits:
                selects which qubits to perform readout on. If measured_qubits == None, it will default
                to measuring the qubit for which there are cal points.

            interleaved_measured_qubits:
            interleaved_delay:
            nr_of_interleaves:

        Returns:

        """

        combinations = (["00"] * reps_per_cal_pt +
                        ["01"] * reps_per_cal_pt +
                        ["10"] * reps_per_cal_pt +
                        ["11"] * reps_per_cal_pt)
        if f_state_cal_pts:
            extra_combs = (['02'] * reps_per_cal_pt + ['20'] * reps_per_cal_pt +
                           ['22'] * reps_per_cal_pt)
            combinations += extra_combs
        if q2 is not None:
            combinations += ["Park_0", "Park_1"]

        if (measured_qubits == None) and (q2 is None):
            measured_qubits = [q0, q1]
        elif (measured_qubits == None):
            measured_qubits = [q0, q1, q2]

        for i, comb in enumerate(combinations):
            k = self.create_kernel('cal{}_{}'.format(i, comb))
            k.prepz(q0)
            k.prepz(q1)
            if q2 is not None:
                k.prepz(q2)
            if interleaved_measured_qubits:
                for j in range(nr_of_interleaves):
                    for q in interleaved_measured_qubits:
                        k.measure(q)
                    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0)
                    if interleaved_delay:
                        k.gate('wait', [0, 1, 2, 3, 4, 5, 6], int(interleaved_delay * 1e9))

            if comb[0] == '0':
                k.gate('i', [q0])
            elif comb[0] == '1':
                k.gate('rx180', [q0])
            elif comb[0] == '2':
                k.gate('rx180', [q0])
                # FIXME: this is a workaround
                # k.gate('rx12', [q0])
                k.gate('cw_31', [q0])

            if comb[1] == '0':
                k.gate('i', [q1])
            elif comb[1] == '1':
                k.gate('rx180', [q1])
            elif comb[1] == '2':
                k.gate('rx180', [q1])
                # FIXME: this is a workaround
                # k.gate('rx12', [q1])
                k.gate('cw_31', [q1])
            if comb[0] == 'P' and comb[-1] == '0':
                k.gate('i', [q2])
            elif comb[0] == 'P' and comb[-1] == '1':
                k.gate('rx180', [q2])

            # Used to ensure timing is aligned
            k.gate('wait', measured_qubits, 0)
            for q in measured_qubits:
                k.measure(q)
            k.gate('wait', measured_qubits, 0)
            self.add_kernel(k)


    #############################################################################
    # Private functions
    #############################################################################

    def _configure_compiler(
            self,
            cqasm_src_filename: str,
            extra_pass_options: List[Tuple[str, str]] = None
    ) -> ql.Compiler:
        # NB: for alternative ways to configure the compiler, see
        # https://openql.readthedocs.io/en/latest/gen/reference_configuration.html#compiler-configuration

        c = self.platform.get_compiler()


        # remove default pass list (this also removes support for most *global* options as defined in
        # https://openql.readthedocs.io/en/latest/gen/reference_options.html, except for 'log_level')
        # NB: this defeats automatic backend selection by OpenQL based on key "eqasm_compiler"
        c.clear_passes()

        # add the passes we need
        compiling_cqasm = cqasm_src_filename != ""
        if compiling_cqasm:
            # cQASM reader as very first step
            c.append_pass(
                'io.cqasm.Read',
                'reader',
                {
                    'cqasm_file': cqasm_src_filename
                }
            )

            # decomposer for legacy decompositions (those defined in the "gate_decomposition" section)
            # FIXME: comment incorrect, also decomposes new-style definitions
            # see https://openql.readthedocs.io/en/latest/gen/reference_passes.html#instruction-decomposer
            c.append_pass(
                'dec.Instructions',
                # NB: don't change the name 'legacy', see:
                # - https://openql.readthedocs.io/en/latest/gen/reference_passes.html#instruction-decomposer
                # - https://openql.readthedocs.io/en/latest/gen/reference_passes.html#predicate-key
                'legacy',
            )
        else:  # FIXME: experimental. Also decompose API input to allow use of new style decompositions
            c.append_pass(
                'dec.Instructions',
                # NB: don't change the name 'legacy', see:
                # - https://openql.readthedocs.io/en/latest/gen/reference_passes.html#instruction-decomposer
                # - https://openql.readthedocs.io/en/latest/gen/reference_passes.html#predicate-key
                'legacy',
            )

        # report the initial qasm
        c.append_pass(
            'io.cqasm.Report',
            'initial',
            {
                'output_suffix': '.cq',
                'with_timing': 'no'
            }
        )

        # schedule
        c.append_pass(
            'sch.ListSchedule',
            'scheduler',
            {
                'resource_constraints': 'yes'
            }
        )

        # report scheduled qasm
        c.append_pass(
            'io.cqasm.Report',
            'scheduled',
            {
                'output_suffix': '.cq',
            }
        )

        if self._arch == 'CC':
            # generate code using CC backend
            # NB: OpenQL >= 0.10 no longer has a CC-light backend
            c.append_pass(
                'arch.cc.gen.VQ1Asm',
                'cc_backend'
            )

        # set compiler pass options
        c.set_option('*.output_prefix', f'{OqlProgram.output_dir}/%N.%P')
        if self._arch == 'CC':
            c.set_option('cc_backend.output_prefix', f'{OqlProgram.output_dir}/%N')
        c.set_option('scheduler.scheduler_target', 'alap')
        if compiling_cqasm:
            c.set_option('cc_backend.run_once', 'yes')  # if you want to loop, write a cqasm loop

        # finally, set user pass options
        if extra_pass_options is not None:
            for opt, val in extra_pass_options:
                c.set_option(opt, val)

        log.debug("\n" + c.dump_strategy())
        return c


##########################################################################
# compatibility functions
# FIXME: these are deprecated, but note that many scripts use these.
#  In many functions we return the program object for legacy
#  compatibility, although we specify a return type of " -> None" for
#  those that use PyCharm or an other tool aware of type inconsistencies
#  (which is highly recommended)
##########################################################################

@deprecated(version='0.4', reason="use class OqlProgram")
def create_program(
        name: str,
        platf_cfg: str,
        nregisters: int = 32
) -> OqlProgram:
    return OqlProgram(name, platf_cfg, nregisters)


@deprecated(version='0.4', reason="use class OqlProgram")
def create_kernel(
        kname: str,
        program: OqlProgram
) -> ql.Kernel:
    return program.create_kernel(kname)


@deprecated(version='0.4', reason="use class OqlProgram")
def compile(
        p: OqlProgram,
        quiet: bool = False,
        extra_openql_options: List[Tuple[str,str]] = None
) -> None:
    p.compile(quiet, extra_openql_options)
    return p # legacy compatibility


@deprecated(version='0.4', reason="use class OqlProgram")
def add_single_qubit_cal_points(
        p: OqlProgram,
        qubit_idx: int,
        f_state_cal_pts: bool = False,
        measured_qubits=None
) -> None:
    p.add_single_qubit_cal_points(qubit_idx, f_state_cal_pts, measured_qubits)
    return p # legacy compatibility


@deprecated(version='0.4', reason="use class OqlProgram")
def add_two_q_cal_points(
        p: OqlProgram,
        q0: int,
        q1: int,
        reps_per_cal_pt: int = 1,
        f_state_cal_pts: bool = False,
#        f_state_cal_pt_cw: int = 31, # FIXME: iold, unused parameter
        measured_qubits=None,
        interleaved_measured_qubits=None,
        interleaved_delay=None,
        nr_of_interleaves=1
) -> None:
    p.add_two_q_cal_points(q0, q1, reps_per_cal_pt, f_state_cal_pts, measured_qubits, interleaved_measured_qubits, interleaved_delay, nr_of_interleaves)
    return p # legacy compatibility


@deprecated(version='0.4', reason="use class OqlProgram")
def add_multi_q_cal_points(
    p: OqlProgram,
    qubits: List[int],
    combinations: List[str] = ["00", "01", "10", "11"],
    reps_per_cal_pnt: int = 1,
    f_state_cal_pt_cw: int = 9,  # 9 is the one listed as rX12 in `mw_lutman`
    nr_flux_dance: int = None,
    flux_cw_list: List[str] = None,
    return_comb=False
):
    """
    Add a list of kernels containing calibration points in the program `p`

    Args:
        p               : OpenQL  program to add calibration points to
        qubits          : list of int
        combinations    : list with the target multi-qubit state
            e.g. ["00", "01", "10", "11"] or
            ["00", "01", "10", "11", "02", "20", "22"] or
            ["000", "010", "101", "111"]
        reps_per_cal_pnt : number of times to repeat each cal point
        f_state_cal_pt_cw: the cw_idx for the pulse to the ef transition.
    Returns:
        p
    """
    kernel_list = []  # Not sure if this is needed
    comb_repetead = []
    for state in combinations:
        comb_repetead += [state] * reps_per_cal_pnt

    state_to_gates = {
        "0": ["i"],
        "1": ["rx180"],
        "2": ["rx180", "cw_{:02}".format(f_state_cal_pt_cw)],
    }

    for i, comb in enumerate(comb_repetead):
        k = create_kernel('cal{}_{}'.format(i, comb), p)

        # NOTE: for debugging purposes of the effect of fluxing on readout,
        #       prepend flux dance before calibration points
        for q_state, q in zip(comb, qubits):
            k.prepz(q)
        k.gate("wait", [], 0)  # alignment

        if nr_flux_dance and flux_cw_list:
            for i in range(int(nr_flux_dance)):
                for flux_cw in flux_cw_list:
                    k.gate(flux_cw, [0])
                k.gate("wait", [], 0)
            # k.gate("wait", [], 20) # prevent overlap of flux with mw gates

        for q_state, q in zip(comb, qubits):
            for gate in state_to_gates[q_state]:
                k.gate(gate, [q])
        k.gate("wait", [], 0)  # alignment
        # k.gate("wait", [], 20) # prevent overlap of flux with measurement pulse

        for q in qubits:
            k.measure(q)
        k.gate('wait', [], 0)  # alignment
        kernel_list.append(k)
        p.add_kernel(k)

    if return_comb:
        return comb_repetead
    else:
        return p

@deprecated(version='0.4', reason="use class OqlProgram")
def add_two_q_cal_points_special_cond_osc(
        p, q0: int, q1: int,
        q2=None,
        reps_per_cal_pt: int = 1,
        f_state_cal_pts: bool = False,
        # f_state_cal_pt_cw: int = 31,
        measured_qubits=None,
        interleaved_measured_qubits=None,
        interleaved_delay=None,
        nr_of_interleaves=1
) -> None:
    p.add_two_q_cal_points_special_cond_osc(
        q0, q1, q2,
        reps_per_cal_pt,
        f_state_cal_pts,
        measured_qubits,
        interleaved_measured_qubits,
        interleaved_delay,
        nr_of_interleaves
    )
    return p # legacy compatibility


# FIXME: move?
#############################################################################
# RamZZ measurement
#############################################################################
def measure_ramzz(k, qubit_idx: int, wait_time_ns: int):
    """
    Helper function that adds a ramsey readout sequence to the specified qubit
    on the specified kernel. Assumes that the qubit was already initialised.

    Input pars:
        k:              Kernel to add ramsey readout sequence to
        qubit_idx:      Qubit to undergo ramsey sequence
        wait_time_ns:   Wait time in-between pi/2 pulses
    Output pars:
        None
    """

    k.gate('ry90', [qubit_idx])
    k.gate('wait', wait_time_ns, [qubit_idx])
    k.gate('rym90', [qubit_idx])
    k.measure(qubit_idx)


#############################################################################
# File modifications
#############################################################################

def is_compatible_openql_version_cc() -> bool:
    """
    test whether OpenQL version is compatible with Central Controller
    """
    return ql.get_version() > '0.10.0'  # NB: 0.10.0 does not contain new CC backend yet


def clocks_to_s(time, clock_cycle=20e-9):
    """
    Converts a time in clocks to a time in s
    """
    return time * clock_cycle


#############################################################################
# Recompilation helpers
#############################################################################

def check_recompilation_needed_hash_based(
        program_fn: str,
        platf_cfg: str,
        clifford_rb_oql: str,
        recompile: bool = True,
):
    raise DeprecationWarning("use OqlProgram.check_recompilation_needed_hash_based")


@deprecated(reason="Use `check_recompilation_needed_hash_based`!")
def check_recompilation_needed(
        program_fn: str,
        platf_cfg: str,
        recompile=True
) -> bool:
    """
    determines if compilation of a file is needed based on it's timestamp
    and an optional recompile option

    The behavior of this function depends on the recompile argument.

    recompile:
        True -> True, the program should be compiled

        'as needed' -> compares filename to timestamp of config
            and checks if the file exists, if required recompile.
        False -> checks if the file exists, if it doesn't
            compilation is required and raises a ValueError.
            Use carefully, only if you know what you are doing!
            Use 'as needed' to stay safe!
    """
    if recompile is True:
        return True  # compilation is enforced
    elif recompile == 'as needed':
        # In case you ever think of a hash-based check mind that this
        # function is called in parallel multiprocessing sometime!!!
        if isfile(program_fn) and is_more_recent(program_fn, platf_cfg):
            return False  # program file is good for using
        else:
            return True  # compilation is required
    elif recompile is False:
        if isfile(program_fn):
            if is_more_recent(platf_cfg, program_fn):
                log.warning("File {}\n is more recent"
                    "than program, use `recompile='as needed'` if you"
                    " don't know what this means!".format(platf_cfg))
            return False
        else:
            raise ValueError('No file:\n{}'.format(platf_cfg))
    else:
        raise NotImplementedError(
            'recompile should be True, False or "as needed"')

#############################################################################
# Multiple program loading helpers
#############################################################################

def load_range_of_oql_programs(
        programs,
        counter_param,
        CC
) -> None:
    """
    This is a helper function for running an experiment that is spread over
    multiple OpenQL programs such as RB.
    """
    program = programs[counter_param()]
    counter_param((counter_param() + 1) % len(programs))
    CC.eqasm_program(program.filename)


def load_range_of_oql_programs_from_filenames(
        programs_filenames: list,
        counter_param,
        CC
) -> None:
    """
    This is a helper function for running an experiment that is spread over
    multiple OpenQL programs such as RB.

    [2020-07-04] this is a modification of the above function such that only
    the filename is passed and not a OpenQL program, allowing for parallel
    program compilations using the multiprocessing of python (only certain
    types of data can be returned from the processing running the
    compilations in parallel)
    """
    fn = programs_filenames[counter_param()]
    counter_param((counter_param() + 1) % len(programs_filenames))
    CC.eqasm_program(fn)


def load_range_of_oql_programs_varying_nr_shots(
        programs,
        counter_param,
        CC,
        detector
) -> None:
    """
    This is a helper function for running an experiment that is spread over
    multiple OpenQL programs of varying length such as GST.

    Everytime the detector is called it will also modify the number of sweep
    points in the detector.
    """
    program = programs[counter_param()]
    counter_param((counter_param() + 1) % len(programs))
    CC.eqasm_program(program.filename)

    detector.nr_shots = len(program.sweep_points)
