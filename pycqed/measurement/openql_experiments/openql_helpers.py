import re
import logging
import json
import numpy as np
from os import remove
from os.path import join, dirname, isfile
from typing import List, Tuple

import openql as ql
from openql import Program, Kernel, Platform

from pycqed.utilities.general import suppress_stdout
from pycqed.utilities.general import is_more_recent
from pycqed.utilities.general import get_file_sha256_hash


log = logging.getLogger(__name__)


class OqlProgram:
    def __init__(
            self,
            name: str,
            platf_cfg: str,
            nregisters: int = 32
    ):
        """
        create OpenQL Program (and Platform)

        :param name: Name of the program
        :param platf_cfg: location of the platform configuration used to construct the OpenQL Platform used.
        :param nregisters: the number of classical registers required in the program.
        """

        output_dir = join(dirname(__file__), 'output')
        ql.initialize()  # reset options, may initialize more functionality in the future
        ql.set_option('output_dir', output_dir)
        ql.set_option('scheduler', 'ALAP')

        self.name = name
        self.nregisters = nregisters  # NB: not available via platform
        self.platform = Platform('OpenQL_Platform', platf_cfg)
        self.nqubits = self.platform.get_qubit_number()
        self.program = Program(
            name,
            self.platform,
            self.nqubits,
            self.nregisters
        )
        self.output_dir = output_dir
        self.filename = ""
        self.sweep_points = None

        # detect OpenQL backend ('eqasm_compiler') used by inspecting platf_cfg
        self.eqasm_compiler = ''
        with open(platf_cfg) as f:
            for line in f:
                if 'eqasm_compiler' in line:
                    m = re.search('"eqasm_compiler" *: *"(.*?)"', line)
                    self.eqasm_compiler = m.group(1)
                    break
        if self.eqasm_compiler == '':
            logging.error(f"key 'eqasm_compiler' not found in file '{platf_cfg}'")

        # determine extension of generated file
        # if self.eqasm_compiler == 'eqasm_backend_cc':
        if 1:  # FIXME: workaround for OpenQL 0.8.1.dev4 resetting values
            ext = '.vq1asm'  # CC
        else:
            ext = '.qisa'  # CC-light, QCC

        # add filename to help finding the output files. NB: file is created by calling compile()
        self.filename = join(self.output_dir, self.name + ext)


    def add_kernel(self, k: Kernel) -> None:
        self.program.add_kernel(k)


    def create_kernel(
            self,
            kname: str
    ) -> Kernel:
        """
        Wrapper around constructor of openQL "Kernel" class.
        """
        kname = kname.translate(
            {ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+ "})

        k = Kernel(kname, self.platform, self.nqubits, self.nregisters)
        return k


    def compile(
            self,
            quiet: bool = False,
            extra_openql_options: List[Tuple[str, str]] = None
    ) -> None:
        """
        Wrapper around OpenQL Program.compile() method.
        """
        ql.set_option('output_dir', self.output_dir)
        if quiet:
            with suppress_stdout():
                self.program.compile()
        else:  # show warnings
            ql.set_option('log_level', 'LOG_ERROR')
            if extra_openql_options is not None:
                for opt, val in extra_openql_options:
                    ql.set_option(opt, val)
            self.program.compile()

        return self  # FIXME: returned unchanged, kept for compatibility for now (PR #638), but we say we return None

    #############################################################################
    # Calibration points
    #
    # FIXME: while changing these from separate functions to class methods, it
    #  was found that most functions returned the program that was provided as a
    #  parameter (which makes no sense), and that the return parameter was mostly
    #  ignored (which makes no difference). The function documentation was
    #  inconsistent with the actual code, probably as a result of earlier
    #  refactoring.
    #  Function 'add_multi_q_cal_points' would return different types dependent
    #  on a boolean parameter 'return_comb', but no cases were found where this
    #  parameter was set to True, so this behaviour was removed
    #############################################################################

    def add_single_qubit_cal_points(
            self,
            qubit_idx: int,
            f_state_cal_pts: bool = False,
            measured_qubits=None
    ) -> None:
        """
        Adds single qubit calibration points to an OpenQL program

        :param qubit_idx:
        :param f_state_cal_pts:
        :param measured_qubits: selects which qubits to perform readout on. If measured_qubits == None, it will default
        to measuring the qubit for which there are cal points.
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

        :param q0: index of first qubit
        :param q1: index of scond qubit
        :param reps_per_cal_pt: number of times to repeat each cal point
        :param f_state_cal_pts: if True, add calibration points for the 2nd exc. state
        :param measured_qubits: selects which qubits to perform readout on if measured_qubits == None, it will default to measuring the qubits for which there are cal points.
        :param interleaved_measured_qubits:
        :param interleaved_delay:
        :param nr_of_interleaves:
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
        Add a list of kernels containing calibration points in the program `p`

        :param qubits: list of int
        :param combinations: list with the target multi-qubit state
                e.g. ["00", "01", "10", "11"] or
                ["00", "01", "10", "11", "02", "20", "22"] or
                ["000", "010", "101", "111"]
        :param reps_per_cal_pnt: number of times to repeat each cal point
        :param f_state_cal_pt_cw: the cw_idx for the pulse to the ef transition.
        :param nr_flux_dance:
        :param flux_cw_list:
        """

        kernel_list = [] # FIXME: not really used (anymore?)
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
            kernel_list.append(k)
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

        :param q0:
        :param q1:
        :param q2:
        :param reps_per_cal_pt: number of times to repeat each cal point
        :param f_state_cal_pts: if True, add calibration points for the 2nd exc. state
        :param measured_qubits: selects which qubits to perform readout on. If measured_qubits == None, it will default to measuring the qubits for which there are cal points.
        :param interleaved_measured_qubits:
        :param interleaved_delay:
        :param nr_of_interleaves:
        :return: kernel_list     : list containing kernels for the calibration points FIXME: incorrect
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
            kernel_list.append(k)
            self.add_kernel(k)


##########################################################################
# compatibility functions (to be deprecated)
##########################################################################

def create_program(
        name: str,
        platf_cfg: str,
        nregisters: int = 32
) -> OqlProgram:
    return OqlProgram(name, platf_cfg, nregisters)


def create_kernel(
        kname: str,
        program: OqlProgram
) -> Kernel:
    return program.create_kernel(kname)


def compile(
        p: OqlProgram,
        quiet: bool = False,
        extra_openql_options: List[Tuple[str,str]] = None
) -> None:
    return p.compile(quiet, extra_openql_options)


def add_single_qubit_cal_points(
        p: OqlProgram,
        qubit_idx: int,
        f_state_cal_pts: bool = False,
        measured_qubits=None
) -> None:
    p.add_single_qubit_cal_points(qubit_idx, f_state_cal_pts, measured_qubits)


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


def add_multi_q_cal_points(
        p: OqlProgram,
        qubits: List[int],
        combinations: List[str] = ["00", "01", "10", "11"],
        reps_per_cal_pnt: int = 1,
        f_state_cal_pt_cw: int = 9,  # 9 is the one listed as rX12 in `mw_lutman`
        nr_flux_dance: int = None,
        flux_cw_list: List[str] = None
) -> None:
    p.add_multi_q_cal_points(qubits, combinations, reps_per_cal_pnt, f_state_cal_pt_cw, nr_flux_dance, flux_cw_list)


def add_two_q_cal_points_special_cond_osc(
        p: OqlProgram,
        q0: int,
        q1: int,
        q2 = None,
        reps_per_cal_pt: int =1,
        f_state_cal_pts: bool=False,
#        f_state_cal_pt_cw: int = 31,
        measured_qubits=None,
        interleaved_measured_qubits=None,
        interleaved_delay=None,
        nr_of_interleaves=1
) -> None:
    p.add_two_q_cal_points_special_cond_osc(q0, q1, q2, reps_per_cal_pt, f_state_cal_pts, measured_qubits, interleaved_measured_qubits, interleaved_delay, nr_of_interleaves)


#############################################################################
# Helpers
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


# FIXME: manage recompilation in this file, not at caller
def check_recompilation_needed_hash_based(
        program_fn: str,
        platf_cfg: str,
        clifford_rb_oql: str,
        recompile: bool = True,
):
    """
    Similar functionality to the deprecated `check_recompilation_needed` but
    based on a file that is generated alongside with the program file
    containing hashes of the files that are relevant to the generation of the
    RB sequences and that might be modified somewhat often

    NB: Not intended for stand alone use!
    The code invoking this function should later invoke:
        `os.rename(recompile_dict["tmp_file"], recompile_dict["file"])`

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
    rb_system_hashes_fn = program_fn + hashes_ext
    tmp_fn = rb_system_hashes_fn + tmp_ext

    platf_cfg_hash = get_file_sha256_hash(platf_cfg, return_hexdigest=True)
    this_file_hash = get_file_sha256_hash(clifford_rb_oql, return_hexdigest=True)
    file_hashes = {platf_cfg: platf_cfg_hash, clifford_rb_oql: this_file_hash}

    def write_hashes_file():
        # We use a temporary file such that for parallel compilations, if the
        # process is interrupted before the end there will be no hash and
        # recompilation will be forced
        with open(tmp_fn, "w") as outfile:
            json.dump(file_hashes, outfile)

    def load_hashes_from_file():
        with open(rb_system_hashes_fn) as json_file:
            hashes_dict = json.load(json_file)
        return hashes_dict

    _recompile = False

    if not isfile(program_fn):
        if recompile is False:
            raise ValueError('No file:\n{}'.format(platf_cfg))
        else:
            # Force recompile, there is no program file
            _recompile |= True

    # Determine if compilation is needed based on the hashed files
    if not isfile(rb_system_hashes_fn):
        # There is no file with the hashes, we must compile to be safe
        _recompile |= True
    else:
        # Hashes exist we use them to determine if recompilations is needed
        hashes_dict = load_hashes_from_file()
        # Remove file to signal a compilation in progress
        remove(rb_system_hashes_fn)

        for fn in file_hashes.keys():
            # Recompile becomes true if any of the hashed files has a different
            # hash now
            _recompile |= hashes_dict.get(fn, "") != file_hashes[fn]

    # Write the updated hashes
    write_hashes_file()

    res_dict = {
        "file": rb_system_hashes_fn,
        "tmp_file": tmp_fn
    }

    if recompile is False:
        if _recompile is True:
            log.warning(
                "`{}` or\n`{}`\n might have been modified! Are you sure you didn't"
                " want to compile?".format(platf_cfg, clifford_rb_oql)
            )
        res_dict["recompile"] = False
    elif recompile is True:
        # Enforce recompilation
        res_dict["recompile"] = True
    elif recompile == "as needed":
        res_dict["recompile"] = _recompile

    return res_dict


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
    log.error("Deprecated! Use `check_recompilation_needed_hash_based`!")

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
