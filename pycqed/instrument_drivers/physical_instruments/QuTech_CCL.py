"""
    File:               QuTech_CCL.py based from ccl.py (NV)
    Author:             Kelvin Loh, QuTech
    Purpose:            Python control of Qutech CC-Light
    Prerequisites:      QCodes, QisaAs, CCLightMicrocode, SCPI
    Usage:
    Bugs:
    Tabs: 4
Revision history:
 *    0.1.0   :            : KKL  : * Created this file.
 *    0.2.0   :            : XFu  : * Refined the intialization process.
 *    0.2.1   : 13-01-2018 : XFu  : * Change the parameters into integer.
"""


from .SCPI import SCPI
from qcodes.instrument.base import Instrument
from ._CCL.CCLightMicrocode import CCLightMicrocode
from qcodes import Parameter
from collections import OrderedDict
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals
import os
import logging
import json
import sys
import traceback
import array
import re


try:
    # qisa_as can be installed from the qisa-as folder in the ElecPrj_CCLight
    # repostiory. Current version is 2.2.0 (Dec 18 2017)
    from qisa_as import QISA_Driver, qisa_qmap
except ImportError as e:
    # Do not raise error to be able to use a dummy CCL when no assembler
    # is installed.
    logging.warning(e)


log = logging.getLogger(__name__)

"""
Provide the definitions for the maximum and minimum of each expected data
types.
"""
INT32_MAX = +2147483647
INT32_MIN = -2147483648
CHAR_MAX = +127
CHAR_MIN = -128

MAX_NUM_INSN = 2**15

class CCL(SCPI):
    """
    This is class is used to serve as the driver between the user and the
    CC-Light hardware. The class starts by querying the hardware via the
    SCPI interface. The hardware then responds by providing the available
    standard qcodes parameters. This class then uses qcodes to automatically
    generate the functions necessary for the user to control the hardware.
    """
    exceptionLevel = logging.CRITICAL

    def __init__(self, name, address, port, log_level=False, **kwargs):
        self.model = name
        self._dummy_instr = False
        self.driver_version = "0.2.1"
        try:
            super().__init__(name, address, port, **kwargs)
        except Exception as e:
            # Setting up the SCPI sometimes fails the first time.  If this
            # happens a second effort to initialize and settup the connection
            # is made
            print("Failed to connect (" + str(e) + "). The system will retry" +
                  " to connect")
            self.remove_instance(self)
            super().__init__(name, address, port, **kwargs)
        self.get_idn()
        self.add_standard_parameters()
        self.add_additional_parameters()
        self._init_submodules()
        self.connect_message()

    def _init_submodules(self):
        """
        The parser helper objects are initialized in this function.
        """
        self.microcode = CCLightMicrocode()
        self.QISA = QISA_Driver()

        """
        Only works with version 4.0.0 of the assembler
        """
        curdir = os.path.dirname(__file__)
        print(f"Assembler version {self.QISA.getVersion()}")
        if self.QISA.getVersion() == '4.0.0':
            raise RuntimeError('Cannot init CC-Light driver; proper assembler for CCL (v2.0) is missing from the environment.')

        self.QISA.enableScannerTracing(False)
        self.QISA.enableParserTracing(False)
        self.QISA.setVerbose(False)

        qmap_fn = os.path.join(curdir, '_CCL', 'qisa_opcode.qmap')
        self.qisa_opcode(qmap_fn)

    def stop(self, getOperationComplete=True):
        self.run(0),
        self.enable(0)
        # Introduced to work around AWG8 triggering issue
        if getOperationComplete:
            self.getOperationComplete()

    def start(self, getOperationComplete=True):
        self.enable(1)
        self.run(1)
        # Introduced to work around AWG8 triggering issue
        if getOperationComplete:
            self.getOperationComplete()

    def add_standard_parameters(self):
        """
        Function to automatically generate the CC-Light specific functions
        from the qcodes parameters. The function uses the add_parameter
        function internally.
        """
        self.parameter_list = self._read_parameters()

        for parameter in self.parameter_list:
            name = parameter["name"]
            del parameter["name"]

            if ("vals" in parameter):
                validator = parameter["vals"]
                try:
                    val_type = validator["type"]

                    if (val_type == "Bool"):
                        parameter["vals"] = vals.Ints(0, 1)
                        parameter['get_parser'] = int

                    elif (val_type == "Non_Neg_Number"):
                        if ("range" in validator):
                            val_min = validator["range"][0]
                            val_max = validator["range"][1]
                        else:
                            val_min = 0
                            val_max = INT32_MAX

                        parameter["vals"] = vals.PermissiveInts(val_min,
                                                                val_max)
                        parameter['get_parser'] = int
                        parameter['set_parser'] = int

                    else:
                        log.warning("Failed to set the validator for the" +
                                    " parameter " + name + ", because of a" +
                                    " unknown validator type: '" + val_type +
                                    "'")

                except Exception as e:
                    log.warning(
                        "Failed to set the validator for the parameter " +
                        name + ".(%s)", str(e))

            try:
                log.info("Adding parameter:")
                for key, value in parameter.items():
                    log.info("\t", key, value)
                log.info("\n")

                self.add_parameter(name, **parameter)

            except Exception as e:
                log.warning("Failed to create the parameter " + name +
                            ", because of a unknown keyword in this" +
                            " parameter.(%s)", str(e))

    def add_additional_parameters(self):
        """
        Certain hardware specific parameters cannot be generated
        automatically. This function generates the functions to upload
        instructions for the user. They are special because
        these functions use the _upload_instructions and _upload_microcode
        functions internally, and they output binary data using the
        SCPI.py driver, which is not qcodes standard. Therefore,
        we have to manually created them specifically for CC-Light.
        """
        self.add_parameter(
            'eqasm_program',
            label=('eQASM program'),
            docstring='Uploads the eQASM program to the CC-Light. ' +
            'Valid input is a string representing the filename.',
            set_cmd=self._upload_instructions,
            vals=vals.Strings()
        )

        self.add_parameter(
            'control_store',
            label=('Control store'),
            docstring='Uploads the microcode to the CC-Light. ' +
            'Valid input is a string representing the filename.',
            set_cmd=self._upload_microcode,
            vals=vals.Strings()
        )

        self.add_parameter(
            'qisa_opcode',
            label=('QISA opcode qmap'),
            docstring='Uploads the opcode.qmap to the CC-Light assembler. ' +
            'Valid input is a string representing the filename.',
            set_cmd=self._upload_opcode_qmap,
            vals=vals.Strings()
        )

        self.add_parameter('last_loaded_instructions',
                           vals=vals.Strings(),
                           initial_value='',
                           parameter_class=ManualParameter)

    def _read_parameters(self):
        """
        This function is the 'magic'. It queries the hardware for all the
        parameters which can be put in standard QCodes parameter form.
        The hardware is expected to produce a json-formatted string which
        gets sent via TCP/IP. This function also writes out the json-file,
        for user inspection. The function returns a json string.
        """
        dir_path = os.path.dirname(os.path.abspath(__file__))

        param_file_dir = os.path.join(dir_path, '_CCL')

        if not os.path.exists(param_file_dir):
            os.makedirs(param_file_dir)

        self.param_file_name = os.path.join(param_file_dir,
                                            'ccl_param_nodes.json')

        open_file_success = False
        try:
            file = open(self.param_file_name, "r")
            open_file_success = True
        except Exception as e:
            log.info("CC-Light local parameter file {} not found ({})".format(
                            self.param_file_name, e))

        read_file_success = False
        if open_file_success:
            try:
                file_content = json.loads(file.read())
                file.close()
                read_file_success = True
            except Exception as e:
                log.info("Error while reading CC-Light local parameter file."
                        " Will update it from the hardware.")

        if read_file_success:
            self.saved_param_version = None
            if "Embedded Software Build Time" in file_content["version"]:
                self.saved_param_version = \
                    file_content["version"]["Embedded Software Build Time"]

            # check if the saved parameters have the same version number
            # as CC-Light, if yes, return the saved one.
            if (('Embedded Software Build Time' in self.version_info and
                  (self.version_info['Embedded Software Build Time'] ==
                  self.saved_param_version)) or
                self._dummy_instr):
                results = file_content["parameters"]
                return results
            else:
                log.info("CC-Light local parameter file out of date."
                    " Will update it from the hardware.")

        try:
            raw_param_string = self.ask('QUTech:PARAMeters?')
        except Exception as e:
            raise ValueError("Failed to retrieve parameter information"
                " from CC-Light hardware: ", e)

        raw_param_string = raw_param_string.replace('\t', '\n')

        try:
            results = json.loads(raw_param_string)["parameters"]
        except Exception as e:
            raise ValueError("Unrecognized parameter information received from "
                "CC-Light: \n {}".format(raw_param_string))

        try:
            # file.write(raw_param_string)
            # load dump combination is to sort and indent
            param_dict = json.loads(raw_param_string)
            file = open(self.param_file_name, 'w')
            par_str = json.dumps(param_dict,
                                  indent=4, sort_keys=True)
            file.write(par_str)
            file.close()
        except Exception as e:
            log.info("Failed to update CC-Light local parameter file:", str(e))

        return results

    def get_idn(self):
        self.version_info = {}
        try:
            id_string = ""
            id_string = self.ask('*IDN?')
            id_string = id_string.replace("'", "\"")
            self.version_info = json.loads(id_string)
        except Exception as e:
            logging.warn('Error: failed to retrive IDN from CC-Light.', str(e))

        self.version_info["Driver Version"] = self.driver_version

        return self.version_info

    def print_readable_idn(self):
        for key, value in self.version_info.items():
            print("{0: >30s} :  {1:}".format(key, value))

    def print_qisa_opcodes(self):
        if self.QISA is None:
            log.info("The assembler of CCLight has not been initialized yet.")
            return

        print(self.QISA.dumpInstructionsSpecification())

    def print_control_store(self):
        if self.microcode is None:
            log.info("The microcode unit of CCLight has not been"
                " initialized yet.")
            return

        self.microcode.dump_microcode()

    def print_qisa_with_control_store(self):
        if self.microcode is None:
            log.info("The microcode unit of CCLight has not been"
                " initialized yet.")
            return

        if self.QISA is None:
            log.info("The assembler of CCLight has not been initialized yet.")
            return

        q_arg = OrderedDict()

        insn_opcodes_str = self.QISA.dumpInstructionsSpecification()
        lines = insn_opcodes_str.split('\n')
        trimed_lines = [line.strip() for line in lines \
                        if line.startswith('def_q')]

        # put every instruction with its opcode into a dict
        for line in trimed_lines:
            name, opcode = line.split('=')
            name = name.strip().lower()
            opcode = opcode.strip().lower()

            # convert the opcode into an integer
            if opcode.startswith("0x"):
                base = 16
            elif opcode.startswith("0o"):
                base = 8
            elif opcode.startswith("0b"):
                base = 2
            else:
                base = 10
            opcode = int(opcode, base)

            if name.startswith("def_q_arg_none"):
                q_arg[name[16:-2]] = opcode
            if name.startswith("def_q_arg_tt"):
                q_arg[name[14:-2]] = opcode
            if name.startswith("def_q_arg_st"):
                q_arg[name[14:-2]] = opcode

        print("Instruction      Codewords")
        for key, value in q_arg.items():
            print('  {:<10s}:  '.format(key), end = '')
            self.microcode.print_cs_line_no_header(value)
            print("")

###############################################################################

#  These are functions which cannot be cast into the standard
#  form or not that I know of.
#  They will be added manually using add_parameter explicitly

###############################################################################

    def _upload_instructions(self, filename):
        """
        _upload_instructions expects the assembly filename and uses the
        QISA_Driver as a parser. The QISA_driver then converts it to a binary
        file which in turn gets read and internally
        converts the bytes read to a bytearray which is required by
        binBlockWrite in SCPI.
        """
        self.stop()
        if not isinstance(filename, str):
            raise ValueError(
                "The parameter filename type({}) is incorrect. "
                "It should be str.".format(type(filename)))

        success_parser = self.QISA.assemble(filename)

        if success_parser is not True:
            print("Error detected while assembling the file {}:".format(
                filename))
            print(self.QISA.getLastErrorMessage())
            raise RuntimeError("Assembling failed.")

        instHex = self.QISA.getInstructionsAsHexStrings(False)

        intarray = []
        for instr in instHex:
            intarray.append(int(instr[2:], 16))

        # add a stop instruction at the end of the program
        intarray.append(0x10000000)

        if len(intarray) > MAX_NUM_INSN:
            raise OverflowError("Failed to upload instructions: program length ({})"
                " exceeds allowed maximum value ({}).".format(len(intarray),
                    MAX_NUM_INSN))
            return


        binBlock = bytearray(array.array('L', intarray))
        # print("binblock size:", len(binBlock))
        # write binblock
        hdr = 'QUTech:UploadInstructions '
        self.binBlockWrite(binBlock, hdr)
        # print("CCL: Sending instructions to the hardware finished.")

        # write to last_loaded_instructions so it can conveniently be read back
        self.last_loaded_instructions(filename)

    def _upload_microcode(self, filename):
        """
        _upload_controls is different from send_instructions because we can
        generate the microcode from a text file and the generation of the
        microcode is done by the CCLightMicrocode.py
        """

        if not isinstance(filename, str):
            raise ValueError(
                "The parameter filename type({}) is incorrect. "
                "It should be str.".format(type(filename)))

        self.microcode.load_microcode(filename)
        binBlock = self.microcode.write_to_bin()
        if not isinstance(binBlock, bytearray):
            raise ValueError(
                "The parameter binBlock type({}) is incorrect. "
                "It should be bytearray.".format(type(binBlock)))

        # write binblock
        hdr = 'QUTech:UploadMicrocode '
        self.binBlockWrite(binBlock, hdr)

    def _upload_opcode_qmap(self, filename: str):
        success = self.QISA.loadQuantumInstructions(filename)
        if not success:
            logging.warning("Error: ", driver.getLastErrorMessage())
            logging.warning("Failed to load quantum instructions from dictionaries.")

        return success

    def _set_vsm_chan_delay(self, chanNum, value):
        """
        This function is available for the user to 'hack' the
        vsm_channel_delay using just a single function name
        """
        self.write('QUTech:VSMChannelDelay%d %d' % (chanNum, value))

    def _get_vsm_chan_delay(self, chanNum):
        """
        This function is available for the user to 'hack' the
        vsm_channel_delay using just a single function name
        """
        strCommand = 'QUTech:VSMChannelDelay%d?' % chanNum
        retval = self.ask_int(strCommand)
        return retval

    def _change_file_ext(self, qumis_name, ext):
        pathname = os.path.dirname(qumis_name)
        base_name = os.path.splitext(os.path.basename(qumis_name))[0]
        fn = os.path.join(pathname, base_name + ext)
        return fn

class dummy_CCL(CCL):
    """
    Dummy CCL all paramaters are manual and all other methods include pass
    statements
    """

    def __init__(self, name, **kw):
        Instrument.__init__(self, name=name, **kw)
        self._socket = None  # exists so close method of IP instrument works
        self._ensure_connection = True
        self._dummy_instr = True
        self.model = name
        self.version_info = self.get_idn()
        self.add_standard_parameters()
        self.add_additional_parameters()
        self.connect_message()
        # required because of annoying IP instrument
        self._port = ''
        self._confirmation = ''
        self._address = ''
        self._terminator = ''
        self._timeout = ''
        self._persistent = ''

    def get_idn(self):
        return {'driver': str(self.__class__), 'name': self.name, 'model': 'CCL'}

    def getOperationComplete(self):
        return True

    def get_operation_complete(self):  # FIXME PR #638
        return True

    def add_standard_parameters(self):
        """
        Dummy version, all are manual parameters
        """
        self.parameter_list = self._read_parameters()

        for parameter in self.parameter_list:
            name = parameter["name"]
            del parameter["name"]
            # Remove these as this is for a Dummy instrument
            if "get_cmd" in parameter:
                del parameter["get_cmd"]
            if "set_cmd" in parameter:
                del parameter["set_cmd"]

            if ("vals" in parameter):
                validator = parameter["vals"]
                try:
                    val_type = validator["type"]

                    if (val_type == "Bool"):
                        # Bool can naturally only have 2 values, 0 or 1...
                        parameter["vals"] = vals.Ints(0, 1)

                    elif (val_type == "Non_Neg_Number"):
                        # Non negative integers
                        try:
                            if ("range" in validator):
                                # if range key is specified in the parameter,
                                # then, the validator is limited to the
                                # specified min,max values
                                val_min = validator["range"][0]
                                val_max = validator["range"][1]

                            parameter["vals"] = vals.Ints(val_min, val_max)

                        except Exception as e:
                            parameter["vals"] = vals.Ints(0, INT32_MAX)
                            log.warning("Range of validator not set correctly")

                    else:
                        log.warning("Failed to set the validator for the" +
                                    " parameter " + name + ", because of a" +
                                    " unknown validator type: '" + val_type +
                                    "'")

                except Exception as e:
                    log.warning(
                        "Failed to set the validator for the parameter " +
                        name + ".(%s)", str(e))

            try:
                self.add_parameter(name, parameter_class=ManualParameter,
                                   **parameter)

            except Exception as e:
                log.warning("Failed to create the parameter " + name +
                            ", because of a unknown keyword in this" +
                            " parameter.(%s)", str(e))

    def add_additional_parameters(self):
        """
        Dummy version, parameters are added as manual parameters
        """
        self.add_parameter(
            'eqasm_program',
            label=('Upload instructions'),
            docstring='It uploads the instructions to the CC-Light. ' +
            'Valid input is a string representing the filename',
            parameter_class=ManualParameter,
            vals=vals.Strings()
        )

        self.add_parameter(
            'control_store',
            label=('Upload microcode'),
            docstring='It uploads the microcode to the CC-Light. ' +
            'Valid input is a string representing the filename',
            parameter_class=ManualParameter,
            vals=vals.Strings()
        )

        self.add_parameter(
            'qisa_opcode',
            parameter_class=ManualParameter,
            vals=vals.Strings()
        )
