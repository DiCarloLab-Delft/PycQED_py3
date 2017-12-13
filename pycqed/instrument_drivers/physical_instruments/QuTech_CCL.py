"""
    File:               QuTech_CCL.py based from ccl.py (NV)
    Author:             Kelvin Loh, QuTech
    Purpose:            Python control of Qutech CC-Light
    Prerequisites:      QCodes, QisaAs, CCLightMicrocode, SCPI
    Usage:
    Bugs:
    Tabs: 4
"""


from .SCPI import SCPI
from qcodes.instrument.base import Instrument
from ._CCL.CCLightMicrocode import CCLightMicrocode
from qcodes import Parameter
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals
import os
import logging
import json
import sys
import traceback
import array
import re

# The assembler needs to be build first before it can be imported.
# The default location is a copy of the build in the _CCL hidden folder next
# to this instrument driver. A better solution is needed to prevent build
# issues, this is documented in issue #65 of the CCL repo.
curdir = (os.path.dirname(__file__))
CCLight_Assembler_dir = os.path.join(curdir, "_CCL", "qisa-as", "build")
sys.path.append(CCLight_Assembler_dir)
try:
    from pyQisaAs import QISA_Driver
except ImportError as e:
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

id_field_table = {
    'vendor'        : 'vendor',
    'model'         : 'model',
    'serial'        : 'serial',
    'fwVersion'     : 'firmware',
    'fwBuild'       : 'Firmware Build Time',
    'swVersion'     : 'Embedded Software Version',
    'swBuild'       : 'Embedded Software Build Time',
    'kmodVersion'   : 'Kernel Module Version',
    'kmodBuild'     : 'Kernel Module Build Time'
}


class CCL(SCPI):
    """
    This is class is used to serve as the driver between the user and the
    CC-Light hardware. The class starts by querying the hardware via the
    SCPI interface. The hardware then responds by providing the available
    standard qcodes parameters. This class then uses qcodes to automatically
    generate the functions necessary for the user to control the hardware.
    """
    exceptionLevel = logging.CRITICAL

    def __init__(self, name, address, port, **kwargs):
        self.model = name
        self._dummy_instr = False
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
        self._initialize_insn_microcode_parsers()
        self.connect_message()

    def _initialize_insn_microcode_parsers(self):
        """
        The parser helper objects are initialized in this function.
        """
        self.microcode = CCLightMicrocode()
        self.QISA = QISA_Driver()
        self.QISA.enableScannerTracing(False)
        self.QISA.enableParserTracing(False)
        self.QISA.setVerbose(False)

    def stop(self):
        self.run(0),
        self.enable(0)

    def start(self):
        self.enable(1)
        self.run(1)

    def add_parameter(self, name, parameter_class=Parameter, **kwargs):
        """
        Function to manually add a qcodes parameter. Useful for nonstandard
        forms of the scpiCmds.
        """
        super(CCL, self).add_parameter(name, parameter_class, **kwargs)

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
                        # Bool can naturally only have 2 values, 0 or 1...
                        parameter["vals"] = vals.Ints(0, 1)
                        parameter['get_parser'] = int

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
                            parameter['get_parser'] = int
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
                self.add_parameter(name, **parameter)

            except Exception as e:
                log.warning("Failed to create the parameter " + name +
                            ", because of a unknown keyword in this" +
                            " parameter.(%s)", str(e))

    def add_additional_parameters(self):
        """
        Certain hardware specific parameters cannot be generated
        automatically. This function generates the upload_instructions and
        upload_microcode parameters for the user. They are special because
        these functions use the _upload_instructions and _upload_microcode
        functions internally, and they output block binary data using the
        SCPI.py driver, which is not qcodes standard. Therefore,
        we have to manually create them specifically for CC-Light.
        """
        self.add_parameter(
            'upload_instructions',
            label=('Upload instructions'),
            docstring='It uploads the instructions to the CC-Light. ' +
            'Valid input is a string representing the filename',
            set_cmd=self._upload_instructions,
            vals=vals.Strings()
        )

        self.add_parameter(
            'upload_microcode',
            label=('Upload microcode'),
            docstring='It uploads the microcode to the CC-Light. ' +
            'Valid input is a string representing the filename',
            set_cmd=self._upload_microcode,
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

        param_file_dir = os.path.join(dir_path, 'qutech_parameter_files')

        if not os.path.exists(param_file_dir):
            os.makedirs(param_file_dir)

        self.param_file_name = os.path.join(param_file_dir,
                                            'ccl_param_nodes.txt')

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
                self.saved_param_version = file_content["version"]["software"]
                read_file_success = True
            except Exception as e:
                log.info("Error while reading CC-Light local parameter file."
                        " Will update it from the hardware.")

        if read_file_success:
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
            log.warning("Failed to retrieve parameter information from CC-Light"
                " hardware: ", e)

        raw_param_string = raw_param_string.replace('\t', '\n')

        try:
            results = json.loads(raw_param_string)["parameters"]
        except Exception as e:
            log.warning("Unrecognized parameter information received from "
                "CC-Light: \n {}".format(raw_param_string))

        try:
            file = open(self.param_file_name, 'w')
            file.write(raw_param_string)
            file.close()
        except Exception as e:
            log.info("Failed to update CC-Light local parameter file:", str(e))

        return results

    def get_idn(self):
        self.version_info = {}
        try:
            id_string = ""
            id_string = self.ask('*IDN?')
        except Exception:
            logging.warn('Error: failed to retrive IDN from CC-Light.')

        # the pattern that contains all symbols used to strip the
        # raw string received from CC-Light
        pattern = re.compile("\s*,\s*|\s*;\s*")

        try:
            # split the raw string into different fields, each field is a
            # string with the format "key=value"
            id_fields = [x.strip() for x in pattern.split(id_string) if x]

            # convert to dictionary
            for field in id_fields:
                a, b = field.split("=")
                if str(a) not in id_field_table:
                    raise ValueError("Unexpected IDN field received from "
                        "CC-Light.")
                self.version_info[id_field_table[str(a)]] = str(b)

        except Exception:
            logging.warn("Bad IDN message received from CC-Light: {}".format(
                id_string))

        return self.version_info

    def print_readable_idn(self):
        for key, value in self.version_info.items():
            print("{0: >30s} :  {1:}".format(key, value))

    def print_qisa_opcodes(self):
        if self.QISA is None:
            log.info("The assembler of CCLight has not been initialized yet.")
            return

        print(self.QISA.dumpOpcodeSpecification())

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

        success_parser = self.QISA.parse(filename)

        if success_parser is not True:
            print("Error detected while assembling the file {}:".format(
                filename))
            print(self.QISA.getLastErrorMessage())
            raise RuntimeError("Assembling failed.")

        instHex = self.QISA.getInstructionsAsHexStrings(False)

        if len(instHex) > 8191:
            log.warning("Failed to upload instructions: program length ({})"
                " exceeds allowed maximum value 8192.".format(len(instHex)))
            return

        intarray = []
        for instr in instHex:
            intarray.append(int(instr[2:], 16))

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
        return {'driver': str(self.__class__), 'name': self.name}

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
            'upload_instructions',
            label=('Upload instructions'),
            docstring='It uploads the instructions to the CC-Light. ' +
            'Valid input is a string representing the filename',
            parameter_class=ManualParameter,
            vals=vals.Strings()
        )

        self.add_parameter(
            'upload_microcode',
            label=('Upload microcode'),
            docstring='It uploads the microcode to the CC-Light. ' +
            'Valid input is a string representing the filename',
            parameter_class=ManualParameter,
            vals=vals.Strings()
        )
