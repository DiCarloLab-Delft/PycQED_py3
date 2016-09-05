import zishell as zis
import zhinst.ziPython as zi
import zhinst.utils as zi_utils
import time

from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals

"""
This contains no driver yet! needs to be done.

Installation instructions for Zurich Instrument Libraries.
1. pip install zhinst (python3 module provided by ZI)
    (this will put it in site-packages)
2. manually past zishell.py in the zhinst directory
3. add zishell.py to the zhinst __init__.py
"""


class UHFQC(Instrument):
    '''
    This is the qcodes driver for the 1.8 Gsample/s UHF-QC developed
    by Zurich Instruments.

    Requirements:
    Installation instructions for Zurich Instrument Libraries.
    1. pip install zhinst (python3 module provided by ZI)
        (this will put it in site-packages)
    2. manually past zishell.py in the zhinst directory
    3. add zishell.py to the zhinst __init__.py
    todo:
    - write all fcuncions for data acquisition
    - write all functions for AWG control
    '''
    def __init__(self, name, server_name, **kw):
        t0 = time.time()
        super().__init__(name, server_name)

        # parameter structure: [ZI node path, data type, min value, max value]
        parameters = [
                    ['quex/deskew/0/col/0','d',-1,1],
                    ['quex/deskew/0/col/1','d',-1,1],
                    ['quex/deskew/1/col/0','d',-1,1],
                    ['quex/deskew/1/col/1','d',-1,1],
                    ['quex/wint/length','i',0,3567587328]
                    ['quex/wint/delay','i',0,1020]
                    ['quex/wint/weights/0/real','v']
                    ['quex/wint/weights/0/imag','v']
                    ['quex/wint/weights/1/real','v']
                    ['quex/wint/weights/1/imag','v']
                    ['quex/wint/weights/2/real','v']
                    ['quex/wint/weights/2/imag','v']
                    ['quex/wint/weights/3/real','v']
                    ['quex/wint/weights/3/imag','v']
                    ['quex/iavg/length','i',0,4096]
                    ['quex/iavg/delay','i',0,1]
                    ['quex/iavg/avgcnt','i',-1,1]
                    ['quex/iavg/acqcnt','d',-1,1]
                    ['quex/iavg/readout','d',-1,1]
                    ['quex/iavg/data/0','d',-1,1]
                    ['quex/iavg/data/1','d',-1,1]
                    ['quex/rot/0/real','d',-1,1]
                    ['quex/rot/0/imag','d',-1,1]
                    ['quex/rot/1/real','d',-1,1]
                    ['quex/rot/1/imag','d',-1,1]
                    ['quex/rot/2/real','d',-1,1]
                    ['quex/rot/2/imag','d',-1,1]
                    ['quex/rot/3/real','d',-1,1]
                    ['quex/rot/3/imag','d',-1,1]
                    ['quex/trans/0/col/0/real','d',-1,1]
                    ['quex/trans/0/col/0/imag','d',-1,1]
                    ['quex/trans/0/col/1/real','d',-1,1]
                    ['quex/trans/0/col/1/imag','d',-1,1]
                    ['quex/trans/0/col/2/real','d',-1,1]
                    ['quex/trans/0/col/2/imag','d',-1,1]
                    ['quex/trans/0/col/3/real','d',-1,1]
                    ['quex/trans/0/col/3/imag','d',-1,1]
                    ['quex/trans/1/col/0/real','d',-1,1]
                    ['quex/trans/1/col/0/imag','d',-1,1]
                    ['quex/trans/1/col/1/real','d',-1,1]
                    ['quex/trans/1/col/1/imag','d',-1,1]
                    ['quex/trans/1/col/2/real','d',-1,1]
                    ['quex/trans/1/col/2/imag','d',-1,1]
                    ['quex/trans/1/col/3/real','d',-1,1]
                    ['quex/trans/1/col/3/imag','d',-1,1]
                    ['quex/trans/2/col/0/real','d',-1,1]
                    ['quex/trans/2/col/0/imag','d',-1,1]
                    ['quex/trans/2/col/1/real','d',-1,1]
                    ['quex/trans/2/col/1/imag','d',-1,1]
                    ['quex/trans/2/col/2/real','d',-1,1]
                    ['quex/trans/2/col/2/imag','d',-1,1]
                    ['quex/trans/2/col/3/real','d',-1,1]
                    ['quex/trans/2/col/3/imag','d',-1,1]
                    ['quex/trans/3/col/0/real','d',-1,1]
                    ['quex/trans/3/col/0/imag','d',-1,1]
                    ['quex/trans/3/col/1/real','d',-1,1]
                    ['quex/trans/3/col/1/imag','d',-1,1]
                    ['quex/trans/3/col/2/real','d',-1,1]
                    ['quex/trans/3/col/2/imag','d',-1,1]
                    ['quex/trans/3/col/3/real','d',-1,1]
                    ['quex/trans/3/col/3/imag','d',-1,1]
                    ['quex/thres/0/level','d',-1,1]
                    ['quex/thres/1/level','d',-1,1]
                    ['quex/thres/2/level','d',-1,1]
                    ['quex/thres/3/level','d',-1,1]
                    ['quex/rl/length','d',-1,1]
                    ['quex/rl/avgcnt','i',-1,1]
                    ['quex/rl/acqcnt','i',-1,1]
                    ['quex/rl/readout','d',-1,1]
                    ['quex/rl/source','d',-1,1]
                    ['quex/rl/data/0','d',-1,1]
                    ['quex/rl/data/1','d',-1,1]
                    ['quex/rl/data/2','d',-1,1]
                    ['quex/rl/data/3','d',-1,1]
                    ['awgs/0/enable','d',-1,1]
                    ['awgs/0/single','d',-1,1]
                    ['awgs/0/time','d',-1,1]
                    ['awgs/0/outputs/0/amplitude','d',-1,1]
                    ['awgs/0/outputs/1/amplitude','d',-1,1]
                    ['awgs/0/outputs/0/mode','d',-1,1]
                    ['awgs/0/outputs/1/mode','d',-1,1]
                    ['awgs/0/sequencer/pc','d',-1,1]
                    ['awgs/0/sequencer/status','d',-1,1]
                    ['awgs/0/sequencer/continue','d',-1,1]
                    ['awgs/0/sequencer/next','d',-1,1]
                    ['awgs/0/sequencer/memoryusage','d',-1,1]
                    ['awgs/0/userregs/0','d',-1,1]
                    ['awgs/0/userregs/1','d',-1,1]
                    ['awgs/0/userregs/2','d',-1,1]
                    ['awgs/0/userregs/3','d',-1,1]
                    ['awgs/0/userregs/4','d',-1,1]
                    ['awgs/0/userregs/5','d',-1,1]
                    ['awgs/0/userregs/6','d',-1,1]
                    ['awgs/0/userregs/7','d',-1,1]
                    ['awgs/0/userregs/8','d',-1,1]
                    ['awgs/0/userregs/9','d',-1,1]
                    ['awgs/0/userregs/10','d',-1,1]
                    ['awgs/0/userregs/11','d',-1,1]
                    ['awgs/0/userregs/12','d',-1,1]
                    ['awgs/0/userregs/13','d',-1,1]
                    ['awgs/0/userregs/14','d',-1,1]
                    ['awgs/0/userregs/15','d',-1,1]
                    ['awgs/0/waveform/index','d',-1,1]
                    ['awgs/0/waveform/memoryusage','d',-1,1]
                    ['awgs/0/elf/checksum','d',-1,1]
                    ['awgs/0/elf/memoryusage','d',-1,1]
                    ['awgs/0/triggers/0/level','d',-1,1]
                    ['awgs/0/triggers/0/hysteresis/absolute','d',-1,1]
                    ['awgs/0/triggers/0/hysteresis/relative','d',-1,1]
                    ['awgs/0/triggers/0/hysteresis/mode','d',-1,1]
                    ['awgs/0/triggers/0/rising','d',-1,1]
                    ['awgs/0/triggers/0/falling','d',-1,1]
                    ['awgs/0/triggers/0/channel','d',-1,1]
                    ['awgs/0/triggers/0/force','d',-1,1]
                    ['awgs/0/triggers/0/gate/enable','d',-1,1]
                    ['awgs/0/triggers/0/gate/inputselect','d',-1,1]
                    ['awgs/0/triggers/1/level','d',-1,1]
                    ['awgs/0/triggers/1/hysteresis/absolute','d',-1,1]
                    ['awgs/0/triggers/1/hysteresis/relative','d',-1,1]
                    ['awgs/0/triggers/1/hysteresis/mode','d',-1,1]
                    ['awgs/0/triggers/1/rising','d',-1,1]
                    ['awgs/0/triggers/1/falling','d',-1,1]
                    ['awgs/0/triggers/1/channel','d',-1,1]
                    ['awgs/0/triggers/1/force','d',-1,1]
                    ['awgs/0/triggers/1/gate/enable','d',-1,1]
                    ['awgs/0/triggers/1/gate/inputselect','d',-1,1]
                    ['awgs/0/auxtriggers/0/channel','d',-1,1]
                    ['awgs/0/auxtriggers/1/channel','d',-1,1]
                    ['awgs/0/elf/data','d',-1,1]
                    ['awgs/0/waveform/descriptors','d',-1,1]
                    ['awgs/0/waveform/data','d',-1,1]
                    ['awgs/0/sequencer/program','d',-1,1]
                    ['awgs/0/sequencer/assembly','d',-1,1]
                    ['awgs/0/elf/name','d',-1,1]
                    ['awgs/0/sweep/awgtrigs/0','d',-1,1]
                    ['awgs/0/sweep/awgtrigs/1','d',-1,1]
                    ['awgs/0/sweep/awgtrigs/2','d',-1,1]
                    ['awgs/0/sweep/awgtrigs/3','d',-1,1]
                    ]
        setd = zis.setd
        getd = zis.getd
        seti = zis.seti
        geti = zis.geti
        setv = zis.setv
        getv = zis.getv

        for parameter in parameters:
            parname=parameter[0].replace("/","_")
            if parameter[1]=='d':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(setd, parameter[0]),
                    get_cmd=self._gen_get_func(getd, parameter[0]),
                    vals=vals.Numbers(parameter[2], parameter[3]))
            elif parameter[1]=='i':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(seti, parameter[0]),
                    get_cmd=self._gen_get_func(geti, parameter[0]),
                    vals=vals.Ints(parameter[2], parameter[3]))
            elif parameter[1]=='v':
                self.add_parameter(
                    parname,
                    set_cmd=self._gen_set_func(seti, parameter[0]),
                    get_cmd=self._gen_get_func(geti, parameter[0]),
                    vals=vals.Anything())
            else:
                print("parameter type not recognized")

            print(parname)

        self.add_parameter('awg_sequence',
                           set_cmd=self._do_set_awg,
                           get_cmd=self._do_get_acquisition_mode,
                           vals=vals.Anything())

        self._daq = zi.ziDAQServer('127.0.0.1', 8004)
        self._device = zi_utils.autoDetect(self._daq)
        zis.connect_server('localhost', 8004)
        zis.autoconnect()
        t1 = time.time()
        print('Initialized UHFQC', self._device,
              'in %.2fs' % (t1-t0))


    def _gen_set_func(self, dev_set_type, cmd_str):
        def set_func(val):
            dev_set_type(cmd_str, val)
            return dev_set_type(cmd_str, value=val)
        return set_func

    def _gen_get_func(self, dev_get_type, ch):
        def get_func():
            return dev_get_type(ch)
        return get_func

    def reconnect(self):
        zi_utils.autoDetect(self._daq)

    def load_AWG_sequence(self, sequence):
        zis.awg(sequence)

