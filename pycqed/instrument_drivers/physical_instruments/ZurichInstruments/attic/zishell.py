# N.B. !!!!!!!
# This file should match the zishell_nh.py but was provided by Yves for 
# debugging in may 2019. Note that this file is not used in the rest of PycQED. 


#!/usr/bin/ipython

################################################################################
##
## $Id: zishell.py 61974 2019-05-29 16:25:11Z sebastiand $
##
################################################################################
##
## Title      : zishell.py
## Project    :
##
################################################################################
##
## Author     : Niels Haandbaek (niels.haandbaek@zhinst.com)
## Company    : Zurich Instruments AG
## Created    : 2014/09/19
## Platform   : Python
## Standard   : none
##
################################################################################
##
## Description: Shell with simplified interface to ziPython
##
################################################################################
##
## Copyright (c) 2014/2015, Zurich Instruments AG
## All rights reserved.
##
################################################################################

import xml.etree.ElementTree as ET
import re
import subprocess
import time
import os
import inspect
import textwrap
import httplib2
import visa

from fnmatch import fnmatch

import zhinst.ziPython as zi
import zhinst.utils as utils

import numpy as np
from subprocess import call
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython.display import display, HTML

def unsigned(signed_value, bitsize):
    return signed_value if signed_value >= 0 else signed_value + (1 << bitsize)

def signed(unsigned_value, bitsize):
    return unsigned_value if unsigned_value < (1 << bitsize-1) else unsigned_value - (1 << bitsize)

def print_timing_diagram(data, bits, line_length=30):
    def _print_timing_diagram(data, bits):
        line_length = 0
        string = ''

        for i in bits:
            string += '       '
            last = (data[0] >> i) & 1
            for d in data:
                curr = (d >> i) & 1
                if last == 0 and curr == 0:
                    string += '  '
                elif last == 0 and curr == 1:
                    string += ' _'
                elif last == 1 and curr == 0:
                    string += '  '
                elif last == 1 and curr == 1:
                    string += '__'
                last = curr
            string += '<br>'
            string += 'Bit {:2d}:'.format(i)

            last = (data[0] >> i) & 1
            for d in data:
                curr = (d >> i) & 1
                if last == 0 and curr == 0:
                    string += '__'
                elif last == 0 and curr == 1:
                    string += '/ '
                elif last == 1 and curr == 0:
                    string += '\\_'
                elif last == 1 and curr == 1:
                    string += '  '
                last = curr
            string += '<br>'

        return string

    last = False
    while len(data) > 0:
        if len(data) > line_length:
            d = data[0:line_length]
            data = data[line_length:]
        else:
            d = data
            data = []
            last = True

        string = _print_timing_diagram(d, bits)
        display(HTML('<p style="line-height:1.0"><font size="-1"><code>{}</code></font></p>'.format(string)))

        if not last:
            #print('')
            #print('       ',end='')
            string = '       ' + '-'*2*line_length
            #print('-'*2*line_length)
            display(HTML('<p style="line-height:1.0"><font size="-1"><code>{}</code></font></p>'.format(string)))

def print_slice_timing_diagram(data, bits, line_length=30):
    def _print_timing_diagram(data, bits):
        line_length = 0
        string = ''

        for i in bits:
            string += '       '
            last = (data[0] >> (9*0+i)) & 1

            for d in data:
                for j in range(2):
                    curr = (d >> (9*j+i)) & 1
                    if last == 0 and curr == 0:
                        string += '  '
                    elif last == 0 and curr == 1:
                        string += ' _'
                    elif last == 1 and curr == 0:
                        string += '  '
                    elif last == 1 and curr == 1:
                        string += '__'
                    last = curr
            string += '<br>'
            string += 'Bit {:2d}:'.format(i)

            last = (data[0] >> (9*0+i)) & 1
            for d in data:
                for j in range(2):
                    curr = (d >> (9*j+i)) & 1
                    if last == 0 and curr == 0:
                        string += '__'
                    elif last == 0 and curr == 1:
                        string += '/ '
                    elif last == 1 and curr == 0:
                        string += '\\_'
                    elif last == 1 and curr == 1:
                        string += '  '
                    last = curr
            string += '<br>'

        return string

    last = False
    while len(data) > 0:
        if len(data) > int(line_length/2):
            d = data[0:int(line_length/2)]
            data = data[int(line_length/2):]
        else:
            d = data
            data = []
            last = True

        string = _print_timing_diagram(d, bits)
        display(HTML('<p style="line-height:1.0"><font size="-1"><code>{}</code></font></p>'.format(string)))

        if not last:
            #print('')
            #print('       ',end='')
            string = '       ' + '-|'*line_length
            #print('-'*2*line_length)
            display(HTML('<p style="line-height:1.0"><font size="-1"><code>{}</code></font></p>'.format(string)))

def plot_timing_diagram(data, bits, line_length=30):
    def _plot_lines(ax, pos, *args, **kwargs):
        if ax == 'x':
            for p in pos:
                plt.axvline(p, *args, **kwargs)
        else:
            for p in pos:
                plt.axhline(p, *args, **kwargs)

    def _plot_timing_diagram(data, bits):
        plt.figure(figsize=(20, 0.5*len(bits)))

        t = np.arange(len(data))
        _plot_lines('y', 2*np.arange(len(bits)), color='.5', linewidth=2)
        _plot_lines('x', t[0:-1:2], color='.5', linewidth=0.5)

        for n, i in enumerate(reversed(bits)):
            line = [((x >> i) & 1) for x in data]
            plt.step(t, np.array(line) + 2*n, 'r', linewidth = 2, where='post')
            plt.text(-0.5, 2*n, str(i))

        plt.xlim([t[0], t[-1]])
        plt.ylim([0, 2*len(bits)+1])

        plt.gca().axis('off')
        plt.show()

    last = False
    while len(data) > 0:
        if len(data) > line_length:
            d = data[0:line_length]
            data = data[line_length:]
        else:
            d = data
            data = []
            last = True

        _plot_timing_diagram(d, bits)

#with open("zishell.py") as f:
#    code = compile(f.read(), "zishell.py", 'exec')
#    exec(code, global_vars, local_vars)

class ziShellError(Exception):
    """Base class for exceptions in this module."""
    pass

class ziShellServerError(ziShellError):
    """Exception raised when no server is configured or the server cannot be reached."""
    pass

class ziShellDAQError(ziShellError):
    """Exception raised when no DAQ has been connected."""
    pass

class ziShellConnectError(ziShellError):
    """Exception raised when a device could not be connected."""
    pass

class ziShellModuleError(ziShellError):
    """Exception raised when a module has not been started."""
    pass

class ziVISADevice:
    def __init__(self, host):
        rm = visa.ResourceManager('@py')
        self.instr = rm.open_resource('TCPIP::{}::INSTR'.format(host))
        print("Connected to " + self.instr.query('*IDN?') + " over TCP/IP")


    def idn(self):
        return self.instr.query('*IDN?')


class RSScope(ziVISADevice):
    def __init__(self, host):
        super().__init__(host)

    def set_trigger_mode(self, normal):
        if normal:
            self.instr.write('TRIG:A:MODE NORMAL')
        else:
            self.instr.write('TRIG:A:MODE AUTO')

    def set_trigger_level(self, level):
        self.instr.write('TRIG:A:LEVEL {}'.format(level))

    def set_trigger_ext(self):
        self.instr.write('TRIG:A:SOURCE EXT')

    def set_trigger_src(self, chan):
        self.instr.write('TRIG:A:SOURCE CH{}'.format(chan))

    def set_trigger_slope(self,slope):
        self.instr.write('TRIG:A:TYPE EDGE')
        if slope == 1:
            self.instr.write('TRIG:A:EDGE:SLOPE POS')
        elif slope == 2:
            self.instr.write('TRIG:A:EDGE:SLOPE NEG')
        else:
            self.instr.write('TRIG:A:EDGE:SLOPE EITHER')

    def set_trigger_coupling(self, dc):
        if dc:
            self.instr.write('TRIG:A:EDGE:COUPLING DC')
        else:
            self.instr.write('TRIG:A:EDGE:COUPLING AC')

    def set_time_scale(self, scale, reference=0):
        self.instr.write('TIM:SCAL {}'.format(scale))
        self.instr.write('TIM:REF {}'.format(reference))

    def set_channel_scale(self, chan, scale, position=0.0, offset=0.0):
        self.instr.write('CHAN{}:SCAL {}'.format(chan, scale))
        self.instr.write('CHAN{}:POS {}'.format(chan, position))
        self.instr.write('CHAN{}:OFF {}'.format(chan, offset))

    def do_single_shot(self):
        self.instr.write('STOP')
        self.instr.write('SING')
        self.instr.query('*OPC?')

    def set_averaging(self, chan, enable):
      if enable:
        self.instr.write('CHAN{}:ARIT AVER'.format(chan))
      else:
        self.instr.write('CHAN{}:ARIT OFF'.format(chan))


    def set_format_real(self):
        self.instr.write('FORM:REAL')

    def set_average_count(self, count):
        self.instr.write('ACQ:COUNT {:d}'.format(count))

    def sync(self):
        self.instr.query('*OPC?')

    def do_get_screenshot(self):
        self.instr.write('HCOP:DEST MMEM')
        self.instr.write('HCOP:LANG PNG')
        self.instr.write('HCOP:COL:SCH COL')
        self.instr.write('MMEM:NAME SCREENSHOT')
        self.instr.write('HCOP:IMM')
        return self.instr.query('MMEM:DATA? SCREENSHOT.PNG')

class RTMScope(RSScope):
    def __init__(self, host):
        super().__init__(host)

    def get_channel_data(self, chans):
      self.instr.write('FORM:DATA ASCII')

      data = []
      for chan in chans:
          xorigin = float(self.instr.query('CHAN{}:DATA:XOR?'.format(chan)))
          xincrement = float(self.instr.query('CHAN{}:DATA:XINC?'.format(chan)))
          y = np.array([float(n) for n in self.instr.query('CHAN{}:DATA?'.format(chan)).split(',')])
          t = np.linspace(xorigin, xorigin + xincrement*(len(y)-1), len(y))
          data.append((t, y))

      return data

    def is_averaging_complete(self):
        res = int(self.instr.query('ACQ:AVER:COMP?'))
        if res == 1:
          return True
        else:
          return False

    def set_maximum_rate(self):
        self.instr.write('ACQ:WRAT MSAM')



class RTOScope(RSScope):
    def __init__(self, host):
        super().__init__(host)

    def get_channel_data(self, chans):
        self.instr.write('FORM:DATA ASCII')

        data = []
        for chan in chans:
            y = np.array([float(n) for n in self.instr.query('CHAN{}:DATA:VALUES?'.format(chan)).split(',')])
            header = [float(n) for n in self.instr.query('CHAN{}:DATA:HEAD?'.format(chan)).split(',')]
            t_start   = header[0]
            t_stop    = header[1]
            n_samples = header[2]
            t = np.linspace(t_start, t_stop, n_samples)
            data.append((t, y))

        return data


class ziShellDevice:
    """Holds all the global variables."""
    def __init__(self, device="", host="", port=8004, interface="1GbE", api_level=5):
        # Paths to automatically skip
        self.skip = ['*system*']

        # Server connection (host, port, api_level)
        self.server = None

        # Connection to the ziDAQ library
        self.daq = None

        # Connected devices
        self.device = device
        self.interface = interface
        self.connected = False

        # Connect to server
        if host and device and interface:
            self.connect_server(host, port, api_level)
        else:
            self.host = host
            self.port = port

    def connect_server(self, host, port=8004, api_level=5):
        self.server = (host, port, api_level)
        print("Connecting to server on host {0}:{1} using API level {2}".format(host, port, api_level))
        self.daq = zi.ziDAQServer(host, port, api_level)
        if not self.daq:
            raise(ziShellDAQError())

        self.daq.setDebugLevel(0)
        self.connected = False

        if self.device and self.interface:
            self.connect_device(self.device, self.interface)

    def reconnect(self):
        self.connect_server(self.server[0], self.server[1], self.server[2])

        if not self.daq:
            raise(ziShellDAQError())

        self.connect_device(self.device, self.interface)

    def connect_device(self, device, interface='1GbE'):
        if not self.daq:
            raise(ziShellDAQError())

        self.daq.connectDevice(device, interface)
        self.device = device
        self.interface = interface
        self.connected = True
        print("Connected to device {} over {}".format(self.device, self.interface))

    def load_settings(self, path):
        if not self.daq:
            raise(ziShellDAQError())

        def _load_settings(r, path, nodes):
            if not len(r):
                # Got leaf
                value = float(r.text)
                total_path = re.sub(r'/ch(\d+)', r'/\1', path + '/' + r.tag)

                # Check to skip
                for s in self.skip:
                    if fnmatch(total_path.lower(), s.lower()):
                        return

                # Check if it exists
                if total_path.upper() in nodes:
                    self.daq.setDouble(total_path, value)
            else:
                for c in r:
                    _load_settings(c, path + '/' + r.tag, nodes)

        try:
            tree = ET.parse(path)
            nodes = self.daq.listNodes('/' + self.device, 15)

            for r in tree.getroot().findall('./deviceSettings/nodes')[0]:
                _load_settings(r, '/' + device, nodes, self.daq, self.skip)

            self.daq.sync()

            return True
        except:
            return False

    def features(self):
        if not self.daq:
            raise(ziShellDAQError())

        rv = []
        rv.append([s for s in str(self.daq.getByte('/' + self.device + '/features/options')).split('\n') if len(s)])
        return rv

    def setd(self, path, value, sync = False):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        # Handle absolute path
        if path[0] == '/':
            if sync:
                self.daq.syncSetDouble(path, value)
            else:
                self.daq.setDouble(path, value)
        else:
            if sync:
                self.daq.syncSetDouble('/' + self.device + '/' + path, value)
            else:
                self.daq.setDouble('/' + self.device + '/' + path, value)

    def seti(self, path, value, sync = False):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        # Handle absolute path
        if path[0] == '/':
            if sync:
                self.daq.syncSetInt(path, value)
            else:
                self.daq.setInt(path, value)
        else:
            if sync:
                self.daq.syncSetInt('/' + self.device + '/' + path, value)
            else:
                self.daq.setInt('/' + self.device + '/' + path, value)

    def setv(self, path, value):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        # Handle absolute path
        if path[0] == '/':
            if "setVector" in dir(self.daq):
                self.daq.setVector(path, value)
            else:
                self.daq.vectorWrite(path, value)
        else:
            if "setVector" in dir(self.daq):
                self.daq.setVector('/' + self.device + '/' + path, value)
            else:
                self.daq.vectorWrite('/' + self.device + '/' + path, value)

    def geti(self, path, deep=True):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()
        if path[0] != '/':
            path = '/' + self.device + '/' + path

        if deep:
            self.daq.getAsEvent(path)
            timeout = 1.0
            while timeout > 0.0:
                tmp = self.daq.poll(0.1, 500, 4, True)
                if path in tmp:
                    return tmp[path]['value'][0]
                else:
                    timeout -= 0.1
            return None
        else:
            return self.daq.getInt(path)

    def getd(self, path, deep=True):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        if path[0] != '/':
            path = '/' + self.device + '/' + path

        if deep:
            self.daq.getAsEvent(path)
            tmp = self.daq.poll(0.1, 500, 4, True)
            if path in tmp:
                return tmp[path]['value'][0]
            else:
                return None
        else:
            return self.daq.getDouble(path)

    def getv(self, path):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()
        if path[0] != '/':
            path = '/' + self.device + '/' + path

        self.daq.getAsEvent(path)
        tmp = self.daq.poll(0.5, 500, 4, True)
        if path in tmp:
            return tmp[path]
        else:
            return None

    def find(self, *args):
        if not self.daq:
            raise(ziShellDAQError())

        nodes = self.daq.listNodes('/' + self.device + '/', 7)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower() for k in nodes if fnmatch(k.lower(), m.lower())]

        return nodes

    def finds(self, *args):
        if not self.daq:
            raise(ziShellDAQError())

        nodes = self.daq.listNodes('/' + self.device + '/', 15)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower() for k in nodes if fnmatch(k.lower(), m.lower())]

        return nodes

    def subs(self, path):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        if path[0] == '/':
            self.daq.subscribe(path)
        else:
            self.daq.subscribe('/' + self.device + '/' + path)

    def unsubs(self, path):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        if path[0] == '/':
            self.daq.unsubscribe(path)
        else:
            self.daq.unsubscribe('/' + self.device + '/' + path)

    def poll(self, length=0.1):
        if not self.daq:
            raise(ziShellDAQError())

        return self.daq.poll(length, 500, 4, True)

    def sync(self):
        global env

        if not self.daq:
            raise(ziShellDAQError())

        self.daq.sync()

    def __str__(self):
        return self.device

class UHFDevice(ziShellDevice):
    def __init__(self, device="", host="", port=8004, interface="1GbE", api_level=5):
        # Modules
        self.awgModule = None
        self.scopeModule = None

        super().__init__(device, host, port, interface, api_level)

    def connect_device(self, device, interface='1GbE'):
        super().connect_device(device, interface)

        # Initialize modules
        self.awgModule = self.daq.awgModule()
        self.awgModule.set('awgModule/device', self.device)
        self.awgModule.execute()
        print("Initialized awgModule")

        self.scopeModule = self.daq.scopeModule()
        self.scopeModule.set('scopeModule/mode', 0)
        self.scopeModule.subscribe('/' + self.device + '/scopes/0/wave')
        self.scopeModule.execute()
        print("Initialized scopeModule")

    def configure_awg_file(self, filename):
        if not self.daq:
            raise(ziShellDAQError())

        if not self.awgModule:
            raise(ziShellModuleError())

        self.awgModule.set('awgModule/index', 0)
        self.awgModule.set('awgModule/compiler/sourcefile', filename)
        self.awgModule.set('awgModule/compiler/start', 1)
        while self.awgModule.get('awgModule/compiler/start')['compiler']['start'][0]:
            time.sleep(0.1)

    def configure_awg_string(self, string):
        if not self.daq:
            raise(ziShellDAQError())

        if not self.awgModule:
            raise(ziShellModuleError())

        self.awgModule.set('awgModule/index', 0)
        self.awgModule.set('awgModule/compiler/sourcestring', string)
        while self.awgModule.get('awgModule/compiler/sourcestring')['compiler']['sourcestring'][0]:
            time.sleep(0.1)

    def read_scope(self, timeout=1.0, enable=True):
        if not self.daq:
            raise(ziShellDAQError())

        if not self.scopeModule:
            raise(ziShellModuleError())

        if enable:
            self.seti('scopes/0/enable', 1)

        local_timeout = timeout
        while self.scopeModule.progress()[0] >= 1.0 and local_timeout > 0:
            time.sleep(0.1)
            local_timeout -= 0.1

        local_timeout = timeout
        while self.scopeModule.progress()[0] < 1.0 and local_timeout > 0:
            time.sleep(0.1)
            local_timeout -= 0.1

        data = self.scopeModule.read()
        if self.device in data:
            if 'scopes' in data[self.device]:
                return data[self.device]['scopes']['0']['wave']

        return None

    def restart(self):
        try:
            seti('/' + self.device + '/raw/system/restart', 1)
            time.sleep(4)
            output = check_output(['timeout', '30', 'make', '-C', os.environ['work'] + '/maven', 'uart'], \
                                  stderr=subprocess.STDOUT)
            if re.search('#DHCP Success', output):
                return True
            else:
                print(output)
        except subprocess.CalledProcessError:
            pass

        return False

class HDDevice(ziShellDevice):
    DIOCTRL         = 44
    DIODLYCALIB     = 45
    SLV0DIOCTRL     = 46
    SLV1DIOCTRL     = 47
    SLV2DIOCTRL     = 48
    SLV3DIOCTRL     = 49
    DACRST          = 50
    DACPHASE        = 51
    DACIFRST        = 52
    DIOCALIB        = 53
    SLVSHIFT        = 54
    DIOFPGA         = 55
    SLVDIODLY       = 56
    SLVRSVRD        = 57
    TPGSTART        = 44+16
    TPGSTARTADDR    = 44+17
    TPGLOG2LENGTH   = 44+18
    TPGBUSY         = 44+19
    TPGSTATE        = 44+20
    TPGCNT          = 44+21
    TPGERRORCNT     = 44+22
    TPGSLV          = 44+23
    TIMEMONSTART    = 44+24
    TIMEMONADDR     = 44+25
    TIMEMONDATA     = 44+26
    FPGADACSETDELAY = 44+27
    FPGADACDELAY10  = 44+28
    FPGADACDELAY32  = 44+29
    FPGADACDELAY54  = 44+30
    FPGADACDELAY76  = 44+31

    def __init__(self, device="", host="", port=8004, interface="1GbE", api_level=5):
        self.awgModule = None

        super().__init__(device, host, port, interface, api_level)

    def connect_device(self, device, interface='1GbE'):
        super().connect_device(device, interface)

        # Initialize modules
        self.awgModule = self.daq.awgModule()
        self.awgModule.set('awgModule/device', self.device)
        self.awgModule.execute()
        print("Initialized awgModule")

    def configure_awg_file(self, filename, awg=0):
        if not self.daq:
            raise(ziShellDAQError())

        if not self.awgModule:
            raise(ziShellModuleError())

        self.awgModule.set('awgModule/index', awg)
        self.awgModule.set('awgModule/compiler/sourcefile', filename)
        self.awgModule.set('awgModule/compiler/start', 1)
        while self.awgModule.get('awgModule/compiler/start')['compiler']['start'][0]:
            time.sleep(0.1)

    def configure_awg_string(self, string, awg):
        if not self.daq:
            raise(ziShellDAQError())

        if not self.awgModule:
            raise(ziShellModuleError())

        self.awgModule.set('awgModule/index', awg)
        self.awgModule.set('awgModule/compiler/sourcestring', string)
        while self.awgModule.get('awgModule/compiler/sourcestring')['compiler']['sourcestring'][0]:
            time.sleep(0.1)

    def on(self, outputs):
        if isinstance(outputs, str):
            self.seti('sigouts/' + outputs + '/on', 1)
        else:
            for o in outputs:
                self.seti('sigouts/' + str(o) + '/on', 1)

    def off(self, outputs):
        if isinstance(outputs, str):
            self.seti('sigouts/' + outputs + '/on', 0)
        else:
            for o in outputs:
                self.seti('sigouts/' + str(o) + '/on', 0)

    def extclk(self, on):
        if on:
            self.seti('awgs/*/enable', 0)
            self.seti('system/extclk', 1)
        else:
            self.seti('system/extclk', 0)

    def sampleclk(self, rate):
        self.seti('awgs/*/enable', 0)
        self.setd('system/sampleclk', rate)

    def mode(self, outputs, amp):
        if isinstance(outputs, str):
            self.seti('raw/sigouts/' + outputs + '/mode', amp)
        else:
            for o in outputs:
                self.seti('raw/sigouts/' + str(o) + '/mode', amp)

    def direct(self, outputs, dir):
        if isinstance(outputs, str):
            self.seti('sigouts/' + outputs + '/direct', dir)
        else:
            for o in outputs:
                self.seti('sigouts/' + str(o) + '/direct', dir)

    def calibrate_dio_protocol(self, awgs):
        pass

    def dio_snapshot(self, bits=range(32), line_width=64, dump=False):
        data = self.getv('raw/dios/0/data')
        plot_timing_diagram(data[0]['vector'], bits, line_width)
        if dump:
            for x in data[0]['vector']:
                print('{:08x}'.format(x))

    def awg_dio_snapshot(self, awg, bits=range(32), line_width=64, dump=False):
        data = self.getv('awgs/{}/dio/data'.format(awg))
        plot_timing_diagram(data[0]['vector'], bits, line_width)
        if dump:
            for x in data[0]['vector']:
                print('{:08x}'.format(x))

    def dio_testmode(self, on, mode=0):
        # Enable the toggle pattern on DIO
        if on:
            self.seti('raw/debug/{}/value'.format(HDDevice.DIOFPGA), 0x0000 + 0x0001)
            if mode == 0:
                self.seti('raw/debug/{}/value'.format(HDDevice.DIOFPGA), 0x0002 + 0x0000)
            elif mode == 1:
                self.seti('raw/debug/{}/value'.format(HDDevice.DIOFPGA), 0x0002 + 0x0004)
            elif mode == 2:
                self.seti('raw/debug/{}/value'.format(HDDevice.DIOFPGA), 0x0002 + 0x0004 + 0x0010)
            elif mode == 3:
                self.seti('raw/debug/{}/value'.format(HDDevice.DIOFPGA), 0x0002 + 0x0020)
        else:
            self.seti('raw/debug/{}/value'.format(HDDevice.DIOFPGA), 0x0000)

class powerSwitches:
    def __init__(self, host = 'powerswitch1.zhinst.com', su = 0) :
        self.host = host
        m = re.match('^powerswitch([0-9]+)\.zhinst\.com$', host)
        if m:
            self.sid = int(m.group(1))-1
            if (self.sid < 0) or (self.sid > 15):
                raise RuntimeError("The requested switch '%s' is outside of the supported range." % host)
        else:
            raise RuntimeError("The given switch parameter '%s' is not a valid powerswitch name." % host)

        if su:
            self.su = 1
            self.su_label = " as superuser"
        else:
            self.su = 0
            self.su_label = ""

    def onOffMultiple(self, channels, on = 0) :
        if on:
            on_label = "ON"
        else:
            on_label = "OFF"

        for channel in channels :
            conn = httplib2.HTTPConnection('powerswitches.zhinst.com')
            conn.request("GET", "/index.php?ajax=set&sid=%d&chid=%d&val=%d&su=%d" % (self.sid, channel, on, self.su))
            response = conn.getresponse()
            if not (response.status == 200) :
                logger.error("HTTP Error %s(%d) during switching %s channel %d %s%s." % (response.reason, response.status, self.host, channel, on_label, self.su_label))
            else:
                resp_body = response.read()
                m = re.match('^ERR:(.*)$', resp_body)
                if m:
                    logger.error("Server responded with '%s' while switching %s channel %d %s%s." % (m.group(1), self.host, channel, on_label, self.su_label))

def _get_edges(value, last_value, mask):
    """
    Given two integer values representing a current and a past value,
    and a bit mask, this function will return two
    integer values representing the bits with rising (re) and falling (fe)
    edges.
    """
    changed = value ^ last_value
    re = changed & value & mask
    fe = changed & ~value & mask
    return re, fe

def _is_dio_strb_symmetric(data, bits):
    count_ok = True

    for bit in bits:
        strobe_mask = 1 << bit
        count_low = False
        count_high = False
        strobe_low = 0
        strobe_high = 0
        last_strobe = None
        for n, d in enumerate(data):
            curr_strobe = (d & strobe_mask) != 0

            if count_high:
              if curr_strobe:
                strobe_high += 1
              else:
                if (strobe_low > 0) and (strobe_low != strobe_high):
                    count_ok = False
                    break

            if count_low:
              if not curr_strobe:
                strobe_low += 1
              else:
                if (strobe_high > 0) and (strobe_low != strobe_high):
                    count_ok = False
                    break

            if (last_strobe != None):
              if (curr_strobe and not last_strobe):
                strobe_high = 0
                count_high = True
                count_low = False
              elif (not curr_strobe and last_strobe):
                strobe_low = 0
                count_low = True
                count_high = False

            last_strobe = curr_strobe

        if not count_ok:
            break

    return count_ok

def _analyze_dio_data(data, strb_mask, strb_slope, vld_mask, vld_polarity, cw_mask, cw_shift):
    """
    Analyzes a list of integer values that represent samples recorded on the DIO interface.
    The function needs information about the protocol used on the DIO interface. Based
    on this information the function will return two lists: the detected codewords
    and the positions where 'timing violations' are found. The codewords are sampled
    according to the protocol configuration. Timing violations occur when a codeword
    bit or the valid bit changes value at the same time as the strobe signal.
    """
    timing_violations = []
    codewords = []
    last_d = None
    for n, d in enumerate(data):
        if n > 0:
            strb_re = False
            strb_fe = False
            if strb_slope == 0:
                strb_re = True
                strb_fe = True
            elif strb_slope == 1:
                strb_re, _ = ZI_HDAWG8._get_edges(d, last_d, strb_mask)
            elif strb_slope == 2:
                _, strb_fe = ZI_HDAWG8._get_edges(d, last_d, strb_mask)
            else:
                strb_re, strb_fe = ZI_HDAWG8._get_edges(d, last_d, strb_mask)

            vld_re = False
            vld_fe = False
            if vld_polarity != 0:
                vld_re, vld_fe = ZI_HDAWG8._get_edges(d, last_d, vld_mask)

            d_re = False
            d_fe = False
            if cw_mask != 0:
                d_re, d_fe =ZI_HDAWG8._get_edges(d, last_d, cw_mask << cw_shift)

            vld_active = ((vld_polarity & 1) and ((d & vld_mask) == 0)) or ((vld_polarity & 2) and ((d & vld_mask) != 0))
            codeword = (d >> cw_shift) & cw_mask

            # Check for timing violation on vld
            if (strb_re or strb_fe) and (vld_re or vld_fe):
                timing_violations.append(n)
            elif (strb_re or strb_fe) and (d_re or d_fe):
                timing_violations.append(n)

            # Get the codewords
            if (strb_re or strb_fe) and vld_active:
                codewords.append(codeword)

        last_d = d

    return codewords, timing_violations

def _find_valid_delays(self, awg, expected_sequence, verbose=False):
    """
    The function loops through the possible delay settings on the DIO interface
    and records and analyzes DIO data for each setting. It then determines whether
    a given delay setting results in valid DIO protocol data being recorded.
    In order for data to be correct, two conditions must be satisfied: First,
    no timing violations are allowed, and, second, the sequence of codewords
    detected on the interface must match the expected sequence.
    """
    if verbose: print("INFO   : Finding valid delays for AWG {}".format(awg))
    vld_mask     = 1 << self._dev.geti('awgs/{}/dio/valid/index'.format(awg))
    vld_polarity = self._dev.geti('awgs/{}/dio/valid/polarity'.format(awg))
    strb_mask    = 1 << self._dev.geti('awgs/{}/dio/strobe/index'.format(awg))
    strb_slope   = self._dev.geti('awgs/{}/dio/strobe/slope'.format(awg))
    cw_mask      = self._dev.geti('awgs/{}/dio/mask/value'.format(awg))
    cw_shift     = self._dev.geti('awgs/{}/dio/mask/shift'.format(awg))

    valid_delays = []
    for delay in range(0, 7):
        if verbose: print("INFO   : Testing delay {} on AWG {}...".format(delay, awg))
        self._set_dio_delay(awg, strb_mask, (cw_mask << cw_shift) | vld_mask, delay)

        data = self._dev.getv('awgs/' + str(awg) + '/dio/data')
        codewords, timing_violations = ZI_HDAWG8._analyze_dio_data(data, strb_mask, strb_slope, vld_mask, vld_polarity, cw_mask, cw_shift)
        timeout_cnt = 0
        while (cw_mask != 0) and len(codewords) == 0:
            if timeout_cnt > 5:
                break
            if verbose: print("WARNING: No codewords detected, trying again!")
            data = self._dev.getv('awgs/' + str(awg) + '/dio/data')
            codewords, timing_violations = ZI_HDAWG8._analyze_dio_data(data, strb_mask, strb_slope, vld_mask, vld_polarity, cw_mask, cw_shift)
            timeout_cnt += 1

        # Compare codewords against sequence
        if (cw_mask != 0) and len(codewords) == 0:
            if verbose: print("WARNING: No codewords detected on AWG {} for delay {}".format(awg, delay))
            continue

        # Can't do nothing with timing violations
        if timing_violations:
            if verbose: print("WARNING: Timing violation detected on AWG {} for delay {}!".format(awg, delay))
            continue

        # Check against expected sequence
        valid_sequence = True
        for n, codeword in enumerate(codewords):
            if n == 0:
                if codeword not in expected_sequence:
                    if verbose: print("WARNING: Codeword {} with value {} not in expected sequence {}!".format(n, codeword, expected_sequence))
                    valid_sequence = False
                    break
                else:
                    index = expected_sequence.index(codeword)
            else:
                last_index = index
                index = (index + 1) % len(expected_sequence)
                if codeword != expected_sequence[index]:
                    if verbose: print("WARNING: Codeword {} with value {} not expected to follow codeword {} in expected sequence {}!".format(n, codeword, expected_sequence[last_index], expected_sequence))
                    valid_sequence = False
                    break

        # If we get to this point the delay is valid
        if valid_sequence:
            valid_delays.append(delay)

    if verbose: print("INFO   : Found valid delays of {}".format(list(valid_delays)))
    return set(valid_delays)

def _set_dio_delay(self, awg, strb_mask, data_mask, delay):
    """
    The function sets the DIO delay for a given FPGA. The valid delay range is
    0 to 6. The delays are created by either delaying the data bits or the strobe
    bit. The data_mask input represents all bits that are part of the codeword or
    the valid bit. The strb_mask input represents the bit that define the strobe.
    """
    if delay < 0:
        print('WARNING: Clamping delay to 0')
    if delay > 6:
        print('WARNING: Clamping delay to 6')
        delay = 6

    strb_delay = 0
    data_delay = 0
    if delay > 3:
        strb_delay = delay-3
    else:
        data_delay = 3-delay

    for i in range(32):
        self._dev.seti('awgs/{}/dio/delay/index'.format(awg), i)
        if strb_mask & (1 << i):
            self._dev.seti('awgs/{}/dio/delay/value'.format(awg), strb_delay)
        elif data_mask & (1 << i):
            self._dev.seti('awgs/{}/dio/delay/value'.format(awg), data_delay)
        else:
            self._dev.seti('awgs/{}/dio/delay/value'.format(awg), 0)

def ensure_symmetric_strobe(self, verbose=False):
    done = False
    good_shots = 0
    bad_shots = 0
    strb_bits = []
    for awg in range(0, 4):
        strb_bits.append(self._dev.geti('awgs/{}/dio/strobe/index'.format(awg)))
    strb_bits = list(set(strb_bits))
    if verbose: print('INFO   : Analyzing strobe bits {}'.format(strb_bits))

    while not done:
        data = self._dev.getv('raw/dios/0/data')
        if ZI_HDAWG8._is_dio_strb_symmetric(data, strb_bits):
            if verbose: print('INFO   : Found good shot')
            bad_shots = 0
            good_shots += 1
            if good_shots > 5:
                done = True
        else:
            if verbose: print('INFO   : Strobe bit(s) are not sampled symmetrically')
            if verbose: print("INFO   :   Disabling AWG's")
            enables = 4*[0]
            for awg in range(0, 4):
                enables[awg] = self._dev.geti('awgs/{}/enable'.format(awg))
                self._dev.seti('awgs/{}/enable'.format(awg), 0)
            if verbose: print("INFO   :   Switching to internal clock")
            self.set('system_extclk', 0)
            time.sleep(5)
            if verbose: print("INFO   :   Switching to external clock")
            self.set('system_extclk', 1)
            time.sleep(5)
            if verbose: print("INFO   :   Enabling AWG's")
            for awg in range(0, 4):
                self._dev.seti('awgs/{}/enable'.format(awg), enables[awg])
            good_shots = 0
            bad_shots += 1
            if bad_shots > 5:
                done = True

    return (good_shots > 0) and (bad_shots == 0)

def calibrate_dio_protocol(self, awgs_and_sequences, verbose=False):
    if verbose: print("INFO   : Calibrating DIO delays")

    if not self.ensure_symmetric_strobe(verbose):
        if verbose: print("ERROR  : Strobe is not symmetric!")
        return False
    else:
        if verbose: print("INFO   : Strobe is symmetric")

    all_valid_delays = []
    for awg, sequence in awgs_and_sequences:
        valid_delays = self._find_valid_delays(awg, sequence, verbose)
        if valid_delays:
            all_valid_delays.append(valid_delays)
        else:
            if verbose: print("ERROR  : Unable to find valid delays for AWG {}!".format(awg))
            return False

    # Figure out which delays are valid
    combined_valid_delays = set.intersection(*all_valid_delays)
    max_valid_delay = max(combined_valid_delays)

    # Print information
    if verbose: print("INFO   : Valid delays are {}".format(combined_valid_delays))
    if verbose: print("INFO   : Setting delay to {}".format(max_valid_delay))

    # And configure the delays
    for awg, _ in awgs_and_sequences:
        vld_mask     = 1 << self._dev.geti('awgs/{}/dio/valid/index'.format(awg))
        strb_mask    = 1 << self._dev.geti('awgs/{}/dio/strobe/index'.format(awg))
        cw_mask      = self._dev.geti('awgs/{}/dio/mask/value'.format(awg))
        cw_shift     = self._dev.geti('awgs/{}/dio/mask/shift'.format(awg))
        if verbose: print("INFO   : Setting delay of AWG {}".format(awg))
        self._set_dio_delay(awg, strb_mask, (cw_mask << cw_shift) | vld_mask, max_valid_delay)

    return True
