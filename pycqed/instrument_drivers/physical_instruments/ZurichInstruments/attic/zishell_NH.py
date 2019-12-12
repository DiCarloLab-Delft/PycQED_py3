#!/usr/bin/ipython

##########################################################################
##
# $Id: zishell.py 44323 2017-04-13 07:26:59Z niels $
##
##########################################################################
##
# Title      : zishell.py
# Project    :
##
##########################################################################
##
# Author     : Niels Haandbaek (niels.haandbaek@zhinst.com)
# Company    : Zurich Instruments AG
# Created    : 2014/09/19
# Platform   : Python
# Standard   : none
##
##########################################################################
##
# Description: Shell with simplified interface to ziPython
##
##########################################################################
##
# Copyright (c) 2014/2015, Zurich Instruments AG
# All rights reserved.
##
##########################################################################

import xml.etree.ElementTree as ET
import re
import subprocess
import time
import os
import textwrap
import httplib2

from fnmatch import fnmatch

import zhinst.ziPython as zi
import zhinst.utils as utils

from numpy import *

from plotly.graph_objs import *
import logging
# with open("zishell.py") as f:
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


class ziShellCompilationError(ziShellError):
    """
    Exception raised when the zi AWG-8 compiler encounters an error.
    """
    pass


class ziShellEnvironment:
    """Holds all the global variables."""

    def __init__(self, host=None, port=8004, api_level=5, device=None):
        # Paths to automatically skip
        self.skip = ['*system*']

        # Server connection (host, port, api_level)
        self.server = None

        # Connection to the ziDAQ library
        self.daq = None

        # Connected devices
        self.devices = set()

        # Optionally connect to server
        if host:
            self.connect_server(host, port, api_level)

        # Optionally connect to device
        if device:
            self.connect_device(device)

    def connect_server(self, host, port=8004, api_level=5):
        self.server = (host, port, api_level)
        self.reconnect()

    def reconnect(self):
        if not self.server:
            raise(ziShellServerError())

        host, port, api_level = self.server
        print("Connecting to server on host {0}:{1} using API level {2}".format(
            host, port, api_level))
        self.daq = zi.ziDAQServer(host, port, api_level)
        # self.daq.setDebugLevel(5)

    def autoconnect(self):
        if not self.daq:
            raise(ziShellDAQError())

        try:
            self.devices.add(utils.autoDetect(self.daq))
        except Exception:
            pass

    def connect_device(self, device, interface='1GbE'):
        if not self.daq:
            raise(ziShellDAQError())

        if device in self.devices:
            self.devices.remove(device)

        self.daq.connectDevice(device, interface)
        self.devices.add(device)

    def connected(self):
        return len(self.devices) != 0

    def disconnect_device(self, device):
        if not self.daq:
            raise(ziShellDAQError())

        if device in self.devices:
            self.daq.disconnectDevice(device)
            self.devices.remove(device)

    def disconnect_devices(self):
        if not self.daq:
            raise(ziShellDAQError())

        for device in self.devices:
            try:
                self.daq.disconnectDevice(device)
            except Exception:
                pass

        self.devices = set()

    def restart_device(self):
        try:
            for device in self.devices:
                seti('/' + device + '/raw/system/restart', 1)
                time.sleep(4)
                output = check_output(['timeout', '30', 'make', '-C', os.environ['work'] + '/maven', 'uart'],
                                      stderr=subprocess.STDOUT)
                if re.search('#DHCP Success', output):
                    return True
                else:
                    print(output)
        except subprocess.CalledProcessError:
            pass

        return False

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
            for device in self.devices:
                nodes = self.daq.listNodes('/' + device, 15)

                for r in tree.getroot().findall('./deviceSettings/nodes')[0]:
                    _load_settings(r, '/' + device, nodes, self.daq, self.skip)

                self.daq.sync()

            return True
        except Exception:
            return False

    def features(self):
        if not self.daq:
            raise(ziShellDAQError())

        rv = []
        for device in self.devices:
            rv.append([s for s in str(self.daq.getByte(
                '/' + device + '/features/options')).split('\n') if len(s)])
        return rv

    def setd(self, path, value):
        if not self.daq:
            raise(ziShellDAQError())

        # Handle absolute path
        if path[0] == '/':
            self.daq.setDouble(path, value)
        else:
            for device in self.devices:
                self.daq.setDouble('/' + device + '/' + path, value)

    def seti(self, path, value):
        if not self.daq:
            raise(ziShellDAQError())

        # Handle absolute path
        if path[0] == '/':
            self.daq.syncSetInt(path, value)
        else:
            for device in self.devices:
                self.daq.syncSetInt('/' + device + '/' + path, value)

    def setv(self, path, value):
        if not self.daq:
            raise(ziShellDAQError())

        # Handle absolute path
        if path[0] == '/':
            if 'setVector' in dir(self.daq):
                self.daq.setVector(path, value)
            else:
                self.daq.vectorWrite(path, value)
        else:
            for device in self.devices:
                if 'setVector' in dir(self.daq):
                    self.daq.setVector('/' + device + '/' + path, value)
                else:
                    self.daq.vectorWrite('/' + device + '/' + path, value)

    def geti(self, paths, deep=True):
        if not self.daq:
            raise(ziShellDAQError())

        if not isinstance(paths, list):
            paths = [paths]
            single = 1
        else:
            single = 0

        values = []
        for p in paths:
            p = p.lower()
            if p[0] == '/':
                if deep:
                    env.daq.getAsEvent(p)
                    timeout = 1.0
                    while timeout > 0.0:
                        tmp = env.daq.poll(0.2, 500, 4, True)
                        if p in tmp:
                            values.append(tmp[p]['value'][0])
                            break
                        else:
                            timeout -= 0.2
                else:
                    values.append(env.daq.getInt(p))
            else:
                for device in self.devices:
                    tmp_p = '/' + device + '/' + p
                    if deep:
                        env.daq.getAsEvent(tmp_p)
                        timeout = 1.0
                        while timeout > 0.0:
                            tmp = env.daq.poll(0.2, 500, 4, True)
                            if tmp_p in tmp:
                                values.append(tmp[tmp_p]['value'][0])
                                break
                            else:
                                timeout -= 0.2
                    else:
                        values.append(env.daq.getInt(tmp_p))
        if single:
            return values[0]
        else:
            return values

    def getd(self, paths, deep=True):
        if not self.daq:
            raise(ziShellDAQError())

        if type(paths) is not list:
            paths = [paths]
            single = 1
        else:
            single = 0

        values = []
        for p in paths:
            if p[0] == '/':
                if deep:
                    env.daq.getAsEvent(p)
                    tmp = env.daq.poll(0.2, 500, 4, True)
                    if p in tmp:
                        values.append(tmp[p]['value'][0])
                else:
                    values.append(env.daq.getDouble(p))
            else:
                for device in self.devices:
                    tmp_p = '/' + device + '/' + p
                    if deep:
                        env.daq.getAsEvent(tmp_p)
                        tmp = env.daq.poll(0.2, 500, 4, True)
                        if tmp_p in tmp:
                            values.append(tmp[tmp_p]['value'][0])
                    else:
                        values.append(env.daq.getDouble(tmp_p))
        if single:
            return values[0]
        else:
            return values

    def getv(self, paths):
        if not self.daq:
            raise(ziShellDAQError())

        if type(paths) is not list:
            paths = [paths]
            single = 1
        else:
            single = 0

        values = []
        for p in paths:
            if p[0] == '/':
                tmp = env.daq.get(p, True, 0)
                if p in tmp:
                    values.append(tmp[p])
            else:
                for device in self.devices:
                    tmp_p = '/' + device + '/' + p
                    tmp = env.daq.get(tmp_p, True, 0)
                    if tmp_p in tmp:
                        values.append(tmp[tmp_p])
        if single:
            return values[0]
        else:
            return values

    def find(self, *args):
        if not self.daq:
            raise(ziShellDAQError())

        nodes = self.daq.listNodes('/', 7)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower()
                         for k in nodes if fnmatch(k.lower(), m.lower())]

        return nodes

    def find_devices(self):
        if not self.daq:
            raise(ziShellDAQError())

        nodes = self.daq.listNodes('/', 0)
        devices = []
        for n in nodes:
            if fnmatch(n.lower(), 'dev*'):
                devices.append(n.lower())
        return devices

    def finds(self, *args):
        if not self.daq:
            raise(ziShellDAQError())

        nodes = self.daq.listNodes('/', 15)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower()
                         for k in nodes if fnmatch(k.lower(), m.lower())]

        return nodes

    def subs(self, paths):
        if not self.daq:
            raise(ziShellDAQError())

        if type(paths) is not list:
            paths = [paths]

        for p in paths:
            if p[0] == '/':
                self.daq.subscribe(p)
            else:
                for device in self.devices:
                    self.daq.subscribe('/' + device + '/' + p)

    def unsubs(self, paths):
        if not self.daq:
            raise(ziShellDAQError())

        if type(paths) is not list:
            paths = [paths]

        for p in paths:
            if p[0] == '/':
                self.daq.unsubscribe(p)
            else:
                for device in env.devices:
                    self.daq.unsubscribe('/' + device + '/' + p)

    def poll(self, length=0.1):
        if not self.daq:
            raise(ziShellDAQError())

        return self.daq.poll(length, 500, 4, True)

    def sync(self):
        global env

        if not self.daq:
            raise(ziShellDAQError())

        self.daq.sync()

    def awg(self, filename):
        for device in self.devices:
            h = self.daq.awgModule()
            h.set('awgModule/device', device)
            h.set('awgModule/index', 0)
            h.execute()
            h.set('awgModule/compiler/sourcefile', filename)
            h.set('awgModule/compiler/start', 1)
            h.set('awgModule/elf/file', '')

# Global environment
env = ziShellEnvironment()


def printdict(base_path, data):
    for key, value in data.iteritems():
        path = base_path + \
            ("" if (not base_path or base_path.isspace()) else "/") + key
        if isinstance(value, dict):
            printdict(path, value)
        elif isinstance(value, list):
            print(path, "= [")
            for i in range(0, len(value)):
                printdict('   ', value[i])
            print("]")
        else:
            print(path, "=", value)


def connect_server(host='localhost', port=8004, api_level=5):
    global env
    env.connect_server(host, port, api_level)


def reconnect():
    global env
    env.reconnect()


def connected():
    global env
    return env.connected()


def connect_device(device, interface='1GbE'):
    global env
    return env.connect_device(device, interface)


def disconnect_devices():
    global env
    env.disconnect_devices()


def restart_device():
    global env
    env.restart_device()


def load_settings(path):
    global env
    return env.load_settings(path)


def find(*args):
    global env
    return env.find(*args)


def find_devices():
    global env
    return env.find_devices()


def finds(*args):
    global env
    return env.finds(*args)


def features():
    global env
    return env.features()


def has_mf_feature():
    return any([s == 'MF' for s in features()])


def has_dig_feature():
    return any([s == 'DIG' for s in features()])


def setd(path, value):
    global env
    env.setd(path, value)


def seti(path, value):
    global env
    env.seti(path, value)


def setv(path, value):
    global env
    env.setv(path, value)


def getd(paths, deep=True):
    global env
    return env.getd(paths, deep)


def geti(paths, deep=True):
    global env
    return env.geti(paths, deep)


def getv(paths):
    global env
    return env.getv(paths)


def sync():
    global env
    env.sync()


def subs(paths):
    global env
    env.subs(paths)


def unsubs(paths):
    global env
    env.unsubs(paths)


def poll(length=0.1):
    global env
    return env.poll(length)


def awg(filename):
    global env
    env.awg(filename)


def powerswitch(sid, channel, onoff):
    conn = httplib2.Http()
    uri = "http://powerswitches.zhinst.com/index.php?ajax=set&sid={}&chid={}&val={}&su=1".format(
        sid, channel, onoff)
    r, c = conn.request(uri, method="GET")
    if r.status != 200:
        raise RuntimeError("Error %s(%d) during switching on channel %d." % (
            r.reason, r.status, channel))

    tmp = c.decode('utf-8')
    if re.match('STAT:\d+\|([\d,]+)', tmp):
        bits = tmp.group(1).split(',')
        if bits[channel] != onoff:
            raise RuntimeError(
                "Channel %d was not correctly set to %d" % (channel, onoff))


def powerswitch_on(sid, channel):
    powerswitch(sid, channel, 1)


def powerswitch_off(sid, channel):
    powerswitch(sid, channel, 0)


def powerswitch_reset(sid, channel):
    powerswitch_off(sid, channel)
    time.sleep(5.0)
    powerswitch_on(sid, channel)


def check_output(*popenargs, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    # Special case for time-out command
    if retcode and retcode != 124:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd)
    return output


def check_status(string, function):
    for l in textwrap.wrap(string, 40):
        print('{0: <70}'.format(l))
    result = function()
    if result:
        print('[SUCCESS]')
    else:
        print('[FAILURE]')
    return result


def print_status(string, success):
    for l in textwrap.wrap(string, 40):
        print('{0: <70}'.format(l)),
    if success:
        print('[SUCCESS]')
    else:
        print('[FAILURE]')


def getwave(timeout=True):
    global env
    path = 'scopes/0/wave'
    subs(path)

    seti('scopes/0/single', 1)
    seti('scopes/0/enable', 1)

    complete_wave = None
    wave = None
    tries = 0
    done = 0
    while not done and (not timeout or tries < 10):
        wave = poll()

        if path in wave:
            wave = wave[path]
            while len(wave) and not complete_wave:
                complete_wave = wave[0]
                wave = wave[1:]
                if complete_wave['blocknumber'] != 0:
                    complete_wave = None

            if not complete_wave:
                continue

            for w in wave:
                if w['sequencenumber'] != complete_wave['sequencenumber']:
                    if w['blocknumber'] == 0:
                        complete_wave = w
                    else:
                        continue

                elif w['blocknumber'] == complete_wave['blocknumber'] + 1 and w['flags'] == 0:
                    complete_wave['blocknumber'] += 1
                    complete_wave['blockmarker'] = w['blockmarker']
                    complete_wave['wave'] = vstack(
                        (complete_wave['wave'], w['wave']))

            if len(complete_wave['wave']) == complete_wave['totalsamples'] and \
                    complete_wave['blockmarker'] == 1 and \
                    complete_wave['flags'] == 0:
                done = True
        else:
            tries += 1

    unsubs(path)
    if timeout and tries >= 10:
        return None
    else:
        return complete_wave


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

        # Modules
        self.awgModule = None
        self.scopeModule = None

        # Connect to server
        if host and device and interface:
            self.connect_server(host, port, api_level)
        else:
            self.host = host
            self.port = port

    def connect_server(self, host, port=8004, api_level=5):
        self.server = (host, port, api_level)
        print("Connecting to server on host {0}:{1} using API level {2}".format(
            host, port, api_level))
        self.daq = zi.ziDAQServer(host, port, api_level)
        if not self.daq:
            raise(ziShellDAQError())

        # self.daq.setDebugLevel(5)
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
        print("Connected to device {} over {}".format(
            self.device, self.interface))

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

    def restart_device(self):
        try:
            seti('/' + self.device + '/raw/system/restart', 1)
            time.sleep(4)
            output = check_output(['timeout', '30', 'make', '-C',
                                   os.environ['work'] + '/maven', 'uart'],
                                  stderr=subprocess.STDOUT)
            if re.search('#DHCP Success', output):
                return True
            else:
                print(output)
        except subprocess.CalledProcessError:
            pass

        return False

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
        except Exception:
            return False

    def features(self):
        if not self.daq:
            raise(ziShellDAQError())

        rv = []
        rv.append([s for s in str(self.daq.getByte(
            '/' + self.device + '/features/options')).split('\n') if len(s)])
        return rv

    def setd(self, path, value, synchronous=False):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        # Handle absolute path
        if path[0] == '/':
            if synchronous:
                self.daq.syncSetDouble(path, value)
            else:
                self.daq.setDouble(path, value)
        else:
            if synchronous:
                self.daq.syncSetDouble('/' + self.device + '/' + path, value)
            else:
                self.daq.setDouble('/' + self.device + '/' + path, value)

    def seti(self, path, value, synchronous=False):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        # Handle absolute path
        if path[0] == '/':
            if synchronous:
                self.daq.syncSetInt(path, value)
            else:
                self.daq.setInt(path, value)
        else:
            if synchronous:
                self.daq.syncSetInt('/' + self.device + '/' + path, value)
            else:
                self.daq.setInt('/' + self.device + '/' + path, value)

    def setv(self, path, value):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        # Handle absolute path
        if path[0] == '/':
            self.daq.setVector(path, value)
        else:
            self.daq.setVector('/' + self.device + '/' + path, value)

    def sets(self, path, value, sync=False):
        if not self.daq:
            raise(ziShellDAQError())

        path = path.lower()

        # Handle absolute path
        if path[0] == '/':
            self.daq.setString(path, value)
        else:
            self.daq.setString('/' + self.device + '/' + path, value)

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
            # The [0]['vector'] is to get strip of the vector stuff around it
            return tmp[path][0]['vector']
        else:
            return None

    def gets(self, path, deep=True):
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
                    return tmp[path][0]
                else:
                    timeout -= 0.1
            return None
        else:
            return self.daq.getString(path)

    def find(self, *args):
        if not self.daq:
            raise(ziShellDAQError())

        nodes = self.daq.listNodes('/' + self.device + '/', 7)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower()
                         for k in nodes if fnmatch(k.lower(), m.lower())]

        return nodes

    def finds(self, *args):
        if not self.daq:
            raise(ziShellDAQError())

        nodes = self.daq.listNodes('/' + self.device + '/', 15)
        if len(args) and args[0]:
            for m in args:
                nodes = [k.lower()
                         for k in nodes if fnmatch(k.lower(), m.lower())]

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

    def configure_awg_from_file(self, filename):
        raise NotImplementedError('Use "configure_awg_from_string" instead')
        if not self.daq:
            raise(ziShellDAQError())

        if not self.awgModule:
            raise(ziShellModuleError())

        self.awgModule.set('awgModule/compiler/sourcefile', filename)
        self.awgModule.set('awgModule/compiler/start', 1)
        self.awgModule.set('awgModule/elf/file', '')

    def configure_awg_from_string(self, awg_nr: int, program_string: str,
                                  timeout: float=15):
        """
        Uploads a program string to one of the AWGs in the AWG8.

        This function is tested to work and give the correct error messages
        when compilation fails.

        N.B. the uploaded program will not work unless the
        "configure_codeword_protocol" method is called on the HDAWG
        """
        t0 = time.time()
        success_and_ready = False
        # This check (and while loop) is added as a workaround for #9
        while not success_and_ready:
            self.seti('awgs/' + str(awg_nr) + '/dio/valid/polarity', 0)
            self.seti('awgs/' + str(awg_nr) + '/dio/strobe/slope', 0)

            print('Configuring AWG_nr {}.'.format(awg_nr))
            if not self.daq:
                raise(ziShellDAQError())

            if not self.awgModule:
                raise(ziShellModuleError())

            self.awgModule.set('awgModule/index', awg_nr)
            self.awgModule.set('awgModule/compiler/sourcestring', program_string)


            succes_msg = 'File successfully uploaded'
            # Success is set to False when either a timeout or a bad compilation
            # message is encountered.
            success = True
            # while ("compilation not completft1ed"):
            while len(self.awgModule.get('awgModule/compiler/sourcestring')
                      ['compiler']['sourcestring'][0]) > 0:
                time.sleep(0.01)
                comp_msg = (self.awgModule.get(
                    'awgModule/compiler/statusstring')['compiler']
                    ['statusstring'][0])
                if (time.time()-t0 >= timeout):
                    success = False
                    # print('Timeout encountered during compilation.')
                    raise TimeoutError()
                    break

            if not comp_msg.endswith(succes_msg):
                success = False

            self.seti('awgs/' + str(awg_nr) + '/dio/valid/polarity', 2)
            self.seti('awgs/' + str(awg_nr) + '/dio/strobe/slope', 2)

            if not success:
                print("Compilation failed, printing program:")
                for i, line in enumerate(program_string.splitlines()):
                    print(i+1, '\t', line)
                print('\n')
                raise ziShellCompilationError(comp_msg)

            # This check (and while loop) is added as a workaround for #9
            for i in range(10):
                if self.geti('awgs/'+str(awg_nr)+'/ready', True)!= 1:
                    logging.warning('AWG not ready')
                    success_and_ready = False
                    time.sleep(0.1)
                else:
                    success_and_ready = True
                    break

        # print('AWG {} ready: {}'.format(awg_nr,
        #                              self.geti('awgs/'+str(awg_nr)+'/ready')))
        t1 = time.time()
        print(self.awgModule.get('awgModule/compiler/statusstring')
              ['compiler']['statusstring'][0] + ' in {:.2f}s'.format(t1-t0))

    def read_from_scope(self, timeout=1.0, enable=True):
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

    def restart_scope_module(self):
        self.scopeModule = self.daq.scopeModule()
        self.scopeModule.set('scopeModule/mode', 0)
        self.scopeModule.subscribe('/' + self.device + '/scopes/0/wave')
        self.scopeModule.execute()
        print("Initialized scopeModule")

    def restart_awg_module(self):
        self.awgModule = self.daq.awgModule()
        self.awgModule.set('awgModule/device', self.device)
        self.awgModule.execute()
        print("Initialized awgModule")

    def __str__(self):
        return self.device
