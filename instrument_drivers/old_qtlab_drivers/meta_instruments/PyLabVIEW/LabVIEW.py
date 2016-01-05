#########################################################################
#
# Custom LabVIEW TCP Server Client Script
#
# Copyright (C) 2002,2003 Jim Kring jim@jimkring.com
#
# Visit OpenG.org for the more info on Open Source LabVIEW Projects
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#########################################################################

import socket as _socket

_serverHost = 'localhost'
_serverPort = 50007
isConnected = 0
_sockobj = None
_server_on = False

def test_connection():
    global _server_on, _sockobj
    try:
        _sockobj = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)      # create socket
        _sockobj.connect((_serverHost, _serverPort))   # connect to LV
        _sockobj.close()
        _server_on = True
    except:
        print('server not on')
        _server_on = False
def connect():
    if _server_on:
        'opens a connection to LabVIEW Server'
        global _sockobj, isConnected
        _sockobj = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)      # create socket
        _sockobj.connect((_serverHost, _serverPort))   # connect to LV
        isConnected = 1
    else:
        print('LV server not on')


def disconnect():
    if _server_on:
        'closes the connection to LabVIEW Server'
        global isConnected
        _sockobj.close()                             # close socket
        isConnected = 0
    else:
        'LV server not on'

def _passCommand(command):
    'passes a command to LabVIEW Server'
    #print len(command+'\r\n')
    if _server_on:
        _sockobj.send(str(len(command+'\r\n'))+'\r\n')
        _sockobj.send(command+'\r\n')
        data = _sockobj.recv(65536)
        execString = "lvdata = " + data
        exec(execString)
        return lvdata
    else:
        'LVserver not on'


class _Instrument:

    def __init__(self, _instrumentName, _functionNames):

        for _functionName in _functionNames:
            _execString = "self." + _functionName + " =_Function('" + _instrumentName + "." + _functionName + "')"
            exec(_execString)


class _Function:

    def __init__(self, name):

        self._name = name

    def __call__(self, *a):

        if isConnected:
            if (a is None):
                return _passCommand(self._name + '()')
            else:
                return _passCommand(self._name + '(%s)' % a)

        elif _server_on: print('Not Connected: Run "%s.connect()" method to connect.'% __name__)

test_connection()
if _server_on:
    connect()

    _instrumentList = _passCommand("System.Get_Instruments()")

    for _instrument in _instrumentList:
        _instrumentName = _instrument[0]
        _instrumentFunctionList = _instrument[1]
        _execString = _instrumentName + ' = _Instrument(' + repr(_instrumentName) + ", " + repr(_instrumentFunctionList) +')'
        exec(_execString)

    disconnect()
