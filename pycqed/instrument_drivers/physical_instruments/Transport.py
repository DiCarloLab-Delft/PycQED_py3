"""
    File:       Transport.py
    Author:     Wouter Vlothuizen, TNO/QuTech
    Purpose:    provide self contained data transport, similar to QCoDeS IPInstrument/VisaInstrument
    Usage:
    Notes:
    Bugs:
    Changelog:

"""

import socket


class Transport:
    """
    abstract base class for data transport to instruments
    """

    def close(self) -> None:
        pass

    def write(self, cmd_str: str) -> None:
        pass

    def write_binary(self, data: bytes) -> None:
        pass

    def read_binary(self, size: int) -> bytes:
        pass

    def readline(self) -> str:
        pass


class IPTransport(Transport):
    """
    Based on: SCPI.py, QCoDeS::IPInstrument
    """

    def __init__(self, host: str, port: int = 5025) -> None:
        """
        establish connection, e.g. IPTransport('192.168.0.16', 4000)
        """
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(1)  # first set timeout (before connect)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # send things immediately

        # beef up buffer, to prevent socket.send() not sending all our data in one go
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512 * 1024)
        self._socket.connect((host, port))

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        self._socket.close()

    def write(self, cmd_str: str) -> None:
        outStr = cmd_str + '\n'
        # FIXME: check return value, maybe encode() can be improved on by not using unicode strings?
        self._socket.send(outStr.encode('ascii'))

    def write_binary(self, data: bytes) -> None:
        exp_len = len(data)
        act_len = self._socket.send(data)
        if(act_len != exp_len):
            # FIXME: handle this case by calling send again. Or enlarge
            # socket.SO_SNDBUF even further
            raise UserWarning(
                'not all data sent: expected %d, actual %d' % (exp_len, act_len))

    def read_binary(self, size: int) -> bytes:
        data = self._socket.recv(size)
        act_len = len(data)
        exp_len = size
        while act_len != exp_len:
            data += self._socket.recv(exp_len - act_len)
            act_len = len(data)
        return data

    def readline(self) -> str:
        return self._socket.makefile().readline()


class VisaTransport(Transport):
    pass


class FileTransport(Transport):
    pass
