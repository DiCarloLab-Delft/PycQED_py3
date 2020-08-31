"""
    File:       Transport.py
    Author:     Wouter Vlothuizen, TNO/QuTech
    Purpose:    provide self contained data transport using several transport mechanisms
    Usage:
    Notes:      handles large data transfers properly
    Bugs:
    Changelog:

"""

import socket


class Transport:
    """
    abstract base class for data transport to instruments
    """

    def __del__(self) -> None:
        self.close()

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

    def __init__(self, host: str,
                 port: int = 5025,
                 timeout = 10.0,
                 snd_buf_size: int = 512 * 1024) -> None:
        """
        establish connection, e.g. IPTransport('192.168.0.16', 4000)
        """
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)  # first set timeout (before connect)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, snd_buf_size) # beef up buffer
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # send things immediately
        self._socket.connect((host, port))

    def close(self) -> None:
        self._socket.close()

    def write(self, cmd_str: str) -> None:
        out_str = cmd_str + '\n'
        self.write_binary(out_str.encode('ascii'))

    def write_binary(self, data: bytes) -> None:
        exp_len = len(data)
        act_len = 0
        while True:
            act_len += self._socket.send(data[act_len:exp_len])
            if act_len == exp_len:
                break

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
    # FIXME: implement
    pass


class FileTransport(Transport):
    def __init__(self, out_file_name: str,
                 in_file_name: str = '') -> None:
        """
        input/output from/to file to support driver testing
        FIXME: we now have inject() instead of in_file_name
        """
        self._out_file = open(out_file_name, "wb+")
        self._inject_data = '1'  # response to "*OPC?"
    def close(self) -> None:
        self._out_file.close()

    def write(self, cmd_str: str) -> None:
        out_str = cmd_str + '\n'
        self.write_binary(out_str.encode('ascii'))

    def write_binary(self, data: bytes) -> None:
        self._out_file.write(data)

    def read_binary(self, size: int) -> bytes:
        return self._inject_data.encode('utf-8')

    def readline(self) -> str:
        return self._inject_data

    def inject(self, data: bytes) -> None:
        """
        inject data to be returned by read*. Same data can be read multiple times
        """
        self._inject_data = data



class DummyTransport(Transport):
    def __init__(self) -> None:
        self._inject_data = '1'  # response to "*OPC?"

    def read_binary(self, size: int) -> bytes:
        return self._inject_data.encode('utf-8')

    def readline(self) -> str:
        return self._inject_data

    def inject(self, data: bytes) -> None:
        """
        inject data to be returned by read*. Same data can be read multiple times
        """
        self._inject_data = data
