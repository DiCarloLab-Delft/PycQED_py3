import socket
import select
from time import sleep, time


class SocketVisa:
    def __init__(self, host, port):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((host, port))
        self._socket.settimeout(20)

    def clear(self):
        rlist, wlist, xlist = select.select([self._socket], [], [], 0)
        if len(rlist) == 0:
            return
        ret = self.read()
        print('Unexpected data before ask(): %r' % (ret, ))

    def write(self, data):
        self.clear()
        if len(data) > 0 and data[-1] != '\r\n':
            data += '\n'
        #if len(data)<100:
            #print 'Writing %s' % (data,)
        self._socket.send(data)

    def read(self,timeouttime=20):
        start = time()
        try:
            ans = ''
            while len(ans) == 0 and (time() - start) < timeouttime or not has_newline(ans):
                ans2 = self._socket.recv(8192)
                ans += ans2
                if len(ans2) == 0:
                    sleep(0.01)
            #print 'Read: %r (len=%s)' % (ans, len(ans))
            AWGlastdataread=ans
        except socket.timeout as e:
            print('Timed out')
            return ''

        if len(ans) > 0:
            ans = ans.rstrip('\r\n')
        return ans

    def ask(self, data):
        self.clear()
        self.write(data)
        return self.read()


def has_newline(ans):
    if len(ans) > 0 and ans.find('\n') != -1:
        return True
    return False