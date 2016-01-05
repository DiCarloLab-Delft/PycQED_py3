from instrument import Instrument
import instruments
import types
import numpy as np




class Resonators(Instrument):

    def __init__(self,name):
        Instrument.__init__(self, name, tags=['virtual'])

        self.add_parameter('res',
                type=list,
                flags=Instrument.FLAG_GETSET)
        self._resonatorlst=[]
        #self.add_function('add_to_reslst')

    def do_get_res(self):
        return self._resonatorlst

    def do_set_res(self,newlst):
        self._resonatorlst = newlst

    def add_res(self, resonator):
        print(self._resonatorlst.append(123))
        self.set_res(self._resonatorlst.append(resonator))

class resonator():
    def __init__(self, name, fres = 0):
        self.name = str(name)
        self.fres = fres