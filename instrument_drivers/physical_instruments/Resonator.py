from instrument import Instrument
import instruments
import types
import numpy as np


class Resonator(Instrument):

    def __init__(self,name):
        Instrument.__init__(self, name, tags=['virtual'])

        self.add_parameter('fbare',
                           type=float,
                            flags=Instrument.FLAG_GETSET,
                            units='GHz',
                            minval=1,
                            maxval=20)
        self.add_parameter('fspan',
                           type=float,
                            flags=Instrument.FLAG_GETSET,
                            units='GHz',
                            minval=1e-6,
                            maxval=10e-3)
        self.add_parameter('reslsts',
                           type=list,
                           flags=Instrument.FLAG_GETSET)
        self._reslsts=[]
        #self.add_function('add_to_reslst')

    def do_get_fbare(self):
        return self._fbare
    def do_set_fbare(self,freq):
        self._fbare = freq

    def do_get_fspan(self):
        return self._fspan
    def do_set_fspan(self,freq):
        self._fspan = freq

    def do_get_reslsts(self):
        return self._reslsts

    def do_set_reslsts(self,newlst):
        self._reslsts = newlst

    # def add_reslst(self,newlst):
    #     if type(newlst).__name__ != 'list':
    #         return False
    #     else:
    #         self._reslsts = self._reslsts + [newlst]
    #         #Add merge
    #         return True