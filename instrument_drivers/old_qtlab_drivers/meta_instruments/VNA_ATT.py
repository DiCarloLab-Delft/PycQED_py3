from instrument import Instrument
import qt
import types
import logging
import numpy as np



class VNA_ATT(Instrument):
    def __init__(self, name, VNA = qt.instruments['VNA'], VATT = qt.instruments['VATT']):

        Instrument.__init__(self, name, tags=['virtual'])

        self.VNA = VNA
        self.VATT = VATT
        [self.get_GET_SET_functions(instr) for instr in [VNA,VATT]]

        #Setting zero attenuation powers:
        zeroattenuationpowers = {'RS_VNA':0, 'Homebuilt_VNA':-30,'FS_VNA':-60}
        self._zeroattenuatpower = zeroattenuationpowers[self.VNA.get_type()]

        pars = self.VNA.get_parameters()
        self.add_parameter('power', type=int,
                flags=Instrument.FLAG_GETSET,
                minval = - self.VATT.get_parameter_options('attenuation')['maxval'] + self._zeroattenuatpower,
                maxval = self._zeroattenuatpower,
                units='dBm')


        #self.get_funcs()

    def do_set_power(self, power):
        if self.VNA.get_type() == 'RS_VNA':
            att = - power + self._zeroattenuatpower
            if att % 2 == 1:
                print('Cannot use uneven power: %d, setting 1 dB lower' % att)
                att -= 1
            self.VATT.set_attenuation(att)
            self._power = power
        elif self.VNA.get_type() == 'FS_VNA':
            power -= self._zeroattenuatpower
            VATT_attenuation = - ((power+9) / 10 * 10)
            VNA_power = (power+9) % 10 - 9
            self.VATT.set_attenuation(VATT_attenuation)
            self.VNA.set_power(VNA_power)


    def do_get_power(self):
        return self._power
    def start_single_sweep(self):
        self.VNA.start_single_sweep()
    def download_trace(self):
        return self.VNA.download_trace()
    def prepare_sweep(self, fsta,fsto, npoints, tint, power,navg):
        self.set_power(power)
        self.VNA.prepare_sweep(fsta,fsto, npoints, tint, 0,navg)

    def get_GET_SET_functions(self, instr):
        pars = instr.get_parameters()
        parkeys = list(pars.keys())
        for removekey in ['power', 'attenuation']:
            if removekey in parkeys:
                parkeys.remove(removekey)
        for par in parkeys:
            try:

                setattr(self,'do_set_%s'%par,getattr(instr, 'set_%s'%par))
            except:
                print('no set_%s'%par)
                pass
            try:
                setattr(self,'do_get_%s'%par,getattr(instr, 'get_%s'%par))
            except:
                print('no get_%s'%par)
                pass

            parsopts = instr.get_parameter_options(par)

            poppedget = parsopts.pop('get_func')
            poppedset = parsopts.pop('set_func')
            #parsopts.pop('flags')


            self.add_parameter(par,**parsopts)
            parsopts['get_func'] = poppedget
            parsopts['set_func'] = poppedset
