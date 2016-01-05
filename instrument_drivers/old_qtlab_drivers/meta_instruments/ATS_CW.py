from instrument import Instrument
import types
import sys
import logging
import qt
import numpy as np
from time import clock as time
ATS = qt.instruments['ATS']

class ATS_CW(Instrument):
    def __init__(self, name, ATS=qt.instruments['ATS']):
        # Initialize wrapper
        logging.info(__name__ + ' : Initializing instrument ATS_CW')
        Instrument.__init__(self, name, tags=['CW_instr'])
        self.ATS = ATS


        self.add_parameter('range', flags=Instrument.FLAG_GETSET,
                           type=float, units='Volt (Vp)')
        self.add_parameter('coupling', flags=Instrument.FLAG_GETSET,
                           type=bytes)
        self.add_parameter('t_int', flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                           type=float, units='ms')
        self.add_parameter('sample_rate', flags=Instrument.FLAG_GETSET,
                           type=int, units='MSPS')
        self.add_parameter('overload', flags=Instrument.FLAG_GET,
                           type=bytes)
        self.add_parameter('number_of_buffers', type=int,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)
        self.add_parameter('trace_length', type=float,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                           units='ms')

        self.set_sample_rate(250)
        self.n_buff = 1
        self.set_range(2.)
        self.set_coupling('AC')
        self.set_t_int(20, optimize=True, silent=True)

    def init(self, optimize=False, silent=False, max_buf=1024):
        #TODO: Figure out if there is something to do with RF PuM
        #TODO: Simply use self.ATS.load_defaults()?
        self.ATS.abort()

        self.ATS.set_sample_rate(self._MSPS*1e3)
        self.ATS.set_ch1_coupling(self._coupling)
        self.ATS.set_ch2_coupling(self._coupling)
        self.ATS.set_ch1_range(self._range)
        self.ATS.set_ch2_range(self._range)
        self.ATS.set_records_per_buffer(1)
        self.set_t_int(self.t_int, optimize=optimize, silent=silent,
                       max_buf=max_buf)


    def get_t_base(self):
        return np.arange(self.ATS.get_points_per_trace())/(self.ATS.get_sample_rate()*1e3)

    def do_set_coupling(self, acdc):
        '''
        'AC' or 'DC'
        '''
        self._coupling = acdc
        self.ATS.set_ch1_coupling(self._coupling)
        self.ATS.set_ch2_coupling(self._coupling)

    def do_get_coupling(self):
        return self._coupling

    def do_set_sample_rate(self, MSPS):
        self._MSPS = MSPS
        self.ATS.set_sample_rate(self._MSPS*1e3)

    def do_get_sample_rate(self):
        return self._MSPS

    def do_get_range(self):
        '''
        Get the range (in volts) of the ATS.
        range is specified as peak voltage.
        '''
        return self._range

    def do_set_range(self, value):
        '''
        Set the range (in volts) of the ATS.
        range is specified as peak voltage.
        Allowed values: 0.04, 0.1, 0.2, 0.4, 1, 2, 4
        '''
        self._range = value
        self.ATS.set_ch1_range(self._range)
        self.ATS.set_ch2_range(self._range)

    def do_get_overload(self):
        ol1 = self.ATS.get_ch1_overload()
        ol2 = self.ATS.get_ch2_overload()
        ch = '-'
        if ol1 or ol2:
            ch = ''
            if ol1:
                ch += '1 '
            elif ol2:
                ch+= '2'
            print('WARNING: overload detected on ch: %s!!!!'%ch)
        return ch

    def average_data(self, channel):
        return self.ATS.average_data(channel)

    def start(self):
        self.ATS.arm()

    def wait_until_ats_ready(self):
        kk = 1
        while self.ATS.get_busy():
            qt.msleep(0.005)
            if kk/100 == kk/100.:
                sys.stdout.write('.')
            kk += 1
        self.ATS.abort()

    def do_set_t_int(self, tint, max_buf=1024, optimize=False, silent=False):
        '''
        sets the integration time per probe shot
        '''
        if optimize is True:
            print('Optimizing ATS readout')
            n_buf, t_int = self.optimize_readout(tint, max_buf, silent=silent)
        self.ATS.set_number_of_buffers(self.get_number_of_buffers())
        sr = self._MSPS*1e3  # KSPS
        self.ATS.set_points_per_trace(
            tint/self.get_number_of_buffers() * sr)
        self.get_trace_length()

    def do_get_t_int(self, **kw):
        self.t_int = self.ATS.get_trace_length() * self.get_number_of_buffers()
        return self.t_int

    def optimize_readout(self, ro_time, max_buf, silent=False):
        '''
        optimizes n_buf and self._length (number of samples)
        ro_time = self._n_buff*self._trace_length (in ms)

        returns the optimum values for opt_n_buf and trace_length
        '''
        if not silent:
            print()
            print('Optimizing ATS CW readout settings ')
        ats = self.ATS

        ro_us = ro_time  # us
        sr = ats.get_sample_rate()  # KSPS
        mppt = ats.get_max_points_per_trace()
        mtrl = mppt/sr  # maximum trace length in ms
        n_buf = max(int(ro_time/mtrl), 2)
        kstart = int(np.log(n_buf)/np.log(2))+1
        k = kstart

        while n_buf <= max_buf:
            tr_len = ro_time/n_buf  # ms
            npoints = ro_time/n_buf*sr
            if not silent:
                print('npoints: %s, trace_length :%.3f, n_buf: %s' % (npoints, tr_len, n_buf))
            ats.set_number_of_buffers(n_buf)
            ats.set_points_per_trace(npoints)
            ats.get_trace_length()
            t0 = time()
            ats.arm()

            t1 = time()-t0
            xy = ats.average_data(1)

            t2 = time()-t1-t0
            t_end = time()-t0

            if k==kstart:
                t_min = t_end
                self.opt_n_buf = n_buf
                self.optimum_points_p_trace = tr_len
            else:
                if t_end < t_min:
                    t_min = t_end
                    self.opt_n_buf = n_buf
                    self.optimum_points_p_trace = tr_len
                else:
                    break
            n_buf = (2**(k))
            k += 1
            if not silent:
                print('t_acq: %.3f, t_proc: %.3f, t_tot: %.3f' % (t1, t2, t_end))
        if not silent:
            print()
            print('Optimal values')
            print('number of buffers: %s' % self.opt_n_buf)
            print('Trace length: %.3f' % self.optimum_points_p_trace)
            print('Processing time: %.3f' % t_min)
            print()
        ats.abort()
        self.do_set_number_of_buffers(self.opt_n_buf)
        ats.set_number_of_buffers(self.opt_n_buf)
        ats.set_points_per_trace(self.optimum_points_p_trace)

        ats.get_trace_length()

        qt.msleep(0.1)

        return self.opt_n_buf, self.optimum_points_p_trace

    def do_set_number_of_buffers(self, n_buff):
        self.n_buff = n_buff
        t_int = self.get_t_int()
        self.set_t_int(t_int)

    def do_get_number_of_buffers(self):
        return self.n_buff

    def do_set_trace_length(self, trace_len):
        '''
        Trace lenght is the integration time per buffer
        '''
        self.set_t_int(self.get_number_of_buffers()*trace_len)

    def do_get_trace_length(self):
        return self.get_t_int()/self.get_number_of_buffers()
