"""
    Qudev specific driver for the UHFQA instrument. 
"""

import logging
import time

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQA_core as uhf

log = logging.getLogger(__name__)

class UHFQA_qudev(uhf.UHFQA_core):
    """This is the Qudev specific PycQED driver for the 1.8 GSa/s UHFQA instrument
    from Zurich Instruments AG.
    """

    USER_REG_FIRST_SEGMENT = 5
    USER_REG_LAST_SEGMENT = 6

    def __init__(self,
                 name,
                 device:                  str,
                 interface:               str = 'USB',
                 address:                 str = '127.0.0.1',
                 port:                    int = 8004,
                 nr_integration_channels: int = 10,
                 server:                  str = '',
                 **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument
            device          (str) the name of the device e.g., "dev8008"
            interface       (str) the name of the interface to use ('1GbE' or 'USB')
            address         (str) the host where the ziDataServer is running (for compatibility)
            port            (int) the port to connect to for the ziDataServer (don't change)
            nr_integration_channels (int) the number of integration channels to use (max 10)
            server:         (str) the host where the ziDataServer is running (if not '' then used instead of address)
        """
        t0 = time.time()
        
        super().__init__(name=name, device=device, interface=interface, address=address,
                         server=server, port=port, nr_integration_channels=nr_integration_channels,
                         **kw)

        t1 = time.time()
        log.info(f'{self.devname}: Initialized UHFQA_qudev in {t1 - t0:.3f}s')

    def acquisition_initialize(self, samples, averages, loop_cnt, channels=(0, 1), mode='rl') -> None:
        # Define the channels to use and subscribe to them
        self._acquisition_nodes = []

        if mode == 'rl':
            for c in channels:
                path = self._get_full_path('qas/0/result/data/{}/wave'.format(c))
                self._acquisition_nodes.append(path)
                self.subs(path)
            # Enable automatic readout
            self.qas_0_result_reset(1)
            self.qas_0_result_enable(0)
            self.qas_0_result_length(samples)
            self.qas_0_result_averages(averages)
            ro_mode = 0
        else:
            for c in channels:
                path = self._get_full_path('qas/0/monitor/inputs/{}/wave'.format(c))
                self._acquisition_nodes.append(path)
                self.subs(path)
            # Enable automatic readout
            self.qas_0_monitor_reset(1)
            self.qas_0_monitor_enable(1)
            self.qas_0_monitor_length(samples)
            self.qas_0_monitor_averages(averages)
            ro_mode = 1

        self.set('awgs_0_userregs_{}'.format(uhf.UHFQA_core.USER_REG_LOOP_CNT), loop_cnt)
        self.set('awgs_0_userregs_{}'.format(uhf.UHFQA_core.USER_REG_RO_MODE), ro_mode)
        if self.wait_dly() > 0 and not self._awg_program_features['wait_dly']:
            raise uhf.ziUHFQCSeqCError('Trying to use a delay of {} using an AWG program that does not use \'wait_dly\'.'.format(self.wait_dly()))
