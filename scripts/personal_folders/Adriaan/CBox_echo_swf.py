from modules.measurement import sweep_functions as swf
from modules.measurement.pulse_sequences import standard_sequences as st_seqs


class CBox_Echo(swf.Hard_Sweep):
    def __init__(self,
                 IF, RO_pulse_length,
                 RO_pulse_delay, RO_trigger_delay, pulse_length,
                 AWG, CBox, cal_points=True,
                 max_seq_duration=100e-6,
                 upload=True):
        super().__init__()
        self.parameter_name = 'tau'
        self.unit = 's'
        self.name = 'T2-echo'
        # Making input pars available to prepare
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.pulse_length = pulse_length
        self.RO_pulse_length = RO_pulse_length
        self.max_seq_duration = max_seq_duration
        # Required instruments
        self.AWG = AWG
        self.CBox = CBox
        self.upload = upload
        self.cal_points = cal_points

    def prepare(self, **kw):
        self.AWG.stop()
        time_tape = []

        pi_elt = 1
        pi2_elt = 3
        mpi2_elt = 5
        Id_elt = 0  # Prepend the identity for time

        if self.cal_points:
            times = self.sweep_points[:-4]
        else:
            times = self.sweep_points
        for i, tau in enumerate(times):
            ta_ns = round((self.max_seq_duration - tau - self.pulse_length/2)
                          * 1e9)
            if ta_ns < 0:
                raise(ValueError('tau "{}" larger than'.format(tau) +
                      'max_seq_duration "{}"'.format(self.max_seq_duration)))
            tb_ns = round((tau/2 - self.pulse_length)*1e9)
            if tb_ns < 0:
                raise(ValueError('tau/2 "{}" smaller than'.format(tau/2) +
                      'pulse length "{}"'.format(self.pulse_length)))

            tc_ns = tb_ns   # + 5*i  # Can be added for "artificial" detuning

            e1 = self.CBox.create_timing_tape_entry(ta_ns, pi2_elt, False,
                                                    prepend_elt=Id_elt)
            e2 = self.CBox.create_timing_tape_entry(tb_ns, pi_elt, False,
                                                    prepend_elt=Id_elt)
            e3 = self.CBox.create_timing_tape_entry(tc_ns, mpi2_elt, True,
                                                    prepend_elt=Id_elt)

            time_tape.extend(e1)
            time_tape.extend(e2)
            time_tape.extend(e3)
        if self.cal_points:
            cal0 = self.CBox.create_timing_tape_entry(0, Id_elt, True,
                                                      prepend_elt=Id_elt)
            cal1 = self.CBox.create_timing_tape_entry(self.max_seq_duration*1e9,
                                                      pi_elt, True,
                                                      prepend_elt=Id_elt)
            time_tape.extend(cal0)
            time_tape.extend(cal0)
            time_tape.extend(cal1)
            time_tape.extend(cal1)

        print('Total tape length', len(time_tape))
        for awg in range(3):
            self.CBox.set('AWG{}_mode'.format(awg), 'Segmented')
            self.CBox.set_segmented_tape(awg, time_tape)
            self.CBox.restart_awg_tape(awg)

        ch3_amp = self.AWG.get('ch3_amp')
        ch4_amp = self.AWG.get('ch3_amp')
        st_seqs.CBox_single_pulse_seq(
            IF=self.IF,
            RO_pulse_delay=self.RO_pulse_delay + self.max_seq_duration+300e-9,
            RO_trigger_delay=self.RO_trigger_delay,
            RO_pulse_length=self.RO_pulse_length)

        self.AWG.set('ch3_amp', ch3_amp)
        self.AWG.set('ch4_amp', ch4_amp)