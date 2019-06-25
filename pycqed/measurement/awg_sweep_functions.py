import numpy as np
import logging
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sqs
from pycqed.measurement.pulse_sequences import single_qubit_2nd_exc_seqs as sqs2
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.measurement.pulse_sequences import calibration_elements as csqs
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mq_sqs
from pycqed.measurement.waveform_control import pulsar as ps



class File(swf.Hard_Sweep):

    def __init__(self, filename, AWG, title=None, NoElements=None, upload=True):
        self.upload = upload
        self.AWG = AWG
        if title:
            self.name = title
        else:
            self.name = filename
        self.filename = filename + '_FILE'
        self.upload = upload
        self.parameter_name = 'amplitude'
        self.unit = 'V'

    def prepare(self, **kw):
        if self.upload:
            self.AWG.set_setup_filename(self.filename)


class awg_seq_swf(swf.Hard_Sweep):
    def __init__(self, awg_seq_func, awg_seq_func_kwargs,
                 parameter_name=None, unit='a.u.',
                 AWG=None,
                 fluxing_channels=[],
                 upload=True, return_seq=False):
        """
        A wrapper for awg sequence generating functions.
        Works as a general awg sweep function.
        """
        super().__init__()
        self.upload = upload
        self.awg_seq_func = awg_seq_func
        self.awg_seq_func_kwargs = awg_seq_func_kwargs
        self.unit = unit
        self.name = 'swf_' + awg_seq_func.__name__
        self.fluxing_channels = fluxing_channels
        self.AWG = AWG

        if parameter_name is not None:
            self.parameter_name = parameter_name
        else:
            self.parameter_name = 'points'

    def prepare(self, **kw):
        if self.parameter_name != 'points':
            self.awg_seq_func_kwargs[self.parameter_name] = self.sweep_points

        if self.upload:
            old_vals = np.zeros(len(self.fluxing_channels))
            for i, ch in enumerate(self.fluxing_channels):
                old_vals[i] = self.AWG.get('{}_amp'.format(ch))
                self.AWG.set('{}_amp'.format(ch), 2)

            self.awg_seq_func(**self.awg_seq_func_kwargs)

            for i, ch in enumerate(self.fluxing_channels):
                self.AWG.set('{}_amp'.format(ch), old_vals[i])

    def set_parameter(self, val, **kw):
        # exists for compatibility reasons with 2D sweeps
        pass


class mixer_skewness_calibration_swf(swf.Hard_Sweep):
    """
    Based on the "UHFQC_integrated_average" detector.
    generates an AWG seq to measure sideband transmission.
    """

    def __init__(self, pulseIch, pulseQch, alpha, phi_skew,
                 f_mod, RO_trigger_channel, RO_pars,
                 amplitude=0.1, RO_trigger_separation=5e-6,
                 verbose=False,  data_points=1, upload=True):
        super().__init__()
        self.name = 'mixer_skewness_calibration_swf'
        self.parameter_name = 'alpha'
        self.unit = 'a.u'
        self.pulseIch = pulseIch
        self.pulseQch = pulseQch
        self.alpha = alpha
        self.phi_skew = phi_skew
        self.f_mod = f_mod
        self.amplitude = amplitude
        self.verbose = verbose
        self.RO_trigger_separation = RO_trigger_separation
        self.RO_trigger_channel = RO_trigger_channel
        self.RO_pars = RO_pars
        self.data_points = data_points
        self.n_measured = 0
        self.seq = None
        self.elts = []
        self.finished = False
        self.upload = upload

    def prepare(self):
        if self.upload:
            sqs.mixer_skewness_cal_sqs(pulseIch=self.pulseIch,
                              pulseQch=self.pulseQch,
                              alpha=self.alpha,
                              phi_skew=self.phi_skew,
                              f_mod=self.f_mod,
                              RO_trigger_channel=self.RO_trigger_channel,
                              RO_pars=self.RO_pars,
                              amplitude=self.amplitude,
                              RO_trigger_separation=self.RO_trigger_separation,
                              data_points=self.data_points)


class arbitrary_variable_swf(swf.Hard_Sweep):

    def __init__(self,control=None,parameter = None):

        super().__init__()
        self.name = 'arbitrary_variable_swf'
        self.parameter_name = 'phi_skew'
        self.unit = 'Deg'
        if control is not None:
            if not hasattr(control,'children'):
                control.children = [self]
            if not hasattr(control,'prepared'):
                control.prepared = [False]
        self.control_swf = control

    def set_parameter(self,value):
        pass

    def prepare(self):
        if self.control_swf is not None:
            #here there has to be a name map in the control swf in order to find
            #the correct swf. in control_swf and set it's prepared value to True.
          pass


class MultiElemSegmentTimingSwf(swf.Hard_Sweep):
    def __init__(self, phases, qbn, op_dict, ramsey_time, nr_wait_elems,
                 elem_type='interleaved', cal_points=((-4, -3), (-2, -1)),
                 upload=True):
        super().__init__()
        self.phases = phases
        self.qbn = qbn
        self.op_dict = op_dict
        self.ramsey_time = ramsey_time
        self.nr_wait_elems = nr_wait_elems
        self.elem_type = elem_type
        self.cal_points = cal_points
        self.upload = upload
        self.name = 'Phase of second pi/2 pulse'
        self.parameter_name = 'phase'
        self.unit = 'rad'

    def prepare(self, **kw):
        if self.upload:
            sqs.multi_elem_segment_timing_seq(
                self.phases,
                self.qbn,
                self.op_dict,
                self.ramsey_time,
                self.nr_wait_elems,
                elem_type=self.elem_type,
                cal_points=self.cal_points,
                upload=True)

class SegmentHardSweep(swf.Hard_Sweep):

    def __init__(self, sequence, upload=True):
        super().__init__()
        self.sequence = sequence
        self.upload = upload

    def prepare(self, **kw):
        if self.upload:
            ps.Pulsar.get_instance().program_awgs(self.sequence)

class Rabi(swf.Hard_Sweep):

    def __init__(self, qb_name, operation_dict, cal_points,
                 upload=True, n=1, preparation_type='wait', post_ro_wait=1e-6,
                 reset_reps=1, final_reset_pulse=True, for_ef=False):
        super().__init__()
        self.qb_name = qb_name
        self.operation_dict = operation_dict
        self.n = n
        self.cal_points = cal_points
        self.upload = upload
        self.preparation_type = preparation_type
        self.post_ro_wait = post_ro_wait
        self.reset_reps = reset_reps
        self.final_reset_pulse = final_reset_pulse
        self.name = 'Rabi'
        self.parameter_name = 'amplitude'
        self.unit = 'V'
        self.for_ef = for_ef

    def prepare(self, **kw):
        if self.upload:
            sqs.rabi_seq_active_reset(amps=self.sweep_points, qb_name=self.qb_name, operation_dict=self.operation_dict,
                                      cal_points=self.cal_points, upload=self.upload, n=self.n,
                                      preparation_type=self.preparation_type, post_ro_wait=self.post_ro_wait,
                                      reset_reps=self.reset_reps, final_reset_pulse=self.final_reset_pulse,
                                      for_ef=self.for_ef)

class Rabi_2nd_exc(swf.Hard_Sweep):

    def __init__(self, pulse_pars, pulse_pars_2nd, RO_pars,
                 last_ge_pulse=True,
                 n=1, cal_points=True, no_cal_points=4, upload=True,
                 return_seq=False):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.n = n
        self.cal_points = cal_points
        self.no_cal_points = no_cal_points
        self.upload = upload
        self.name = 'Rabi 2nd excited state'
        self.parameter_name = 'amplitude'
        self.unit = 'V'
        self.last_ge_pulse = last_ge_pulse
        self.return_seq = return_seq

    def prepare(self, **kw):
        if self.upload:
            sqs2.rabi_2nd_exc_seq(amps=self.sweep_points,
                                  last_ge_pulse=self.last_ge_pulse,
                                  pulse_pars=self.pulse_pars,
                                  pulse_pars_2nd=self.pulse_pars_2nd,
                                  RO_pars=self.RO_pars, upload=self.upload,
                                  cal_points=self.cal_points,
                                  no_cal_points=self.no_cal_points,
                                  n=self.n, return_seq=self.return_seq)

class two_qubit_tomo_cardinal(swf.Hard_Sweep):

    def __init__(self, cardinal, q0_pulse_pars, q1_pulse_pars,
                 RO_pars, timings_dict, upload=True, return_seq=False):
        super().__init__()
        self.cardinal = cardinal
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.RO_pars = RO_pars
        self.timings_dict = timings_dict
        self.upload = upload
        self.return_seq = return_seq
        self.name = 'Tomo2Q_%d' % cardinal
        self.parameter_name = 'Tomo Pulses'
        self.unit = 'V'

    def prepare(self, **kw):
        if self.upload:
            self.seq = mq_sqs.two_qubit_tomo_cardinal(cardinal=self.cardinal,
                                                      q0_pulse_pars=self.q0_pulse_pars,
                                                      q1_pulse_pars=self.q1_pulse_pars,
                                                      RO_pars=self.RO_pars,
                                                      timings_dict=self.timings_dict,
                                                      upload=self.upload,
                                                      return_seq=self.return_seq)

class two_qubit_tomo_bell(swf.Hard_Sweep):

    def __init__(self, bell_state, q0_pulse_pars, q1_pulse_pars,
                 q0_flux_pars, q1_flux_pars,
                 RO_pars, distortion_dict, AWG,
                 timings_dict, CPhase=True, upload=True, return_seq=False):
        super().__init__()
        self.bell_state = bell_state
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.q0_flux_pars = q0_flux_pars
        self.q1_flux_pars = q1_flux_pars
        self.RO_pars = RO_pars
        self.CPhase = CPhase
        self.distortion_dict = distortion_dict
        self.timings_dict = timings_dict
        self.AWG = AWG
        self.upload = upload
        self.return_seq = return_seq
        self.name = 'Tomo2Q_%d' % bell_state
        self.parameter_name = 'Tomo Pulses'
        self.unit = 'V'

    def prepare(self, **kw):
        if self.upload:
            old_val_qS = self.AWG.get(
                '{}_amp'.format(self.q0_flux_pars['channel']))
            old_val_qCP = self.AWG.get(
                '{}_amp'.format(self.q1_flux_pars['channel']))

            # Rescaling the AWG channel amp is done to ensure that the dac
            # values of the flux pulses (including kernels) are defined on
            # a 2Vpp scale.
            self.AWG.set(
                '{}_amp'.format(self.q1_flux_pars['channel']), 2.)
            self.AWG.set(
                '{}_amp'.format(self.q0_flux_pars['channel']), 2.)

            self.seq = mq_sqs.two_qubit_tomo_bell(bell_state=self.bell_state,
                                                      q0_pulse_pars=self.q0_pulse_pars,
                                                      q1_pulse_pars=self.q1_pulse_pars,
                                                      q0_flux_pars=self.q0_flux_pars,
                                                      q1_flux_pars=self.q1_flux_pars,
                                                      RO_pars=self.RO_pars,
                                                      distortion_dict=self.distortion_dict,
                                                      timings_dict=self.timings_dict,
                                                      CPhase=self.CPhase,
                                                      upload=self.upload,
                                                      return_seq=self.return_seq)
            self.AWG.set('{}_amp'.format(self.q1_flux_pars['channel']),
                         old_val_qCP)

            self.AWG.set('{}_amp'.format(self.q0_flux_pars['channel']),
                         old_val_qS)
            self.upload = False

class Flipping(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars, n=1, upload=True, return_seq=False):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.n = n
        self.upload = upload
        self.name = 'Rabi'
        self.parameter_name = 'amplitude'
        self.unit = 'V'
        self.return_seq = return_seq

    def prepare(self, **kw):
        if self.upload:
            sqs.Flipping_seq(pulse_pars=self.pulse_pars,
                             RO_pars=self.RO_pars,
                             n=self.n, return_seq=self.return_seq)


class Rabi_amp90(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars, n=1, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.n = n
        self.upload = upload
        self.name = 'Rabi_amp90'
        self.parameter_name = 'ratio_amp90_amp180'
        self.unit = ''

    def prepare(self, **kw):
        if self.upload:
            sqs.rabi_amp90_seq(scales=self.sweep_points,
                               pulse_pars=self.pulse_pars,
                               RO_pars=self.RO_pars,
                               n=self.n)

# class chevron_length(swf.Hard_Sweep):

#     def __init__(self, operation_dict,
#                  # mw_pulse_pars, RO_pars,
#                  # flux_pulse_pars,
#                  dist_dict, AWG, upload=True,
#                  return_seq=False):
#         super().__init__()
#         # self.mw_pulse_pars = mw_pulse_pars
#         # self.RO_pars = RO_pars
#         # self.flux_pulse_pars = flux_pulse_pars
#         self.operation_dict = operation_dict
#         self.fluxing_channel = fluxing_channel
#         self.dist_dict = dist_dict
#         self.upload = upload
#         self.name = 'Chevron'
#         self.parameter_name = 'Time'
#         self.unit = 's'
#         self.return_seq = return_seq
#         self.AWG = AWG

#     def prepare(self, **kw):
#         if self.upload:
#             old_val = self.AWG.get(
#                 '{}_amp'.format(self.flux_pulse_pars['channel']))
#             # Rescaling the AWG channel amp is done to ensure that the dac
#             # values of the flux pulses (including kernels) are defined on
#             # a 2Vpp scale.
#             self.AWG.set('{}_amp'.format(self.flux_pulse_pars['channel']), 2.)
#             fsqs.chevron_seq(self.mw_pulse_pars,
#                              self.RO_pars,
#                              self.flux_pulse_pars,
#                              pulse_lengths=self.sweep_points,
#                              distortion_dict=self.dist_dict)
#             self.AWG.set('{}_amp'.format(self.flux_pulse_pars['channel']),
#                          old_val)

#     def set_parameter(self, val, **kw):
#         pass


class chevron_single_element(swf.Soft_Sweep):

    def __init__(self, pulse_length, mw_pulse_pars, RO_pars,
                 flux_pulse_pars, dist_dict, AWG, upload=True,
                 return_seq=False):
        super().__init__()
        self.mw_pulse_pars = mw_pulse_pars
        self.RO_pars = RO_pars
        self.flux_pulse_pars = flux_pulse_pars
        self.dist_dict = dist_dict
        self.upload = upload
        self.name = 'Chevron'
        self.parameter_name = 'Time'
        self.unit = 's'
        self.return_seq = return_seq
        self.AWG = AWG
        self.pulse_length = pulse_length

    def prepare(self, **kw):
        if self.upload:
            old_val = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars['channel']))
            # Rescaling the AWG channel amp is done to ensure that the dac
            # values of the flux pulses (including kernels) are defined on
            # a 2Vpp scale.
            self.AWG.set('{}_amp'.format(self.flux_pulse_pars['channel']), 2.)
            fsqs.chevron_seq(self.mw_pulse_pars,
                             self.RO_pars,
                             self.flux_pulse_pars,
                             pulse_lengths=[self.pulse_length],
                             distortion_dict=self.dist_dict,
                             cal_points=False)
            self.AWG.set('{}_amp'.format(self.flux_pulse_pars['channel']),
                         old_val)

    def set_parameter(self, val, **kw):
        pass

class swap_swap_wait(swf.Hard_Sweep):

    def __init__(self, mw_pulse_pars, RO_pars,
                 flux_pulse_pars, dist_dict, AWG,
                 inter_swap_wait=100e-9,
                 upload=True,
                 return_seq=False):
        super().__init__()
        self.mw_pulse_pars = mw_pulse_pars
        self.RO_pars = RO_pars
        self.flux_pulse_pars = flux_pulse_pars
        self.dist_dict = dist_dict
        self.upload = upload
        self.name = 'swap-wait-swap'
        self.parameter_name = 'phase'
        self.unit = 'deg'
        self.return_seq = return_seq
        self.AWG = AWG
        self.inter_swap_wait = inter_swap_wait

    def prepare(self, **kw):
        if self.upload:
            old_val = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars['channel']))
            # Rescaling the AWG channel amp is done to ensure that the dac
            # values of the flux pulses (including kernels) are defined on
            # a 2Vpp scale.
            self.AWG.set('{}_amp'.format(self.flux_pulse_pars['channel']), 2.)
            fsqs.swap_swap_wait(self.mw_pulse_pars,
                                self.RO_pars,
                                self.flux_pulse_pars,
                                phases=self.sweep_points,
                                inter_swap_wait=self.inter_swap_wait,
                                distortion_dict=self.dist_dict)
            self.AWG.set('{}_amp'.format(self.flux_pulse_pars['channel']),
                         old_val)

    def set_parameter(self, val, **kw):
        pass

class swap_CP_swap_2Qubits(swf.Hard_Sweep):

    def __init__(self,
                 mw_pulse_pars_qCP, mw_pulse_pars_qS,
                 flux_pulse_pars_qCP, flux_pulse_pars_qS,
                 RO_pars,
                 dist_dict,
                 AWG,
                 CPhase=True,
                 excitations='both',
                 inter_swap_wait=100e-9,
                 upload=True,
                 identity=False,
                 return_seq=False,
                 reverse_control_target=False):
        super().__init__()
        self.mw_pulse_pars_qCP = mw_pulse_pars_qCP
        self.mw_pulse_pars_qS = mw_pulse_pars_qS
        self.flux_pulse_pars_qCP = flux_pulse_pars_qCP
        self.flux_pulse_pars_qS = flux_pulse_pars_qS
        self.RO_pars = RO_pars
        self.dist_dict = dist_dict

        self.CPhase = CPhase
        self.excitations = excitations
        self.inter_swap_wait = inter_swap_wait

        self.upload = upload
        self.name = 'swap-CP-swap'
        self.parameter_name = 'phase'
        self.unit = 'deg'
        self.return_seq = return_seq
        self.AWG = AWG
        self.reverse_control_target=reverse_control_target

    def prepare(self, **kw):
        if self.upload:
            old_val_qS = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars_qS['channel']))
            old_val_qCP = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars_qCP['channel']))

            # Rescaling the AWG channel amp is done to ensure that the dac
            # values of the flux pulses (including kernels) are defined on
            # a 2Vpp scale.
            self.AWG.set(
                '{}_amp'.format(self.flux_pulse_pars_qCP['channel']), 2.)
            self.AWG.set(
                '{}_amp'.format(self.flux_pulse_pars_qS['channel']), 2.)
            fsqs.swap_CP_swap_2Qubits(
                mw_pulse_pars_qCP=self.mw_pulse_pars_qCP,
                mw_pulse_pars_qS=self.mw_pulse_pars_qS,
                flux_pulse_pars_qCP=self.flux_pulse_pars_qCP,
                flux_pulse_pars_qS=self.flux_pulse_pars_qS,
                RO_pars=self.RO_pars,
                distortion_dict=self.dist_dict,
                CPhase=self.CPhase,
                excitations=self.excitations,
                phases=self.sweep_points,
                inter_swap_wait=self.inter_swap_wait,
                reverse_control_target=self.reverse_control_target
                )
            self.AWG.set('{}_amp'.format(self.flux_pulse_pars_qCP['channel']),
                         old_val_qCP)

            self.AWG.set('{}_amp'.format(self.flux_pulse_pars_qS['channel']),
                         old_val_qS)

    def set_parameter(self, val, **kw):
        pass

class swap_CP_swap_2Qubits_1qphasesweep(swf.Hard_Sweep):

    def __init__(self,
                 mw_pulse_pars_qCP, mw_pulse_pars_qS,
                 flux_pulse_pars_qCP, flux_pulse_pars_qS,
                 RO_pars,
                 dist_dict,
                 timings_dict,
                 AWG,
                 CPhase=True,
                 excitations='both',
                 inter_swap_wait=100e-9,
                 upload=True,
                 identity=False,
                 return_seq=False,
                 reverse_control_target=False,
                 sweep_q=0):
        super().__init__()
        self.mw_pulse_pars_qCP = mw_pulse_pars_qCP
        self.mw_pulse_pars_qS = mw_pulse_pars_qS
        self.flux_pulse_pars_qCP = flux_pulse_pars_qCP
        self.flux_pulse_pars_qS = flux_pulse_pars_qS
        self.RO_pars = RO_pars
        self.dist_dict = dist_dict
        self.timings_dict = timings_dict

        self.CPhase = CPhase
        self.excitations = excitations
        self.inter_swap_wait = inter_swap_wait

        self.upload = upload
        self.name = 'swap-CP-swap'
        self.parameter_name = 'phase'
        self.unit = 'deg'
        self.return_seq = return_seq
        self.AWG = AWG
        self.sweep_q = sweep_q
        self.reverse_control_target=reverse_control_target

    def prepare(self, **kw):
        if self.upload:
            old_val_qS = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars_qS['channel']))
            old_val_qCP = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars_qCP['channel']))

            # Rescaling the AWG channel amp is done to ensure that the dac
            # values of the flux pulses (including kernels) are defined on
            # a 2Vpp scale.
            self.AWG.set(
                '{}_amp'.format(self.flux_pulse_pars_qCP['channel']), 2.)
            self.AWG.set(
                '{}_amp'.format(self.flux_pulse_pars_qS['channel']), 2.)
            self.last_seq = fsqs.swap_CP_swap_2Qubits_1qphasesweep(
                mw_pulse_pars_qCP=self.mw_pulse_pars_qCP,
                mw_pulse_pars_qS=self.mw_pulse_pars_qS,
                flux_pulse_pars_qCP=self.flux_pulse_pars_qCP,
                flux_pulse_pars_qS=self.flux_pulse_pars_qS,
                RO_pars=self.RO_pars,
                distortion_dict=self.dist_dict,
                timings_dict=self.timings_dict,
                CPhase=self.CPhase,
                excitations=self.excitations,
                sphasesweep=self.sweep_points,
                inter_swap_wait=self.inter_swap_wait,
                reverse_control_target=self.reverse_control_target,
                sweep_q=self.sweep_q)
            self.AWG.set('{}_amp'.format(self.flux_pulse_pars_qCP['channel']),
                         old_val_qCP)

            self.AWG.set('{}_amp'.format(self.flux_pulse_pars_qS['channel']),
                         old_val_qS)
        return self.last_seq

    def set_parameter(self, val, **kw):
        pass

class swap_CP_swap_2Qubits_1qphasesweep_amp(swf.Hard_Sweep):

    def __init__(self,
                 mw_pulse_pars_qCP, mw_pulse_pars_qS,
                 flux_pulse_pars_qCP, flux_pulse_pars_qS,
                 RO_pars,
                 dist_dict,
                 timings_dict,
                 AWG,
                 CPhase=True,
                 excitations='both',
                 inter_swap_wait=100e-9,
                 upload=True,
                 identity=False,
                 return_seq=False,
                 reverse_control_target=False,
                 sweep_q=0):
        super().__init__()
        self.mw_pulse_pars_qCP = mw_pulse_pars_qCP
        self.mw_pulse_pars_qS = mw_pulse_pars_qS
        self.flux_pulse_pars_qCP = flux_pulse_pars_qCP
        self.flux_pulse_pars_qS = flux_pulse_pars_qS
        self.RO_pars = RO_pars
        self.dist_dict = dist_dict
        self.timings_dict = timings_dict

        self.CPhase = CPhase
        self.excitations = excitations
        self.inter_swap_wait = inter_swap_wait

        self.upload = upload
        self.name = 'swap-CP-swap'
        self.parameter_name = 'phase'
        self.unit = 'deg'
        self.return_seq = return_seq
        self.AWG = AWG
        self.sweep_q = sweep_q
        self.reverse_control_target=reverse_control_target

    def prepare(self, **kw):
        if self.upload:
            old_val_qS = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars_qS['channel']))
            old_val_qCP = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars_qCP['channel']))

            # Rescaling the AWG channel amp is done to ensure that the dac
            # values of the flux pulses (including kernels) are defined on
            # a 2Vpp scale.
            self.AWG.set(
                '{}_amp'.format(self.flux_pulse_pars_qCP['channel']), 2.)
            self.AWG.set(
                '{}_amp'.format(self.flux_pulse_pars_qS['channel']), 2.)
            self.last_seq = fsqs.swap_CP_swap_2Qubits_1qphasesweep_amp(
                mw_pulse_pars_qCP=self.mw_pulse_pars_qCP,
                mw_pulse_pars_qS=self.mw_pulse_pars_qS,
                flux_pulse_pars_qCP=self.flux_pulse_pars_qCP,
                flux_pulse_pars_qS=self.flux_pulse_pars_qS,
                RO_pars=self.RO_pars,
                distortion_dict=self.dist_dict,
                timings_dict=self.timings_dict,
                CPhase=self.CPhase,
                excitations=self.excitations,
                sphasesweep=self.sweep_points,
                inter_swap_wait=self.inter_swap_wait,
                reverse_control_target=self.reverse_control_target,
                sweep_q=self.sweep_q)
            self.AWG.set('{}_amp'.format(self.flux_pulse_pars_qCP['channel']),
                         old_val_qCP)

            self.AWG.set('{}_amp'.format(self.flux_pulse_pars_qS['channel']),
                         old_val_qS)
        return self.last_seq

    def set_parameter(self, val, **kw):
        pass

class chevron_with_excited_bus_2Qubits(swf.Hard_Sweep):

    def __init__(self,
                 mw_pulse_pars_qCP, mw_pulse_pars_qS,
                 flux_pulse_pars_qCP, flux_pulse_pars_qS,
                 RO_pars,
                 dist_dict,
                 AWG,
                 CPhase=True,
                 excitations=1,
                 upload=True,
                 return_seq=False):
        super().__init__()
        self.mw_pulse_pars_qCP = mw_pulse_pars_qCP
        self.mw_pulse_pars_qS = mw_pulse_pars_qS
        self.flux_pulse_pars_qCP = flux_pulse_pars_qCP
        self.flux_pulse_pars_qS = flux_pulse_pars_qS
        self.RO_pars = RO_pars
        self.dist_dict = dist_dict
        self.CPhase = CPhase
        self.excitations = excitations
        self.upload = upload
        self.name = 'swap-CP length'
        self.parameter_name = 'swap-CP length'
        self.unit = 's'
        self.return_seq = return_seq
        self.AWG = AWG

    def prepare(self, **kw):
        if self.upload:

            old_val_qS = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars_qS['channel']))
            old_val_qCP = self.AWG.get(
                '{}_amp'.format(self.flux_pulse_pars_qCP['channel']))
            # Rescaling the AWG channel amp is done to ensure that the dac
            # values of the flux pulses (including kernels) are defined on
            # a 2Vpp scale.
            self.AWG.set(
                '{}_amp'.format(self.flux_pulse_pars_qCP['channel']), 2.)
            self.AWG.set(
                '{}_amp'.format(self.flux_pulse_pars_qS['channel']), 2.)
            fsqs.chevron_with_excited_bus_2Qubits(
                mw_pulse_pars_qCP=self.mw_pulse_pars_qCP,
                mw_pulse_pars_qS=self.mw_pulse_pars_qS,
                flux_pulse_pars_qCP=self.flux_pulse_pars_qCP,
                flux_pulse_pars_qS=self.flux_pulse_pars_qS,
                RO_pars=self.RO_pars,
                distortion_dict=self.dist_dict,
                excitations=self.excitations,
                chevron_pulse_lengths=self.sweep_points,
            )
            self.AWG.set('{}_amp'.format(self.flux_pulse_pars_qCP['channel']),
                         old_val_qCP)

            self.AWG.set('{}_amp'.format(self.flux_pulse_pars_qS['channel']),
                         old_val_qS)

    def set_parameter(self, val, **kw):
        pass


class chevron_cphase_length(swf.Hard_Sweep):
    # TODO: Delete this function it is deprecasted

    def __init__(self, length_vec, mw_pulse_pars, RO_pars,
                 flux_pulse_pars, cphase_pulse_pars, phase_2, dist_dict, AWG,
                 upload=True, return_seq=False, cal_points=True,
                 toggle_amplitude_sign=False):
        super().__init__()
        self.length_vec = length_vec
        self.mw_pulse_pars = mw_pulse_pars
        self.RO_pars = RO_pars
        self.flux_pulse_pars = flux_pulse_pars
        self.dist_dict = dist_dict
        self.artificial_detuning = 4./length_vec[-1]
        self.upload = upload
        self.name = 'Chevron'
        self.parameter_name = 'Time'
        self.unit = 's'
        self.return_seq = return_seq
        self.cphase_pulse_pars = cphase_pulse_pars
        self.phase_2 = phase_2
        self.AWG = AWG
        self.cal_points = cal_points
        self.toggle_amplitude_sign = toggle_amplitude_sign

    def prepare(self, **kw):
        if self.upload:
            fsqs.chevron_seq_cphase(lengths=self.length_vec,
                                    mw_pulse_pars=self.mw_pulse_pars,
                                    RO_pars=self.RO_pars,
                                    flux_pulse_pars=self.flux_pulse_pars,
                                    cphase_pulse_pars=self.cphase_pulse_pars,
                                    artificial_detuning=self.artificial_detuning,
                                    phase_2=self.phase_2,
                                    distortion_dict=self.dist_dict,
                                    toggle_amplitude_sign=self.toggle_amplitude_sign,
                                    cal_points=self.cal_points)

    def pre_upload(self, **kw):
        self.seq = fsqs.chevron_seq_cphase(lengths=self.length_vec,
                                           mw_pulse_pars=self.mw_pulse_pars,
                                           RO_pars=self.RO_pars,
                                           flux_pulse_pars=self.flux_pulse_pars,
                                           cphase_pulse_pars=self.cphase_pulse_pars,
                                           artificial_detuning=self.artificial_detuning,
                                           phase_2=self.phase_2,
                                           distortion_dict=self.dist_dict,
                                           toggle_amplitude_sign=self.toggle_amplitude_sign,
                                           cal_points=self.cal_points,
                                           return_seq=True)



class BusT2(swf.Hard_Sweep):

    def __init__(self, times_vec, mw_pulse_pars, RO_pars,
                 flux_pulse_pars, dist_dict, AWG, upload=True,
                 return_seq=False):
        super().__init__()
        self.times_vec = times_vec
        self.mw_pulse_pars = mw_pulse_pars
        self.RO_pars = RO_pars
        self.flux_pulse_pars = flux_pulse_pars
        self.dist_dict = dist_dict
        self.upload = upload
        self.name = 'Chevron'
        self.parameter_name = 'Time'
        self.unit = 's'
        self.return_seq = return_seq
        self.AWG = AWG

    def prepare(self, **kw):
        if self.upload:
            fsqs.BusT2(self.times_vec,
                       self.mw_pulse_pars,
                       self.RO_pars,
                       self.flux_pulse_pars,
                       distortion_dict=self.dist_dict)

    def pre_upload(self, **kw):
        self.seq = fsqs.BusT2(self.times_vec,
                              self.mw_pulse_pars,
                              self.RO_pars,
                              self.flux_pulse_pars,
                              distortion_dict=self.dist_dict, return_seq=True)


class BusEcho(swf.Hard_Sweep):

    def __init__(self, times_vec, mw_pulse_pars, RO_pars, artificial_detuning,
                 flux_pulse_pars, dist_dict, AWG, upload=True,
                 return_seq=False):
        super().__init__()
        self.times_vec = times_vec
        self.mw_pulse_pars = mw_pulse_pars
        self.RO_pars = RO_pars
        self.flux_pulse_pars = flux_pulse_pars
        self.dist_dict = dist_dict
        self.artificial_detuning = artificial_detuning
        self.upload = upload
        self.name = 'Chevron'
        self.parameter_name = 'Time'
        self.unit = 's'
        self.return_seq = return_seq
        self.AWG = AWG

    def prepare(self, **kw):
        if self.upload:
            fsqs.BusEcho(self.times_vec,
                         self.mw_pulse_pars,
                         self.RO_pars,
                         self.artificial_detuning,
                         self.flux_pulse_pars,
                         distortion_dict=self.dist_dict)

    def pre_upload(self, **kw):
        self.seq = fsqs.BusEcho(self.times_vec,
                                self.mw_pulse_pars,
                                self.RO_pars,
                                self.artificial_detuning,
                                self.flux_pulse_pars,
                                distortion_dict=self.dist_dict, return_seq=True)

class cphase_fringes(swf.Hard_Sweep):

    def __init__(self, phases, q0_pulse_pars, q1_pulse_pars, RO_pars,
                 swap_pars_q0, cphase_pars_q1, timings_dict,
                 dist_dict, upload=True, return_seq=False):
        super().__init__()
        self.phases = phases,
        self.q0_pulse_pars = q0_pulse_pars,
        self.q1_pulse_pars = q1_pulse_pars,
        self.RO_pars = RO_pars,
        self.swap_pars_q0 = swap_pars_q0,
        self.cphase_pars_q1 = cphase_pars_q1,
        self.timings_dict = timings_dict,
        self.dist_dict = dist_dict
        self.upload = upload
        self.name = 'CPhase'
        self.parameter_name = 'Phase'
        self.unit = 'deg'
        self.return_seq = return_seq

    def prepare(self, **kw):
        if self.upload:
            mq_sqs.cphase_fringes(phases=self.phases,
                                  q0_pulse_pars=self.q0_pulse_pars,
                                  q1_pulse_pars=self.q1_pulse_pars,
                                  RO_pars=self.RO_pars,
                                  swap_pars_q0=self.swap_pars_q0,
                                  cphase_pars_q1=self.cphase_pars_q1,
                                  timings_dict=self.timings_dict,
                                  distortion_dict=self.dist_dict)

    def pre_upload(self, **kw):
        self.seq = mq_sqs.cphase_fringes(phases=self.phases,
                                         q0_pulse_pars=self.q0_pulse_pars,
                                         q1_pulse_pars=self.q1_pulse_pars,
                                         RO_pars=self.RO_pars,
                                         swap_pars_q0=self.swap_pars_q0,
                                         cphase_pars_q1=self.cphase_pars_q1,
                                         timings_dict=self.timings_dict,
                                         distortion_dict=self.dist_dict,
                                         return_seq=True)


class T1(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars, cal_points=True, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points

        self.name = 'T1'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs.T1_seq(times=self.sweep_points,
                       pulse_pars=self.pulse_pars,
                       RO_pars=self.RO_pars,
                       cal_points=self.cal_points,
                       upload=self.upload)

class T1_2nd_exc(swf.Hard_Sweep):

    def __init__(self, pulse_pars, pulse_pars_2nd, RO_pars, no_cal_points=6,
                 cal_points=True, upload=True, last_ge_pulse=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.cal_points = cal_points
        self.no_cal_points=no_cal_points
        self.last_ge_pulse = last_ge_pulse
        self.upload = upload
        self.name = 'T1 2nd excited state'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs2.T1_2nd_exc_seq(times=self.sweep_points,
                                no_cal_points=self.no_cal_points,
                                pulse_pars=self.pulse_pars,
                                pulse_pars_2nd=self.pulse_pars_2nd,
                                RO_pars=self.RO_pars,
                                cal_points=self.cal_points,
                                last_ge_pulse=self.last_ge_pulse)


class AllXY(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars, double_points=False, upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.double_points = double_points
        self.upload = upload

        self.parameter_name = 'AllXY element'
        self.unit = '#'
        self.name = 'AllXY'
        if not double_points:
            self.sweep_points = np.arange(21)
        else:
            self.sweep_points = np.arange(42)

    def prepare(self, **kw):
        if self.upload:
            sqs.AllXY_seq(pulse_pars=self.pulse_pars,
                          RO_pars=self.RO_pars,
                          double_points=self.double_points)


class SingleLevel(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars, pulse_pars_2nd=None, upload=True,
                 RO_spacing=300e-9, level='e', nr_samples=2, preselection=False):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.RO_spacing = RO_spacing
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.name = level
        self.preselection = preselection
        self.sweep_points = np.arange(nr_samples)

    def prepare(self, **kw):
        if self.upload:
            sqs.single_level_seq(pulse_pars=self.pulse_pars, RO_pars=self.RO_pars,
                                 pulse_pars_2nd=self.pulse_pars_2nd,
                                 RO_spacing=self.RO_spacing,
                                 level=self.name, preselection=self.preselection)


class Butterfly(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars, initialize=False, upload=True,
                 post_msmt_delay=2000e-9):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'Buttefly element'
        self.unit = '#'
        self.name = 'Butterfly'
        self.sweep_points = np.arange(2)
        self.initialize = initialize
        self.post_msmt_delay = post_msmt_delay

    def prepare(self, **kw):
        if self.upload:
            sqs.Butterfly_seq(pulse_pars=self.pulse_pars,
                              post_msmt_delay=self.post_msmt_delay,
                              RO_pars=self.RO_pars, initialize=self.initialize)


class Resonator_spectroscopy(swf.Soft_Sweep):

    def __init__(self, RO_channel, RO_MWG, RO_IF,
                 marker_interval=3e-6, marker_length=5e-9,
                 upload=True):

        super().__init__()
        self.RO_channel = RO_channel
        self.marker_interval = marker_interval
        self.marker_length = marker_length
        self.upload = upload
        self.RO_MWG = RO_MWG
        self.RO_IF = RO_IF
        self.parameter_name = 'RO frequency'
        self.unit = 'Hz'
        self.name = 'Resonator_spectroscopy'

        self.prepare(upload=self.upload)

    def prepare(self, upload=False, **kw):
        if upload:
            sqs.Resonator_spec_seq(RO_channel=self.RO_channel,
                                   marker_interval=self.marker_interval,
                                   marker_length=self.marker_length,
                                   upload=True)

    def set_parameter(self, val):
        self.RO_MWG.frequency(val - self.RO_IF)


class Qubit_spectroscopy(swf.Soft_Sweep):

    def __init__(self, spec_pars, RO_pars,
                 drive_MWG, drive_IF,
                 upload=True):

        super().__init__()
        self.spec_pars = spec_pars
        self.RO_pars = RO_pars
        self.drive_MWG = drive_MWG
        self.drive_IF = drive_IF
        self.upload = upload
        self.parameter_name = 'Drive frequency'
        self.unit = 'Hz'
        self.name = 'Qubit_spectroscopy'

        self.prepare(upload=self.upload)

    def prepare(self, upload=False, **kw):
        if upload:
            sqs.Pulsed_spec_seq(spec_pars=self.spec_pars,
                                RO_pars=self.RO_pars,
                                upload=upload)

    def set_parameter(self, val):
        self.drive_MWG.frequency(val)


class Randomized_Benchmarking_nr_cliffords(swf.Soft_Sweep):

    def __init__(self, sweep_control='soft', upload=True,
                 RB_sweepfunction=None):
        super().__init__()

        self.sweep_control = sweep_control
        self.RB_sweepfunction = RB_sweepfunction
        self.upload = upload
        self.is_first_sweeppoint = True
        self.name = 'Randomized_Benchmarking_nr_cliffords'
        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'


    def set_parameter(self, val):
        self.RB_sweepfunction.nr_cliffords_value = val
        self.RB_sweepfunction.upload = self.upload
        self.RB_sweepfunction.prepare(upload_all=self.is_first_sweeppoint)
        self.is_first_sweeppoint = False


class Randomized_Benchmarking_one_length(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars,
                 nr_cliffords_value, #int
                 gate_decomposition='HZ',
                 interleaved_gate=None,
                 cal_points=True,
                 seq_name=None,
                 upload=False):

        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.nr_cliffords_value = nr_cliffords_value
        self.cal_points = cal_points
        self.seq_name = seq_name
        self.gate_decomposition = gate_decomposition
        self.interleaved_gate = interleaved_gate

        self.parameter_name = 'Nr of Seeds'
        self.unit = '#'
        self.name = 'Randomized_Benchmarking_one_length'

    def prepare(self, upload_all=True, **kw):
        if self.upload:
            sqs.Randomized_Benchmarking_seq_one_length(
                self.pulse_pars, self.RO_pars,
                nr_cliffords_value=self.nr_cliffords_value,
                gate_decomposition=self.gate_decomposition,
                interleaved_gate=self.interleaved_gate,
                nr_seeds=self.sweep_points,
                upload_all=upload_all,
                upload=True,
                cal_points=self.cal_points,
                seq_name=self.seq_name)


class Randomized_Benchmarking(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars,
                 nr_seeds, nr_cliffords,
                 cal_points=True,
                 double_curves=False, seq_name=None,
                 upload=True):
        # If nr_cliffords is None it still needs to be specfied when setting
        # the experiment
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.nr_seeds = nr_seeds
        self.cal_points = cal_points
        self.sweep_points = nr_cliffords
        self.double_curves = double_curves
        self.seq_name = seq_name

        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'
        self.name = 'Randomized_Benchmarking'
        self.sweep_points = nr_cliffords
        if double_curves:
            nr_cliffords = np.repeat(nr_cliffords, 2)

        if self.cal_points:
            self.sweep_points = np.concatenate([nr_cliffords,
                                                [nr_cliffords[-1]+.2,
                                                 nr_cliffords[-1]+.3,
                                                 nr_cliffords[-1]+.7,
                                                 nr_cliffords[-1]+.8]])

    def prepare(self, **kw):
        if self.upload:
            sqs.Randomized_Benchmarking_seq(
                self.pulse_pars, self.RO_pars,
                nr_cliffords=self.sweep_points,
                nr_seeds=self.nr_seeds,
                cal_points=self.cal_points,
                double_curves=self.double_curves,
                seq_name=self.seq_name)


class Ramsey(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars,
                 artificial_detuning=None,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning

        self.name = 'Ramsey'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs.ramsey_seq(times=self.sweep_points,
                           pulse_pars=self.pulse_pars,
                           RO_pars=self.RO_pars,
                           artificial_detuning=self.artificial_detuning,
                           cal_points=self.cal_points)


class Ramsey_multiple_detunings(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars,
                 artificial_detunings=None,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detunings = artificial_detunings

        self.name = 'Ramsey_mult_det'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs.ramsey_seq_multiple_detunings(times=self.sweep_points,
                           pulse_pars=self.pulse_pars,
                           RO_pars=self.RO_pars,
                           artificial_detunings=self.artificial_detunings,
                           cal_points=self.cal_points)


class Ramsey_2nd_exc(swf.Hard_Sweep):

    def __init__(self, pulse_pars, pulse_pars_2nd, RO_pars,
                 artificial_detuning=None, return_seq=False,
                 n=1, cal_points=True, upload=True, no_cal_points=6,
                 last_ge_pulse=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.artificial_detuning = artificial_detuning
        self.n = n
        self.cal_points = cal_points
        self.upload = upload
        self.name = 'Ramsey 2nd excited state'
        self.parameter_name = 't'
        self.unit = 's'
        self.return_seq = return_seq
        self.last_ge_pulse = last_ge_pulse
        self.no_cal_points = no_cal_points

    def prepare(self, **kw):
        if self.upload:
            sqs2.ramsey_2nd_exc_seq(times=self.sweep_points,
                                    pulse_pars=self.pulse_pars,
                                    pulse_pars_2nd=self.pulse_pars_2nd,
                                    RO_pars=self.RO_pars,
                                    artificial_detuning =
                                            self.artificial_detuning,
                                    n=self.n, cal_points=self.cal_points,
                                    no_cal_points=self.no_cal_points,
                                    upload=self.upload,
                                    return_seq=self.return_seq,
                                    last_ge_pulse=self.last_ge_pulse)

class Ramsey_2nd_exc_multiple_detunings(swf.Hard_Sweep):
    def __init__(self, pulse_pars, pulse_pars_2nd, RO_pars,
                 artificial_detunings=None, return_seq=False,
                 n=1, cal_points=True, upload=True, no_cal_points=6,
                 last_ge_pulse=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.artificial_detunings = artificial_detunings
        self.n = n
        self.cal_points = cal_points
        self.upload = upload
        self.name = 'Rabi 2nd excited state'
        self.parameter_name = 't'
        self.unit = 's'
        self.return_seq = return_seq
        self.last_ge_pulse = last_ge_pulse
        self.no_cal_points = no_cal_points

    def prepare(self, **kw):
        if self.upload:
            sqs2.ramsey_2nd_exc_seq_multiple_detunings(times=self.sweep_points,
                                                       pulse_pars=self.pulse_pars,
                                                       pulse_pars_2nd=self.pulse_pars_2nd,
                                                       RO_pars=self.RO_pars,
                                                       n=self.n,
                                                       cal_points=self.cal_points,
                                                       artificial_detunings=self.artificial_detunings,
                                                       upload=self.upload,
                                                       return_seq=self.return_seq,
                                                       last_ge_pulse=self.last_ge_pulse)


class FluxDetuning(swf.Hard_Sweep):

    def __init__(self, flux_params, pulse_pars, RO_pars,
                 artificial_detuning=None,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.flux_params = flux_params
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning

        self.name = 'FluxDetuning'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            seq=sqs.Ramsey_seq(flux_pars=self.flux_params, times=self.sweep_points,
                           pulse_pars=self.pulse_pars,
                           RO_pars=self.RO_pars,
                           artificial_detuning=self.artificial_detuning,
                           cal_points=self.cal_points,
                           upload=False,
                           return_seq=True)

class Echo(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars,
                 artificial_detuning=None,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning
        self.name = 'Echo'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs.echo_seq(times=self.sweep_points,
                         pulse_pars=self.pulse_pars,
                         RO_pars=self.RO_pars,
                         artificial_detuning=self.artificial_detuning,
                         cal_points=self.cal_points)


class Echo_2nd_exc(swf.Hard_Sweep):

    def __init__(self, pulse_pars, pulse_pars_2nd, RO_pars,
                 artificial_detuning=None, return_seq=False,
                 cal_points=True, upload=True, no_cal_points=6,
                 last_ge_pulse=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.no_cal_points = no_cal_points
        self.artificial_detuning = artificial_detuning
        self.last_ge_pulse = last_ge_pulse
        self.return_seq = return_seq
        self.name = 'Echo 2nd excited state'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs2.echo_2nd_exc_seq(times=self.sweep_points,
                                  pulse_pars=self.pulse_pars,
                                  pulse_pars_2nd=self.pulse_pars_2nd,
                                  RO_pars=self.RO_pars,
                                  artificial_detuning=self.artificial_detuning,
                                  cal_points=self.cal_points,
                                  no_cal_points=self.no_cal_points,
                                  upload=self.upload,
                                  return_seq=self.return_seq,
                                  last_ge_pulse=self.last_ge_pulse)


class Motzoi_XY(swf.Hard_Sweep):

    def __init__(self, motzois, pulse_pars, RO_pars, upload=True):
        '''
        Measures 2 points per motzoi value specified in motzois and adds 4
        calibration points to it.
        '''
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.name = 'Motzoi_XY'
        self.parameter_name = 'motzoi'
        self.unit = ' '
        sweep_pts = np.repeat(motzois, 2)
        self.sweep_points = np.append(sweep_pts,
                                      [motzois[-1]+(motzois[-1]-motzois[-2])]*4)

    def prepare(self, **kw):
        if self.upload:
            sqs.Motzoi_XY(motzois=self.sweep_points,
                          pulse_pars=self.pulse_pars,
                          RO_pars=self.RO_pars)

class QScale(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars, upload=True,
                 cal_points=True):
        '''
        Measures 3 number of points per QScale parameter value specified
        in qscales and adds 4 calibration points to it.
        '''
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.name = 'QScale'
        self.parameter_name = 'QScale_factor'
        self.unit = ''
        self.cal_points = cal_points

    def prepare(self, **kw):
        if self.upload:
            sqs.qscale(qscales=self.sweep_points,
                       pulse_pars=self.pulse_pars,
                       RO_pars=self.RO_pars,
                       cal_points=self.cal_points)

class QScale_2nd_exc(swf.Hard_Sweep):

    def __init__(self, qscales, pulse_pars, pulse_pars_2nd, RO_pars,
                 cal_points=True, upload=True, return_seq=False,
                 last_ge_pulse=True, no_cal_points=6):
        super().__init__()
        self.qscales = qscales
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.cal_points = cal_points
        self.upload = upload
        self.name = 'Qscale 2nd excited state'
        self.parameter_name = 'motzoi'
        self.unit = ''
        self.return_seq = return_seq
        self.last_ge_pulse = last_ge_pulse
        self.no_cal_points=no_cal_points

    def prepare(self, **kw):
        if self.upload:
            sqs2.qscale_2nd_exc_seq(qscales=self.qscales,
                                      pulse_pars=self.pulse_pars,
                                      pulse_pars_2nd=self.pulse_pars_2nd,
                                      RO_pars=self.RO_pars,
                                      upload=self.upload,
                                      cal_points=self.cal_points,
                                      no_cal_points=self.no_cal_points,
                                      return_seq=self.return_seq,
                                      last_ge_pulse=self.last_ge_pulse)


class Freq_XY(swf.Hard_Sweep):

    def __init__(self, freqs, pulse_pars, RO_pars, upload=True):
        '''
        Measures 2 points per motzoi value specified in freqs and adds 4
        calibration points to it.
        '''
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.name = 'Motzoi_XY'
        self.parameter_name = 'motzoi'
        self.unit = ' '
        sweep_pts = np.repeat(freqs, 2)
        self.sweep_points = np.append(sweep_pts,
                                      [freqs[-1]+(freqs[-1]-freqs[-2])]*4)

    def prepare(self, **kw):
        if self.upload:
            sqs.Motzoi_XY(motzois=self.sweep_points,
                          pulse_pars=self.pulse_pars,
                          RO_pars=self.RO_pars)


class CBox_T1(swf.Hard_Sweep):

    def __init__(self, IF, RO_pulse_delay, RO_trigger_delay, mod_amp, AWG,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.name = 'T1'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.AWG = AWG
        self.mod_amp = mod_amp
        self.upload = upload

    def prepare(self, **kw):
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch4_amp')
            st_seqs.CBox_T1_marker_seq(IF=self.IF, times=self.sweep_points,
                                       RO_pulse_delay=self.RO_pulse_delay,
                                       RO_trigger_delay=self.RO_trigger_delay,
                                       verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class CBox_v3_T1(swf.Hard_Sweep):

    def __init__(self, CBox, upload=True):
        super().__init__()
        self.name = 'T1'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.upload = upload
        self.CBox = CBox

    def prepare(self, **kw):
        if self.upload:
            self.CBox.AWG0_mode('Codeword-trigger mode')
            self.CBox.AWG1_mode('Codeword-trigger mode')
            self.CBox.AWG2_mode('Codeword-trigger mode')
            self.CBox.set_master_controller_working_state(0, 0, 0)
            self.CBox.load_instructions('CBox_v3_test_program\T1.asm')
            self.CBox.set_master_controller_working_state(1, 0, 0)


class CBox_v3_T1(swf.Hard_Sweep):

    def __init__(self, CBox, upload=True):
        super().__init__()
        self.name = 'T1'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.upload = upload
        self.CBox = CBox

    def prepare(self, **kw):
        if self.upload:
            self.CBox.AWG0_mode('Codeword-trigger mode')
            self.CBox.AWG1_mode('Codeword-trigger mode')
            self.CBox.AWG2_mode('Codeword-trigger mode')
            self.CBox.set_master_controller_working_state(0, 0, 0)
            self.CBox.load_instructions('CBox_v3_test_program\T1.asm')
            self.CBox.set_master_controller_working_state(1, 0, 0)


class CBox_Ramsey(swf.Hard_Sweep):

    def __init__(self, IF, RO_pulse_length,
                 RO_pulse_delay, RO_trigger_delay, pulse_delay,
                 AWG, CBox, cal_points=True,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.pulse_delay = pulse_delay
        self.RO_pulse_length = RO_pulse_length
        self.name = 'T2*'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.AWG = AWG
        self.CBox = CBox
        self.upload = upload
        self.cal_points = cal_points

    def prepare(self, **kw):
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_Ramsey_marker_seq(
                IF=self.IF, times=self.sweep_points,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_pulse_length=self.RO_pulse_length,
                RO_trigger_delay=self.RO_trigger_delay,
                pulse_delay=self.pulse_delay,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)

        # gets assigned in MC.set sweep_points
        nr_elts = len(self.sweep_points)
        if self.cal_points:  # append the calibration points to the tape
            tape = [3, 3] * (nr_elts-4) + [0, 0, 0, 0, 0, 1, 0, 1]
        else:
            tape = [3, 3] * nr_elts

        self.AWG.stop()
        # TODO Change to segmented tape if we have the new timing tape
        self.CBox.AWG0_mode.set('segmented tape')
        self.CBox.AWG1_mode.set('segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', tape)
        self.CBox.set('AWG1_tape', tape)


class CBox_Echo(swf.Hard_Sweep):

    def __init__(self, IF,
                 RO_pulse_delay, RO_trigger_delay, pulse_delay,
                 AWG, CBox, cal_points=True,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.pulse_delay = pulse_delay
        self.name = 'T2-echo'
        self.parameter_name = 'tau'
        self.unit = 's'
        self.AWG = AWG
        self.CBox = CBox
        self.upload = upload
        self.cal_points = cal_points
        logging.warning('Underlying sequence is not implemented')
        logging.warning('Replace it with the multi-pulse sequence')

    def prepare(self, **kw):
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_Echo_marker_seq(
                IF=self.IF, times=self.sweep_points,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)

        # gets assigned in MC.set sweep_points
        nr_elts = len(self.sweep_points)
        if self.cal_points:
            tape = [3, 3] * (nr_elts-4) + [0, 1]
        else:
            tape = [3, 3] * nr_elts

        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', tape)
        self.CBox.set('AWG1_tape', tape)


class CBox_OffOn(swf.Hard_Sweep):

    def __init__(self, IF, RO_pulse_delay, RO_trigger_delay,
                 RO_pulse_length,
                 AWG, CBox,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.parameter_name = 'Tape element'
        self.unit = ''
        self.name = 'Off-On'
        self.tape = [0, 1]
        self.sweep_points = np.array(self.tape)  # array for transpose in MC
        self.AWG = AWG
        self.CBox = CBox
        self.RO_pulse_length = RO_pulse_length
        # would actually like to check if file is already loaded
        # filename can be get using AWG.get('setup_filename')
        self.upload = upload

    def prepare(self, **kw):
        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', self.tape)
        self.CBox.set('AWG1_tape', self.tape)

        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_single_pulse_seq(
                IF=self.IF,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay,
                RO_pulse_length=self.RO_pulse_length,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)
            # Prevents reloading, potentially bug prone as reusing the swf
            # does not rest the upload flag
            self.upload = False


class CBox_AllXY(swf.Hard_Sweep):

    def __init__(self, IF, pulse_delay,
                 RO_pulse_delay,
                 RO_trigger_delay,
                 RO_pulse_length,
                 AWG, CBox,
                 double_points=True,
                 upload=True):
        '''
        Generates a sequence for the AWG to trigger the CBox and sets the tape
        in the CBox to measure an AllXY.

        double_points: True will measure the tape twice per element, this
            should give insight wether the deviation is real.
        '''
        super().__init__()
        self.parameter_name = 'AllXY element'
        self.unit = '#'
        self.name = 'AllXY'
        # would actually like to check if file is already loaded
        # filename can be get using AWG.get('setup_filename')
        self.upload = upload

        # The AllXY tape
        self.tape = np.array([0, 0, 1, 1,  # 1, 2
                              2, 2, 1, 2,  # 3, 4
                              2, 1, 3, 0,  # 5, 6
                              4, 0, 3, 4,  # 7, 8
                              4, 3, 3, 2,  # 9, 10
                              4, 1, 1, 4,  # 11, 12
                              2, 3, 3, 1,  # 13, 14
                              1, 3, 4, 2,  # 15, 16
                              2, 4, 1, 0,  # 17, 18
                              2, 0, 3, 3,  # 19, 20
                              4, 4])       # 21
        if double_points:
            double_tape = []
            for i in range(len(self.tape)//2):
                for j in range(2):
                    double_tape.extend((self.tape[2*i:2*i+2]))
            self.tape = double_tape
        self.sweep_points = np.arange(
            int(len(self.tape)/2))  # 2 pulses per elt

        # Making input pars available to prepare
        # Required instruments
        self.AWG = AWG
        self.CBox = CBox
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.RO_pulse_length = RO_pulse_length
        self.pulse_delay = pulse_delay

    def prepare(self, **kw):
        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', self.tape)
        self.CBox.set('AWG1_tape', self.tape)

        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')

            st_seqs.CBox_two_pulse_seq(
                IF=self.IF,
                pulse_delay=self.pulse_delay,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_pulse_length=self.RO_pulse_length,
                RO_trigger_delay=self.RO_trigger_delay, verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class CBox_multi_element_tape(swf.Hard_Sweep):

    def __init__(self, n_pulses, tape,
                 pulse_delay,
                 IF, RO_pulse_delay, RO_trigger_delay,
                 RO_pulse_length,
                 AWG, CBox,
                 upload=True):
        '''
        Sets an arbitrary tape as a sequence
        n_pulses is the number of pulses per element in the sequence
        by default
        '''
        super().__init__()
        self.n_pulses = n_pulses
        self.parameter_name = 'Element'
        self.unit = '#'
        self.name = 'multi-element tape'
        self.tape = tape

        self.upload = upload

        self.sweep_points = np.arange(int(len(self.tape)/n_pulses))
        self.AWG = AWG
        self.CBox = CBox
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.RO_pulse_length = RO_pulse_length
        self.pulse_delay = pulse_delay

    def prepare(self, **kw):
        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', self.tape)
        self.CBox.set('AWG1_tape', self.tape)
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_multi_pulse_seq(
                n_pulses=self.n_pulses, pulse_delay=self.pulse_delay,
                IF=self.IF,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay,
                RO_pulse_length=self.RO_pulse_length,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class Resetless_tape(swf.Hard_Sweep):

    def __init__(self, n_pulses, tape,
                 pulse_delay, resetless_interval,
                 IF, RO_pulse_delay, RO_trigger_delay,
                 RO_pulse_length,
                 AWG, CBox,
                 upload=True):
        super().__init__()
        self.IF = IF
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        self.parameter_name = 'Tape element'
        self.unit = ''
        self.name = 'Resetless_tape'
        self.tape = tape
        # array for transpose in MC these values are bs
        self.sweep_points = np.array(self.tape)
        self.AWG = AWG
        self.CBox = CBox
        self.RO_pulse_length = RO_pulse_length
        # would actually like to check if file is already loaded
        # filename can be get using AWG.get('setup_filename')
        self.upload = upload

        self.n_pulses = n_pulses
        self.resetless_interval = resetless_interval
        self.pulse_delay = pulse_delay

    def prepare(self, **kw):
        self.AWG.stop()
        self.CBox.AWG0_mode.set('Segmented tape')
        self.CBox.AWG1_mode.set('Segmented tape')
        self.CBox.restart_awg_tape(0)
        self.CBox.restart_awg_tape(1)
        self.CBox.set('AWG0_tape', self.tape)
        self.CBox.set('AWG1_tape', self.tape)
        if self.upload:
            ch3_amp = self.AWG.get('ch3_amp')
            ch4_amp = self.AWG.get('ch3_amp')
            st_seqs.CBox_resetless_multi_pulse_seq(
                n_pulses=self.n_pulses, pulse_delay=self.pulse_delay,
                resetless_interval=self.resetless_interval,
                IF=self.IF,
                RO_pulse_delay=self.RO_pulse_delay,
                RO_trigger_delay=self.RO_trigger_delay,
                RO_pulse_length=self.RO_pulse_length,
                verbose=False)
            self.AWG.set('ch3_amp', ch3_amp)
            self.AWG.set('ch4_amp', ch4_amp)


class CBox_RB_sweep(swf.Hard_Sweep):

    def __init__(self,
                 IF, RO_pulse_length,
                 RO_pulse_delay, RO_trigger_delay,
                 pulse_delay,
                 AWG, CBox, LutMan,
                 cal_points=True,
                 nr_cliffords=[1, 3, 5, 10, 20],
                 nr_seeds=3, max_seq_duration=15e-6,
                 safety_margin=500e-9,
                 upload=True):
        super().__init__()
        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'
        self.name = 'Randomized_Benchmarking'
        self.safety_margin = safety_margin
        # Making input pars available to prepare
        # Required instruments
        self.AWG = AWG
        self.CBox = CBox
        self.LutMan = LutMan
        self.nr_seeds = nr_seeds
        self.cal_points = [0, 0, 1, 1]
        self.nr_cliffords = np.array(nr_cliffords)
        self.max_seq_duration = max_seq_duration
        self.pulse_delay_ns = pulse_delay*1e9

        self.IF = IF
        self.RO_pulse_length = RO_pulse_length
        self.RO_pulse_delay = RO_pulse_delay
        self.RO_trigger_delay = RO_trigger_delay
        # Funny last sweep point values are to make the cal points appear
        # in sensible (visible) places in the plot
        self.sweep_points = np.concatenate([nr_cliffords,
                                            [nr_cliffords[-1]+.2,
                                             nr_cliffords[-1]+.3,
                                             nr_cliffords[-1]+.7,
                                             nr_cliffords[-1]+.8]])

    def prepare(self, upload_tek_seq=True, **kw):
        self.AWG.stop()
        n_cls = self.nr_cliffords
        time_tape = []
        pulse_length = self.LutMan.gauss_width.get()*4
        for seed in range(self.nr_seeds):
            for n_cl in n_cls:
                cliffords = rb.randomized_benchmarking_sequence(n_cl)
                cl_tape = rb.convert_clifford_sequence_to_tape(
                    cliffords,
                    self.LutMan.lut_mapping.get())
                for i, tape_elt in enumerate(cl_tape):
                    if i == 0:
                        # wait_time is in ns
                        wait_time = (self.max_seq_duration*1e9 -
                                     (len(cl_tape)-1)*self.pulse_delay_ns -
                                     pulse_length)
                    else:
                        wait_time = self.pulse_delay_ns - pulse_length
                    end_of_marker = (i == (len(cl_tape)-1))
                    entry = self.CBox.create_timing_tape_entry(
                        wait_time, tape_elt, end_of_marker, prepend_elt=0)
                    time_tape.extend(entry)

            for cal_pt in self.cal_points:
                wait_time = self.max_seq_duration*1e9 - pulse_length
                time_tape.extend(self.CBox.create_timing_tape_entry(
                    wait_time, cal_pt, True, prepend_elt=0))
        for awg in range(3):
            self.CBox.set('AWG{}_mode'.format(awg), 'Segmented')
            self.CBox.set_segmented_tape(awg, time_tape)
            self.CBox.restart_awg_tape(awg)
        if upload_tek_seq:
            self.upload_tek_seq()

    def upload_tek_seq(self):
        st_seqs.CBox_single_pulse_seq(
            IF=self.IF,
            RO_pulse_delay=self.RO_pulse_delay +
            self.max_seq_duration+self.safety_margin,
            RO_trigger_delay=self.RO_trigger_delay,
            RO_pulse_length=self.RO_pulse_length)


class Two_d_CBox_RB_seq(swf.Soft_Sweep):

    def __init__(self, CBox_RB_sweepfunction):
        super().__init__()
        self.parameter_name = 'Idx'
        self.unit = '#'
        self.name = 'Randomized_Benchmarking_random_seeds'
        self.CBox_RB_sweepfunction = CBox_RB_sweepfunction

    def set_parameter(self, val):
        '''
        Uses the CBox RB sweepfunction to upload a new tape of random cliffords
        explicitly does not reupload the AWG sequence.
        '''
        self.CBox_RB_sweepfunction.prepare(upload_tek_seq=False)


class Load_Sequence_Tek(swf.Hard_Sweep):

    def __init__(self, AWG, sequence_name, seq_elements, upload=True):
        super().__init__()
        self.sweep_points = seq_elements
        self.len = len(seq_elements)
        self.name = sequence_name
        self.parameter_name = 'amplitude'
        self.unit = 'V'
        self.upload = upload
        self.sequence_name = sequence_name
        self.AWG = AWG

    def prepare(self, **kw):
        if self.upload:
            self.AWG.set_setup_filename(self.sequence_name)



class Dynamic_phase(swf.Hard_Sweep):

    def __init__(self, qb_name, CZ_pulse_name,
                 operation_dict,
                 upload=True, cal_points=True):
        '''
        Ramsey type measurement with interleaved fluxpulse
        for more detailed description see fsqs.Ramsey_with_flux_pulse_meas_seq(..)
        sequence function

        inputs:
            qb: qubit object
            X90_separation (float): separation of the pi/2 pulses
            upload (bool): if False, sequences are NOT uploaded onto the AWGs
        '''
        super().__init__()
        # self.thetas = thetas
        self.qb_name = qb_name
        self.CZ_pulse_name = CZ_pulse_name
        self.operation_dict = operation_dict
        self.upload = upload
        self.cal_points = cal_points

        self.name = 'Dynamic_phase'
        self.parameter_name = 'theta'
        self.unit = 'rad'

    def prepare(self, **kw):
        if self.upload:
            fsqs.dynamic_phase_meas_seq(thetas=self.sweep_points,
                                        qb_name=self.qb_name,
                                        CZ_pulse_name=self.CZ_pulse_name,
                                        operation_dict=self.operation_dict,
                                        cal_points=self.cal_points)


class Ramsey_interleaved_fluxpulse_sweep(swf.Hard_Sweep):

    def __init__(self, qb, X90_separation, upload=True):
        '''
        Ramsey type measurement with interleaved fluxpulse
        for more detailed description see fsqs.Ramsey_with_flux_pulse_meas_seq(..)
        sequence function

        inputs:
            qb: qubit object
            X90_separation (float): separation of the pi/2 pulses
            upload (bool): if False, sequences are NOT uploaded onto the AWGs
        '''
        super().__init__()
        self.qb = qb
        self.X90_separation = X90_separation
        self.upload = upload

        self.name = 'Ramsey with interleaved flux pulse'
        self.parameter_name = 'theta'
        self.unit = 'rad'

    def prepare(self, **kw):
        if self.upload:
            fsqs.Ramsey_with_flux_pulse_meas_seq(
                thetas=self.sweep_points, qb=self.qb,
                X90_separation=self.X90_separation
            )


class Ramsey_fluxpulse_ampl_sweep(swf.Soft_Sweep):

    def __init__(self, qb, hard_sweep):
        '''
        flux pulse amplitude sweep for Ramsey type measurement with interleaved
        fluxpulse; for more detailed description see
        the fsqs.Ramsey_with_flux_pulse_meas_seq(..) sequence function

        Args:
            qb: qubit object
            hard_sweep: hard sweep function
        '''
        super().__init__()
        self.name = 'Ramsey interleaved fluxpulse amplitude sweep'
        self.parameter_name = 'Fluxpulse amplitude'
        self.unit = 'V'
        self.hard_sweep = hard_sweep
        self.qb = qb

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        self.qb.flux_pulse_amp(val)
        self.hard_sweep.upload = True
        self.hard_sweep.prepare()

    def finish(self):
        pass

class Ramsey_fluxpulse_delay_sweep(swf.Soft_Sweep):

    def __init__(self, qb, hard_sweep):
        '''
        flux pulse delay sweep for Ramsey type measurement with interleaved
        fluxpulse; for more detailed description see fsqs.Ramsey_with_flux_pulse_\
        meas_seq(..)
        sequence function

        Args:
            qb: qubit object
            hard_sweep: hard_sweep object with method prepare()
        '''
        super().__init__()
        self.name = 'Ramsey interleaved fluxpulse delay sweep'
        self.parameter_name = 'Fluxpulse delay'
        self.unit = 'delay (s)'
        self.hard_sweep = hard_sweep
        self.qb = qb
    def prepare(self):
        pass
    def set_parameter(self, val, **kw):
        self.qb.flux_pulse_delay(val)
        self.hard_sweep.prepare()
    def finish(self):
        pass


class Chevron_length_hard_swf(swf.Hard_Sweep):

    def __init__(self, qb_control, qb_target, spacing=50e-9,
                 cal_points=False, upload=True):
        '''
        Sweep function class for a single slice of the Chevron experiment where
        the length of the fluxpulse is swept (hard sweep).
        For details on the experiment see documentation of
        'fsqs.Chevron_flux_pulse_length_seq(...)'

        Args:
            qb_control (QuDev_Transmon): control qubit (fluxed qubit)
            qb_target (QuDev_Transmon): target qubit (non-fluxed qubit)
            spacing (float): safety spacing between drive pulses and flux pulse
            cal_points (bool): if True, calibration points are measured
            upload (bool): if False, the sequences are NOT uploaded onto the AWGs
        '''
        super().__init__()
        self.qb_control = qb_control
        self.qb_target = qb_target
        self.spacing = spacing
        self.upload = upload
        self.cal_points = cal_points

        self.name = 'Chevron flux pulse length sweep'
        self.parameter_name = 'length'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            fsqs.Chevron_flux_pulse_length_seq(
                lengths=self.sweep_points, qb_control=self.qb_control,
                qb_target=self.qb_target, spacing=self.spacing,
                cal_points=self.cal_points
                )


class Chevron_length_swf_new(swf.Hard_Sweep):

    def __init__(self, hard_sweep_dict, soft_sweep_dict,
                 qbc_name, qbt_name, qbr_name,
                 CZ_pulse_name, operation_dict, readout_qbt=None,
                 verbose=False, cal_points=False,
                 upload=True, return_seq=False):
        '''
        Sweep function class for a single slice of the Chevron experiment where
        the length of the fluxpulse is swept (hard sweep).
        For details on the experiment see documentation of
        'fsqs.Chevron_length_seq(...)'''

        super().__init__()
        self.hard_sweep_dict = hard_sweep_dict
        self.soft_sweep_dict = soft_sweep_dict
        self.qbc_name = qbc_name
        self.qbt_name = qbt_name
        self.qbr_name = qbr_name
        self.readout_qbt = readout_qbt
        self.CZ_pulse_name = CZ_pulse_name
        self.operation_dict = operation_dict
        self.upload = upload
        self.cal_points = cal_points
        self.verbose = verbose
        self.return_seq = return_seq

        self.name = 'Chevron flux pulse length sweep'
        self.parameter_name = 'Fluxpulse length'
        self.unit = 's'

    def prepare(self, upload_all=True, **kw):
        if self.upload:
            fsqs.Chevron_length_seq_new(
                hard_sweep_dict=self.hard_sweep_dict,
                soft_sweep_dict=self.soft_sweep_dict,
                qbc_name=self.qbc_name,
                qbt_name=self.qbt_name,
                qbr_name=self.qbr_name,
                upload_all=upload_all,
                CZ_pulse_name=self.CZ_pulse_name,
                operation_dict=self.operation_dict,
                verbose=self.verbose, cal_points=self.cal_points,
                upload=self.upload, return_seq=self.return_seq)


class Chevron_frequency_hard_swf(swf.Hard_Sweep):

    def __init__(self, frequencies, length, flux_pulse_amp,
                 qbc_name, qbt_name, qbr_name,
                 CZ_pulse_name, operation_dict, readout_qbt=None,
                 verbose=False, cal_points=False,
                 upload=True, upload_all=True, return_seq=False):

        super().__init__()
        self.length = length
        self.flux_pulse_amp = flux_pulse_amp
        self.frequencies = frequencies
        self.qbc_name = qbc_name
        self.qbt_name = qbt_name
        self.qbr_name = qbr_name
        self.readout_qbt = readout_qbt
        self.CZ_pulse_name = CZ_pulse_name
        self.operation_dict = operation_dict
        self.upload = upload
        self.upload_all = upload_all
        self.cal_points = cal_points
        self.verbose = verbose
        self.return_seq = return_seq

        self.name = 'Chevron flux pulse frequency sweep'
        self.parameter_name = 'Fluxpulse frequency'
        self.unit = 'Hz'

    def prepare(self, upload_all=None, **kw):
        if upload_all is None:
            upload_all = self.upload_all
        if self.upload:
            fsqs.Chevron_frequency_seq(
                frequencies=self.frequencies,
                length=self.length,
                flux_pulse_amp=self.flux_pulse_amp,
                qbc_name=self.qbc_name,
                qbt_name=self.qbt_name,
                qbr_name=self.qbr_name,
                upload_all=upload_all,
                CZ_pulse_name=self.CZ_pulse_name,
                operation_dict=self.operation_dict,
                verbose=self.verbose, cal_points=self.cal_points,
                upload=self.upload, return_seq=self.return_seq)


class Chevron_ampl_swf_new(swf.Soft_Sweep):

    def __init__(self, hard_sweep, parameter_name='', unit=''):
        '''
        Sweep function class (soft sweep) for 2D Chevron experiment where
        the amplitude of the fluxpulse is swept. Used in combination with
        the Chevron_length_hard_swf class.

        Args:
           qb_control (QuDev_Transmon): control qubit (fluxed qubit)
           qb_target (QuDev_Transmon): target qubit (non-fluxed qubit)
           hard_sweep: hard sweep function (fast axis sweep function)
        '''
        super().__init__()
        self.name = 'Chevron flux pulse amplitude sweep'
        self.parameter_name = parameter_name
        self.unit = unit
        self.hard_sweep = hard_sweep
        self.is_first_sweeppoint = True

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        first_key = list(self.hard_sweep.soft_sweep_dict)[0]
        # TODO: do this properly!
        self.hard_sweep.soft_sweep_dict[first_key] = val
        self.hard_sweep.upload = True
        self.hard_sweep.prepare(upload_all=self.is_first_sweeppoint)
        self.is_first_sweeppoint = False

    def finish(self):
        pass


class Chevron_general_soft_swf(swf.Soft_Sweep):

    def __init__(self, hard_sweep, parameter_name='', unit=''):
        super().__init__()
        self.name = 'Chevron flux pulse {} sweep'.format(parameter_name)
        self.parameter_name = parameter_name
        self.unit = unit
        self.hard_sweep = hard_sweep
        self.is_first_sweeppoint = True

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        first_key = list(self.hard_sweep.soft_sweep_dict)[0]
        # TODO: do this properly!
        self.hard_sweep.soft_sweep_dict[first_key] = val
        self.hard_sweep.upload = True
        self.hard_sweep.prepare(upload_all=self.is_first_sweeppoint)
        self.is_first_sweeppoint = False

    def finish(self):
        pass

class Chevron_freq_swf_new(swf.Soft_Sweep):

    def __init__(self, hard_sweep):
        '''
        Text to be updated!!!!!

        Args:
        '''
        super().__init__()
        self.name = 'Chevron flux pulse frequency sweep'
        self.parameter_name = 'Fluxpulse frequency'
        self.unit = 'Hz'
        self.hard_sweep = hard_sweep
        self.is_first_sweeppoint = True

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        self.hard_sweep.frequency = val
        self.hard_sweep.upload = True
        self.hard_sweep.prepare(upload_all=self.is_first_sweeppoint)
        self.is_first_sweeppoint = False

    def finish(self):
        pass

class Chevron_alpha_swf_new(swf.Soft_Sweep):

    def __init__(self, hard_sweep):
        '''
        Text to be updated!!!!!

        Args:
        '''
        super().__init__()
        self.name = 'Chevron NZ flux pulse alpha sweep'
        self.parameter_name = 'Fluxpulse alpha'
        self.unit = ''
        self.hard_sweep = hard_sweep
        self.is_first_sweeppoint = True

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        self.hard_sweep.alpha = val
        self.hard_sweep.upload = True
        self.hard_sweep.prepare(upload_all=self.is_first_sweeppoint)
        self.is_first_sweeppoint = False

    def finish(self):
        pass

class Chevron_ampl_hard_swf(swf.Hard_Sweep):

    def __init__(self, qb_control, qb_target, spacing=50e-9,
                 cal_points=False, upload=True):
        '''
        Sweep function class for a single slice of the Chevron experiment where
        the amplitude of the fluxpulse is swept (hard sweep).
        For details on the experiment see documentation of
        'fsqs.Chevron_flux_pulse_ampl_seq(...)'

        Args:
            qb_control (QuDev_Transmon): control qubit (fluxed qubit)
            qb_target (QuDev_Transmon): target qubit (non-fluxed qubit)
            spacing (float): safety spacing between drive pulses and flux pulse
            cal_points (bool): if True, calibration points are measured
            upload (bool): if False, the sequences are NOT uploaded onto the AWGs
        '''
        super().__init__()
        self.qb_control = qb_control
        self.qb_target = qb_target
        self.spacing = spacing
        self.upload = upload
        self.cal_points = cal_points

        self.name = 'Chevron flux pulse amplitude sweep'
        self.parameter_name = 'amplitude'
        self.unit = 'V'

    def prepare(self, **kw):
        if self.upload:
            fsqs.Chevron_flux_pulse_ampl_seq(
                ampls=self.sweep_points, qb_control=self.qb_control,
                qb_target=self.qb_target, spacing=self.spacing,
                cal_points=self.cal_points,
            )


class Chevron_ampl_swf(swf.Soft_Sweep):

    def __init__(self, qb_control, qb_target, hard_sweep):
        '''
        Sweep function class (soft sweep) for 2D Chevron experiment where
        the amplitude of the fluxpulse is swept. Used in combination with
        the Chevron_length_hard_swf class.

        Args:
           qb_control (QuDev_Transmon): control qubit (fluxed qubit)
           qb_target (QuDev_Transmon): target qubit (non-fluxed qubit)
           hard_sweep: hard sweep function (fast axis sweep function)
        '''
        super().__init__()
        self.name = 'Chevron flux pulse amplitude sweep'
        self.parameter_name = 'Fluxpulse amplitude'
        self.unit = 'V'
        self.hard_sweep = hard_sweep
        self.qb_control = qb_control
        self.qb_target = qb_target

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        self.qb_control.flux_pulse_amp(val)
        self.hard_sweep.upload = True
        self.hard_sweep.prepare()

    def finish(self):
        pass


class Chevron_freq_swf(swf.Soft_Sweep):

    def __init__(self, qb_control, qb_target, hard_sweep):
        '''
        Sweep function class (soft sweep) for 2D Chevron experiment where
        the amplitude of the fluxpulse is swept. Used in combination with
        the Chevron_length_hard_swf class.

        Args:
           qb_control (QuDev_Transmon): control qubit (fluxed qubit)
           qb_target (QuDev_Transmon): target qubit (non-fluxed qubit)
           hard_sweep: hard sweep function (fast axis sweep function)
        '''
        super().__init__()
        self.name = 'Chevron flux pulse frequency sweep'
        self.parameter_name = 'Fluxpulse frequency'
        self.unit = 'Hz'
        self.hard_sweep = hard_sweep
        self.qb_control = qb_control
        self.qb_target = qb_target

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        self.qb_control.flux_pulse_frequency(val)
        self.hard_sweep.upload = True
        self.hard_sweep.prepare()

    def finish(self):
        pass


class Chevron_ampl_fast_swf(swf.Soft_Sweep):

    def __init__(self, qb_control, qb_target, AWG, pulsar, channel=None):
        '''
        Sweep function class (soft sweep) for 2D Chevron experiment where
        the amplitude of the fluxpulse is swept. This 'fast' version uses
        a trick, that instead of uploading new sequences with different amplitudes,
        the ouput scaling of the AWG is changed.
        Used in combination with the Chevron_length_hard_swf class.

        Args:
            qb_control (qubit): control qubit (qubit with flux pulse)
            qb_target (qubit): target qubit
            AWG: AWG used for the flux pulses
            channel (str): (optional), flux pulse channel of the AWG (e.g. 'ch1')
        '''
        super().__init__()
        if channel is None:
            channel = qb_control.flux_pulse_channel()[-3:]
        self.channel = channel
        self.AWG = AWG
        self.name = 'Chevron flux pulse amplitude sweep'
        self.parameter_name = 'Fluxpulse amplitude'
        self.unit = 'V'
        self.qb_control = qb_control
        self.qb_target = qb_target
        self.flux_pulse_ampl_backup = self.qb_control.flux_pulse_amp()
        self.sign_pulse = 1.
        self.pulsar = pulsar

    def prepare(self):

        if np.sign(np.min(self.sweep_points)) != \
                np.sign(np.max(self.sweep_points)):
            logging.warning('Please do not use flux pulse'
                            'amplitudes with different signs!')
        else:
            self.sign_pulse = np.sign(self.sweep_points[0])

        if type(self.channel) == int:
            self.qb_control.flux_pulse_amp(0.25*self.sign_pulse*self.AWG.get(
                'ch{}_amp'.format(self.channel))) # used for correct amplitude
        else:
            self.qb_control.flux_pulse_amp(0.25*self.sign_pulse*self.AWG.get(
                '{}_amp'.format(self.channel))) # used for correct amplitude

    def set_parameter(self, val, **kw):

        self.pulsar.stop()
        if type(self.channel) == int:
            exec('self.AWG.ch{}_amp({})'.format(self.channel, 4*val*self.sign_pulse))
            # factor of 4 for correct amplitude
        else:
            exec('self.AWG.{}_amp({})'.format(self.channel, 4*val*self.sign_pulse))
            # factor of 4 for correct amplitude
        #self.pulsar.start()

    def finish(self):
        self.qb_control.flux_pulse_amp(self.flux_pulse_ampl_backup)


class Flux_pulse_CPhase_meas_hard_swf(swf.Hard_Sweep):

    def __init__(self, qb_control, qb_target, sweep_mode='length', X90_phase=0,
                 spacing=50e-9, cal_points=False, upload=True,
                 measurement_mode='excited_state',
                 reference_measurements=False,
                 upload_AWGs='all',
                 upload_channels='all'):
        '''
            Flexible sweep function class for a single slice of the CPhase
            experiment (hard sweep) that can either sweep the amplitude or
            length of the flux pulse or the phase of the second X90 pulse.
            For details on the experiment see documentation of
            'fsqs.flux_pulse_CPhase_seq(...)'

        Args:
            qb_control: instance of the qubit class (control qubit)
            qb_target: instance of the qubit class (target qubit)
            sweep_mode: string, either 'length', 'amplitude' or 'amplitude'
            X90_phase: float, phase of the second X90 pulse in rad
            spacing: float, spacing between first and second X90 pulse
            measurement_mode (str): either 'excited_state', 'ground_state'
            reference_measurement (bool): if True, appends a reference measurement
                                          IMPORTANT: you need to double
                                          the hard sweep points!
                                          e.g. thetas = np.concatenate((thetas,thetas))
        '''
        super().__init__()
        self.qb_control = qb_control
        self.qb_target = qb_target
        self.spacing = spacing
        self.X90_phase = X90_phase
        self.upload = upload
        self.cal_points = cal_points
        self.sweep_mode = sweep_mode
        self.measurement_mode = measurement_mode
        self.reference_measurements = reference_measurements
        self.upload_AWGs = upload_AWGs
        self.upload_channels = upload_channels

        self.name = 'flux_pulse_CPhase_measurement_{}_sweep'.format(sweep_mode)
        self.parameter_name = sweep_mode
        if sweep_mode == 'length':
            self.unit = 's'
        elif sweep_mode == 'amplitude':
            self.unit = 'V'
        elif sweep_mode == 'phase':
            self.unit = 'rad'


    def prepare(self, X90_phase=None, **kw):

        if X90_phase is not None:
            self.X90_phase = X90_phase
        if self.upload:
            fsqs.flux_pulse_CPhase_seq(
                sweep_points=self.sweep_points, qb_control=self.qb_control,
                qb_target=self.qb_target,
                sweep_mode=self.sweep_mode,
                X90_phase=self.X90_phase, spacing=self.spacing,
                cal_points=self.cal_points,
                measurement_mode=self.measurement_mode,
                reference_measurements=self.reference_measurements,
                upload_AWGs=self.upload_AWGs,
                upload_channels=self.upload_channels
            )

class Flux_pulse_CPhase_hard_swf_new(swf.Hard_Sweep):

    def __init__(self, phases, qbc_name, qbt_name, qbr_name, CZ_pulse_name,
                 CZ_pulse_channel, operation_dict,
                 max_flux_length, cal_points=False,
                 upload=True, reference_measurements=False):
        '''
            Flexible sweep function class for a single slice of the CPhase
            experiment (hard sweep) that can either sweep the amplitude or
            length of the flux pulse or the phase of the second X90 pulse.
            For details on the experiment see documentation of
            'fsqs.flux_pulse_CPhase_seq(...)'

        Args:
            qb_control: instance of the qubit class (control qubit)
            qb_target: instance of the qubit class (target qubit)
            sweep_mode: string, either 'length', 'amplitude' or 'amplitude'
            X90_phase: float, phase of the second X90 pulse in rad
            spacing: float, spacing between first and second X90 pulse
            measurement_mode (str): either 'excited_state', 'ground_state'
            reference_measurement (bool): if True, appends a reference measurement
                                          IMPORTANT: you need to double
                                          the hard sweep points!
                                          e.g. thetas = np.concatenate((thetas,thetas))
        '''
        super().__init__()
        self.phases = phases
        self.qbc_name = qbc_name
        self.qbt_name = qbt_name
        self.qbr_name = qbr_name
        self.operation_dict = operation_dict
        self.CZ_pulse_name = CZ_pulse_name
        self.CZ_pulse_channel = CZ_pulse_channel
        self.upload = upload
        self.cal_points = cal_points
        self.reference_measurements = reference_measurements
        self.name = 'flux_pulse_CPhase_measurement_phase_sweep'
        self.parameter_name = 'phase'
        self.unit = 'rad'
        self.max_flux_length = max_flux_length
        self.flux_length = None
        self.flux_amplitude = None
        self.frequency = None
        self.values_complete = False
        self.first_data_point = True

    def prepare(self, flux_params=None, **kw):
        if flux_params is None:
            return

        if self.upload:
            print('Uploaded CPhase Sequence')
            fsqs.flux_pulse_CPhase_seq_new(
                phases=self.phases,
                flux_params=flux_params,
                max_flux_length = self.max_flux_length,
                qbc_name=self.qbc_name,
                qbt_name=self.qbt_name,
                qbr_name=self.qbr_name,
                operation_dict=self.operation_dict,
                CZ_pulse_name=self.CZ_pulse_name,
                CZ_pulse_channel=self.CZ_pulse_channel,
                cal_points=self.cal_points,
                reference_measurements=self.reference_measurements,
                upload=self.upload,
                return_seq=True,
                first_data_point=self.first_data_point
                )
            self.first_data_point = False

    def set_parameter(self, flux_val, **kw):
        val_type = kw.pop('val_type', None)
        if val_type is None:
            logging.warning('CPhase hard sweep set_parameter method was called '
                          'without a value type!')
            return
        elif val_type == 'length':
            self.flux_length = flux_val
        elif val_type == 'amplitude':
            self.flux_amplitude = flux_val
        elif val_type == 'frequency':
            self.frequency = flux_val
        else:
            logging.error('CPhase hard sweep does not recognize value type '
                          'handed by set_parameter() method!')
        if self.flux_length is not None and self.flux_amplitude is not None:
            self.prepare(flux_params=[self.flux_length, self.flux_amplitude])
            self.flux_length = None
            self.flux_amplitude = None


class CPhase_NZ_hard_swf(swf.Hard_Sweep):

    def __init__(self, phases, qbc_name, qbt_name, CZ_pulse_name,
                 CZ_pulse_channel, operation_dict, num_soft_sweepparams,
                 max_flux_length, num_cz_gates=1,
                 cal_points=False, first_data_point=True,
                 num_cal_points=4, upload=True, reference_measurements=False):

        super().__init__()
        self.phases = phases
        self.qbc_name = qbc_name
        self.qbt_name = qbt_name
        self.operation_dict = operation_dict
        self.num_soft_sweepparams = num_soft_sweepparams
        self.CZ_pulse_name = CZ_pulse_name
        self.CZ_pulse_channel = CZ_pulse_channel
        self.upload = upload
        self.cal_points = cal_points
        self.num_cal_points = num_cal_points
        self.reference_measurements = reference_measurements
        self.name = 'flux_pulse_CPhase_measurement_phase_sweep'
        self.parameter_name = 'phase'
        self.unit = 'rad'
        self.max_flux_length = max_flux_length
        self.num_cz_gates = num_cz_gates
        self.flux_params_dict = {}
        self.values_complete = False
        self.first_data_point = first_data_point

    def prepare(self, flux_params_dict={}, **kw):
        print('flux params dict in hard swf ',
              flux_params_dict)
        if flux_params_dict == {}:
            return
        print(self.upload)
        if self.upload:
            print('Uploading CPhase Sequence...')
            fsqs.cphase_nz_seq(
                phases=self.phases,
                flux_params_dict=flux_params_dict,
                max_flux_length=self.max_flux_length,
                num_cz_gates=self.num_cz_gates,
                qbc_name=self.qbc_name,
                qbt_name=self.qbt_name,
                operation_dict=self.operation_dict,
                CZ_pulse_name=self.CZ_pulse_name,
                CZ_pulse_channel=self.CZ_pulse_channel,
                cal_points=self.cal_points,
                num_cal_points=self.num_cal_points,
                upload=self.upload,
                return_seq=True,
                first_data_point=self.first_data_point
            )
            self.first_data_point = False

    def set_parameter(self, flux_val, **kw):
        val_type = kw.pop('val_type', None)
        print('val_type hard swf ', val_type)
        if val_type is None:
            logging.warning('CPhase hard sweep set_parameter method was called '
                            'without a value type!')
            return
        else:
            self.flux_params_dict.update({val_type: flux_val})

        if len(self.flux_params_dict) == self.num_soft_sweepparams:
            self.prepare(flux_params_dict=self.flux_params_dict)
            self.flux_params_dict = {}


class Flux_pulse_CPhase_hard_swf_frequency(swf.Hard_Sweep):

    def __init__(self, phases, qbc_name, qbt_name,qbr_name,CZ_pulse_name,
                 flux_length, amplitude,
                 CZ_pulse_channel,operation_dict,
                 max_flux_length,cal_points=False,
                 upload=True,reference_measurements=False):
        '''
            Update!!!

        Args:
            qb_control: instance of the qubit class (control qubit)
            qb_target: instance of the qubit class (target qubit)
            sweep_mode: string, either 'length', 'amplitude' or 'amplitude'
            X90_phase: float, phase of the second X90 pulse in rad
            spacing: float, spacing between first and second X90 pulse
            measurement_mode (str): either 'excited_state', 'ground_state'
            reference_measurement (bool): if True, appends a reference measurement
                                          IMPORTANT: you need to double
                                          the hard sweep points!
                                          e.g. thetas = np.concatenate((thetas,thetas))
        '''
        super().__init__()
        self.phases = phases
        self.qbc_name = qbc_name
        self.qbt_name = qbt_name
        self.qbr_name = qbr_name
        self.operation_dict = operation_dict
        self.CZ_pulse_name = CZ_pulse_name
        self.CZ_pulse_channel = CZ_pulse_channel
        self.upload = upload
        self.cal_points = cal_points
        self.reference_measurements = reference_measurements
        self.name = 'flux_pulse_CPhase_measurement_phase_sweep'
        self.parameter_name ='phase'
        self.unit = 'rad'
        self.max_flux_length = max_flux_length
        self.flux_length=flux_length
        self.flux_amplitude=amplitude
        self.frequency=None
        self.values_complete = False
        self.first_data_point = True

    def prepare(self,flux_params=None,**kw):
        if flux_params is None:
            return

        if self.upload:
            print('Uploaded CPhase Sequence')
            fsqs.flux_pulse_CPhase_seq_new(
                phases=self.phases,
                flux_params=flux_params,
                max_flux_length = self.max_flux_length,
                qbc_name=self.qbc_name,
                qbt_name=self.qbt_name,
                qbr_name=self.qbr_name,
                operation_dict=self.operation_dict,
                CZ_pulse_name=self.CZ_pulse_name,
                CZ_pulse_channel=self.CZ_pulse_channel,
                cal_points=self.cal_points,
                reference_measurements=self.reference_measurements,
                upload=self.upload,
                return_seq= True,
                first_data_point = self.first_data_point
            )
            self.first_data_point = False

    def set_parameter(self,flux_val,**kw):
        val_type = kw.pop('val_type',None)
        if val_type is None:
            logging.warning('CPhase hard sweep set_parameter method was called '
                            'without a value type!')
            return
        elif val_type == 'length':
            self.flux_length = flux_val
        elif val_type == 'amplitude':
            self.flux_amplitude = flux_val
        elif val_type == 'frequency':
            self.frequency = flux_val
        else:
            logging.error('CPhase hard sweep does not recognize value type handed'
                          'by set_parameter() method!')
        if self.frequency is not None:
            self.prepare(flux_params=[self.flux_length,
                                      self.flux_amplitude,
                                      self.frequency])
            self.frequency = None

class Flux_pulse_CPhase_soft_swf(swf.Soft_Sweep):

    def __init__(self, hard_sweep, sweep_param, unit='', upload=True):
        '''
            Flexible soft sweep function class for 2D CPhase
            experiments that can either sweep the amplitude or
            length of the flux pulse or the phase of the second X90 pulse.
            For details on the experiment see documentation of
            'fsqs.flux_pulse_CPhase_seq(...)'

        Args:
            hard_sweep: 1D hard sweep
        '''
        super().__init__()
        self.sweep_param = sweep_param
        self.name = 'flux_pulse_CPhase_measurement_{}_2D_sweep'.format(
            sweep_param)
        self.unit = unit
        self.parameter_name = sweep_param
        self.hard_sweep = hard_sweep
        self.upload = upload

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        print('sweep_param set_param in soft swf ', self.sweep_param)
        self.hard_sweep.upload = self.upload
        self.hard_sweep.set_parameter(flux_val=val, val_type=self.sweep_param)

    def finish(self):
        pass


class Flux_pulse_CPhase_meas_2D(swf.Soft_Sweep):

    def __init__(self, qb_control, qb_target, hard_sweep,
                 sweep_mode='amplitude', upload=True):
        '''
            Flexible soft sweep function class for 2D CPhase
            experiments that can either sweep the amplitude or
            length of the flux pulse or the phase of the second X90 pulse.
            For details on the experiment see documentation of
            'fsqs.flux_pulse_CPhase_seq(...)'

        Args:
            qb_control: instance of the qubit class (control qubit)
            qb_target: instance of the qubit class (target qubit)
            hard_sweep: 1D hard sweep
            sweep_mode: string, either 'length', 'amplitude' or 'amplitude'
        '''
        super().__init__()
        self.name = 'flux_pulse_CPhase_measurement_{}_2D_sweep'.format(sweep_mode)
        self.parameter_name = sweep_mode
        self.sweep_mode = sweep_mode
        if self.sweep_mode == 'length':
            self.unit = 's'

        elif self.sweep_mode == 'amplitude':
            self.unit = 'V'
        elif self.sweep_mode == 'phase':
            self.unit = 'rad'
        self.hard_sweep = hard_sweep
        self.qb_control = qb_control
        self.qb_target = qb_target
        self.upload = upload

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):

        self.hard_sweep.upload = self.upload

        if self.sweep_mode == 'length':
            self.qb_control.flux_pulse_length(val)
            self.hard_sweep.prepare()
        elif self.sweep_mode == 'amplitude':
            self.qb_control.flux_pulse_amp(val)
            self.hard_sweep.prepare()
        elif self.sweep_mode == 'phase':
            self.hard_sweep.prepare(X90_phase=val)

    def finish(self):
        pass


class Fluxpulse_scope_swf(swf.Hard_Sweep):

    def __init__(self, qb, cal_points=False, upload=True,
                 spacing=30e-9
                 ):
        '''
        Flux pulse scope hard sweep (sweeps the relative delay of drive pulse and
        flux pulse)

        Args:
            qb (QuDev_Transmon): qubit
            cal_points (bool): if True, the measurement is done with calibration points
            upload (bool): if False, the sequences are NOT uploaded to the AWGs
            spacing (float): safety spacing between flux pulse and RO pulse
        '''
        super().__init__()
        self.qb = qb
        self.upload = upload
        self.cal_points = cal_points
        self.spacing=spacing

        self.name = 'Fluxpulse_scope_{}'.format(self.qb.name)
        self.parameter_name = 'delay'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            fsqs.fluxpulse_scope_sequence(delays=self.sweep_points, qb=self.qb,
                                          cal_points=self.cal_points,
                                          spacing=self.spacing)


class Fluxpulse_scope_drive_freq_sweep(swf.Soft_Sweep):

    def __init__(self, qb):
        '''
        soft sweep function that sweeps the frequency of the drive LO
        Args:
            qb (QuDev_Transmon): qubit
        '''
        super().__init__()
        self.name = 'Fluxpulse scope drive freq sweep'
        self.parameter_name = 'qubit drive freq.'
        self.unit = 'Hz'
        self.qb = qb
        self.stored_f_qubit = self.qb.f_qubit()
        self.f_mod = self.qb.f_pulse_mod()

    def prepare(self):
        pass

    def set_parameter(self, val, **kw):
        self.qb.cw_source.frequency(val - self.f_mod)

    def finish(self):
        self.qb.f_qubit(self.stored_f_qubit)


class Fluxpulse_scope_nzcz_alpha_hard_swf(swf.Hard_Sweep):

    def __init__(self, qb_name, nzcz_alpha,
                 CZ_pulse_name, operation_dict,
                 cal_points=False, upload=True,
                 upload_all=True, spacing=30e-9
                 ):
        '''
        Flux pulse scope hard sweep (sweeps the relative delay of drive pulse and
        flux pulse)

        Args:
            qb (QuDev_Transmon): qubit
            cal_points (bool): if True, the measurement is done with calibration points
            upload (bool): if False, the sequences are NOT uploaded to the AWGs
            spacing (float): safety spacing between flux pulse and RO pulse
        '''
        super().__init__()
        self.qb_name = qb_name
        self.nzcz_alpha = nzcz_alpha
        self.CZ_pulse_name = CZ_pulse_name
        self.operation_dict = operation_dict
        self.upload = upload
        self.upload_all = upload_all
        self.cal_points = cal_points
        self.spacing=spacing

        self.name = 'Fluxpulse_scope_nzcz_alpha_{}'.format(qb_name)
        self.parameter_name = 'delay'
        self.unit = 's'

    def prepare(self, upload_all=None, **kw):
        if upload_all is None:
            upload_all = self.upload_all
        if self.upload:
            fsqs.fluxpulse_scope_alpha_sequence(
                delays=self.sweep_points, qb_name=self.qb_name,
                nzcz_alpha=self.nzcz_alpha,
                CZ_pulse_name=self.CZ_pulse_name,
                operation_dict=self.operation_dict,
                upload_all=upload_all,
                cal_points=self.cal_points,
                spacing=self.spacing)


class Fluxpulse_scope_nzcz_alpha_soft_sweep(swf.Soft_Sweep):

    def __init__(self, hard_sweep, upload=True):
        super().__init__()
        self.name = 'Fluxpulse scope nzCZ alpha sweep'
        self.parameter_name = 'nzCZ alpha'
        self.unit = ''
        self.hard_sweep = hard_sweep
        self.upload = upload
        self.is_first_sweeppoint = True

    def set_parameter(self, val):
        self.hard_sweep.nzcz_alpha = val
        self.hard_sweep.upload = self.upload
        self.hard_sweep.prepare(upload_all=self.is_first_sweeppoint)
        self.is_first_sweeppoint = False


class Readout_pulse_scope_swf(swf.Hard_Sweep):
    def __init__(self, delays, pulse_pars, RO_pars, RO_separation,
                 cal_points=((-4, -3), (-2, -1)), prep_pulses=None,
                 comm_freq=225e6, verbose=False, upload=True):
        super().__init__()
        self.delays = delays
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.RO_separation = RO_separation
        self.cal_points = cal_points
        self.prep_pulses = prep_pulses
        self.comm_freq = comm_freq
        self.verbose = verbose
        self.upload = upload

        self.name = 'Readout pulse scope'
        self.parameter_name = 'delay'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            csqs.readout_pulse_scope_seq(
                delays=self.delays,
                pulse_pars=self.pulse_pars,
                RO_pars=self.RO_pars,
                RO_separation=self.RO_separation,
                cal_points=self.cal_points,
                prep_pulses=self.prep_pulses,
                comm_freq=self.comm_freq,
                verbose=self.verbose)

class readout_photons_in_resonator_swf(swf.Hard_Sweep):
    def __init__(self, delay_to_relax, delay_buffer, ramsey_times,
                 RO_pars, pulse_pars, cal_points=((-4, -3), (-2, -1)),
                 verbose=False, upload=True, return_seq=False,
                 artificial_detuning=None):
        super().__init__()
        self.delay_to_relax = delay_to_relax
        self.delay_buffer = delay_buffer
        self.ramsey_times = ramsey_times
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning
        self.verbose = verbose
        self.upload = upload
        self.return_seq = return_seq

        self.name = 'read out photons in RO'
        self.parameter_name = 'ramsey_times'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            csqs.readout_photons_in_resonator_seq(
                delay_to_relax=self.delay_to_relax,
                pulse_pars=self.pulse_pars,
                RO_pars=self.RO_pars,
                delay_buffer=self.delay_buffer,
                cal_points=self.cal_points,
                ramsey_times=self.ramsey_times,
                return_seq=self.return_seq,
                artificial_detuning=self.artificial_detuning,
                verbose=self.verbose)

class readout_photons_in_resonator_soft_swf(swf.Soft_Sweep):

    def __init__(self, hard_sweep):

        super().__init__()
        self.hard_sweep = hard_sweep
        self.name = 'read out photons in RO'
        self.parameter_name = 'delay_to_relax'
        self.unit = 's'

    def set_parameter(self, val):
        self.hard_sweep.upload = True 
        self.hard_sweep.delay_to_relax = val
        self.hard_sweep.prepare()


class Custom_single_qubit_swf(swf.Hard_Sweep):

    def __init__(self, seq_func, pulse_pars, RO_pars, upload=True):
        super().__init__()
        self.seq_func = seq_func
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload

        self.name = 'Custom_swf'
        self.parameter_name = 'seg_num'
        self.unit = '#'

    def prepare(self, **kw):
        if self.upload:
            sqs.custom_seq(seq_func=self.seq_func,
                           sweep_points=self.sweep_points,
                           pulse_pars=self.pulse_pars,
                           RO_pars=self.RO_pars, upload=True)

class Ramsey_decoupling_swf(swf.Hard_Sweep):

    def __init__(self, seq_func, pulse_pars, RO_pars,
                 artificial_detuning=None, nr_echo_pulses=4,
                 cpmg_scheme=True,
                 upload=True, cal_points=True):
        super().__init__()
        self.seq_func = seq_func
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning
        self.nr_echo_pulses = nr_echo_pulses
        self.cpmg_scheme = cpmg_scheme

        self.name = 'Ramsey_decoupling'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            self.seq_func(times=self.sweep_points,
                          pulse_pars=self.pulse_pars,
                          RO_pars=self.RO_pars,
                          artificial_detuning=self.artificial_detuning,
                          cal_points=self.cal_points,
                          nr_echo_pulses=self.nr_echo_pulses,
                          cpmg_scheme=self.cpmg_scheme)


class CZ_bleed_through_phase_hard_sweep(swf.Hard_Sweep):

    def __init__(self, qb_name, CZ_pulse_name, CZ_separation, operation_dict,
                 oneCZ_msmt=False, verbose=False, upload=True, nr_cz_gates=1,
                 return_seq=False, cal_points=True):

        super().__init__()
        self.qb_name = qb_name
        self.CZ_pulse_name = CZ_pulse_name
        self.CZ_separation = CZ_separation
        self.oneCZ_msmt = oneCZ_msmt
        self.nr_cz_gates = nr_cz_gates
        self.operation_dict = operation_dict
        self.verbose = verbose
        self.upload = upload
        self.return_seq = return_seq
        self.cal_points = cal_points

        self.name = 'CZ_bleed_through_phase'
        self.parameter_name = 'theta'
        self.unit = 'rad'

    def prepare(self, upload_all=True, **kw):
        if self.upload:
            fsqs.CZ_bleed_through_phase_seq(
                phases=self.sweep_points,
                qb_name=self.qb_name,
                CZ_pulse_name=self.CZ_pulse_name,
                CZ_separation=self.CZ_separation,
                oneCZ_msmt=self.oneCZ_msmt,
                nr_cz_gates=self.nr_cz_gates,
                operation_dict=self.operation_dict,
                verbose=self.verbose,
                upload=self.upload,
                return_seq=self.return_seq,
                upload_all=upload_all,
                cal_points=self.cal_points)


class CZ_bleed_through_separation_time_soft_sweep(swf.Soft_Sweep):

    def __init__(self, hard_sweep, upload=True):

        super().__init__()
        self.hard_sweep = hard_sweep
        self.upload = upload
        self.is_first_sweeppoint = True
        self.name = 'CZ_bleed_through_separation_time'
        self.parameter_name = 't_sep'
        self.unit = 's'

    def set_parameter(self, val):
        self.hard_sweep.CZ_separation = val
        self.hard_sweep.upload = self.upload
        # self.hard_sweep.prepare(upload_all=self.is_first_sweeppoint)
        self.hard_sweep.prepare(upload_all=True)
        self.is_first_sweeppoint = False


class CZ_bleed_through_nr_cz_gates_soft_sweep(swf.Soft_Sweep):

    def __init__(self, hard_sweep, upload=True):

        super().__init__()
        self.hard_sweep = hard_sweep
        self.upload = upload
        self.is_first_sweeppoint = True
        self.name = 'CZ_bleed_through_nr_cz_gates'
        self.parameter_name = 'Nr. CZ gates'
        self.unit = '#'

    def set_parameter(self, val):
        self.hard_sweep.nr_cz_gates = val
        self.hard_sweep.upload = self.upload
        # self.hard_sweep.prepare(upload_all=self.is_first_sweeppoint)
        self.hard_sweep.prepare(upload_all=True)
        self.is_first_sweeppoint = False