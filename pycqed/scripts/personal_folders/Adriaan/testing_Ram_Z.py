# Example of how to view a tektronix sequence
import qcodes as qc
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs

fsqs.station = qc.station


reload_mod_stuff()

seq, elts = fsqs.Ram_Z_seq(AncT.get_operation_dict(), 'AncT',
               dist_dict, t_dict, upload=False, return_seq=True)


ax = seq_viewer.show_element_dclab(element=elts[0])
ax = seq_viewer.show_element_dclab(element=elts[4], ax=ax)
ax = seq_viewer.show_element_dclab(element=elts[10], ax=ax)


ax.set_xlim(0, 2600)
ax.set_ylim(-3, 3)






reload_mod_stuff()
#AllXY on Data top
AncT.RO_acq_averages(1024)
int_avg_det = det.UHFQC_integrated_average_detector(
            UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
            channels=[DataT.RO_acq_weight_function_I(),
                      AncT.RO_acq_weight_function_I()],
            nr_averages=AncT.RO_acq_averages(),
            integration_length=AncT.RO_acq_integration_length(),
            cross_talk_suppression=True)

pulse_dict = AncT.get_operation_dict()

ram_Z_sweep = awg_swf.awg_seq_swf(fsqs.Ram_Z_seq,
    awg_seq_func_kwargs={'pulse_dict':pulse_dict, 'q0':'AncT',
                         'inter_pulse_delay':100e-9,
                         'distortion_dict': dist_dict},
                         parameter_name='times')


MC.set_sweep_function(ram_Z_sweep)
MC.set_sweep_points(np.arange(-100e-9, 200e-9, 5e-9))
MC.set_detector_function(int_avg_det)
MC.run('Ram_Z_AncT')
ma.MeasurementAnalysis()