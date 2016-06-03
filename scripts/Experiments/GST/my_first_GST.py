import qcodes as qc
from modules.measurement import sweep_functions as swf
from modules.measurement import detector_functions as det
import modules.measurement.pulse_sequences.gate_set_tomography as gsts
gsts.station = station


station = station
MC = MC
CBox = station.components['CBox']

# calibrate_RO_threshold_no_rotation()

# l = 32
# nr_elts = 1185 # for 32
l = 2048
nr_elts = 3889
nr_logs = 10000

reps_per_log = int(8000/nr_elts)
log_length = (nr_elts*reps_per_log)
nr_shots_per_point = reps_per_log*nr_logs

set_trigger_slow()

fn = (PyCQEDpath+'\modules\measurement\pulse_sequences\_pygsti_Gatesequences\Gatesequences_maxlength={}.txt'.format(l))
seq, elts = gsts.GST_from_textfile(pulse_pars=pulse_pars_duplex, RO_pars=RO_pars, filename=fn)
print('starting?')
MC.live_plot_enabled = False
CBox.log_length(log_length)
MC.set_sweep_function(swf.None_Sweep(sweep_control='hard') )
MC.set_sweep_points(np.arange(log_length))
MC.set_sweep_function_2D(swf.None_Sweep(sweep_control='soft'))
MC.set_sweep_points_2D(np.arange(nr_logs))
MC.set_detector_function(det.CBox_digitizing_shots_det(CBox, AWG,
                                                       threshold=CBox.sig0_threshold_line()))
AWG.start()
data = MC.run('GST_l{}_2D'.format(l), mode='2D')

MC.live_plot_enabled = True
