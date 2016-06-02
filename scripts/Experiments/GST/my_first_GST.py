import qcodes as qc
from modules.measurement import sweep_functions as swf
from modules.measurement import detector_functions as det

station = qc.station
MC = station.MC
CBox = station.components['CBox']

calibrate_RO_threshold_no_rotation()
fn = (PyCQEDpath+'\modules\measurement\pulse_sequences\_pygsti_Gatesequences\Gatesequences_maxlength=1.txt')
seq, elts = gsts.GST_from_textfile(pulse_pars=pulse_pars_duplex, RO_pars=RO_pars, filename=fn)
l=92
log_length = (l*int(8000/l))

CBox.log_length(log_length)
MC.set_sweep_function(swf.None_Sweep(sweep_control='hard') )
MC.set_sweep_points(np.arange(log_length))
MC.set_sweep_function_2D(swf.None_Sweep(sweep_control='soft'))
MC.set_sweep_points_2D(np.arange(40))
MC.set_detector_function(det.CBox_digitizing_shots_det(CBox, AWG, threshold=CBox.sig0_threshold_line()))
data = MC.run('GST_l{}_2D'.format(l), mode='2D')

