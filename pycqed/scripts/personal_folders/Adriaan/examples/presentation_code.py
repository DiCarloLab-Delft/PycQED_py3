pulse_pars=pulse_pars
RO_pars=RO_pars
MC= MC
awg_swf = awg_swf
CBox=CBox
AWG=AWG
import numpy as np
det =det
ma=ma

t_start=t_start
t_stop = t_stop
t_step = t_step
import qcodes as qc
T1_seq=T1_seq
qca = qca

MC.set_sweep_function(awg_swf.T1(
                      pulse_pars=pulse_pars, RO_pars=RO_pars))
MC.set_sweep_points(np.linspace(t_start, t_stop, t_step))
MC.set_detector_function(det.CBox_integrated_average_detector(CBox, AWG))
MC.run(name='T1_qubit_1')
a = ma.T1_Analysis(auto=True, close_fig=False)

data = qc.Loop(T1_seq(pulse_pars, RO_pars)[t_start, t_stop, t_step]).each(
    CBox.integrated_avg_result()).run('T1_qubit_1')
qca.T1_Analysis(data)
