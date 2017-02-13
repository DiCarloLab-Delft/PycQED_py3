from pycqed.scripts.Experiments.Five_Qubits import CZ_tuneup as czt
# script for cphases run on 20170131

mq_mod.measure_SWAPN(S5, 'DataT', swap_amps=np.arange(1.04, 1.06, 0.001))
# amp obtained was 1.048 Vpp
DataT.SWAP_amp(1.048)

# next the cphase amplitude
nested_MC.soft_avg(3)
CZ_amps = np.linspace(1.03, 1.07, 21)
AWG.ch4_amp(DataT.SWAP_amp())
corr_amps = np.arange(.0, .3, 0.01)
MC.set_sweep_function(AWG.ch3_amp)
MC.set_sweep_points(CZ_amps)
d=czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
MC.set_detector_function(d)
MC.run('CZ_cost_function')
# amplitude obtained was 1.056 Vpp

# next the 1Q phase corrections
AncT.CZ_channel_amp(1.056)
AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())
scan_range = {'DataT': np.linspace(0, .15, 60),
              'AncT': np.linspace(0, .21, 60)}
for s_q in ['DataT', 'AncT']:
    mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                        scan_range[s_q], sweep_qubit=s_q, excitations=0)

# nice functions
def fix_phase_qcp():
    label = 'SWAP_CP_SWAP'
    a = ma.MeasurementAnalysis(label=label, auto=False)
    a.get_naming_and_values()
    x = a.sweep_points[:-4]
    cp_acq_weight = 0
    y = a.measured_values[cp_acq_weight, :-4]
    return a_tools.find_min(x, y)


def fix_phase_qs():
    label = 'SWAP_CP_SWAP'
    a = ma.MeasurementAnalysis(label=label, auto=False)
    a.get_naming_and_values()
    x = a.sweep_points[:-4]
    qs_acq_weigth = 1
    y = a.measured_values[qs_acq_weigth, :-4]
    return a_tools.find_min(x, y)


def fix_phase_2Q():
    label = 'CZ_cost_function'
    a = ma.MeasurementAnalysis(label=label, auto=False)
    a.get_naming_and_values()
    x = a.sweep_points[:]
    cp_acq_weight = 0
    y = a.measured_values[cp_acq_weight, :]
    return x[np.argmax(y)],y[np.argmax(y)]
#     return a_tools.find_min(x, -y)

# all toghether
qs_corr = AncT.CZ_corr_amp()
qcp_corr = DataT.SWAP_corr_amp()
amp_2Q = AncT.CZ_channel_amp()
print('********************************')
print('CPhase Amp = %.3f Vpp'%amp_2Q)
print('qS phase corr = %.3f'%qs_corr)
print('qCP phase corr = %.3f'%qcp_corr)
print('********************************')

CZ_amps = np.linspace(-0.002, .002, 5) + amp_2Q
AWG.ch4_amp(DataT.SWAP_amp())
corr_amps = np.arange(.0, .3, 0.01)
MC.set_sweep_function(AWG.ch3_amp)
MC.set_sweep_points(CZ_amps)
d=czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
MC.set_detector_function(d)
MC.run('CZ_cost_function')
amp_2Q = fix_phase_2Q()[0]
AncT.CZ_channel_amp(amp_2Q)
AWG.ch3_amp(AncT.CZ_channel_amp())
print('********************************')
print('CPhase Amp = %.3f Vpp'%amp_2Q)
print('qS phase corr = %.3f'%qs_corr)
print('qCP phase corr = %.3f'%qcp_corr)
print('********************************')

mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                    np.linspace(0, .13, 60), sweep_qubit='DataT', excitations=0)
qs_corr = fix_phase_qs()[0]
DataT.SWAP_corr_amp(qs_corr)
print('********************************')
print('CPhase Amp = %.3f Vpp'%amp_2Q)
print('qS phase corr = %.3f'%qs_corr)
print('qCP phase corr = %.3f'%qcp_corr)
print('********************************')
mq_mod.measure_SWAP_CZ_SWAP(S5, 'DataT', 'AncT',
                    np.linspace(0, .18, 60), sweep_qubit='AncT', excitations=0)
qcp_corr = fix_phase_qcp()[0]
AncT.CZ_corr_amp(qcp_corr)
print('********************************')
print('CPhase Amp = %.3f Vpp'%amp_2Q)
print('qS phase corr = %.3f'%qs_corr)
print('qCP phase corr = %.3f'%qcp_corr)
print('********************************')


#tomo
from pycqed.analysis import tomography as tomo

mq_mod.tomo2Q_bell(bell_state=0, device=S5, qS_name='DataT', qCZ_name='AncT',
                   nr_shots=1024, nr_rep=1)
tomo.analyse_tomo(MLE=False, target_bell=0)

# plot sequence
# vw = None
vw.clear()
for i in [24, 4]:
    vw = viewer.show_element_pyqt(
        station.pulsar.last_elements[i], vw, color_idx=i%7)