'''
# Below is taken from a note by Leo in one note
In this note, I will summarize for myself how the "automated" tuneup of the
two-qubit phase (phi_2Q) and the single-qubit phases (phi_1,cp for Qcp and
phi_1,s for Qs) was done in Weeks 2 and 3.

There are three sequences, and they are typically executed in this order

    - Tuneup of Phi_2Q.
    - Tuneup of Phi_1,cp.
    - Tuneup of Phi_1,s.
    - Tuneup of Phi_1,cp.
    - Tuneup of Phi_1,s.
'''

import numpy as np
from pycqed.scripts.Experiments.Five_Qubits import cost_functions_Leo_optimization as cl
import pycqed.measurement.multi_qubit_module as mq_mod


# 2D scan, V versus lambda 1
AncT.RO_acq_averages(512)
MC.live_plot_enabled(True)
nested_MC.soft_avg(1)
th90 = np.pi/2
reload(det)
int_avg_det = det.UHFQC_integrated_average_detector(
    UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
    channels=[AncT.RO_acq_weight_function_I(),
              DataT.RO_acq_weight_function_I()],
    nr_averages=AncT.RO_acq_averages(),
    integration_length=AncT.RO_acq_integration_length(),
    cross_talk_suppression=True)
phases2 = np.tile(np.linspace(0, 720, 16), 2)
phases2 = np.concatenate([phases2, [730, 740, 750, 760]])


# pulse_lengths=[30]
reload(ca)
reload(det)

#######
# AAAAAAARGH too many magic numbers
#######
for pulse_length in [40]:
    flux_pulse_pars_AncT = AncT.get_flux_pars()
    flux_pulse_pars_AncT['pulse_type'] = 'MartinisFluxPulse'
    # flux_pulse_pars_AncT['pulse_type']='SquareFluxPulse'
    flux_pulse_pars_AncT['amplitude'] = .45
    flux_pulse_pars_AncT['flux_pulse_length'] = pulse_length*1e-9
    flux_pulse_pars_AncT['lambda_coeffs'] = [1, 0, 0]  # [.1, .0]
    flux_pulse_pars_AncT['theta_f'] = th90  # .3
    flux_pulse_pars_AncT['square_pulse_length'] = pulse_length*1e-9
    flux_pulse_pars_AncT['E_c'] = AncT.EC()
    flux_pulse_pars_AncT['f_bus'] = 4.8e9
    flux_pulse_pars_AncT['f_01_max'] = AncT.f_qubit()
    flux_pulse_pars_AncT['dac_flux_coefficient'] = 0.679*2
    flux_pulse_pars_AncT['g2'] = 33.3e6  # 1/(120e-9/(14.5/2))
    flux_pulse_pars_AncT['pulse_buffer'] = 0
    flux_pulse_pars_AncT['phase_corr_pulse_length'] = 10e-9
    flux_pulse_pars_AncT['phase_corr_pulse_amp'] = -0.0

    # preparing flux parameters for the swap pulse
    flux_pulse_pars_DataT = DataT.get_flux_pars()
    flux_pulse_pars_DataT['phase_corr_pulse_length'] = 10e-9
    flux_pulse_pars_DataT['phase_corr_pulse_amp'] = -0.00


# 2D Sweep
d = cl.CPhase_cost_func_det(qCP=AncT, qS=DataT, dist_dict=dist_dict,
                            MC_nested=nested_MC, phases=phases2,
                            flux_pulse_pars_qCP=flux_pulse_pars_AncT,
                            flux_pulse_pars_qS=flux_pulse_pars_DataT,
                            int_avg_det=int_avg_det, CPhase=True,
                            single_qubit_phase_cost=True,
                            reverse_control_target=False,
                            inter_swap_wait=10e-9)

AWG.ch3_amp(1.025)  # (1.145)
AWG.ch4_amp(1.1575)

MC.set_sweep_function(AWG.ch3_amp)
# MC.set_sweep_function_2D(cl.lambda1_sweep(d))
MC.set_sweep_function_2D(cl.lambda2_sweep(d))
MC.set_detector_function(d)
MC.set_sweep_points(np.linspace(0.9, 1.1, 51)*AWG.ch3_amp())
MC.set_sweep_points_2D(np.linspace(0.0, .30, 2))
label = 'CP_amp3_lambda2_{}'.format(pulse_length)
MC.run(label, mode='2D')
ma.TwoD_Analysis(label=label, plot_all=True)

############################################################################
# Self contained example for the params

# Flux pars come from above, these need to be put in qubit object proper
phases = np.tile(np.linspace(0, 720, 16), 2)
phases = np.concatenate([phases, [730, 740, 750, 760]])


qCP_pulse_pars = AncT.get_pulse_pars()[0]
qS_pulse_pars = DataT.get_pulse_pars()[0]
RO_pars = RO_pars_AncT
flux_pulse_pars_AncT = AncT.get_operation_dict()['CZ AncT']
flux_pulse_pars_DataT = DataT.get_operation_dict()['SWAP DataT']
AWG.ch3_amp(AncT.CZ_channel_amp())


s = awg_swf.swap_CP_swap_2Qubits(
    qCP_pulse_pars, qS_pulse_pars,
    flux_pulse_pars_qCP=flux_pulse_pars_AncT, flux_pulse_pars_qS=flux_pulse_pars_DataT,
    RO_pars=RO_pars,
    dist_dict=dist_dict, AWG=AWG,  inter_swap_wait=100e-9,
    excitations='both', CPhase=True,
    reverse_control_target=False)
d = int_avg_det

MC.set_sweep_function(s)
MC.set_detector_function(d)
MC.set_sweep_points(phases)

s.prepare()
s.upload = False
# AWG.ch3_amp(AncT.swap_amp())
# AncT.swap_amp(1.03)
# AWG.ch4_amp(DataT.swap_amp())
MC.run('test')
cl.SWAP_Cost()


############################################################################
############################################################################
################# Start of the new IARPA call notebook #####################
############################################################################
############################################################################


def measure_phase_qcp(points=np.arange(0., 0.15+0.005*4, 0.005), soft_avg=10):
    flux_pulse_pars_AncT = AncT.get_operation_dict()['CZ AncT']
    flux_pulse_pars_DataT = DataT.get_operation_dict()['SWAP DataT']
    AWG.ch3_amp(AncT.CZ_channel_amp())

    int_avg_det = det.UHFQC_integrated_average_detector(
        UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
        channels=[
            AncT.RO_acq_weight_function_I(), DataT.RO_acq_weight_function_I()],
        nr_averages=AncT.RO_acq_averages(),
        integration_length=AncT.RO_acq_integration_length(), cross_talk_suppression=True)
    cphasesweep = points

    d = cl.CPhase_cost_func_det_Ramiro(qCP=AncT, qS=DataT, dist_dict=dist_dict,
                                       MC_nested=nested_MC, sphasesweep=cphasesweep,
                                       flux_pulse_pars_qCP=flux_pulse_pars_AncT,
                                       flux_pulse_pars_qS=flux_pulse_pars_DataT,
                                       int_avg_det=int_avg_det, CPhase=True,
                                       single_qubit_phase_corr=True,
                                       reverse_control_target=True,
                                       inter_swap_wait=10e-9,
                                       cost_function_choice=0, sweep_q=0)
    # lambda2 not actrive, cphae off and reverse on

    d.MC_nested.soft_avg(soft_avg)
    d.s.upload = True
    d.acquire_data_point()
    return d.last_seq


###############################################################################

def measure_phase_qs(points=np.arange(0., 0.15+0.005*4, 0.005), soft_avg=10):
    flux_pulse_pars_AncT = AncT.get_operation_dict()['CZ AncT']
    flux_pulse_pars_DataT = DataT.get_operation_dict()['SWAP DataT']
    AWG.ch3_amp(AncT.CZ_channel_amp())

    int_avg_det = det.UHFQC_integrated_average_detector(
        UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
        channels=[
            AncT.RO_acq_weight_function_I(), DataT.RO_acq_weight_function_I()],
        nr_averages=AncT.RO_acq_averages(),
        integration_length=AncT.RO_acq_integration_length(), cross_talk_suppression=True)
    sphasesweep = points

    from pycqed.scripts.Experiments.Five_Qubits import cost_functions_Leo_optimization as ca

    d = cl.CPhase_cost_func_det_Ramiro(qCP=AncT, qS=DataT, dist_dict=dist_dict,
                                       MC_nested=nested_MC, sphasesweep=sphasesweep,
                                       flux_pulse_pars_qCP=flux_pulse_pars_AncT,
                                       flux_pulse_pars_qS=flux_pulse_pars_DataT,
                                       int_avg_det=int_avg_det, CPhase=True,
                                       single_qubit_phase_corr=True, reverse_control_target=True,
                                       inter_swap_wait=10e-9, cost_function_choice=0, sweep_q=1)  # lambda2 not actrive, cphae off and reverse on

    d.MC_nested.soft_avg(soft_avg)
    d.s.upload = True
    d.acquire_data_point()
    return d.last_seq
#################################################################


def measure_cphase_amp(points=np.arange(1.03, 1.07, 0.005), soft_avg=10):
    flux_pulse_pars_AncT = AncT.get_operation_dict()['CZ AncT']
    flux_pulse_pars_DataT = DataT.get_operation_dict()['SWAP DataT']
    AWG.ch3_amp(AncT.CZ_channel_amp())
    nested_MC.soft_avg(soft_avg)
    th90 = np.pi/2
    int_avg_det = det.UHFQC_integrated_average_detector(
        UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
        channels=[
            AncT.RO_acq_weight_function_I(), DataT.RO_acq_weight_function_I()],
        nr_averages=AncT.RO_acq_averages(),
        integration_length=AncT.RO_acq_integration_length(), cross_talk_suppression=True)
    dt = 0.005  # 25
    sphasesweep = np.tile(np.arange(0., 0.21, dt), 2)
    sphasesweep = np.concatenate([sphasesweep, [0.22, 0.23, 0.24, 0.25]])

    # pulse_lengths=[30]
    for pulse_length in [40]:

        d = cl.CPhase_cost_func_det_Ramiro(qCP=AncT, qS=DataT, dist_dict=dist_dict,
                                           MC_nested=nested_MC, sphasesweep=sphasesweep,
                                           flux_pulse_pars_qCP=flux_pulse_pars_AncT,
                                           flux_pulse_pars_qS=flux_pulse_pars_DataT,
                                           int_avg_det=int_avg_det, CPhase=True,
                                           single_qubit_phase_corr=True, reverse_control_target=False,
                                           inter_swap_wait=10e-9, cost_function_choice=1, sweep_q=1)  # lambda2 not actrive, cphae off and reverse on

        # lambda_swf =
        MC.set_sweep_function(AWG.ch3_amp)

        MC.set_detector_function(d)
        MC.set_sweep_points(points)
        d.MC_nested.soft_avg(soft_avg)
    #     MC.set_sweep_points(np.arange(1.04,1.051,0.001))

        label = 'CP_amp3sweep_1D'
        MC.run(label)
        ma.MeasurementAnalysis(label=label, plot_all=True)
    return d.last_seq

    #################################################################


def fix_phase_qcp():
    label = 'swap_CP_swap'
    a = ma.MeasurementAnalysis(label=label, auto=False)
    a.get_naming_and_values()
    x = a.sweep_points[:-4]
    cp_acq_weight = 0
    y = a.measured_values[cp_acq_weight, :-4]
    return a_tools.find_min(x, y)


def fix_phase_qs():
    label = 'swap_CP_swap'
    a = ma.MeasurementAnalysis(label=label, auto=False)
    a.get_naming_and_values()
    x = a.sweep_points[:-4]
    qs_acq_weigth = 1
    y = a.measured_values[qs_acq_weigth, :-4]
    return a_tools.find_min(x, y)


def fix_phase_2Q():
    label = 'CP_amp3sweep_1D'
    folder = a_tools.latest_data(label)
    splitted = folder.split('\\')
    scan_start = splitted[-2]+'_'+splitted[-1][:6]
    scan_stop = scan_start
    opt_dict = {'scan_label': label}

    pdict = {'I': 'I',
             'times': 'sweep_points'}
    nparams = ['I', 'Q', 'times']
    amp_scans = ca.quick_analysis(t_start=scan_start, t_stop=scan_stop, options_dict=opt_dict,
                                  params_dict_TD=pdict, numeric_params=nparams)
    return amp_scans.TD_dict['times'][0][np.argmax(amp_scans.TD_dict['I'][0, :])]

    #################################################################

# SWAPN


def SWAPN_DataT(swap_amps=np.arange(1.150, 1.171, 0.001)):
    qubit = DataT
    qubit.swap_amp(1.161)
    # qubit.swap_time(11e-9) #initial guess, one of the parameters
    # qubit.swap_amp(1.152) # comes from scaler
    num_iter = 30
    AncT.RO_acq_averages(1024)
    int_avg_det = det.UHFQC_integrated_average_detector(
        UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
        channels=[DataT.RO_acq_weight_function_I(),
                  AncT.RO_acq_weight_function_I()],
        nr_averages=AncT.RO_acq_averages(),
        integration_length=AncT.RO_acq_integration_length(),
        cross_talk_suppression=True)

    times_vec = np.arange(num_iter)*2
    cal_points = 4
    lengths_cal = times_vec[-1] + \
        np.arange(1, 1+cal_points)*(times_vec[1]-times_vec[0])
    lengths_vec = np.concatenate((times_vec, lengths_cal))

    flux_pulse_pars = DataT.get_operation_dict()['SWAP DataT']
    mw_pulse_pars, RO_pars = qubit.get_pulse_pars()
    repSWAP = awg_swf.SwapN(mw_pulse_pars,
                            RO_pars,
                            flux_pulse_pars,
                            dist_dict=dist_dict,
                            AWG=AWG,
                            upload=True)

    qubit.RO_acq_averages(1024)

    MC.set_sweep_function(repSWAP)
    MC.set_sweep_points(lengths_vec)

    MC.set_sweep_function_2D(AWG.ch4_amp)
    MC.set_sweep_points_2D(swap_amps)

    MC.set_detector_function(int_avg_det)
    AWG.set('%s_amp' % qubit.fluxing_channel(), qubit.swap_amp())
    MC.run('SWAPN_%s' % qubit.name, mode='2D')
    ma.TwoD_Analysis(auto=True)


for lll in range(1):
    th90 = np.pi/2
    flux_pulse_pars_AncT = AncT.get_flux_pars()
    flux_pulse_pars_AncT['pulse_type'] = 'MartinisFluxPulse'
    # flux_pulse_pars_AncT['pulse_type']='SquareFluxPulse'
    flux_pulse_pars_AncT['flux_pulse_length'] = pulse_length*1e-9
    flux_pulse_pars_AncT['lambda_coeffs'] = [1, 0.2, 0]  # [.1, .0]
    flux_pulse_pars_AncT['theta_f'] = th90  # .3
    flux_pulse_pars_AncT['square_pulse_length'] = pulse_length*1e-9
    flux_pulse_pars_AncT['E_c'] = AncT.EC()
    flux_pulse_pars_AncT['f_bus'] = 4.8e9
    flux_pulse_pars_AncT['f_01_max'] = AncT.f_qubit()
    flux_pulse_pars_AncT['dac_flux_coefficient'] = 0.679*2
    flux_pulse_pars_AncT['g2'] = 33.3e6  # 1/(120e-9/(14.5/2))
    flux_pulse_pars_AncT['pulse_buffer'] = 0
    flux_pulse_pars_AncT['phase_corr_pulse_length'] = 10e-9
    flux_pulse_pars_AncT['phase_corr_delay'] = 10e-9
    flux_pulse_pars_AncT[
        'phase_corr_pulse_amp'] = phase_qcp_vec[-1]  # -0.14475

    # preparing flux parameters for the swap pulse
    flux_pulse_pars_DataT = DataT.get_flux_pars()
    flux_pulse_pars_DataT['phase_corr_pulse_length'] = 10e-9
    flux_pulse_pars_DataT['phase_corr_pulse_amp'] = phase_qs_vec[-1]  # -0.117

    cphase_ch_amp = 1.05  # 1.0462
    swap_ch_amp = 1.154
    rep = 1
    qcp_ch_amp = np.zeros(rep)
    phase_qs_vec = np.zeros(rep)
    phase_qcp_vec = np.zeros(rep)
    # qcp_ch_amp[0] = 1.05

    AWG.ch3_amp(cphase_ch_amp)
    AWG.ch4_amp(swap_ch_amp)
    seq_2q = measure_cphase_amp(soft_avg=1)
    cphase_ch_amp = fix_phase_2Q()
    print('############################################################')
    print(cphase_ch_amp)
    print('############################################################')
    for k in range(rep):
        # k = 0
        # scanning cphase phase compensation
        AWG.ch3_amp(cphase_ch_amp)
        AWG.ch4_amp(swap_ch_amp)
        points_qcp = np.concatenate(
            [np.linspace(0., 0.21, 60), [0.211, 0.212, 0.213, 0.214]])
#         points_qcp = np.concatenate([np.linspace(0.,0.19,60),[0.191,0.192,0.193,0.194]])
        seq_cp = measure_phase_qcp(points=points_qcp, soft_avg=5)
        phase_qcp_vec[k] = fix_phase_qcp()
        flux_pulse_pars_AncT['phase_corr_pulse_amp'] = phase_qcp_vec[k]
        print('############################################################')
        print('Phase qCP=', phase_qcp_vec[k])
        print('############################################################')
        # scanning swap phase compensation
        AWG.ch3_amp(cphase_ch_amp)
        AWG.ch4_amp(swap_ch_amp)
        points_qs = np.concatenate(
            [np.linspace(0., 0.18, 60), [0.181, 0.182, 0.183, 0.184]])
#         points_qs = np.concatenate([np.linspace(0.,0.18,60),[0.181,0.182,0.183,0.184]])
        seq_qs = measure_phase_qs(points=points_qs, soft_avg=5)
        phase_qs_vec[k] = fix_phase_qs()
        flux_pulse_pars_DataT['phase_corr_pulse_amp'] = phase_qs_vec[k]
        print('############################################################')
        print('Phase qS=', phase_qs_vec[k])
        print('############################################################')

    # update the values
    flux_pulse_pars_AncT['phase_corr_pulse_amp'] = phase_qcp_vec[-1]
    flux_pulse_pars_DataT['phase_corr_pulse_amp'] = phase_qs_vec[-1]
    AWG.ch3_amp(cphase_ch_amp)
    AWG.ch4_amp(swap_ch_amp)
    # reload stuff
    reload_mod_stuff()
    # run
    t_dict = {'buffer_MW_FLUX': 10e-9,
              'buffer_MW_MW': 10e-9,
              'buffer_FLUX_FLUX': 10e-9,
              'buffer_FLUX_MW': 10e-9}
    nr_shots = 1024

    MC.live_plot_enabled(False)
    AncT.RO_acq_averages(nr_shots)
    DataT.RO_acq_averages(nr_shots)

    # TOMO snippet
    for i in range(2):
        AWG.ch3_amp(AncT.CZ_channel_amp())
        AWG.ch4_amp(DataT.SWAP_amp())
        for target_bell in range(4):
            MC.soft_avg(1)
            seq_bell = mq_mod.tomo2Q_bell(target_bell, DataT, AncT,
                                          DataT.get_operation_dict()[
                                              'SWAP DataT'],
                                          AncT.get_operation_dict()['CZ AncT'],
                                          timings_dict=t_dict,
                                          distortion_dict=dist_dict,
                                          CPhase=True, nr_shots=nr_shots,
                                          nr_rep=1, mmt_label='BellTomo')
            tomo.analyse_tomo(MLE=False, target_bell=1)
        MC.live_plot_enabled(True)


def reload_mod_stuff():
    from pycqed.measurement.waveform_control_CC import waveform as wf
    reload(wf)

    from pycqed.measurement.pulse_sequences import fluxing_sequences as fqqs
    reload(fqqs)
    from pycqed.scripts.Experiments.Five_Qubits import cost_functions_Leo_optimization as ca
    reload(ca)
    from pycqed.measurement.waveform_control import pulse_library as pl
    reload(pl)
    from pycqed.measurement.pulse_sequences import standard_elements as ste
    reload(ste)

    from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mqs
    reload(mqs)
    from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_mswf
    reload(awg_mswf)
    reload(awg_swf)
    mqs.station = station
    fqqs.station = station
    reload(mq_mod)
    mq_mod.station = station

    reload(fsqs)
    reload(awg_swf)
    fsqs.station = station
    reload(det)
    reload(ca)
reload_mod_stuff()

##########################
# Calibration snippet
##########################


measure_cphase_amp()
AncT.CZ_channel_amp(fix_phase_2Q())
AWG.ch3_amp(AncT.CZ_channel_amp())
for i in range(2):
    measure_phase_qcp()
    AncT.CZ_phase_corr_amp(fix_phase_qcp())
    measure_phase_qs()
    DataT.SWAP_phase_corr_amp(fix_phase_qs())

for i in range(2):
    AWG.ch3_amp(AncT.CZ_channel_amp())
    AWG.ch4_amp(DataT.SWAP_amp())
    for target_bell in range(4):
        MC.soft_avg(1)
        seq_bell = mq_mod.tomo2Q_bell(target_bell, DataT, AncT,
                                      DataT.get_operation_dict()['SWAP DataT'],
                                      AncT.get_operation_dict()['CZ AncT'],
                                      timings_dict=t_dict,
                                      distortion_dict=dist_dict,
                                      CPhase=True, nr_shots=nr_shots,
                                      nr_rep=1, mmt_label='BellTomo')
        tomo.analyse_tomo(MLE=False, target_bell=target_bell)
    MC.live_plot_enabled(True)
