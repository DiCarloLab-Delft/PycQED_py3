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


t_dict = {}
t_dict['buffer_FLUX_MW'] = 40e-9
t_dict['buffer_MW_FLUX'] = 40e-9
t_dict['buffer_MW_MW'] = 40e-9
t_dict['buffer_FLUX_FLUX'] = 40e-9

# 2D scan, V versus lambda 1
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
flux_pulse_pars_AncT = AncT.get_operation_dict()['CZ AncT']

flux_pulse_pars_DataT = DataT.get_operation_dict()['SWAP DataT']
# 2D Sweep
d = cl.CPhase_cost_func_det(qCP=AncT, qS=DataT, dist_dict=DataT.dist_dict(),
                            MC_nested=nested_MC, phases=phases2,
                            flux_pulse_pars_qCP=flux_pulse_pars_AncT,
                            flux_pulse_pars_qS=flux_pulse_pars_DataT,
                            int_avg_det=int_avg_det, CPhase=True,
                            single_qubit_phase_cost=True,
                            reverse_control_target=False,
                            inter_swap_wait=t_dict['buffer_FLUX_FLUX']*2)

AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())
MC.set_sweep_function(AWG.ch3_amp)
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
# AWG.ch3_amp(AncT.SWAP_amp())
# AncT.SWAP_amp(1.03)
# AWG.ch4_amp(DataT.SWAP_amp())
MC.run('test')
cl.SWAP_Cost()


############################################################################
############################################################################
################# Start of the new IARPA call notebook #####################
############################################################################
############################################################################


def measure_phase_qcp(points=np.arange(0., 0.15+0.005*4, 0.005)):
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
                                       timings_dict=t_dict,
                                       flux_pulse_pars_qCP=flux_pulse_pars_AncT,
                                       flux_pulse_pars_qS=flux_pulse_pars_DataT,
                                       int_avg_det=int_avg_det, CPhase=True,
                                       single_qubit_phase_corr=True,
                                       reverse_control_target=True,
                                       inter_swap_wait=10e-9,
                                       cost_function_choice=0, sweep_q=0)
    # lambda2 not actrive, cphae off and reverse on

    d.s.upload = True
    d.acquire_data_point()
    return d.last_seq


###############################################################################

def measure_phase_qs(points=np.arange(0., 0.15+0.005*4, 0.005)):
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
                                       timings_dict=t_dict,
                                       flux_pulse_pars_qCP=flux_pulse_pars_AncT,
                                       flux_pulse_pars_qS=flux_pulse_pars_DataT,
                                       int_avg_det=int_avg_det, CPhase=True,
                                       single_qubit_phase_corr=True, reverse_control_target=True,
                                       inter_swap_wait=10e-9, cost_function_choice=0, sweep_q=1)  # lambda2 not actrive, cphae off and reverse on

    d.s.upload = True
    d.acquire_data_point()
    return d.last_seq
#################################################################


def measure_cphase_amp(points=np.arange(1.03, 1.07, 0.005)):
    flux_pulse_pars_AncT = AncT.get_operation_dict()['CZ AncT']
    flux_pulse_pars_DataT = DataT.get_operation_dict()['SWAP DataT']
    AWG.ch3_amp(AncT.CZ_channel_amp())
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
                                           timings_dict=t_dict,
                                           flux_pulse_pars_qCP=flux_pulse_pars_AncT,
                                           flux_pulse_pars_qS=flux_pulse_pars_DataT,
                                           int_avg_det=int_avg_det, CPhase=True,
                                           single_qubit_phase_corr=True, reverse_control_target=False,
                                           inter_swap_wait=10e-9, cost_function_choice=1, sweep_q=1)  # lambda2 not actrive, cphae off and reverse on

        # lambda_swf =
        MC.set_sweep_function(AWG.ch3_amp)

        MC.set_detector_function(d)
        MC.set_sweep_points(points)

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


def SWAPN(qubit, swap_amps=np.arange(1.150, 1.171, 0.001),
          number_of_swaps=30,
          int_avg_det=None):
    # These are the sweep points
    swap_vec = np.arange(number_of_swaps)*2
    cal_points = 4
    lengths_cal = swap_vec[-1] + \
        np.arange(1, 1+cal_points)*(swap_vec[1]-swap_vec[0])
    swap_vec = np.concatenate((swap_vec, lengths_cal))

    if int_avg_det is None:
        int_avg_det = det.UHFQC_integrated_average_detector(
            UHFQC=AncT._acquisition_instr, AWG=AncT.AWG,
            channels=[DataT.RO_acq_weight_function_I(),
                      AncT.RO_acq_weight_function_I()],
            nr_averages=AncT.RO_acq_averages(),
            integration_length=AncT.RO_acq_integration_length(),
            cross_talk_suppression=True)

    op_dict = qubit.get_operation_dict()
    flux_pulse_pars = op_dict['SWAP '+qubit.name]
    mw_pulse_pars, RO_pars = qubit.get_pulse_pars()
    dist_dict = qubit.dist_dict()
    AWG = qubit.AWG

    repSWAP = awg_swf.SwapN(mw_pulse_pars,
                            RO_pars,
                            flux_pulse_pars,
                            dist_dict=dist_dict,
                            AWG=AWG,
                            upload=True)
    MC.set_sweep_function(repSWAP)
    MC.set_sweep_points(swap_vec)

    MC.set_sweep_function_2D(AWG.ch4_amp)
    MC.set_sweep_points_2D(swap_amps)

    MC.set_detector_function(int_avg_det)
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
    seq_2q = measure_cphase_amp()
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
        seq_cp = measure_phase_qcp(points=points_qcp, )
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
        seq_qs = measure_phase_qs(points=points_qs, )
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



##########################
# Calibration snippet
##########################


AWG.ch3_amp(AncT.CZ_channel_amp())
AWG.ch4_amp(DataT.SWAP_amp())

measure_cphase_amp(np.arange(1.044, 1.05, .001))
AncT.CZ_channel_amp(fix_phase_2Q())
AWG.ch3_amp(AncT.CZ_channel_amp())

for i in range(2):
    measure_phase_qcp(points=np.linspace(0.1, 0.25, 64))
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


