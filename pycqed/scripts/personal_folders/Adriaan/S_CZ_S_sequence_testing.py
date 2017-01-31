
reload_mod_stuff()
################################
# Reloading qubit snippet
################################
import qcodes as qc
station = qc.station
from qcodes.utils import validators as vals
from pycqed.instrument_drivers.meta_instrument.qubit_objects import qubit_object as qo
from pycqed.instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as cbt
from pycqed.instrument_drivers.meta_instrument.qubit_objects import Tektronix_driven_transmon as qbt

reload(qo)
reload(cbt)
reload(qbt)


for i, name in enumerate(['AncT', 'DataT']):
    q = station.components[name]
    q.close()
    del station.components[name]
    q = qbt.Tektronix_driven_transmon(name, LO=LO, cw_source=Spec_source,
                                      td_source=Qubit_LO,
                                      IVVI=IVVI, rf_RO_source=RF,
                                      AWG=AWG,
                                      heterodyne_instr=HS,
                                      FluxCtrl=Flux_Control,
                                      MC=MC)
    station.add_component(q)

    if i == 0:
        q.dac_channel(1)
        q.RO_acq_weight_function_I(1)
        q.RO_acq_weight_function_Q(1)
    else:
        q.dac_channel(3)

        q.RO_acq_weight_function_I(0)
        q.RO_acq_weight_function_Q(0)


AncT = station.components['AncT']
DataT = station.components['DataT']

##


AncT.add_operation('CZ')
# AncT.add_operation('CZ_phase_corr') # to be added as separate later
AncT.add_pulse_parameter('CZ', 'CZ_pulse_amp', 'amplitude', initial_value=.5)
AncT.add_pulse_parameter('CZ', 'fluxing_operation_type', 'operation_type',
                         initial_value='Flux', vals=vals.Strings())
AncT.add_pulse_parameter('CZ', 'CZ_channel_amp', 'channel_amplitude',
                         initial_value=2.)
AncT.link_param_to_operation('CZ', 'fluxing_channel', 'channel')
AncT.link_param_to_operation('CZ', 'E_c', 'E_c')
AncT.add_pulse_parameter('CZ', 'CZ_pulse_type', 'pulse_type',
                         initial_value='MartinisFluxPulse', vals=vals.Strings())
AncT.add_pulse_parameter('CZ', 'CZ_dac_flux_coeff', 'dac_flux_coefficient',
                         initial_value=1.358)
AncT.add_pulse_parameter('CZ', 'CZ_dead_time', 'dead_time',
                         initial_value=3e-6)
AncT.link_param_to_operation('CZ', 'f_qubit', 'f_01_max')
AncT.add_pulse_parameter('CZ', 'CZ_bus', 'f_bus', 4.8e9)
AncT.add_pulse_parameter('CZ', 'CZ_length', 'length', 40e-9)
# AncT.link_param_to_operation('CZ', 'CZ_length', 'flux_pulse_length')

AncT.add_pulse_parameter('CZ', 'g2', 'g2', 33.3e6)
AncT.add_pulse_parameter('CZ', 'CZ_lambda_coeffs', 'lambda_coeffs',
                         np.array([1, 0, 0]),
                         vals=vals.Arrays())
AncT.link_param_to_operation('CZ', 'mw_to_flux_delay', 'mw_to_flux_delay')#, 0)


AncT.add_pulse_parameter('CZ', 'CZ_pulse_delay',
                         'pulse_delay', 0e-9)
AncT.add_pulse_parameter('CZ', 'CZ_refpoint',
                         'refpoint', 'end', vals=vals.Strings())

# AncT.add_pulse_parameter('CZ', 'CZ_square_pulse_buffer',
#                          'square_pulse_buffer', 100e-9)
# AncT.add_pulse_parameter('CZ', 'CZ_square_pulse_length',
#                          'square_pulse_length', 40e-9)
AncT.add_pulse_parameter('CZ', 'CZ_theta', 'theta_f', np.pi/2)


AncT.add_operation('CZ_corr')
AncT.link_param_to_operation('CZ_corr', 'fluxing_operation_type', 'operation_type')
AncT.link_param_to_operation('CZ_corr', 'fluxing_channel', 'channel')

AncT.link_param_to_operation('CZ_corr', 'CZ_refpoint', 'refpoint')

AncT.add_pulse_parameter('CZ_corr', 'CZ_corr_amp', 'amplitude', 0)
AncT.add_pulse_parameter('CZ_corr', 'CZ_corr_length',
                         'square_pulse_length', 10e-9)
#
AncT.add_pulse_parameter('CZ_corr', 'CZ_corr_pulse_type', 'pulse_type',
                         initial_value='SquareFluxPulse',
                         vals=vals.Strings())
AncT.add_pulse_parameter('CZ_corr', 'CZ_corr_pulse_delay',
                         'pulse_delay', 0)

DataT.add_operation('SWAP')
DataT.add_pulse_parameter('SWAP', 'fluxing_operation_type', 'operation_type',
                          initial_value='Flux', vals=vals.Strings())
DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_amp', 'amplitude',
                          initial_value=0.5)
DataT.link_param_to_operation('SWAP', 'fluxing_channel', 'channel')

DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_type', 'pulse_type',
                          initial_value='SquareFluxPulse', vals=vals.Strings())
DataT.add_pulse_parameter('SWAP', 'SWAP_refpoint',
                          'refpoint', 'end', vals=vals.Strings())
DataT.link_param_to_operation('SWAP', 'SWAP_amp', 'SWAP_amp')
DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_buffer',
                          'pulse_buffer', 0e-9)

DataT.link_param_to_operation('SWAP', 'SWAP_time', 'square_pulse_length')


DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_delay',
                          'pulse_delay', 0e-9)

DataT.add_operation('SWAP_corr')
DataT.add_pulse_parameter(
    'SWAP_corr', 'SWAP_corr_amp', 'amplitude', 0)
DataT.link_param_to_operation('SWAP_corr', 'fluxing_operation_type', 'operation_type')
DataT.link_param_to_operation('SWAP_corr', 'fluxing_channel', 'channel')
DataT.link_param_to_operation('SWAP_corr', 'SWAP_refpoint', 'refpoint')
DataT.add_pulse_parameter('SWAP_corr', 'SWAP_corr_length',
                          'square_pulse_length', 10e-9)
# DataT.link_param_to_operation('SWAP_corr', 'SWAP_corr_amp', 'amplitude')
# DataT.link_param_to_operation('SWAP_corr', 'SWAP_corr_length', 'square_pulse_length')
DataT.add_pulse_parameter('SWAP_corr', 'SWAP_corr_pulse_type', 'pulse_type',
                          initial_value='SquareFluxPulse', vals=vals.Strings())
DataT.add_pulse_parameter('SWAP_corr', 'SWAP_corr_pulse_delay',
                          'pulse_delay', 0)


gen.load_settings_onto_instrument(AncT)
gen.load_settings_onto_instrument(DataT)


DataT.RO_acq_weight_function_I(0)
DataT.RO_acq_weight_function_Q(0)
AncT.RO_acq_weight_function_I(1)
AncT.RO_acq_weight_function_Q(1)


# Reloading device type object
from pycqed.instrument_drivers.meta_instrument import device_object as do
reload(do)
# print(S5)
try:
    S5 = station.components['S5']
    S5.close()
    del station.components['S5']
except:
    pass
S5 = do.DeviceObject('S5')
station.add_component(S5)
S5.add_qubits([AncT, DataT])

S5.Buffer_Flux_Flux(10e-9)
S5.Buffer_Flux_MW(40e-9)
S5.Buffer_MW_MW(10e-9)
S5.Buffer_MW_Flux(10e-9)
station.sequencer_config = S5.get_operation_dict()['sequencer_config']


# Required for the Niels naming scheme
q0 = DataT
q1 = AncT


dist_dict = {'ch_list': ['ch4', 'ch3'],
             'ch4': k0.kernel(),
             'ch3': k1.kernel()}

DataT.dist_dict(dist_dict)
AncT.dist_dict(dist_dict)

AWG.ch4_amp(DataT.SWAP_amp())
AWG.ch3_amp(AncT.CZ_channel_amp())

reload(fsqs)
reload(awg_swf)
operation_dict = S5.get_operation_dict()


corr_amps = np.arange(.0, .1, 0.01)
CZ_amps = np.linspace(1.03, 1.07, 21)
MC.set_sweep_function(AWG.ch3_amp)
MC.set_sweep_points(CZ_amps)

d=czt.CPhase_cost_func_det(S5, DataT, AncT, nested_MC, corr_amps)
MC.set_detector_function(d)
MC.run('CZ_cost_function')

# S_CZ_S_swf = awg_swf.awg_seq_swf(
#         fsqs.SWAP_CZ_SWAP_phase_corr_swp,
#         # parameter_name='phase_corr_amps',
#         parameter_name='rec_phases',
#         unit='V',
#         AWG=DataT.AWG,
#         fluxing_channels=[DataT.fluxing_channel(), AncT.fluxing_channel()],
#         awg_seq_func_kwargs={'operation_dict': operation_dict,
#                              'qS': DataT.name,
#                              'qCZ': AncT.name,
#                              'sweep_qubit': AncT.name,
#                              'RO_target': AncT.name,
#                              'distortion_dict': DataT.dist_dict()})

# amps = np.tile(np.linspace(0, .2, 21),2)
# phases = np.tile(np.linspace(0,710,21),2)
# MC.set_sweep_function(S_CZ_S_swf)
# MC.set_detector_function(int_avg_det)
# # MC.set_sweep_points(amps)
# MC.set_sweep_points(phases)

# MC.run('test_S_CZ_S')
# ma.MeasurementAnalysis()





# kwargs = {'operation_dict': operation_dict,
#                              'qS': DataT.name,
#                              'qCZ': AncT.name,
#                              'phase_corr_amps': np.linspace(0, 0.5, 21),
#                              'sweep_qubit': AncT.name,
#                              'RO_target': AncT.name,
#                              'distortion_dict': DataT.dist_dict()}

# seq, elts = fsqs.SWAP_CZ_SWAP_phase_corr_swp(**kwargs)