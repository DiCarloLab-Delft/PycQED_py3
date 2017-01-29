################################
# Reloading qubit snippet
################################
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
AncT.link_param_to_operation('CZ', 'fluxing_amp', 'amplitude', )
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
AncT.link_param_to_operation('CZ', 'CZ_length', 'flux_pulse_length')

AncT.add_pulse_parameter('CZ', 'g2', 'g2', 33.3e6)
AncT.add_pulse_parameter('CZ', 'CZ_lambda_coeffs', 'lambda_coeffs',
                         np.array([1, 0, 0]),
                         vals=vals.Arrays())
AncT.link_param_to_operation('CZ', 'mw_to_flux_delay', 'mw_to_flux_delay')#, 0)
AncT.add_pulse_parameter('CZ', 'CZ_phase_corr_amp', 'phase_corr_pulse_amp', 0)
AncT.add_pulse_parameter('CZ', 'CZ_phase_corr_length',
                         'phase_corr_pulse_length', 10e-9)
AncT.add_pulse_parameter('CZ', 'CZ_pulse_buffer',
                         'pulse_buffer', 0)
AncT.add_pulse_parameter('CZ', 'CZ_pulse_delay',
                         'pulse_delay', 10e-9)
AncT.add_pulse_parameter('CZ', 'CZ_refpoint',
                         'refpoint', 'end', vals=vals.Strings())

AncT.add_pulse_parameter('CZ', 'CZ_square_pulse_buffer',
                         'square_pulse_buffer', 100e-9)
AncT.add_pulse_parameter('CZ', 'CZ_square_pulse_length',
                         'square_pulse_length', 40e-9)
AncT.add_pulse_parameter('CZ', 'CZ_swap_amp',
                         'swap_amp', 1.0)  # Should not be CZ
AncT.add_pulse_parameter('CZ', 'CZ_theta', 'theta_f', np.pi/2)


AncT.add_operation('Z')

AncT.link_param_to_operation('Z', 'fluxing_channel', 'channel')
AncT.link_param_to_operation('Z', 'CZ_refpoint', 'refpoint')
AncT.link_param_to_operation('Z', 'CZ_phase_corr_amp', 'amplitude')
AncT.link_param_to_operation('Z', 'CZ_phase_corr_length', 'square_pulse_length')
AncT.link_param_to_operation('Z', 'CZ_pulse_buffer', 'pulse_buffer')
AncT.add_pulse_parameter('Z', 'Z_pulse_type', 'pulse_type',
                         initial_value='SquareFluxPulse',
                         vals=vals.Strings())
AncT.add_pulse_parameter('Z', 'Z_pulse_delay',
                         'pulse_delay', 0)

DataT.add_operation('SWAP')
DataT.link_param_to_operation('SWAP', 'fluxing_amp', 'amplitude')
DataT.link_param_to_operation('SWAP', 'fluxing_channel', 'channel')
DataT.add_pulse_parameter('SWAP', 'SWAP_dead_time', 'dead_time',
                          initial_value=3e-6)
DataT.link_param_to_operation('SWAP', 'mw_to_flux_delay', 'mw_to_flux_delay')
DataT.add_pulse_parameter(
    'SWAP', 'SWAP_phase_corr_amp', 'phase_corr_pulse_amp', 0)
DataT.add_pulse_parameter('SWAP', 'SWAP_phase_corr_length',
                          'phase_corr_pulse_length', 10e-9)
DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_type', 'pulse_type',
                          initial_value='SquareFluxPulse', vals=vals.Strings())
DataT.add_pulse_parameter('SWAP', 'SWAP_refpoint',
                          'refpoint', 'end', vals=vals.Strings())
DataT.link_param_to_operation('SWAP', 'SWAP_amp', 'SWAP_amp')
DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_buffer',
                          'pulse_buffer', 100e-9)
DataT.add_pulse_parameter('SWAP', 'SWAP_square_pulse_buffer',
                          'square_pulse_buffer', 100e-9)
DataT.add_pulse_parameter('SWAP', 'SWAP_square_pulse_length',
                          'square_pulse_length', 40e-9)

DataT.add_pulse_parameter('SWAP', 'SWAP_pulse_delay',
                          'pulse_delay', 10e-9)

DataT.add_operation('Z')
DataT.link_param_to_operation('Z', 'fluxing_channel', 'channel')
DataT.link_param_to_operation('Z', 'SWAP_refpoint', 'refpoint')
DataT.link_param_to_operation('Z', 'SWAP_phase_corr_amp', 'amplitude')
DataT.link_param_to_operation('Z', 'SWAP_phase_corr_length', 'square_pulse_length')
DataT.link_param_to_operation('Z', 'SWAP_pulse_buffer', 'pulse_buffer')
DataT.add_pulse_parameter('Z', 'Z_pulse_type', 'pulse_type',
                         initial_value='SquareFluxPulse', vals=vals.Strings())
DataT.add_pulse_parameter('Z', 'Z_pulse_delay',
                         'pulse_delay', 0)


gen.load_settings_onto_instrument(AncT, timestamp='020234')
gen.load_settings_onto_instrument(DataT, timestamp='020234')


DataT.RO_acq_weight_function_I(0)
DataT.RO_acq_weight_function_Q(0)
AncT.RO_acq_weight_function_I(1)
AncT.RO_acq_weight_function_Q(1)


# Reloading device type object
from pycqed.instrument_drivers.meta_instrument import device_object as do
reload(do)
try:
    S5.close()
    del station.components['S5']
except:
    pass
S5 = do.DeviceObject('S5')
station.add_component(S5)
S5.add_qubits([AncT, DataT])

# Required for the Niels naming scheme
q0 = DataT
q1 = AncT