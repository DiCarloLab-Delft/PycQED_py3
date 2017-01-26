################################
# Reloading qubit snippet
################################

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
    gen.load_settings_onto_instrument(q)
    if i ==0:
        q.dac_channel(1)
        q.fluxing_channel(3)
        q.RO_acq_weight_function_I(1)
        q.RO_acq_weight_function_Q(1)
    else:
        q.dac_channel(3)
        q.fluxing_channel(4)

        q.RO_acq_weight_function_I(0)
        q.RO_acq_weight_function_Q(0)


AncT = station.components['AncT']
DataT = station.components['DataT']

##



DataT.add_operation('CZ')
DataT.add_operation('CZ_phase_corr')
DataT.link_param_to_operation('fluxing_amp', 'amplitude', 'CZ')
DataT.link_param_to_operation('fluxing_channel', 'channel', 'CZ')
DataT.add_pulse_parameter('CZ_pulse_type', 'pulse_type','CZ', initial_value='SquareFluxPulse', vals=vals.Strings())
DataT.add_pulse_parameter('CZ_amp', 'channel_amp', 'CZ', 1.05)

DataT.link_param_to_operation('fluxing_amp', 'amplitude', 'CZ')
DataT.link_param_to_operation('fluxing_channel', 'channel', 'CZ_phase_corr')
DataT.add_pulse_parameter('CZ_phase_corr_pulse_type', 'pulse_type','CZ_phase_corr',
                          initial_value='SquareFluxPulse', vals=vals.Strings())


## Reloading device type object

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