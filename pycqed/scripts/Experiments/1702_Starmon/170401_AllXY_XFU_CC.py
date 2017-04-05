from pycqed.instrument_drivers.meta_instrument.qubit_objects import CC_transmon as qb

def reload_CC_qubit(qubit):
    try:
        qubit_name = qubit.name
        qubit.close()
        del station.components[qubit_name]

    except Exception as e:
        logging.warning(e)
    qubit = qb.CBox_v3_driven_transmon(qubit_name)
    station.add_component(qubit)
    gen.load_settings_onto_instrument(qubit)
    return qubit

from pycqed.scripts.Experiments.intel_demo import qasm_helpers as qh
reload(det)
reload(qh)
reload(qb)
QL = reload_CC_qubit(QL)
QR = reload_CC_qubit(QR)

QR.RO_LutMan('CBox_RO_LutMan')
QR.RO_awg_nr(2)
QR.acquisition_instrument('CBox')
QR.RO_pulse_type('MW_IQmod_pulse_CBox')
QR.RO_pulse_length(640e-9)

CBox.upload_standard_weights(QR.f_RO_mod())

