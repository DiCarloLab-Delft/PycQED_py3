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

reload(qb)
QL = reload_CC_qubit(QL)
QR = reload_CC_qubit(QR)

