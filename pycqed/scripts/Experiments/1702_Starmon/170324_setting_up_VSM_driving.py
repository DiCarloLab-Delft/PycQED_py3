import pycqed.instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
from pycqed.instruments_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dt


from pycqed.instrument_drivers.physical_instruments import QuTech_Duplexer as qdux
VSM = qdux.QuTech_Duplexer('VSM', address='TCPIP0::192.168.0.100')


def reload_VSM_qubit(qubit):
    reload(qbt)
    reload(dt)
    try:
        qubit_name = qubit.name
        qubit.close()
        del station.components[qubit_name]

    except Exception as e:
        logging.warning(e)
    qubit = dt.Duplexer_tek_transmon(qubit_name)
    station.add_component(qubit)
    gen.load_settings_onto_instrument(qubit)
    return qubit

station.add_component(QL)