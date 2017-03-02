# This script contains an example of reloading a qubit object.
# similar scripts can be made for any instrument


def reload_qubit(qubit):
    # Start by reloading the modules that you have changed
    reload(sqqs)
    reload(qo)
    reload(qbt)
    reload(qb)
    # Try removing the object if you have previously created it
    try:
        qubit_name = qubit.name
        qubit.close()
        del station.components[qubit_name]

    except Exception as e:
        logging.warning(e)

    if qubit.name == 'QL':
        cw_source = QR_LO
        td_source = QL_LO
    else:
        cw_source = QL_LO
        td_source = QR_LO

    qubit = qbt.Tektronix_driven_transmon('QL', LO=LO, cw_source=cw_source,
                                       td_source=td_source,
                                       IVVI=IVVI, rf_RO_source=RF,
                                       AWG=AWG,
                                       heterodyne_instr=HS,
                                       FluxCtrl=None,
                                       MC=MC,
                                       server_name=None)
    station.add_component(qubit)
    gen.load_settings_onto_instrument(qubit)
    return qubit
QL = reload_qubit(QL)
QR = reload_qubit(QR)