from importlib import reload
import qcodes as qc
import logging
station = qc.station
from pycqed.utilities import general as gen
from pycqed.instrument_drivers.meta_instrument.qubit_objects import CC_transmon as qb

def reload_CC_qubit(qubit):
    reload(qb)
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


QL_CC = qb.CBox_v3_driven_transmon('QL_CC')
station.add_component(QL_CC)


from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBox_v3 as qcb
CBox = qcb.QuTech_ControlBox_v3(
    'CBox', address='Com6', run_tests=False, server_name=None)
station.add_component(CBox)

QL_CC = reload_CC_qubit(QL_CC)

QL_CC.CBox(CBox.name)
QL_CC.LO(LO.name)
QL_CC.RF_RO_source(RF.name)
QL_CC.cw_source('QL_LO')
QL_CC.td_source('QR_LO')

QL_CC.MC(MC.name)
QL_CC.RO_acq_weight_function_I(0)
QL_CC.RO_acq_weight_function_Q(1)
# apparently only possible after setting integration weight channels
QL_CC.acquisition_instrument(UHFQC_1.name)

gen.load_settings_onto_instrument(QL_CC, load_from_instr='QL')
IM.update()


# Testing the heterodyne sequence
CW_RO_sequence = sqqs.CW_RO_sequence(QL_CC.name,
                                     QL_CC.RO_acq_period_cw())
CW_RO_sequence_asm = qta.qasm_to_asm(CW_RO_sequence.name,
                                     QL_CC.get_operation_dict())

qumis_file = CW_RO_sequence_asm
CBox.load_instructions(qumis_file.name)
CBox.run_mode('run')



def check_keyboard_interrupt():
    try:  # Try except statement is to make it work on non windows pc
        if msvcrt.kbhit():
            key = msvcrt.getch()
            print(key)
            if b'q' in key:
                raise KeyboardInterrupt('Human interupt q')
    except Exception:
        pass