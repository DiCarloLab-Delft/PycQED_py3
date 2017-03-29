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


QL = qb.CBox_v3_driven_transmon('QL')
station.add_component(QL)


from pycqed.instrument_drivers.physical_instruments import QuTech_ControlBox_v3 as qcb
CBox = qcb.QuTech_ControlBox_v3(
    'CBox', address='Com6', run_tests=False, server_name=None)
station.add_component(CBox)


############ Reloading the QL_CC
QL = reload_CC_qubit(QL)
gen.load_settings_onto_instrument('QL', timestamp ='20170322_204816')
# QL_CC.CBox(CBox.name)
# QL_CC.LO(LO.name)
# QL_CC.RF_RO_source(RF.name)
# QL_CC.cw_source('QL_LO')
# QL_CC.td_source('QR_LO')
# QL_CC.MC(MC.name)
# QL_CC.acquisition_instrument(UHFQC_1.name)

# # gen.load_settings_onto_instrument(QL_CC, load_from_instr='QL')
# QL_CC.spec_pulse_type('gated')
IM.update()


import pycqed.instrument_drivers.meta_instrument.CBox_LookuptableManager as cbl
reload(cbl)
try:
    CBox_LutMan.close()
    del station.components['CBox_LutMan']
except:
    pass
CBox_LutMan = cbl.ControlBox_LookuptableManager('CBox_LutMan')
CBox_LutMan.CBox(CBox.name)
station.add_component(CBox_LutMan)

QL.find_resonator_frequency()
QL.find_frequency()
QL.find_frequency(pulsed=True)

SH = sh.SignalHound_USB_SA124B('Signal hound', server_name=None)

QL.measure_rabi()