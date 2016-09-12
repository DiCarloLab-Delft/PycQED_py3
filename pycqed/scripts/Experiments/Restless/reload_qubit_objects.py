reload(awg_swf)
from instrument_drivers.meta_instrument.qubit_objects import qubit_object as qb
reload(qb)

from instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as qcb
reload(qcb)

from instrument_drivers.meta_instrument.qubit_objects import Tektronix_driven_transmon as tqb
reload(tqb)
from instrument_drivers.meta_instrument.qubit_objects import duplexer_tek_transmon as dqb
reload(dqb)

VIP_mon_2_dux = dt.Duplexer_tek_transmon('VIP_mon_2_dux', LO=LO,
                                         cw_source=Qubit_LO,
                                         td_source=Qubit_LO,
                                         IVVI=IVVI, AWG=AWG, CBox=CBox,
                                         heterodyne_instr=HS, MC=MC, Mux=Dux,
                                         rf_RO_source=RF, server_name=None)

station.VIP_mon_2_dux = VIP_mon_2_dux
gen.load_settings_onto_instrument(VIP_mon_2_dux)