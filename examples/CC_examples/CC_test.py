from pycqed.instrument_drivers.library.Transport import IPTransport
from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC
ip_cc = '192.168.0.241'
cc = CC('cc', IPTransport(ip_cc))
cc.init()
cc.set_seqbar_cnt(2,5)
