find optimal pump power
    set pump freq
    set pump power at chip <-80 dBm
    measure s21, while sweeping pump power

find optimal pump frequency
    find dispersive feature with network analyzer(close to 8)
    sweep pump power as above, away from disspersive feature(6 GHz). Choose ~2 dBm less than fall off point
    gain vs pump frequency, with pump power as found before.


-------------------------------------------------




import visa
rm = visa.ResourceManager()
rm.list_resources()

import visa
inst = rm.open_resource('GPIB0::12::INSTR')
SYSTem:COMMunicate:NETWork:IPADdress:MODE STAT\n
SYSTem:COMMunicate:NETWork:IPADdress:GATeway '192.168.0.1'\n
SYSTem:COMMunicate:NETWork:IPADdress:SUBNet:MASK '255.255.255.0'\n
SYSTem:COMMunicate:NETWork:IPADdress '192.168.0.86'\n