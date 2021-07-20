import logging

if 0:
    root_formatter = logging.Formatter('{asctime}.{msecs:03.0f} {levelname:7s} {name:32.32s}  {message}',
                                       '%Y%m%d %H:%M:%S',
                                       '{')

    class LoggingNameFilter(logging.Filter):
        def filter(self, record):
            record.name = record.name[-30:] # right trim name
            return True

    # configure root logger
    root_logger = logging.getLogger('')
    root_sh = logging.StreamHandler()
    root_sh.setLevel(logging.DEBUG)  # set log level of handler
    root_sh.setFormatter(root_formatter)
    root_sh.addFilter(LoggingNameFilter())
    root_logger.addHandler(root_sh)
    root_logger.setLevel(logging.WARNING)  # set log level of logger

    # configure pycqed logger
    pycqed_logger = logging.getLogger('pycqed')
    pycqed_logger.setLevel(logging.DEBUG)  # set log level of logger




from pycqed.instrument_drivers.library.Transport import IPTransport
import pycqed.instrument_drivers.library.DIO as DIO
from pycqed.instrument_drivers.physical_instruments.QuTech.QWG import QWG,QWGMultiDevices

from pycqed.instrument_drivers.physical_instruments.QuTech.CC import CC


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


cc = CC('cc', IPTransport('192.168.0.241'))
print(cc.get_identity())
cc.init()


qwg_21 = QWG('qwg_21', IPTransport('192.168.0.179'))
#qwg_22 = QWG('qwg_22', IPTransport('192.168.0.178'))
qwg_8 = QWG('qwg_8', IPTransport('192.168.0.192'))
qwgs = [qwg_21, qwg_8]
#qwgs = [qwg_22, qwg_21] # reversed

for qwg in qwgs:
    print(qwg.get_identity())
    qwg.init()
    qwg.run_mode('CODeword')
    qwg.cfg_codeword_protocol('awg8-mw-direct-iq')

qwgs[0].dio_mode('MASTER')
qwgs[1].dio_mode('SLAVE')


if 1:
    qwg_multi = QWGMultiDevices(qwgs)
    DIO.calibrate(sender=cc,receiver=qwg_multi,sender_dio_mode='awg8-mw-direct-iq')
else:
    for qwg in qwgs:
        DIO.calibrate(sender=cc,receiver=qwg,sender_dio_mode='awg8-mw-direct-iq')


# for qwg in qwgs:
#     print(f"QWG '{qwg.name}'' DIO calibration report:")
#     print(qwg.dio_calibration_report())

if 0:
    for qwg in qwgs:
        qwg.ch1_state(True)
        qwg.ch2_state(True)
        qwg.ch3_state(True)
        qwg.ch4_state(True)
        qwg.start()
