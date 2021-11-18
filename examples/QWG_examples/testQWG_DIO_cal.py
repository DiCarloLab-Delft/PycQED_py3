import logging

if 1:
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

if 1:
    cc = CC('cc', IPTransport('192.168.0.241'))
    print(cc.get_identity())
    cc.init()
else:
    cc = None

if 0:
    qwg_21 = QWG('qwg_21', IPTransport('192.168.0.179'))
    #qwg_22 = QWG('qwg_22', IPTransport('192.168.0.178'))
    qwg_8 = QWG('qwg_8', IPTransport('192.168.0.192'))
    qwgs = [qwg_21, qwg_8]
    #qwgs = [qwg_22, qwg_21] # reversed

if 1: # 20210907, development setup Wouter, slot 0 and 1
    qwg_9 = QWG('qwg_9', IPTransport('192.168.0.191'))      # slot 0
    qwg_19 = QWG('qwg_19', IPTransport('192.168.0.181'))    # slot 1
    qwgs = [qwg_19, qwg_9]
if 0: # 20210907, development setup Wouter, slot 2 and 3
    qwg_14 = QWG('qwg_14', IPTransport('192.168.0.186'))    # slot 2
    qwg_10 = QWG('qwg_10', IPTransport('192.168.0.190'))    # slot 3
    qwgs = [qwg_10, qwg_14]




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
