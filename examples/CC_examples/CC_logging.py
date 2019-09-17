import logging
import sys

if 0:
    root_formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-40.39s  %(message)s', '%Y%m%d %H:%M:%S')
else:
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

# configure ZI_base_instrument logger
zibi_logger = logging.getLogger('pycqed.instrument_drivers.physical_instruments.ZurichInstruments')
zibi_logger.setLevel(logging.INFO)  # set log level of logger

# configure print logger
print_logger = logging.getLogger('print')
print_logger.setLevel(logging.DEBUG)

### redirect stdout (Python print statements, and e.g. ziPython) to log
# we need some special treatment of ziPython output which does not call the Python print function
def print_logger_write(msg):
    if(msg.strip() != ''): # ignore messages with white space only
        lines = msg.split('\n')
        # try to extract some useful info from string. FIXME: this is flaky
        for line in lines:
            if 'failed' in line.lower() or 'error' in line.lower():
                print_logger.error(line)
            elif 'warning' in line.lower():
                print_logger.warning(line)
            else:
                print_logger.info(line)

sys.stdout.write = print_logger_write  # reroute stdout write (and also __stdout__write if __stdout__ equals stdout)
