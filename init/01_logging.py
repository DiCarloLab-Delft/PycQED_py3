import logging
from lib import config

def _setup_logging():
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)',
        datefmt='%Y-%m-%d %H:%M',
        filename=os.path.join(config.get_execdir(), 'qtlab.log'),
        filemode='a+')
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(name)s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def set_debug(enable):
    logger = logging.getLogger()
    if enable:
        logger.setLevel(logging.DEBUG)
        logging.info('Set logging level to DEBUG')
    else:
        logger.setLevel(logging.INFO)
        logging.info('Set logging level to INFO')

_setup_logging()
