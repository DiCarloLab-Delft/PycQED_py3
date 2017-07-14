import os
import logging
import pycqed as pq
from uuid import getnode as get_mac
from pycqed.init.config import setup_dict


def get_default_datadir():
    """
    Returns the default datadir by first looking in the setup dict and
    searching for a location defined by the mac address. If there is no
    location specified, it will fall back to pycqed/data as a default location.
    """
    try:
        # Try extracting the location from the setup_dict
        mac = get_mac()
        setup_name = setup_dict.mac_dict[str(mac)]
        datadir = setup_dict.data_dir_dict[setup_name]
    except Exception:
        # If the mac_address is unknown
        # Stores data in the default data location (pycqed_py3/data/)
        datadir = os.path.abspath(os.path.join(
            os.path.dirname(pq.__file__), os.pardir, 'data'))
        print(datadir)
        logging.info('Setting datadir to default location: {}'.format(
            datadir))
    return datadir
