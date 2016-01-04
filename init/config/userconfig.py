# This file contains user-specific settings for qtlab.
# It is run as a regular python script.
import os
import sys
from uuid import getnode as get_mac
import qt
# Do not change the following line unless you know what you are doing
config.remove([
            'datadir',
            'startdir',
            'scriptdirs',
            'user_ins_dir',
            'startgui',
            'gnuplot_terminal',
            ])

PycQEDdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(PycQEDdir)
sys.path.append(os.path.join(PycQEDdir, 'modules'))

# Put here because PycQED dir needs to be appended to path first
from init.config import setup_dict

# execfile(PycQEDdir+'/init/config/setup_dict.py')
# loads dictionary containing mac addresses and datadirs

mac = get_mac()
config['PycQEDdir'] = PycQEDdir
config['mac_address'] = mac

try:
    setup_name = setup_dict.mac_dict[str(mac)]
    print('Setup identified as "%s"' % setup_name)
    datadir = setup_dict.data_dir_dict[setup_name]
    print('Datadir set to "%s"' % datadir)
    qt.config['datadir'] = datadir


except:
    print('Warning setup with mac: "%s" , not identified. Add setup to init/config/setup_dict.py and create custom config' %mac)
    print('Using default config')
    setup_name = 'default_config'
    qt.config['datadir'] = 'D:\Experiments\Data'


qt.config['user_instrument_directories'] = ['instrument_drivers/physical_instruments',
        'instrument_drivers/meta_instruments',
        'instrument_drivers/container_instruments',
        'instrument_drivers/dummy_instruments'
        ]

config['allowed_ips'] = []
config['instance_name'] = 'qtlab_n1'
config['setup name'] = setup_name
config['setup_config_dir'] = PycQEDdir+'/init/config/'+setup_name+'.py'