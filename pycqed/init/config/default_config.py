'''
Default qt.config
'''
import qt

qt.config['instance_name'] = 'PycQED_qtlab'
qt.config['port'] = 12002


# A list of allowed IP ranges for remote connections
qt.config['allowed_ips'] = (
   '130.161.*.*',
   '145.94.*.*',
)

# Start instrument server to share with instruments with remote QTLab?
qt.config['instrument_server'] = False


# This sets a default directory for qtlab to start in
qt.config['startdir'] = os.path.join(qt.config['PycQEDdir'], 'scripts')

# A default script (or list of scripts) to run after qtlab started
qt.config['startscript'] = []  # e.g. 'initscript1.py'

# A default script (or list of scripts) to run when qtlab closes
qt.config['exitscript'] = []   # e.g. ['closescript1.py', 'closescript2.py']

# Add directories containing scripts here. All scripts will be added to the
# global namespace as functions.
qt.config['scriptdirs'] = [
        str(os.path.join(qt.config['PycQEDdir']+'/scripts')),
        str(os.path.join(qt.config['PycQEDdir']+'/scripts/testing'))
]

# This sets a user instrument directory
# Any instrument drivers placed here will take
# preference over the general instrument drivers
qt.config['user_instrument_directories'] = ['instrument_drivers/physical_instruments',
                                            'instrument_drivers/meta_instruments',
                                            'instrument_drivers/dummy_instruments',
                                            'instrument_drivers/container_instruments'
                                            ]
    # str(os.path.join(qt.config['execdir']+'/instrument_plugins'))



# For adding additional folders to the 'systm path'
# so python can find your modules
#import sys
sys.path.append(os.path.join(qt.config['PycQEDdir'], 'modules'))

# Whether to start the GUI automatically
qt.config['startgui'] = True

# Enter a filename here to log all IPython commands
qt.config['ipython_logfile'] = ''      #e.g. 'command.log'
