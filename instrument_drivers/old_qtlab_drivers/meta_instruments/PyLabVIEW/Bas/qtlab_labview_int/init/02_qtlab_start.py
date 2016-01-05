import types
from lib.misc import exact_time, get_ipython
from time import sleep

#set_debug(True)
from lib.network import object_sharer as objsh
objsh.root.set_instance_name('python_lv_int')

# Set exception handler
try:
    import qtflow
    # Note: This does not seem to work for 'KeyboardInterrupt',
    # likely it is already caught by ipython itself.
    get_ipython().set_custom_exc((Exception, ), qtflow.exception_handler)
except Exception as e:
    print('Error: %s' % str(e))

# Other functions should be registered using qt.flow.register_exit_handler
from lib.misc import register_exit
import qtflow
register_exit(qtflow.qtlab_exit)
