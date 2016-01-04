# qtflow.py, handle 'flow control' in the QT lab environment
# Pieter de Groot, <pieterdegroot@gmail.com>, 2008
# Reinier Heeres, <reinier@heeres.eu>, 2008
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import gobject
import gtk
import logging
import time
from gettext import gettext as _L
from .lib.misc import exact_time, get_traceback
from .lib.network.object_sharer import SharedGObject
import os

AutoFormattedTB = get_traceback()

class FlowControl(SharedGObject):
    '''
    Class for flow control of the QT measurement environment.
    '''

    __gsignals__ = {
            'measurement-start': (gobject.SIGNAL_RUN_FIRST,
                gobject.TYPE_NONE,()),
            'measurement-end': (gobject.SIGNAL_RUN_FIRST,
                gobject.TYPE_NONE,()),
            'measurement-idle': (gobject.SIGNAL_RUN_FIRST,
                gobject.TYPE_NONE,()),
            'stop-request': (gobject.SIGNAL_RUN_FIRST,
                gobject.TYPE_NONE,()),
            'close-gui': (gobject.SIGNAL_RUN_FIRST,
                gobject.TYPE_NONE,()),
    }

    STATUS_STOPPED = 0
    STATUS_RUNNING = 1

    def __init__(self):
        SharedGObject.__init__(self, 'flow')
        self._status = 'starting'
        self._measurements_running = 0
        self._abort = False
        self._pause = False
        self._exit_handlers = []
        self._callbacks = {}

    #########
    ### signals
    #########

    def measurement_start(self):
        '''
        Indicate the start of a measurement.

        FIXME: The following is disabled due to lack of exception catching.
        This will increment the internal measurement counter, and if a
        measurement was not running, it will emit the 'measurement-start'
        signal.
        '''

        self._measurements_running += 1
        if self._measurements_running == 1:
            self._set_status('running')
            self.emit('measurement-start')

            # Handle callbacks
            self.run_mainloop(1, wait=False)

    def measurement_end(self, abort=False):
        '''
        Indicate the end of a measurement.

        FIXME: The following is disabled due to lack of exception catching.
        This will decrement the internal measurement counter if abort=False,
        and set it to 0 in abort=True. If the counter reached zero (e.g. the
        last measurement was stopped, it will emit the 'measurement-end'
        signal.
        '''

        if abort:
            self._measurements_running = 0
        elif self._measurements_running > 0:
            self._measurements_running -= 1

        if self._measurements_running == 0:
            self._set_status('stopped')
            self.emit('measurement-end')

            # Handle callbacks
            self.run_mainloop(1, wait=False)

    def run_mainloop(self, delay, wait=True, exact=False):
        '''
        Run mainloop for a maximum of <delay> seconds.
        If wait is True (default), sleep until <delay> seconds have passed.
        '''
        start = exact_time()
        dt = 0
	# TODO possibly this implementation of event handling using threads
	# can be done in a better way using ipython-0.11 inputhook support?
        gtk.gdk.threads_enter()
        while gtk.events_pending() and (not exact or (dt + 0.001) < delay):
            gtk.main_iteration_do(False)
            dt = exact_time() - start
        gtk.gdk.threads_leave()

        if delay > dt and wait:
            time.sleep(delay - dt)

    def measurement_idle(self, delay=0.0, exact=False, emit_interval=1):
        '''
        Indicate that the measurement is idle and handle events.

        This function will check whether an abort has been requested and
        handle queued events for a time up to 'delay' (in seconds).

        It starts by emitting the 'measurement-idle' signal to allow callbacks
        to be executed by the time this function handles the event queue.
        After that it handles events and sleeps for periods of 10msec. Every
        <emit_interval> seconds it will emit another measurement-idle signal.

        If exact=True, timing should be a bit more precise, but in this case
        a delay <= 1msec will result in NO gui interaction.
        '''

        start = exact_time()

        self.emit('measurement-idle')
        lastemit = exact_time()

        while self._pause:
            self.check_abort()
            self.run_mainloop(0.01)

        while True:
            self.check_abort()

            curtime = exact_time()
            if curtime - lastemit > emit_interval:
                self.emit('measurement-idle')
                lastemit = curtime

            dt = exact_time() - start
            self.run_mainloop(delay-dt, wait=False, exact=exact)

            dt = exact_time() - start
            if dt + 0.01 < delay:
                time.sleep(0.01)
            else:
                time.sleep(max(0, delay - dt))
                return

    def _run_script(self, scriptfile):
        return exec(compile(open(scriptfile).read(), scriptfile, 'exec'))

    def register_exit_script(self, scriptfile):
        self.register_exit_handler(lambda: self._run_script(scriptfile))

    def register_exit_handler(self, func):
        if func not in self._exit_handlers:
            self._exit_handlers.append(func)

    def exit_request(self):
        '''Run all registered exit handlers.'''
        for func in self._exit_handlers:
            try:
                func()
            except Exception as e:
                print('Error in func %s: %s' % (func.__name__, str(e)))

    def register_callback(self, time_msec, func, handle=None):
        '''
        Register a function to be called every time_msec miliseconds.

        <handle> is a name you can use to refer to it when removing the
        callback using 'remove_callback'. If you don't specify a specific
        name, a handle will be generated.

        Returns: callback handle
        '''

        hid = gobject.timeout_add(time_msec, func)
        if handle is None:
            handle = hid
        self._callbacks[handle] = hid
        return handle

    def remove_callback(self, handle):
        '''
        Remove a callback that was created with 'register_callback'
        '''
        if handle not in self._callbacks:
            logging.warning('Callback %s not found')
            return False

        gobject.source_remove(self._callbacks[handle])
        del self._callbacks[handle]
        return True

    ############
    ### status
    ############

    def get_status(self):
        '''Get status, one of "running", "stopped", "starting" '''
        return self._status

    def _set_status(self, val):
        self._status = val

    def finished_starting(self):
        self._status = "stopped"

    def is_measuring(self):
        return self.get_status() == 'running'

    def get_live_plot(self):
        import qt
        return qt.config.get('live-plot', True)

    def set_live_plot(self, val):
        import qt
        qt.config.set('live-plot', val)

    def toggle_live_plot(self):
        self.set_live_plot(not self.get_live_plot())

    def check_abort(self):
        '''Check whether an abort has been requested.'''

        if self._abort:
            self._abort = False
            self.emit('stop-request')
            self.measurement_end(abort=True)
            raise ValueError(_L('Human abort'))

    def set_abort(self):
        '''Request an abort.'''
        self._abort = True

    def is_paused(self):
        return self._pause

    def set_pause(self, pause):
        '''Set / unset pause state.'''
        self._pause = pause

    def start_gui(self):
        import qt

        curdir = os.getcwd()
        os.chdir(qt.config['execdir'])

        args = ['-p', str(qt.config['port']), '--name', qt.config['instance_name']]
        if os.name == 'nt':
            args.insert(0, 'qtlabgui.bat')
            os.spawnv(os.P_NOWAIT, 'qtlabgui.bat', args)
        if os.name == 'posix':
            args.insert(0, 'qtlabgui')
            pid = os.spawnv(os.P_NOWAIT, 'qtlabgui', args)

        os.chdir(curdir)

    def close_gui(self):
        logging.info('Emitting close-gui signal')
        self.emit('close-gui')

def exception_handler(self, etype, value, tb, tb_offset=None):
    # when the 'tb_offset' keyword argument is omitted above, ipython
    # raises an error.
    TB = AutoFormattedTB(mode='Context', color_scheme='Linux', tb_offset=1)

    fc = get_flowcontrol()

    # for the unlikely event that an error occured after pushing the
    # stop button
    fc._abort = False

    if fc.is_measuring():
        # make sure that open objects get closed properly (like files)
        fc.emit('stop-request')

        # put qtlab back in 'stopped' state
        fc.measurement_end(abort=True)

    TB(etype, value, tb)

try:
    _flowcontrol
except NameError:
    _flowcontrol = FlowControl()

def get_flowcontrol():
    global _flowcontrol
    return _flowcontrol

def qtlab_exit():
    print("Closing QTlab...")

    import qt
    qt.flow.exit_request()

