import re
import logging
log = logging.getLogger(__name__)


##################################################
#### This module creates an analysis pipeline ####
##################################################


class ProcessingPipeline(list):
    def __init__(self, node_type=None, **params):
        super().__init__()
        if node_type is not None:
            # self.append(eval(('add_' + node_type + '_node')(**params)))
            self.append(getattr(self, 'add_' + node_type + '_node')(**params))

    def add_node(self, node_type, **params):
        if hasattr(self, 'add_' + node_type + '_node'):
            self.append(getattr(self, 'add_' + node_type + '_node')(**params))
        else:
            params['node_type'] = node_type
            self.append(params)

    def check_keys_in(self, keys_in=None):
        if keys_in == 'previous':
            if len(self) > 0:
                keys_in = self[-1]['keys_out']
            else:
                raise ValueError('This is the first node in the pipeline. '
                                 'keys_in must be specified.')
        return keys_in

    def add_filter_data_node(self, reset_reps, keys_in=None, **params):
        keys_in = self.check_keys_in(keys_in=keys_in)
        return {'node_type': 'filter_data',
                'keys_in': keys_in,
                'keys_out': [k+' filtered' for k in keys_in],
                'data_filter': f'lambda x: x[{reset_reps}::{reset_reps}+1]',
                **params}

    def add_average_node(self, num_bins, keys_in=None, **params):
        keys_in = self.check_keys_in(keys_in)
        return {'node_type': 'average',
                'keys_in': keys_in,
                'keys_out': [k + ' averaged' for k in keys_in],
                'num_bins': num_bins,
                **params}

    def add_get_std_deviation_node(self, num_bins, keys_in=None,
                                   **params):
        keys_in = self.check_keys_in(keys_in)
        return {'node_type': 'get_std_deviation',
                'keys_in': keys_in,
                'keys_out': [k + ' std' for k in keys_in],
                'num_bins': num_bins,
                **params}

    def add_rotate_iq_node(self, keys_in=None, meas_obj_name='', **params):
        keys_in = self.check_keys_in(keys_in)
        return {'node_type': 'rotate_iq',
                'keys_in': keys_in,
                'keys_out': ['rotated data ' + '-'.join(k) for k in keys_in],
                'meas_obj_name': meas_obj_name,
                **params}

    def add_rotate_1d_array_node(self, keys_in=None, meas_obj_name='',
                                 **params):
        keys_in = self.check_keys_in(keys_in)
        return {'node_type': 'rotate_1d_array',
                'keys_in': keys_in,
                'keys_out': [f'rotated data {k}' for k in keys_in],
                'meas_obj_name': meas_obj_name,
                **params}

    ######################################
    #### plot dicts preparation nodes ####
    ######################################

    def add_prepare_1d_plot_dicts_node(
            self, keys_in=None, meas_obj_name='', fig_name='',
            do_plotting=False, **params):
        keys_in = self.check_keys_in(keys_in)
        return {'node_type': 'prepare_1d_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_name': meas_obj_name,
                'fig_name': fig_name,
                'do_plotting': do_plotting,
                **params}

    def add_prepare_raw_data_plot_dicts_node(
            self, keys_in='previous', meas_obj_name='', fig_name='',
            do_plotting=False, **params):
        keys_in = self.check_keys_in(keys_in)
        return {'node_type': 'prepare_raw_data_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_name': meas_obj_name,
                'fig_name': fig_name,
                'do_plotting': do_plotting,
                **params}

    def add_prepare_cal_states_plot_dicts_node(
            self, keys_in='previous', meas_obj_name='', fig_name='',
            do_plotting=False, **params):
        keys_in = self.check_keys_in(keys_in)
        return {'node_type': 'prepare_cal_states_plot_dicts',
                'keys_in': keys_in,
                'meas_obj_name': meas_obj_name,
                'fig_name': fig_name,
                'do_plotting': do_plotting,
                **params}

    ################################
    #### nodes that are classes ####
    ################################

    def add_RabiAnalysis_node(self, meas_obj_name, keys_in='previous', **params):
        keys_in = self.check_keys_in(keys_in)
        return {'node_type': 'RabiAnalysis',
                'keys_in': keys_in,
                'meas_obj_name': meas_obj_name,
                **params}

    def add_SingleQubitRBAnalysis_node(self, meas_obj_name, keys_in='previous',
                                       std_keys=None, **params):
        keys_in = self.check_keys_in(keys_in)
        return {'node_type': 'SingleQubitRBAnalysis',
                'keys_in': keys_in,
                'std_keys': std_keys,
                'meas_obj_name': meas_obj_name,
                **params}



