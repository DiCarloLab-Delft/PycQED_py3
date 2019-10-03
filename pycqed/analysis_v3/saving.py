import logging
log = logging.getLogger(__name__)

import os
import json
import lmfit
import h5py
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import OrderedDict
from pycqed.utilities.general import NumpyJsonEncoder
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement.hdf5_data import write_dict_to_hdf5


def save_data(self, data_dict, savedir: str = None, savebase: str = None,
              tag_tstamp: bool = True, fmt: str = 'json', key_list='auto'):
    '''
    Saves the data from data_dict to file.

    Args:
        savedir (string):
                Directory where the file is saved. If this is None, the
                file is saved in self.raw_data_dict['folder'] or the
                working directory of the console.
        savebase (string):
                Base name for the saved file.
        tag_tstamp (bool):
                Whether to append the timestamp of the first to the base
                name.
        fmt (string):
                File extension for the format in which the file should
                be saved.
        key_list (list or 'auto'):
                Specifies which keys from self.raw_data_dict are saved.
                If this is 'auto' or None, all keys-value pairs are
                saved.
    '''
    if savedir is None:
        savedir = self.raw_data_dict.get('folder', '')
        if isinstance(savedir, list):
            savedir = savedir[0]
    if savebase is None:
        savebase = ''
    if tag_tstamp:
        tstag = '_' + self.raw_data_dict['timestamp'][0]
    else:
        tstag = ''

    if key_list == 'auto' or key_list is None:
        key_list = data_dict.keys()

    save_dict = {}
    for k in key_list:
        save_dict[k] = data_dict[k]

    try:
        os.mkdir(savedir)
    except FileExistsError:
        pass

    filepath = os.path.join(savedir, savebase + tstag + '.' + fmt)
    if self.verbose:
        log.info('Saving raw data to %s' % filepath)
    with open(filepath, 'w') as file:
        json.dump(save_dict, file, cls=NumpyJsonEncoder, indent=4)
    log.info('Data saved to "{}".'.format(filepath))


def save_processed_data(self, key=None, overwrite=True):
    """
    Saves data from the processed data dictionary to the hdf5 file

    Args:
        key: key of the data to save. All processed data is saved by
             default.
    """

    if isinstance(key, (list, set)):
        for k in key:
            self.save_processed_data(k)
        return

    # Check weather there is any data to save
    if hasattr(self, 'proc_data_dict') and self.proc_data_dict is not None \
            and key in self.proc_data_dict:
        fn = self.options_dict.get('analysis_result_file', False)
        if fn == False:
            if isinstance(self.raw_data_dict, tuple):
                timestamp = self.raw_data_dict[0]['timestamp']
            else:
                timestamp = self.raw_data_dict['timestamp']
            fn = a_tools.measurement_filename(a_tools.get_folder(
                timestamp))
        try:
            os.mkdir(os.path.dirname(fn))
        except FileExistsError:
            pass

        if self.verbose:
            log.info('Saving fitting results to %s' % fn)

        with h5py.File(fn, 'a') as data_file:
            try:
                analysis_group = data_file.create_group('Analysis')
            except ValueError:
                # If the analysis group already exists.
                analysis_group = data_file['Analysis']

            try:
                proc_data_group = \
                    analysis_group.create_group('Processed data')
            except ValueError:
                # If the processed data group already exists.
                proc_data_group = analysis_group['Processed data']

            if key in proc_data_group.keys():
                del proc_data_group[key]

            d = {key: self.proc_data_dict[key]}
            write_dict_to_hdf5(d, entry_point=proc_data_group,
                               overwrite=overwrite)


def save_fit_results(data_dict, fit_res_dict, **params):
    """
    Saves the fit results
    """
    timestamp = data_dict['timestamp']
    fn = a_tools.measurement_filename(a_tools.get_folder(timestamp))

    try:
        os.mkdir(os.path.dirname(fn))
    except FileExistsError:
        pass

    if params.get('verbose', False):
        log.info('Saving fitting results to %s' % fn)

    with h5py.File(fn, 'a') as data_file:
        try:
            analysis_group = data_file.create_group('Analysis')
        except ValueError:
            # If the analysis group already exists.
            analysis_group = data_file['Analysis']

        # Iterate over all the fit result dicts as not to overwrite
        # old/other analysis
        for fr_key, fit_res in fit_res_dict.items():
            try:
                fr_group = analysis_group.create_group(fr_key)
            except ValueError:
                # If the analysis sub group already exists
                # (each fr_key should be unique)
                # Delete the old group and create a new group
                # (overwrite).
                del analysis_group[fr_key]
                fr_group = analysis_group.create_group(fr_key)

            d = _convert_dict_rec(deepcopy(fit_res))
            write_dict_to_hdf5(d, entry_point=fr_group)


def save_figures(data_dict, figs, **params):

    keys_in = params.get('keys_in', 'auto')
    fmt = params.get('fmt', 'png')
    dpi = params.get('dpi', 300)
    tag_tstamp = params.get('tag_tstamp', True)
    savebase = params.get('savebase', None)
    savedir = params.get('savedir', None)
    if savedir is None:
        if isinstance(data_dict, tuple):
            savedir = data_dict[0].get('folder', '')
        else:
            savedir = data_dict.get('folder', '')

        if isinstance(savedir, list):
            savedir = savedir[0]
        if isinstance(savedir, list):
            savedir = savedir[0]
    if savebase is None:
        savebase = ''
    if tag_tstamp:
        if isinstance(data_dict, tuple):
            tstag = '_' + data_dict[0]['timestamp']
        else:
            tstag = '_' + data_dict['timestamp']
    else:
        tstag = ''

    if keys_in == 'auto' or keys_in is None:
        keys_in = figs.keys()

    try:
        os.mkdir(savedir)
    except FileExistsError:
        pass

    if params.get('verbose', False):
        log.info('Saving figures to %s' % savedir)

    for key in keys_in:
        if params.get('presentation_mode', False):
            savename = os.path.join(savedir, savebase + key + tstag +
                                    'presentation' + '.' + fmt)
            figs[key].savefig(savename, bbox_inches='tight',
                                   fmt=fmt, dpi=dpi)
            savename = os.path.join(savedir, savebase + key + tstag +
                                    'presentation' + '.svg')
            figs[key].savefig(savename, bbox_inches='tight', fmt='svg')
        else:
            savename = os.path.join(savedir, savebase + key + tstag
                                    + '.' + fmt)
            figs[key].savefig(savename, bbox_inches='tight',
                                   fmt=fmt, dpi=dpi)
        if params.get('close_figs', True):
            plt.close(figs[key])


def _convert_dict_rec(obj):
    try:
        # is iterable?
        for k in obj:
            obj[k] = _convert_dict_rec(obj[k])
    except TypeError:
        if isinstance(obj, lmfit.model.ModelResult):
            obj = _flatten_lmfit_modelresult(obj)
        else:
            obj = str(obj)
    return obj


def _flatten_lmfit_modelresult(model):
    assert type(model) is lmfit.model.ModelResult
    dic = OrderedDict()
    dic['success'] = model.success
    dic['message'] = model.message
    dic['params'] = {}
    for param_name in model.params:
        dic['params'][param_name] = {}
        param = model.params[param_name]
        for k in param.__dict__:
            if k == '_val':
                dic['params'][param_name]['value'] = getattr(param, k)
            else:
                if not k.startswith('_') and k not in ['from_internal', ]:
                    dic['params'][param_name][k] = getattr(param, k)
    return dic
