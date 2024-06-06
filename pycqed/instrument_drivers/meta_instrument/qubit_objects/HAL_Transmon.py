"""
File:   HAL_Transmon.py (originally CCL_Transmon.py)
Note:   a lot code was moved around within this file in December 2021. As a consequence, the author information provided
        by 'git blame' makes little sense. See GIT tag 'release_v0.3' for the original file.
"""

import time
import logging
import numpy as np
import warnings
import pytest
import cma
import datetime
import multiprocessing
from deprecated import deprecated
from typing import List, Union, Optional

from pycqed.instrument_drivers.meta_instrument.HAL.HAL_ShimSQ import HAL_ShimSQ

from pycqed.measurement import calibration_toolbox as cal_toolbox
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det

from pycqed.measurement.mc_parameter_wrapper import wrap_par_to_swf
import pycqed.measurement.composite_detector_functions as cdf
from pycqed.measurement.optimization import nelder_mead

from pycqed.measurement.openql_experiments import single_qubit_oql as sqo
import pycqed.measurement.openql_experiments.multi_qubit_oql as mqo
from pycqed.measurement.openql_experiments import clifford_rb_oql as cl_oql
from pycqed.measurement.openql_experiments import pygsti_oql
from pycqed.measurement.openql_experiments import openql_helpers as oqh
from pycqed.measurement.openql_experiments.openql_helpers import \
    load_range_of_oql_programs, load_range_of_oql_programs_from_filenames

from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.tools import cryoscope_tools as ct
from pycqed.analysis.tools import plotting as plt_tools
from pycqed.analysis_v2 import measurement_analysis as ma2

from pycqed.utilities.general import gen_sweep_pts
from pycqed.utilities.learnerND_minimizer import LearnerND_Minimizer, \
    mk_minimization_loss_func, mk_minimization_goal_func

# Imported for type annotations
from pycqed.measurement.measurement_control import MeasurementControl

from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter

log = logging.getLogger(__name__)


class HAL_Transmon(HAL_ShimSQ):
    """
    The HAL_Transmon (formerly known as CCL_Transmon)
    Setup configuration:
        Drive:                 CC controlling AWG8's (and historically a VSM)
        Acquisition:           UHFQC
        Readout pulse configuration: LO modulated using UHFQC AWG
    """

    def __init__(self, name, **kw):
        t0 = time.time()
        super().__init__(name, **kw)

        self.add_parameters()
        self.connect_message(begin_time=t0)

    ##########################################################################
    # Overrides for class Qubit
    ##########################################################################

    def add_instrument_ref_parameters(self):
        # NB: these are now handled in class HAL_ShimSQ
        pass

    def add_ro_parameters(self):
        """
        Adding the parameters relevant for readout.
        """


    def add_mw_parameters(self):
        # parameters for *MW_LutMan
        self.add_parameter(
            'mw_channel_amp',  # FIXME: actually sets a (dimensionless) *gain* relative to the available hardware amplitude, not an *amplitude*
            label='AWG channel amplitude. WARNING: Check your hardware specific limits!',
            unit='',
            initial_value=.5,
            vals=vals.Numbers(min_value=0, max_value=1.6),
            parameter_class=ManualParameter)

        # parameters for *MW_LutMan: pulse attributes
        self.add_parameter(
            'mw_amp180',  # FIXME: appears to have been replaced by mw_channel_amp (to allow iterating without waveform reloading), but is still present all over the place
            label='Pi-pulse amplitude',
            unit='V',
            initial_value=.8,
            parameter_class=ManualParameter)
        self.add_parameter(
            'mw_amp90_scale',
            label='pulse amplitude scaling factor',
            unit='',
            initial_value=.5,
            vals=vals.Numbers(min_value=0, max_value=1.0),
            parameter_class=ManualParameter)
        self.add_parameter(
            'mw_ef_amp',
            label='Pi-pulse amplitude ef-transition',
            unit='V',
            initial_value=.4,
            parameter_class=ManualParameter)
        self.add_parameter(
            'mw_gauss_width', unit='s',
            initial_value=10e-9,
            parameter_class=ManualParameter)
        self.add_parameter(
            'mw_motzoi', label='Motzoi parameter',
            unit='',
            initial_value=0,
            parameter_class=ManualParameter)

    def add_spec_parameters(self):
        # parameters for *MW_LutMan
        self.add_parameter(
            'spec_amp',
            unit='V',
            docstring=(
                'Amplitude of the spectroscopy pulse in the mw LutMan. '
                'The power of the spec pulse should be controlled through '
                'the vsm amplitude "spec_vsm_amp"'),  # FIXME: outdated
            vals=vals.Numbers(0, 1),
            parameter_class=ManualParameter,
            initial_value=0.8)

        # other parameters
        # FIXME: unused
        # self.add_parameter(
        #     'spec_vsm_amp',
        #     label='VSM amplitude for spec pulses',
        #     vals=vals.Numbers(0.1, 1.0),
        #     initial_value=1.0,
        #     parameter_class=ManualParameter)

        self.add_parameter(
            'spec_pulse_length',
            label='Pulsed spec pulse duration',
            unit='s',
            vals=vals.Numbers(0e-9, 50e-6),  # FIXME validator: should be multiple of 20e-9
            initial_value=500e-9,
            parameter_class=ManualParameter)

        # FIXME: unused
        # self.add_parameter(
        #     'spec_type',
        #     parameter_class=ManualParameter,
        #     docstring=(
        #         'determines what kind of spectroscopy to do, \n'
        #         '"CW":  opens the relevant VSM channel to always let the tone '
        #         'through. \n'
        #         '"vsm_gated":  uses the  VSM in external mode to gate the spec '
        #         'source. \n '
        #         '"IQ" uses the TD source and AWG8 to generate a spec pulse'),
        #     initial_value='CW',
        #     vals=vals.Enum('CW', 'IQ', 'vsm_gated'))

        # NB: only used in _measure_spectroscopy_pulsed_marked()
        self.add_parameter(
            'spec_wait_time',
            unit='s',
            vals=vals.Numbers(0, 100e-6),
            parameter_class=ManualParameter,
            initial_value=0)

    def add_flux_parameters(self):
        # fl_dc_ is the prefix for DC flux bias related params
        self.add_parameter(
            'fl_dc_polycoeff',
            docstring='Polynomial coefficients for current to frequency conversion',
            vals=vals.Arrays(),
            # initial value is chosen to not raise errors
            initial_value=np.array([0, 0, -1e12, 0, 6e9]),
            parameter_class=ManualParameter)

        # FIXME: unused
        # self.add_parameter(
        #     'fl_ac_polycoeff',
        #     docstring='Polynomial coefficients for current to frequency conversion',
        #     vals=vals.Arrays(),
        #     # initial value is chosen to not raise errors
        #     initial_value=np.array([0, 0, -1e12, 0, 6e9]),
        #     parameter_class=ManualParameter)

        self.add_parameter(
            'fl_dc_I_per_phi0',
            label='Flux bias I/Phi0',
            docstring='Conversion factor for flux bias, current per flux quantum',
            vals=vals.Numbers(),
            unit='A',
            initial_value=10e-3,
            parameter_class=ManualParameter)
        self.add_parameter(
            'fl_dc_I',
            label='Flux bias',
            unit='A',
            docstring='Current flux bias setting',
            vals=vals.Numbers(),
            initial_value=0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'fl_dc_I0',
            unit='A',
            label='Flux bias sweet spot',
            docstring=('Flux bias offset corresponding to the sweetspot'),
            vals=vals.Numbers(),
            initial_value=0,
            parameter_class=ManualParameter)

        if 0:  # FIXME: unused
            # Currently this has only the parameters for 1 CZ gate.
            # in the future there will be 5 distinct flux operations for which
            # parameters have to be stored.
            # cz to all nearest neighbours (of which 2 are only phase corr) and
            # the "park" operation.
            self.add_parameter(
                'fl_cz_length',
                vals=vals.Numbers(),
                unit='s',
                initial_value=35e-9,
                parameter_class=ManualParameter)
            self.add_parameter(
                'fl_cz_lambda_2',
                vals=vals.Numbers(),
                initial_value=0,
                parameter_class=ManualParameter)
            self.add_parameter(
                'fl_cz_lambda_3',
                vals=vals.Numbers(),
                initial_value=0,
                parameter_class=ManualParameter)
            self.add_parameter(
                'fl_cz_theta_f',
                vals=vals.Numbers(),
                unit='deg',
                initial_value=80,
                parameter_class=ManualParameter)
            self.add_parameter(
                'fl_cz_V_per_phi0',
                vals=vals.Numbers(),
                unit='V',
                initial_value=1,
                parameter_class=ManualParameter)
            self.add_parameter(
                'fl_cz_freq_01_max',
                vals=vals.Numbers(),
                unit='Hz',
                parameter_class=ManualParameter)
            self.add_parameter(
                'fl_cz_J2',
                vals=vals.Numbers(),
                unit='Hz',
                initial_value=50e6,
                parameter_class=ManualParameter)
            self.add_parameter(
                'fl_cz_freq_interaction',
                vals=vals.Numbers(),
                unit='Hz',
                parameter_class=ManualParameter)
            self.add_parameter(
                'fl_cz_phase_corr_length',
                unit='s',
                initial_value=5e-9,
                vals=vals.Numbers(),
                parameter_class=ManualParameter)
            self.add_parameter(
                'fl_cz_phase_corr_amp',
                unit='V',
                initial_value=0, vals=vals.Numbers(),
                parameter_class=ManualParameter)

    def add_config_parameters(self):
        # FIXME: unused
        # self.add_parameter(
        #     'cfg_trigger_period',
        #     label='Trigger period',
        #     docstring=(
        #         'Time between experiments, used to initialize all'
        #         ' qubits in the ground state'),
        #     unit='s',
        #     initial_value=200e-6,
        #     parameter_class=ManualParameter,
        #     vals=vals.Numbers(min_value=1e-6, max_value=327668e-9))

        self.add_parameter(
            'cfg_openql_platform_fn',
            label='OpenQL platform configuration filename',
            parameter_class=ManualParameter,
            vals=vals.Strings())

        self.add_parameter(
            'cfg_qubit_freq_calc_method',
            initial_value='latest',
            parameter_class=ManualParameter,
            vals=vals.Enum('latest', 'flux'))

        self.add_parameter(
            'cfg_rb_calibrate_method',
            initial_value='restless',
            parameter_class=ManualParameter,
            vals=vals.Enum('restless', 'ORBIT'))

        self.add_parameter(
            'cfg_cycle_time',
            initial_value=20e-9,
            unit='s',
            parameter_class=ManualParameter,
            # this is to effectively hardcode the cycle time
            vals=vals.Enum(20e-9))

    def add_generic_qubit_parameters(self):
        self.add_parameter(
            'E_c',
            unit='Hz',
            initial_value=300e6,
            parameter_class=ManualParameter,
            vals=vals.Numbers())
        self.add_parameter(
            'E_j',
            unit='Hz',
            parameter_class=ManualParameter,
            vals=vals.Numbers())
        self.add_parameter(
            'T1',
            unit='s',
            parameter_class=ManualParameter,
            vals=vals.Numbers(0, 200e-6))
        self.add_parameter(
            'T2_echo',
            unit='s',
            parameter_class=ManualParameter,
            vals=vals.Numbers(0, 200e-6))
        self.add_parameter(
            'T2_star',
            unit='s',
            parameter_class=ManualParameter,
            vals=vals.Numbers(0, 200e-6))

        self.add_parameter(
            'freq_max',
            label='qubit sweet spot frequency',
            unit='Hz',
            parameter_class=ManualParameter)
        self.add_parameter(
            'freq_res',
            label='Resonator frequency',
            unit='Hz',
            parameter_class=ManualParameter)

        self.add_parameter(
            'asymmetry',
            unit='',
            docstring='Asymmetry parameter of the SQUID loop',
            initial_value=0,
            parameter_class=ManualParameter)
        self.add_parameter(
            'anharmonicity',
            unit='Hz',
            label='Anharmonicity',
            docstring='Anharmonicity, negative by convention',
            parameter_class=ManualParameter,
            # typical target value
            initial_value=-300e6,
            vals=vals.Numbers())
        self.add_parameter(
            'dispersive_shift',
            label='Resonator dispersive shift',
            unit='Hz',
            parameter_class=ManualParameter,
            vals=vals.Numbers())

        # output parameters for some experiments
        self.add_parameter(
            'F_ssro',
            initial_value=0,
            label='Single shot readout assignment fidelity',
            vals=vals.Numbers(0.0, 1.0),
            parameter_class=ManualParameter)
        self.add_parameter('F_init',
            initial_value=0,
            label='Single shot readout initialization fidelity',
            vals=vals.Numbers(0.0, 1.0),
            parameter_class=ManualParameter)
        self.add_parameter(
            'F_discr',
            initial_value=0,
            label='Single shot readout discrimination fidelity',
            vals=vals.Numbers(0.0, 1.0),
            parameter_class=ManualParameter)

        self.add_parameter(
            'ro_rel_events',
            initial_value=0,
            label='relaxation errors from ssro fit',
            vals=vals.Numbers(0.0, 1.0),
            parameter_class=ManualParameter)
        self.add_parameter(
            'ro_res_ext',
            initial_value=0,
            label='residual excitation errors from ssro fit',
            vals=vals.Numbers(0.0, 1.0),
            parameter_class=ManualParameter)
        self.add_parameter('F_RB',
            initial_value=0,
            label='RB single-qubit Clifford fidelity',
            vals=vals.Numbers(0, 1.0),
            parameter_class=ManualParameter)
        # I believe these were first added by Miguel. 
        # To my knowledge, only Quantum Inspire uses them.
        # LDC, 2022/06/24
        for cardinal in ['NW','NE','SW','SE']:
            self.add_parameter(f'F_2QRB_{cardinal}',
                initial_value=0,
                label=f'RB two-qubit Clifford fidelity for edge {cardinal}',
                vals=vals.Numbers(0, 1.0),
                parameter_class=ManualParameter)
        # LDC adding parameter to keep track of two-qubit phases. 
        # These are used by Quantum Inspire. 
        # 2022/06/24.
        for cardinal in ['NW','NE','SW','SE']:
            self.add_parameter(f'CZ_two_qubit_phase_{cardinal}',
                initial_value=0,
                label=f'Two-qubit phase for CZ on edge {cardinal}',
                vals=vals.Numbers(0, 360),
                parameter_class=ManualParameter)

    ##########################################################################
    # find_ functions (HAL_Transmon specific)
    ##########################################################################

    def find_frequency_adaptive(
            self,
            f_start=None,
            f_span=1e9,
            f_step=0.5e6,
            MC: Optional[MeasurementControl] = None,
            update=True,
            use_max=False,
            spec_mode='pulsed_marked',
            verbose=True
    ) -> bool:
        # USED_BY: device_dependency_graphs.py
        """
        'Adaptive' measurement for finding the qubit frequency. Will look with
        a range of the current frequency estimate, and if it does not find a
        peak it will move and look f_span Hz above and below the estimate. Will
        continue to do such a shift until a peak is found.
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        if f_start is None:
            f_start = self.freq_qubit()

        # Set high power and averages to be sure we find the peak.
        # FIXME: code commented out
        # self.spec_pow(-30)
        # self.ro_pulse_amp_CW(0.025)
        # old_avg = self.ro_acq_averages()
        # self.ro_acq_averages(2**15)

        # Repeat measurement while no peak is found:
        success = False
        f_center = f_start
        n = 0
        while not success:
            success = None
            f_center += f_span * n * (-1) ** n
            n += 1
            if verbose:
                cfreq, cunit = plt_tools.SI_val_to_msg_str(f_center, 'Hz', float)
                sfreq, sunit = plt_tools.SI_val_to_msg_str(f_span, 'Hz', float)
                print('Doing adaptive spectroscopy around {:.3f} {} with a '
                      'span of {:.0f} {}.'.format(cfreq, cunit, sfreq, sunit))

            freqs = np.arange(f_center - f_span / 2, f_center + f_span / 2, f_step)

            self.measure_spectroscopy(MC=MC, freqs=freqs, mode=spec_mode,
                                      analyze=False)
            label = 'spec'

            # Use 'try' because it can give a TypeError when no peak is found
            try:
                analysis_spec = ma.Qubit_Spectroscopy_Analysis(
                    label=label,
                    close_fig=True,
                    qb_name=self.name
                )
            except TypeError:
                logging.warning('TypeError in Adaptive spectroscopy')
                continue

            # Check for peak and check its height
            freq_peak = analysis_spec.peaks['peak']
            offset = analysis_spec.fit_res.params['offset'].value
            peak_height = np.amax(analysis_spec.data_dist)

            # Check if peak is not another qubit, and if it is, move that qubit away
            for qubit_name in self.instr_device.get_instr().qubits():
                qubit = self.instr_device.get_instr().find_instrument(qubit_name)
                if qubit.name != self.name and qubit.freq_qubit() is not None:
                    if np.abs(qubit.freq_qubit() - freq_peak) < 5e6:
                        if verbose:
                            logging.warning('Peak found at frequency of {}. '
                                            'Adjusting currents'
                                            .format(qubit.name))
                        fluxcurrent = self.instr_FluxCtrl.get_instr()
                        old_current = fluxcurrent[qubit.fl_dc_ch()]()
                        fluxcurrent[qubit.fl_dc_ch()](5e-3)
                        n -= 1
                        success = False

            if success is None:
                if freq_peak is None:
                    success = False
                elif peak_height < 4 * offset:
                    success = False
                elif peak_height < 3 * np.mean(analysis_spec.data_dist):
                    success = False
                else:
                    success = True

        # self.ro_acq_averages(old_avg)
        if update:
            if use_max:
                self.freq_qubit(analysis_spec.peaks['peak'])
            else:
                self.freq_qubit(analysis_spec.fitted_freq)
            return True

    def find_qubit_sweetspot(
            self,
            freqs=None,
            dac_values=None,
            update=True,
            set_to_sweetspot=True,
            method='DAC',
            fluxChan=None,
            spec_mode='pulsed_marked'
    ):
        # USED_BY: device_dependency_graphs.py
        """
        Should be edited such that it contains reference to different measurement
        methods (tracking / 2D scan / broad spectroscopy)

        method = 'DAC' - uses ordinary 2D DAC scan
                 'tracked - uses tracked spectroscopy (not really implemented)'
        TODO: If spectroscopy does not yield a peak, it should discard it
        """

        if freqs is None:
            freq_center = self.freq_qubit()
            freq_range = 50e6
            freqs = np.arange(freq_center - freq_range, freq_center + freq_range, 1e6)
        if dac_values is None:
            if self.fl_dc_I0() is not None:
                dac_values = np.linspace(self.fl_dc_I0() - 1e-3,
                                         self.fl_dc_I0() + 1e-3, 8)
            else:
                dac_values = np.linspace(-0.5e3, 0.5e-3, 10)

        if fluxChan is None:
            if self.fl_dc_ch() is not None:
                fluxChan = self.fl_dc_ch()
            else:
                logging.error('No fluxchannel found or specified. Please '
                              'specify fluxChan')

        if method == 'DAC':
            t_start = time.strftime('%Y%m%d_%H%M%S')
            self.measure_qubit_frequency_dac_scan(
                freqs=freqs,
                dac_values=dac_values,
                fluxChan=fluxChan,
                analyze=False,
                mode=spec_mode,
                nested_resonator_calibration=False,
                # nested_resonator_calibration_use_min=False,
                resonator_freqs=np.arange(-5e6, 5e6, 0.2e6) + self.freq_res()
            )

            timestamp = a_tools.get_timestamps_in_range(
                t_start,
                label='Qubit_dac_scan' + self.msmt_suffix)
            timestamp = timestamp[0]
            a = ma2.da.DAC_analysis(timestamp=timestamp)
            self.fl_dc_polycoeff(a.dac_fit_res['fit_polycoeffs'])
            sweetspot_current = a.dac_fit_res['sweetspot_dac']

        elif method == 'tracked':
            t_start = time.strftime('%Y%m%d_%H%M%S')

            for i, dac_value in enumerate(dac_values):
                self.instr_FluxCtrl.get_instr()[self.fl_dc_ch()](dac_value)
                if i == 0:
                    self.find_frequency(freqs=freqs, update=True)
                else:
                    self.find_frequency(update=True)

            t_end = time.strftime('%Y%m%d_%H%M%S')

            a = ma2.DACarcPolyFit(
                t_start=t_start,
                t_stop=t_end,
                label='spectroscopy__' + self.name,
                dac_key='Instrument settings.fluxcurrent.' + self.fl_dc_ch(),
                degree=2)

            pc = a.fit_res['fit_polycoeffs']

            self.fl_dc_polycoeff(pc)
            sweetspot_current = -pc[1] / (2 * pc[0])

        else:
            logging.error('Sweetspot method {} unknown. Use "DAC" or "tracked".'.format(method))

        if update:
            self.fl_dc_I0(sweetspot_current)
            self.freq_max(self.calc_current_to_freq(sweetspot_current))
        if set_to_sweetspot:
            self.instr_FluxCtrl.get_instr()[self.fl_dc_ch()](sweetspot_current)

        # Sanity check: does this peak move with flux?
        check_vals = [self.calc_current_to_freq(np.min(dac_values)),
                      self.calc_current_to_freq(self.fl_dc_I0()),
                      self.calc_current_to_freq(np.max(dac_values))]

        if check_vals[0] == pytest.approx(check_vals[1], abs=0.5e6):
            if check_vals[0] == pytest.approx(check_vals[2], abs=0.5e6):
                if check_vals[1] == pytest.approx(check_vals[2], abs=0.5e6):
                    logging.warning('No qubit shift found with varying flux. Peak is not a qubit')
                    return False

        if self.fl_dc_polycoeff()[1] < 1e6 and self.fl_dc_polycoeff()[2] < 1e6:
            logging.warning('No qubit shift found with varying flux. Peak is not a qubit')
            return False

        return True

    def find_qubit_sweetspot_1D(self, freqs=None, dac_values=None):

        # self.spec_pow(-30)
        self.ro_acq_averages(2 ** 14)

        if dac_values is None:
            if self.fl_dc_I0() is not None:
                dac_values = np.linspace(self.fl_dc_I0() - 1e-3,
                                         self.fl_dc_I0() + 1e-3, 8)
            else:
                dac_values = np.linspace(-1e3, 1e-3, 8)

        if freqs is None:
            freq_center = self.freq_qubit()
            freq_range = 50e6
            freqs = np.arange(freq_center - freq_range, freq_center + freq_range, 0.5e6)
        Qubit_frequency = []
        Reson_frequency = []
        flux_channel = self.fl_dc_ch()

        for dac_value in dac_values:
            # Set Flux Current
            self.instr_FluxCtrl.get_instr()[flux_channel](dac_value)

            # Find Resonator
            self.find_resonator_frequency(freqs=np.arange(-5e6, 5.1e6, .1e6) + self.freq_res(), use_min=True)
            # Find Qubit frequency
            self.find_frequency(freqs=freqs)

            Qubit_frequency.append(self.freq_qubit())
            Reson_frequency.append(self.freq_res())

        # Fit sweetspot with second degree polyfit
        fit_coefs = np.polyfit(dac_values, Qubit_frequency, deg=2)
        sweetspot_current = fit_coefs[1] / (2 * fit_coefs[0])

        # Set Flux Current to sweetspot
        self.instr_FluxCtrl.get_instr()[flux_channel](sweetspot_current)
        self.find_resonator_frequency(freqs=np.arange(-5e6, 5.1e6, .1e6) + self.freq_res(),
                                      use_min=True)
        frequency_sweet_spot = self.find_frequency(
            freqs=np.arange(-50e6, 50e6, .5e6) + self.freq_qubit())

        return frequency_sweet_spot

    def find_anharmonicity_estimate(
            self, freqs=None,
            anharmonicity=None,
            mode='pulsed_marked',
            update=True,
            power_12=10
    ):
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs.py
        """
        Finds an estimate of the anharmonicity by doing a spectroscopy around
        150 MHz below the qubit frequency.

        TODO: if spec_pow is too low/high, it should adjust it to approx the
              ideal spec_pow + 25 dBm
        """

        if anharmonicity is None:
            # Standard estimate, negative by convention
            anharmonicity = self.anharmonicity()

        f02_estimate = self.freq_qubit() * 2 + anharmonicity

        if freqs is None:
            freq_center = f02_estimate / 2
            freq_range = 175e6
            freqs = np.arange(freq_center - 1 / 2 * freq_range, self.freq_qubit() + 1 / 2 * freq_range, 0.5e6)
        old_spec_pow = self.spec_pow()
        self.spec_pow(self.spec_pow() + power_12)

        self.measure_spectroscopy(freqs=freqs, mode=mode, analyze=False)

        a = ma.Qubit_Spectroscopy_Analysis(label=self.msmt_suffix,
                                           analyze_ef=True)
        self.spec_pow(old_spec_pow)
        f02 = 2 * a.params['f0_gf_over_2'].value
        if update:
            self.anharmonicity(f02 - 2 * self.freq_qubit())
            return True

    def find_bus_frequency(
            self,
            freqs,
            spec_source_bus,
            bus_power,
            f01=None,
            label='',
            close_fig=True,
            analyze=True,
            MC: Optional[MeasurementControl] = None,
            prepare_for_continuous_wave=True
    ):
        """
        Drive the qubit and sit at the spectroscopy peak while the bus is driven with
        bus_spec_source

        Args:
            freqs (array):
                list of frequencies of the second drive tone (at bus frequency)

            spec_source_bus (RohdeSchwarz_SGS100A):
                rf source used for the second spectroscopy tone

            bus_power (float):
                power of the second spectroscopy tone

            f_01 (float):
                frequency of 01 transition (default: self.freq_qubit())

            analyze (bool):
                indicates whether to look for peas in the data and perform a fit

            label (str):
                suffix to append to the measurement label

            prepare_for_continuous_wave (bool):
                indicates whether to regenerate a waveform
                generating a readout tone and set all the instruments according
                to the parameters stored in the qubit object
        """

        if f01 is None:
            f01 = self.freq_qubit()

        if prepare_for_continuous_wave:
            self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()

        self.hal_acq_spec_mode_on()

        p = sqo.pulsed_spec_seq(
            qubit_idx=self.cfg_qubit_nr(),
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        spec_source = self.instr_spec_source.get_instr()
        spec_source.on()
        spec_source.frequency(f01)
        # spec_source.power(self.spec_pow())

        spec_source_bus.on()
        spec_source_bus.power(bus_power)

        MC.set_sweep_function(spec_source_bus.frequency)
        MC.set_sweep_points(freqs)
        if self.cfg_spec_mode():
            print('Enter loop')
            MC.set_detector_function(self.UHFQC_spec_det)
        else:
            self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
            MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='Bus_spectroscopy_' + self.msmt_suffix + label)
        spec_source_bus.off()

        self.hal_acq_spec_mode_off()

        if analyze:
            ma.Qubit_Spectroscopy_Analysis(label=self.msmt_suffix,
                                           close_fig=close_fig,
                                           qb_name=self.name)

    ##########################################################################
    # calibrate_ functions (HAL_Transmon specific)
    ##########################################################################

    def calibrate_ro_pulse_amp_CW(self, freqs=None, powers=None, update=True):
        # USED_BY: device_dependency_graphs.py
        """
        Does a resonator power scan and determines at which power the low power
        regime is exited. If update=True, will set the readout power to this
        power.
        """

        if freqs is None:
            freq_center = self.freq_res()
            freq_range = 10e6
            freqs = np.arange(freq_center - freq_range / 2,
                              freq_center + freq_range / 2,
                              0.1e6)

        if powers is None:
            powers = np.arange(-40, 0.1, 8)

        self.measure_resonator_power(freqs=freqs, powers=powers, analyze=False)
        fit_res = ma.Resonator_Powerscan_Analysis(label='Resonator_power_scan',
                                                  close_fig=True)
        if update:
            ro_pow = 10 ** (fit_res.power / 20)
            self.ro_pulse_amp_CW(ro_pow)
            self.ro_pulse_amp(ro_pow)
            self.freq_res(fit_res.f_low)
            if self.freq_qubit() is None:
                f_qubit_estimate = self.freq_res() + (65e6) ** 2 / (fit_res.shift)
                logging.info('No qubit frquency found. Updating with RWA to {}'
                             .format(f_qubit_estimate))
                self.freq_qubit(f_qubit_estimate)

        return True

    def calibrate_mw_pulse_amplitude_coarse(
            self,
            amps=None,
            close_fig=True,
            verbose=False,
            MC: Optional[MeasurementControl] = None,
            update=True,
            all_modules=False
    ):
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs.py
        """
        Calibrates the pulse amplitude using a single rabi oscillation.
        Depending on self.cfg_with_vsm uses VSM or AWG channel amplitude
        to sweep the amplitude of the pi pulse

        For details see self.measure_rabi
        """
        if amps is None:
            if self.cfg_with_vsm():
                amps = np.linspace(0.1, 1, 31)
            else:
                amps = np.linspace(0, 1, 31)

        self.measure_rabi(amps=amps, MC=MC, analyze=False, all_modules=all_modules)

        a = ma.Rabi_Analysis(close_fig=close_fig, label='rabi')

        # update QCDeS parameter
        try:
            # FIXME: move to HAL_ShimSQ
            if self.cfg_with_vsm():
                self.mw_vsm_G_amp(a.rabi_amplitudes['piPulse'])
            else:
                self.mw_channel_amp(a.rabi_amplitudes['piPulse'])
        except(ValueError):
            warnings.warn("Extracted piPulse amplitude out of parameter range. "
                          "Keeping previous value.")
        return True

    def calibrate_mw_pulse_amplitude_coarse_ramzz(
            self,
            measurement_qubit,
            ramzz_wait_time_ns,
            amps=None,
            close_fig=True,
            verbose=False,
            MC: Optional[MeasurementControl] = None,
            update=True,
            all_modules=False
    ):
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs.py
        """
        Calibrates the pulse amplitude using a single rabi oscillation.
        Depending on self.cfg_with_vsm uses VSM or AWG channel amplitude
        to sweep the amplitude of the pi pulse

        For details see self.measure_rabi
        """
        if amps is None:
            if self.cfg_with_vsm():
                amps = np.linspace(0.1, 1, 31)
            else:
                amps = np.linspace(0, 1, 31)

        self.measure_rabi_ramzz(amps=amps,
                                measurement_qubit = measurement_qubit,
                                ramzz_wait_time_ns = ramzz_wait_time_ns,
                                MC=MC,
                                analyze=False,
                                all_modules=all_modules)

        a = ma.Rabi_Analysis(close_fig=close_fig, label='rabi')

        # update QCDeS parameter
        try:
            # FIXME: move to HAL_ShimSQ
            if self.cfg_with_vsm():
                self.mw_vsm_G_amp(a.rabi_amplitudes['piPulse'])
            else:
                self.mw_channel_amp(a.rabi_amplitudes['piPulse'])
        except(ValueError):
            warnings.warn("Extracted piPulse amplitude out of parameter range. "
                          "Keeping previous value.")
        return True

    # FIXME: code contains errors
    # def calibrate_mw_pulse_amplitude_coarse_test(self,
    #                                              amps=None,
    #                                              close_fig=True, verbose=False,
    #                                              MC: Optional[MeasurementControl] = None, update=True,
    #                                              all_modules=False):
    #     """
    #     Calibrates the pulse amplitude using a single rabi oscillation.
    #     Depending on self.cfg_with_vsm uses VSM or AWG channel amplitude
    #     to sweep the amplitude of the pi pulse
    #
    #     For details see self.measure_rabi
    #     """
    #     self.ro_acq_averages(2**10)
    #     self.ro_soft_avg(3)
    #     # self.mw_gauss_width(10e-9)
    #     # self.mw_pulse_duration()=4*self.mw_gauss_width()
    #     if amps is None:
    #         if self.cfg_with_vsm():
    #             amps = np.linspace(0.1, 1, 31)
    #         else:
    #             amps = np.linspace(0, 1, 31)
    #
    #     self.measure_rabi(amps=amps, MC=MC, analyze=False,
    #                       all_modules=all_modules)
    #     a = ma.Rabi_Analysis(close_fig=close_fig, label='rabi')
    #     old_gw = self.mw_gauss_width()
    #     if a.rabi_amplitudes['piPulse'] > 1 or a.rabi_amplitudes['piHalfPulse'] > a.rabi_amplitudes['piPulse']:
    #         self.mw_gauss_width(2*old_gw)
    #         self.prepare_for_timedomain()
    #         mw_lutman.load_waveforms_onto_AWG_lookuptable(
    #             force_load_sequencer_program=False)
    #
    #     try:
    #         if self.cfg_with_vsm():
    #             self.mw_vsm_G_amp(a.rabi_amplitudes['piPulse'])
    #         else:
    #             self.mw_channel_amp(a.rabi_amplitudes['piPulse'])
    #     except(ValueError):
    #         warnings.warn("Extracted piPulse amplitude out of parameter range. "
    #                       "Keeping previous value.")
    #     return True

    def calibrate_mw_vsm_delay(self):
        """
        Uploads a sequence for calibrating the vsm delay.
        The experiment consists of a single square pulse of 20 ns that
        triggers both the VSM channel specified and the AWG8.

        Note: there are two VSM markers, align with the first of two.

        By changing the "mw_vsm_delay" parameter the delay can be calibrated.
        N.B. Ensure that the signal is visible on a scope or in the UFHQC
        readout first!
        """
        self.prepare_for_timedomain()

        p = sqo.vsm_timing_cal_sequence(
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn())
        CC = self.instr_CC.get_instr()
        CC.eqasm_program(p.filename)
        CC.start()
        print('CC program is running. Parameter "mw_vsm_delay" can now be calibrated by hand.')

    def calibrate_mixer_skewness_drive(
            self,
            MC: Optional[MeasurementControl] = None,
            mixer_channels: list = ['G', 'D'],
            x0: list = [1.0, 0.0],
            cma_stds: list = [.15, 10],
            maxfevals: int = 250,
            update: bool = True
    ) -> bool:
        # USED_BY: device_dependency_graphs.py
        """
        Calibrates the mixer skewness and updates values in the qubit object.

        Args:
            MC (MeasurementControl):
                instance of Measurement Control

            mixer_channels (list):
                list of strings indicating what channels to
                calibrate. In VSM case 'G' and/or 'D' can be specified.
                In no-VSM case mixer_channels is alway set to ['G'].

            update (bool):
                if True updates values in the qubit object.

        Return:
            success (bool):
                returns True if succesful. Currently always
                returns True (i.e., no sanity check implemented)
        """

        # turn relevant channels on
        if MC == None:
            MC = self.instr_MC.get_instr()

        # Load the sequence
        p = sqo.CW_tone(
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn())
        CC = self.instr_CC.get_instr()
        CC.eqasm_program(p.filename)
        CC.start()

        if self.cfg_with_vsm():  # FIXME: move to HAL
            # Open the VSM channel
            VSM = self.instr_VSM.get_instr()
            ch_in = self.mw_vsm_ch_in()
            # module 8 is hardcoded for use mixer calls (signal hound)
            VSM.set('mod8_marker_source'.format(ch_in), 'int')
            VSM.set('mod8_ch{}_marker_state'.format(ch_in), 'on')
            VSM.set('mod8_ch{}_gaussian_amp'.format(ch_in), 1.0)
            VSM.set('mod8_ch{}_derivative_amp'.format(ch_in), 1.0)
        else:
            mixer_channels = ['G']

        mw_lutman = self.instr_LutMan_MW.get_instr()
        mw_lutman.mixer_apply_predistortion_matrix(True)
        # # Define the parameters that will be varied
        for mixer_ch in mixer_channels:
            if self.cfg_with_vsm():
                alpha = mw_lutman.parameters['{}_mixer_alpha'.format(mixer_ch)]
                phi = mw_lutman.parameters['{}_mixer_phi'.format(mixer_ch)]
                if mixer_ch == 'G':
                    mw_lutman.sq_G_amp(.5)
                    mw_lutman.sq_D_amp(0)
                elif mixer_ch == 'D':
                    mw_lutman.sq_G_amp(0)
                    mw_lutman.sq_D_amp(.5)
            else:
                alpha = mw_lutman.parameters['mixer_alpha']
                phi = mw_lutman.parameters['mixer_phi']
                mw_lutman.sq_amp(.5)

            spurious_sideband_freq = self.freq_qubit() - 2 * self.mw_freq_mod()

            # This is to ensure the square waveform is pulse 10!
            mw_lutman.set_default_lutmap()

            if self._using_QWG():
                prepare_function = mw_lutman.apply_mixer_predistortion_corrections
                prepare_function_kwargs = {'wave_dict': {}}
            else:
                def load_square():
                    AWG = mw_lutman.AWG.get_instr()
                    AWG.stop()
                    # When using real-time modulation, mixer_alpha is encoded in channel amplitudes.
                    # Loading amplitude ensures new amplitude will be calculated with mixer_alpha.
                    if mw_lutman.cfg_sideband_mode() == 'real-time':
                        mw_lutman._set_channel_amp(mw_lutman._get_channel_amp())

                    # Codeword 10 is hardcoded in the generate CCL config
                    # mw_lutman.load_waveform_realtime(wave_id='square')
                    mw_lutman.load_waveforms_onto_AWG_lookuptable(
                        force_load_sequencer_program=False)
                    AWG.start()

                prepare_function = load_square
                prepare_function_kwargs = {}

            detector = det.Signal_Hound_fixed_frequency(
                self.instr_SH.get_instr(), spurious_sideband_freq,
                prepare_for_each_point=True,
                Navg=5,
                prepare_function=prepare_function,
                prepare_function_kwargs=prepare_function_kwargs)
            # mw_lutman.load_waveform_realtime,
            # prepare_function_kwargs={'waveform_key': 'square', 'wf_nr': 10})
            ad_func_pars = {'adaptive_function': cma.fmin,
                            'x0': x0,
                            'sigma0': 1,
                            'minimize': True,
                            'noise_handler': cma.NoiseHandler(N=2),
                            'options': {'cma_stds': cma_stds,
                                        'maxfevals': maxfevals}}  # Should be enough for mixer skew

            MC.set_sweep_functions([alpha, phi])
            # MC.set_sweep_function(alpha)
            MC.set_detector_function(detector)  # sets test_detector
            MC.set_adaptive_function_parameters(ad_func_pars)
            MC.set_sweep_points(np.linspace(0, 2, 300))
            MC.run(
                name='Spurious_sideband_{}{}'.format(
                    mixer_ch, self.msmt_suffix),
                mode='adaptive')
            # For the figure
            ma.OptimizationAnalysis_v2()
            a = ma.OptimizationAnalysis(auto=True, label='Spurious_sideband')
            alpha = a.optimization_result[0][0]
            phi = a.optimization_result[0][1]
            if update:
                self.set('mw_{}_mixer_alpha'.format(mixer_ch), alpha)
                self.set('mw_{}_mixer_phi'.format(mixer_ch), phi)

        return True

    # def calibrate_mixer_skewness_RO(self, update=True):
    #     """
    #     Calibrates the mixer skewness using mixer_skewness_cal_UHFQC_adaptive
    #     see calibration toolbox for details

    #     Args:
    #         update (bool):
    #             if True updates values in the qubit object.

    #     Return:
    #         success (bool):
    #             returns True if succesful. Currently always
    #             returns True (i.e., no sanity check implemented)
    #     """

    #     # using the restless tuning sequence
    #     self.prepare_for_timedomain()
    #     p = sqo.randomized_benchmarking(
    #         self.cfg_qubit_nr(), self.cfg_openql_platform_fn(),
    #         nr_cliffords=[1],
    #         net_clifford=1, nr_seeds=1, restless=True, cal_points=False)
    #     self.instr_CC.get_instr().eqasm_program(p.filename)
    #     self.instr_CC.get_instr().start()

    #     LutMan = self.instr_LutMan_RO.get_instr()
    #     LutMan.mixer_apply_predistortion_matrix(True)
    #     MC = self.instr_MC.get_instr()
    #     S1 = swf.lutman_par_UHFQC_dig_trig(
    #         LutMan, LutMan.mixer_alpha, single=False, run=True)
    #     S2 = swf.lutman_par_UHFQC_dig_trig(
    #         LutMan, LutMan.mixer_phi, single=False, run=True)

    #     detector = det.Signal_Hound_fixed_frequency(
    #         self.instr_SH.get_instr(), frequency=(self.instr_LO_ro.get_instr().frequency() -
    #                                               self.ro_freq_mod()),
    #         Navg=5, delay=0.0, prepare_for_each_point=False)

    #     ad_func_pars = {'adaptive_function': nelder_mead,
    #                     'x0': [1.0, 0.0],
    #                     'initial_step': [.15, 10],
    #                     'no_improv_break': 15,
    #                     'minimize': True,
    #                     'maxiter': 500}
    #     MC.set_sweep_functions([S1, S2])
    #     MC.set_detector_function(detector)  # sets test_detector
    #     MC.set_adaptive_function_parameters(ad_func_pars)
    #     MC.run(name='Spurious_sideband', mode='adaptive')
    #     a = ma.OptimizationAnalysis(auto=True, label='Spurious_sideband')
    #     alpha = a.optimization_result[0][0]
    #     phi = a.optimization_result[0][1]

    #     if update:
    #         self.ro_pulse_mixer_phi.set(phi)
    #         self.ro_pulse_mixer_alpha.set(alpha)
    #         LutMan.mixer_alpha(alpha)
    #         LutMan.mixer_phi(phi)

    def calibrate_mixer_skewness_RO(self, update=True):
        """
        Calibrates the mixer skewness using mixer_skewness_cal_UHFQC_adaptive
        see calibration toolbox for details FIXME: outdated

        Args:
            update (bool):
                if True updates values in the qubit object.

        Return:
            success (bool):
                returns True if succesful. Currently always
                returns True (i.e., no sanity check implemented)
        """
        p = sqo.CW_RO_sequence(
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn())
        CC = self.instr_CC.get_instr()
        CC.eqasm_program(p.filename)
        CC.start()

        # using the restless tuning sequence
        # self.prepare_for_timedomain()
        # p = sqo.randomized_benchmarking(
        #     self.cfg_qubit_nr(), self.cfg_openql_platform_fn(),
        #     nr_cliffords=[1],
        #     net_clifford=1, nr_seeds=1, restless=True, cal_points=False)
        # self.instr_CC.get_instr().eqasm_program(p.filename)
        # self.instr_CC.get_instr().start()

        LutMan = self.instr_LutMan_RO.get_instr()
        LutMan.mixer_apply_predistortion_matrix(True)
        MC = self.instr_MC.get_instr()
        S1 = swf.lutman_par_UHFQC_dig_trig(LutMan, LutMan.mixer_alpha, single=False, run=True)
        S2 = swf.lutman_par_UHFQC_dig_trig(LutMan, LutMan.mixer_phi, single=False, run=True)

        detector = det.Signal_Hound_fixed_frequency(
            self.instr_SH.get_instr(),
            frequency=self.ro_freq() - 2 * self.ro_freq_mod(),
            Navg=5, delay=0.0,
            prepare_for_each_point=False)

        ad_func_pars = {'adaptive_function': nelder_mead,
                        'x0': [1.0, 0.0],
                        'initial_step': [.15, 10],
                        'no_improv_break': 15,
                        'minimize': True,
                        'maxiter': 500}
        MC.set_sweep_functions([S1, S2])
        MC.set_detector_function(detector)  # sets test_detector
        MC.set_adaptive_function_parameters(ad_func_pars)
        MC.run(name='Spurious_sideband', mode='adaptive')
        a = ma.OptimizationAnalysis(auto=True, label='Spurious_sideband')
        alpha = a.optimization_result[0][0]
        phi = a.optimization_result[0][1]

        if update:
            self.ro_pulse_mixer_phi.set(phi)
            self.ro_pulse_mixer_alpha.set(alpha)
            LutMan.mixer_alpha(alpha)
            LutMan.mixer_phi(phi)

    def calibrate_mixer_offsets_RO(
            self, update: bool = True,
            ftarget=-110
    ) -> bool:
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs.py
        # USED_BY: device_dependency_graphs

        """
        Calibrates the mixer offset and updates the I and Q offsets in
        the qubit object.

        Args:
            update (bool):
                if True updates values in the qubit object.

            ftarget (float): power of the signal at the LO frequency
                for which the optimization is terminated

        Return:
            success (bool):
                returns True if succesful. Currently always
                returns True (i.e., no sanity check implemented)
        """

        chI_par = self.instr_acquisition.get_instr().sigouts_0_offset
        chQ_par = self.instr_acquisition.get_instr().sigouts_1_offset

        offset_I, offset_Q = cal_toolbox.mixer_carrier_cancellation(
            SH=self.instr_SH.get_instr(),
            source=self.instr_LO_ro.get_instr(),
            MC=self.instr_MC.get_instr(),
            chI_par=chI_par,
            chQ_par=chQ_par,
            x0=(0.05, 0.05),
            ftarget=ftarget)

        if update:
            self.ro_pulse_mixer_offs_I(offset_I)
            self.ro_pulse_mixer_offs_Q(offset_Q)
        return True

    def calibrate_mw_pulses_basic(
            self,
            cal_steps=['offsets', 'amp_coarse', 'freq',
                       'drag', 'amp_fine', 'amp_fine',
                       'amp_fine'],
            kw_freqs={'steps': [1, 3, 10, 30, 100,
                                300, 1000]},
            kw_amp_coarse={'amps': np.linspace(0, 1, 31)},
            kw_amp_fine={'update': True},
            soft_avg_allxy=3,
            kw_offsets={'ftarget': -120},
            kw_skewness={},
            kw_motzoi={'update': True},
            f_target_skewness=-120
    ):

        """
        Performs a standard calibration of microwave pulses consisting of
        - mixer offsets
        - mixer skewness
        - pulse ampl coarse (rabi)
        - frequency (ramsey)
        - motzoi
        - ampl fine (flipping)

        - AllXY (to verify)

        Note that this is a basic calibration and does not involve fine tuning
        to ~99.9% and only works if the qubit is well behaved.
        """
        for this_step in cal_steps:
            if this_step == 'offsets':
                self.calibrate_mixer_offsets_drive(**kw_offsets)
            elif this_step == 'skewness':
                self.calibrate_mixer_skewness_drive(**kw_skewness)
            elif this_step == 'amp_coarse':
                self.calibrate_mw_pulse_amplitude_coarse(**kw_amp_coarse)
            elif this_step == 'freq':
                self.find_frequency('ramsey', **kw_freqs)
            elif this_step == 'drag':
                self.calibrate_motzoi(**kw_motzoi)
            elif this_step == 'amp_fine':
                self.measure_flipping(**kw_amp_fine)
        old_soft_avg = self.ro_soft_avg()
        self.ro_soft_avg(soft_avg_allxy)
        self.measure_allxy()
        self.ro_soft_avg(old_soft_avg)
        return True

    def calibrate_ssro_coarse(
            self,
            MC: Optional[MeasurementControl] = None,
            nested_MC: Optional[MeasurementControl] = None,
            freqs=None,
            amps=None,
            analyze: bool = True,
            update: bool = True,
            disable_metadata = False
    ):
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs

        '''
        Performs a 2D sweep of <qubit>.ro_freq and <qubit>.ro_pulse_amp and
        measures SSRO parameters (SNR, F_a, F_d).
        After the sweep is done, it sets the parameters for which the assignment
        fidelity was maximum.

        Args:
            freq (array):
                Range of frequencies of sweep.

            amps (array):
                Range of amplitudes of sweep.
        '''

        if MC is None:
            MC = self.instr_MC.get_instr()

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        if freqs is None:
            if self.dispersive_shift() is not None:
                freqs = np.arange(-2 * abs(self.dispersive_shift()),
                                  abs(self.dispersive_shift()), .5e6) + self.freq_res()
            else:
                raise ValueError('self.dispersive_shift is None. Please specify\
                                 range of sweep frequencies.')

        if amps is None:
            amps = np.linspace(.001, .5, 31)

        self.prepare_for_timedomain()

        ro_lm = self.find_instrument(self.instr_LutMan_RO())
        q_idx = self.cfg_qubit_nr()
        swf1 = swf.RO_freq_sweep(name='RO frequency',
                                 qubit=self,
                                 ro_lutman=ro_lm,
                                 idx=q_idx,
                                 parameter=self.ro_freq)

        nested_MC.set_sweep_function(swf1)
        nested_MC.set_sweep_points(freqs)
        nested_MC.set_sweep_function_2D(self.ro_pulse_amp)
        nested_MC.set_sweep_points_2D(amps)

        d = det.Function_Detector(self.measure_ssro,
                                  result_keys=['SNR', 'F_a', 'F_d'],
                                  value_names=['SNR', 'F_a', 'F_d'],
                                  value_units=['a.u.', 'a.u.', 'a.u.'],
                                  msmt_kw={'prepare': True}
                                  )
        nested_MC.set_detector_function(d)
        nested_MC.run(name='RO_coarse_tuneup', mode='2D', disable_snapshot_metadata = disable_metadata)

        if analyze is True:
            # Analysis
            a = ma.TwoD_Analysis(label='RO_coarse_tuneup', auto=False)
            # Get best parameters
            a.get_naming_and_values_2D()
            arg = np.argmax(a.measured_values[1])
            index = np.unravel_index(arg, (len(a.sweep_points),
                                           len(a.sweep_points_2D)))
            best_freq = a.sweep_points[index[0]]
            best_amp = a.sweep_points_2D[index[1]]
            a.run_default_analysis()
            print('Frequency: {}, Amplitude: {}'.format(best_freq, best_amp))

            if update is True:
                self.ro_freq(best_freq)
                self.ro_pulse_amp(best_amp)

            return True

    def calibrate_ssro_pulse_duration(
            self,
            MC: Optional[MeasurementControl] = None,
            nested_MC: Optional[MeasurementControl] = None,
            amps=None,
            amp_lim=None,
            times=None,
            use_adaptive: bool = True,
            n_points: int = 80,
            analyze: bool = True,
            update: bool = True
    ):
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs

        '''
        Calibrates the RO pulse duration by measuring the assignment fidelity of
        SSRO experiments as a function of the RO pulse duration and amplitude.
        For each set of parameters, the routine calibrates optimal weights and
        then extracts readout fidelity.
        This measurement can be performed using an adaptive sampler
        (use_adaptive=True) or a regular 2D parameter sweep (use_adaptive=False).
        Designed to be used in the GBT node 'SSRO Pulse Duration'.

        Args:
            amps (array):
                If using 2D sweep:
                    Set of RO amplitudes sampled in the 2D sweep.
                If using adaptive sampling:
                    Minimum and maximum (respectively) of the RO amplitude range
                    used in the adaptive sampler.

            times (array):
                If using 2D sweep:
                    Set of RO pulse durations sampled in the 2D sweep.
                If using adaptive sampling:
                    Minimum and maximum (respectively) of the RO pulse duration
                    range used in the adaptive sampler.

            use_adaptive (bool):
                Boolean that sets the sampling mode. Set to "False" for a
                regular 2D sweep or set to "True" for adaptive sampling.

            n_points:
                Only relevant in the adaptive sampling mode. Sets the maximum
                number of points sampled.
        '''

        if MC is None:
            MC = self.instr_MC.get_instr()

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        if times is None:
            times = np.arange(10e-9, 401e-9, 10e-9)

        if amps is None:
            amps = np.linspace(.01, .25, 11)
        if amp_lim is None:
            amp_lim = (0.01, 0.2)
        ######################
        # Experiment
        ######################
        nested_MC.set_sweep_functions([self.ro_pulse_length,
                                       self.ro_pulse_amp])
        d = det.Function_Detector(self.calibrate_optimal_weights,
                                  result_keys=['F_a', 'F_d', 'SNR'],
                                  value_names=['F_a', 'F_d', 'SNR'],
                                  value_units=['a.u.', 'a.u.', 'a.u.'])
        nested_MC.set_detector_function(d)
        # Use adaptive sampling
        if use_adaptive is True:
            # Adaptive sampler cost function
            loss_per_simplex = mk_minimization_loss_func()
            goal = mk_minimization_goal_func()

            nested_MC.set_adaptive_function_parameters(
                {'adaptive_function': LearnerND_Minimizer,
                 'goal': lambda l: goal(l) or l.npoints > n_points,
                 'loss_per_simplex': loss_per_simplex,
                 'bounds': [(10e-9, 400e-9), amp_lim],
                 'minimize': False
                 })
            nested_MC.run(name='RO_duration_tuneup_{}'.format(self.name),
                          mode='adaptive')
        # Use standard 2D sweep
        else:
            nested_MC.set_sweep_points(times)
            nested_MC.set_sweep_points_2D(amps)
            nested_MC.run(name='RO_duration_tuneup_{}'.format(self.name),
                          mode='2D')
        #####################
        # Analysis
        #####################
        if analyze is True:
            if use_adaptive is True:
                A = ma2.Readout_landspace_Analysis(label='RO_duration_tuneup')
                optimal_pulse_duration = A.qoi['Optimal_parameter_X']
                optimal_pulse_amplitude = A.qoi['Optimal_parameter_Y']
                self.ro_pulse_length(optimal_pulse_duration)
                self.ro_pulse_amp(optimal_pulse_amplitude)
            else:
                A = ma.TwoD_Analysis(label='RO_duration_tuneup', auto=True)
            return True

    def calibrate_ssro_fine(
            self,
            MC: Optional[MeasurementControl] = None,
            nested_MC: Optional[MeasurementControl] = None,
            nr_shots_per_case: int = 2 ** 13,  # 8192
            start_freq=None,
            start_amp=None,
            start_freq_step=None,
            start_amp_step=None,
            optimize_threshold: float = .99,
            check_threshold: float = .90,
            analyze: bool = True,
            update: bool = True,
            disable_metadata = False
    ):
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs

        '''
        Runs an optimizer routine on the SSRO assignment fidelity of the
        <qubit>.ro_freq and <qubit>.ro_pulse_amp parameters.
        Intended to be used in the "SSRO Optimization" node of GBT.

        Args:
            start_freq (float):
                Starting frequency of the optmizer.

            start_amp (float):
                Starting amplitude of the optimizer.

            start_freq_step (float):
                Starting frequency step of the optmizer.

            start_amp_step (float):
                Starting amplitude step of the optimizer.

            threshold (float):
                Fidelity threshold after which the optimizer stops iterating.
        '''

        ## check single-qubit ssro first, if assignment fidelity below 92.5%, run optimizer
        self.measure_ssro(nr_shots_per_case=nr_shots_per_case, post_select=True)
        if self.F_ssro() > check_threshold:
            return True

        if MC is None:
            MC = self.instr_MC.get_instr()

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        if start_freq_step is None:
            if start_freq is None:
                start_freq = self.ro_freq()
                start_freq_step = 0.1e6
            else:
                raise ValueError('Must provide start frequency step if start\
                                frequency is specified.')

        if start_amp_step is None:
            if start_amp is None:
                start_amp = self.ro_pulse_amp()
                start_amp_step = 0.01
            else:
                raise ValueError('Must provide start amplitude step if start\
                                amplitude is specified.')

        if start_amp is None:
            start_amp = self.ro_pulse_amp()

        nested_MC.set_sweep_functions([self.ro_freq, self.ro_pulse_amp])

        d = det.Function_Detector(self.calibrate_optimal_weights,
                                  result_keys=['F_a'],
                                  value_names=['F_a'],
                                  value_units=['a.u.'])
        nested_MC.set_detector_function(d)

        ad_func_pars = {'adaptive_function': nelder_mead,
                        'x0': [self.ro_freq(), self.ro_pulse_amp()],
                        'initial_step': [start_freq_step, start_amp_step],
                        'minimize': False,
                        'maxiter': 20,
                        'f_termination': optimize_threshold}
        nested_MC.set_adaptive_function_parameters(ad_func_pars)

        nested_MC.set_optimization_method('nelder_mead')
        nested_MC.run(name='RO_fine_tuneup', mode='adaptive', disable_snapshot_metadata = disable_metadata)

        if analyze is True:
            ma.OptimizationAnalysis(label='RO_fine_tuneup')
            return True

    def calibrate_ro_acq_delay(
            self,
            MC: Optional[MeasurementControl] = None,
            analyze: bool = True,
            prepare: bool = True,
            disable_metadata: bool = False
    ):
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs

        """
        Calibrates the ro_acq_delay parameter for the readout.
        For that it analyzes the transients.

        """

        self.ro_acq_delay(0)  # set delay to zero
        old_pow = self.ro_pulse_amp()
        self.ro_pulse_amp(0.5)

        if MC is None:
            MC = self.instr_MC.get_instr()
        # if plot_max_time is None:
        #     plot_max_time = self.ro_acq_integration_length()+250e-9

        if prepare:
            self.prepare_for_timedomain()
            p = sqo.off_on(
                qubit_idx=self.cfg_qubit_nr(), pulse_comb='off',
                initialize=False,
                platf_cfg=self.cfg_openql_platform_fn())
            self.instr_CC.get_instr().eqasm_program(p.filename)
        else:
            p = None  # object needs to exist for the openql_sweep to work

        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='Transient time', unit='s',
                             upload=prepare)
        MC.set_sweep_function(s)

        if 'UHFQC' in self.instr_acquisition():
            sampling_rate = 1.8e9  # FIXME: get from instrument
        else:
            raise NotImplementedError()

        MC.set_sweep_points(np.arange(self.input_average_detector.nr_samples) / sampling_rate)
        MC.set_detector_function(self.input_average_detector)
        MC.run(name=f'Measure_Acq_Delay_{self.msmt_suffix}', disable_snapshot_metadata=disable_metadata)

        self.ro_pulse_amp(old_pow)

        if analyze:
            a = ma2.RO_acquisition_delayAnalysis(qubit_name=self.name)
            # Delay time is averaged over the two quadratures.
            delay_time = (a.proc_data_dict['I_pulse_start'] +
                          a.proc_data_dict['Q_pulse_start']) / 2
            self.ro_acq_delay(delay_time)
            return True

    def calibrate_mw_gates_restless(
            self,
            MC: Optional[MeasurementControl] = None,
            parameter_list: list = ['G_amp', 'D_amp', 'freq'],
            initial_values: list = None,
            initial_steps: list = [0.05, 0.05, 1e6],
            nr_cliffords: int = 80, nr_seeds: int = 200,
            verbose: bool = True, update: bool = True,
            prepare_for_timedomain: bool = True
    ):
        """
        Refs:
            Rol PR Applied 7, 041001 (2017)
        """

        return self.calibrate_mw_gates_rb(
            MC=None,
            parameter_list=parameter_list,
            initial_values=initial_values,
            initial_steps=initial_steps,
            nr_cliffords=nr_cliffords, nr_seeds=nr_seeds,
            verbose=verbose, update=update,
            prepare_for_timedomain=prepare_for_timedomain,
            method='restless')

    def calibrate_mw_gates_rb(
            self,
            MC: Optional[MeasurementControl] = None,
            parameter_list: list = ['G_amp', 'D_amp', 'freq'],
            initial_values: list = None,
            initial_steps: list = [0.05, 0.05, 1e6],
            nr_cliffords: int = 80, nr_seeds: int = 200,
            verbose: bool = True, update: bool = True,
            prepare_for_timedomain: bool = True,
            method: bool = None,
            optimizer: str = 'NM'
    ):
        """
        Calibrates microwave pulses using a randomized benchmarking based
        cost-function.
        requirements for restless:
        - Digitized readout (calibrated)
        requirements for ORBIT:
        - Optimal weights such that minimizing correspond to 0 state.
        """
        if method is None:
            method = self.cfg_rb_calibrate_method()
        if method == 'restless':
            restless = True
        else:  # ORBIT
            restless = False

        if MC is None:
            MC = self.instr_MC.get_instr()

        if initial_steps is None:
            initial_steps: list = [0.05, 0.05, 1e6]

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        if parameter_list is None:
            # parameter_list = ['G_amp', 'D_amp']
            parameter_list = ['G_amp', 'D_amp', 'freq']

        mw_lutman = self.instr_LutMan_MW.get_instr()

        G_amp_par = wrap_par_to_swf(
            mw_lutman.parameters['channel_amp'],
            retrieve_value=True)
        D_amp_par = swf.QWG_lutman_par(LutMan=mw_lutman,
                                       LutMan_parameter=mw_lutman.mw_motzoi)

        freq_par = self.instr_LO_mw.get_instr().frequency

        sweep_pars = []
        for par in parameter_list:
            if par == 'G_amp':
                sweep_pars.append(G_amp_par)
            elif par == 'D_amp':
                sweep_pars.append(D_amp_par)
            elif par == 'freq':
                sweep_pars.append(freq_par)
            else:
                raise NotImplementedError(
                    "Parameter {} not recognized".format(par))

        if initial_values is None:
            # use the current values of the parameters being varied.
            initial_values = [G_amp_par.get(), mw_lutman.mw_motzoi.get(), freq_par.get()]

        # Preparing the sequence
        if restless:
            net_clifford = 3  # flipping sequence
            d = det.UHFQC_single_qubit_statistics_logging_det(
                self.instr_acquisition.get_instr(),
                self.instr_CC.get_instr(), nr_shots=4 * 4095,
                integration_length=self.ro_acq_integration_length(),
                channel=self.ro_acq_weight_chI(),
                statemap={'0': '1', '1': '0'})
            minimize = False
            msmt_string = f'Restless_tuneup_{nr_cliffords}Cl_{nr_seeds}seeds' + self.msmt_suffix

        else:
            net_clifford = 0  # not flipping sequence
            d = self.int_avg_det_single
            minimize = True
            msmt_string = f'ORBIT_tuneup_{nr_cliffords}Cl_{nr_seeds}seeds' + self.msmt_suffix

        p = sqo.randomized_benchmarking(
            self.cfg_qubit_nr(), self.cfg_openql_platform_fn(),
            nr_cliffords=[nr_cliffords],
            net_clifford=net_clifford, nr_seeds=nr_seeds,
            restless=restless, cal_points=False)
        self.instr_CC.get_instr().eqasm_program(p.filename)
        self.instr_CC.get_instr().start()

        MC.set_sweep_functions(sweep_pars)

        MC.set_detector_function(d)

        if optimizer == 'CMA':
            ad_func_pars = {'adaptive_function': cma.fmin,
                            'x0': initial_values,
                            'sigma0': 1,
                            # 'noise_handler': cma.NoiseHandler(len(initial_values)),
                            'minimize': minimize,
                            'options': {'cma_stds': initial_steps}}

        elif optimizer == 'NM':
            ad_func_pars = {'adaptive_function': nelder_mead,
                            'x0': initial_values,
                            'initial_step': initial_steps,
                            'no_improv_break': 50,
                            'minimize': minimize,
                            'maxiter': 1500}

        MC.set_adaptive_function_parameters(ad_func_pars)
        MC.run(name=msmt_string,
               mode='adaptive')
        a = ma.OptimizationAnalysis(label=msmt_string)

        if update:
            if verbose:
                print("Updating parameters in qubit object")

            opt_par_values = a.optimization_result[0]
            for par in parameter_list:
                if par == 'G_amp':
                    G_idx = parameter_list.index('G_amp')
                    self.mw_channel_amp(opt_par_values[G_idx])
                elif par == 'D_amp':
                    D_idx = parameter_list.index('D_amp')
                    self.mw_vsm_D_amp(opt_par_values[D_idx])
                elif par == 'D_phase':
                    D_idx = parameter_list.index('D_phase')
                    self.mw_vsm_D_phase(opt_par_values[D_idx])
                elif par == 'freq':
                    freq_idx = parameter_list.index('freq')
                    # We are varying the LO frequency in the opt, not the q freq.
                    self.freq_qubit(opt_par_values[freq_idx] + self.mw_freq_mod.get())

    def calibrate_mw_gates_allxy(
            self,
            nested_MC: Optional[MeasurementControl] = None,
            start_values=None,
            initial_steps=None,
            parameter_list=None,
            termination_opt=0.01
    ):
        # FIXME: this tuneup does not update the qubit object parameters
        #  update: Fixed on the the pagani set-up

        # FIXME: this tuneup does not return True upon success
        #  update: Fixed on the pagani set-up

        if initial_steps is None:
            if parameter_list is None:
                initial_steps = [1e6, 0.05, 0.05]
            else:
                raise ValueError(
                    "must pass initial steps if setting parameter_list")

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        if parameter_list is None:
            if self.cfg_with_vsm():
                parameter_list = ["freq_qubit",
                                  "mw_vsm_G_amp",
                                  "mw_vsm_D_amp"]
            else:
                parameter_list = ["freq_qubit",
                                  "mw_channel_amp",
                                  "mw_motzoi"]

        nested_MC.set_sweep_functions([
            self.__getattr__(p) for p in parameter_list])

        if start_values is None:
            # use current values
            start_values = [self.get(p) for p in parameter_list]

        d = det.Function_Detector(self.measure_allxy,
                                  value_names=['AllXY cost'],
                                  value_units=['a.u.'], )
        nested_MC.set_detector_function(d)

        ad_func_pars = {'adaptive_function': nelder_mead,
                        'x0': start_values,
                        'initial_step': initial_steps,
                        'no_improv_break': 10,
                        'minimize': True,
                        'maxiter': 500,
                        'f_termination': termination_opt}

        nested_MC.set_adaptive_function_parameters(ad_func_pars)
        nested_MC.set_optimization_method('nelder_mead')
        nested_MC.run(name='gate_tuneup_allxy', mode='adaptive')
        a2 = ma.OptimizationAnalysis(label='gate_tuneup_allxy')

        if a2.optimization_result[1][0] > termination_opt:
            return False
        else:
            return True

    def calibrate_mw_gates_allxy2(
            self,
            nested_MC: Optional[MeasurementControl] = None,
            start_values=None,
            initial_steps=None, f_termination=0.01
    ):
        '''
        FIXME! Merge both calibrate allxy methods.
        Optimizes ALLXY sequency by tunning 2 parameters:
        mw_channel_amp and mw_motzoi.

        Used for Graph based tune-up in the ALLXY node.
        '''
        old_avg = self.ro_acq_averages()
        self.ro_acq_averages(2 ** 14)

        VSM = self.instr_VSM.get_instr()
        # Close all vsm channels
        modules = range(8)
        for module in modules:
            VSM.set('mod{}_marker_source'.format(module + 1), 'int')
            for channel in [1, 2, 3, 4]:
                VSM.set('mod{}_ch{}_marker_state'.format(
                    module + 1, channel), 'off')
        # Open intended channel
        VSM.set('mod{}_marker_source'.format(self.mw_vsm_mod_out()), 'int')
        VSM.set('mod{}_ch{}_marker_state'.format(
            self.mw_vsm_mod_out(), self.mw_vsm_ch_in()), 'on')

        if initial_steps is None:
            initial_steps = [0.05, 0.05]

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        if self.cfg_with_vsm():
            parameter_list = ["mw_vsm_G_amp",
                              "mw_vsm_D_amp"]
        else:
            parameter_list = ["mw_channel_amp",
                              "mw_motzoi"]

        nested_MC.set_sweep_functions([
            self.__getattr__(p) for p in parameter_list])

        if start_values is None:
            # use current values
            start_values = [self.get(p) for p in parameter_list]

        d = det.Function_Detector(self.measure_allxy,
                                  value_names=['AllXY cost'],
                                  value_units=['a.u.'], )
        nested_MC.set_detector_function(d)

        ad_func_pars = {'adaptive_function': nelder_mead,
                        'x0': start_values,
                        'initial_step': initial_steps,
                        'no_improv_break': 10,
                        'minimize': True,
                        'maxiter': 500,
                        'f_termination': f_termination}

        nested_MC.set_adaptive_function_parameters(ad_func_pars)
        nested_MC.set_optimization_method('nelder_mead')
        nested_MC.run(name='gate_tuneup_allxy', mode='adaptive')
        a2 = ma.OptimizationAnalysis(label='gate_tuneup_allxy')
        self.ro_acq_averages(old_avg)
        # Open all vsm channels
        for module in modules:
            VSM.set('mod{}_marker_source'.format(module + 1), 'int')
            for channel in [1, 2, 3, 4]:
                VSM.set('mod{}_ch{}_marker_state'.format(
                    module + 1, channel), 'on')

        if a2.optimization_result[1][0] > f_termination:
            return False
        else:
            return True

    def calibrate_RO(
            self,
            nested_MC: Optional[MeasurementControl] = None,
            start_params=None,
            initial_step=None,
            threshold=0.05
    ):
        '''
        Optimizes the RO assignment fidelity using 2 parameters:
        ro_freq and ro_pulse_amp.

        Args:
            start_params:   Starting parameters for <qubit>.ro_freq and
                            <qubit>.ro_pulse_amp. These have to be passed on in
                            the aforementioned order, that is:
                            [ro_freq, ro_pulse_amp].

            initial_steps:  These have to be given in the order:
                            [ro_freq, ro_pulse_amp]

            threshold:      Assignment fidelity error (1-F_a) threshold used in
                            the optimization.

        Used for Graph based tune-up.
        '''

        # FIXME: Crashes whenever it tries to set the pulse amplitude higher
        #        than 1.

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        if start_params is None:
            start_params = [self.ro_freq(), self.ro_pulse_amp()]

        if initial_step is None:
            initial_step = [1.e6, .05]

        nested_MC.set_sweep_functions([self.ro_freq, self.ro_pulse_amp])

        def wrap_func():
            error = 1 - self.calibrate_optimal_weights()['F_a']
            return error

        d = det.Function_Detector(wrap_func,
                                  value_names=['F_a error'],
                                  value_units=['a.u.'])
        nested_MC.set_detector_function(d)

        ad_func_pars = {'adaptive_function': nelder_mead,
                        'x0': start_params,
                        'initial_step': initial_step,
                        'no_improv_break': 10,
                        'minimize': True,
                        'maxiter': 20,
                        'f_termination': threshold}
        nested_MC.set_adaptive_function_parameters(ad_func_pars)

        nested_MC.set_optimization_method('nelder_mead')
        nested_MC.run(name='RO_tuneup', mode='adaptive')

        a = ma.OptimizationAnalysis(label='RO_tuneup')

        if a.optimization_result[1][0] > 0.05:  # Fidelity 0.95
            return False
        else:
            return True

    def calibrate_depletion_pulse(
            self, 
            nested_MC=None, 
            two_par=True,
            amp0=None,
            amp1=None, 
            phi0=180, 
            phi1=0, 
            initial_steps=None,
            max_iterations=100,
            depletion_optimization_window=None, 
            depletion_analysis_plot=False,
            use_RTE_cost_function=False,
            use_adaptive_optimizer=False,
            adaptive_loss_weight=5,
            target_cost=0.02
            ):
        """
        this function automatically tunes up a two step, four-parameter
        depletion pulse.
        It uses the averaged transients for ground and excited state for its
        cost function.

        Refs:
        Bultnik PR Applied 6, 034008 (2016)

        Args:
            two_par:    if readout is performed at the symmetry point and in the
                        linear regime two parameters will suffice. Othen, four
                        paramters do not converge.
                        First optimizaing the amplitudes (two paramters) and
                        then run the full 4 paramaters with the correct initial
                        amplitudes works.
            optimization_window:  optimization window determins which part of
                        the transients will be
                        nulled in the optimization. By default it uses a
                        window of 500 ns post depletiona with a 50 ns buffer.
            initial_steps:  These have to be given in the order
                           [phi0,phi1,amp0,amp1] for 4-par tuning and
                           [amp0,amp1] for 2-par tunining
        """

        # FIXME: this calibration does not update the qubit object params
        # FIXME2: this calibration does not return a boolean upon success

        # tuneup requires nested MC as the transients detector will use MC
        self.ro_pulse_type('up_down_down')
        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        # setting the initial depletion amplitudes
        if amp0 is None:
            amp0 = 2*self.ro_pulse_amp()
        if amp1 is None:
            amp1 = 0.5*self.ro_pulse_amp()

        if depletion_optimization_window is None:
            depletion_optimization_window = [
                self.ro_pulse_length()+self.ro_pulse_down_length0()
                + self.ro_pulse_down_length1()+50e-9,
                self.ro_pulse_length()+self.ro_pulse_down_length0()
                + self.ro_pulse_down_length1()+550e-9]

        if two_par:
            nested_MC.set_sweep_functions([
                self.ro_pulse_down_amp0,
                self.ro_pulse_down_amp1])
        else:
            nested_MC.set_sweep_functions([self.ro_pulse_down_phi0,
                                           self.ro_pulse_down_phi1,
                                           self.ro_pulse_down_amp0,
                                           self.ro_pulse_down_amp1])

        # prepare here once, instead of every time in the detector function
        self.prepare_for_timedomain()

        if use_RTE_cost_function:
            d = det.Function_Detector(
                self.measure_error_fraction,
                msmt_kw={'net_gate': 'pi',
                        'feedback': False,
                        'sequence_type': 'echo'},
                value_names=['error fraction'],
                value_units=['au'],
                result_keys=['error fraction'])
        else:
            # preparation needs to be done in detector function 
            # as we are only sweeping parameters here!
            d = det.Function_Detector(
                self.measure_transients,
                msmt_kw={'depletion_analysis': True,
                        'depletion_analysis_plot': depletion_analysis_plot,
                        'depletion_optimization_window': depletion_optimization_window,
                        'prepare': True},
                value_names=['depletion cost'],
                value_units=['au'],
                result_keys=['depletion_cost'])
        nested_MC.set_detector_function(d)

        if two_par:
            if initial_steps is None:
                initial_steps = [-0.5*amp0, -0.5*amp1]
            if use_adaptive_optimizer:
                goal = mk_min_threshold_goal_func(
                    max_pnts_beyond_threshold=2) 
                loss = mk_minimization_loss_func(
                    max_no_improve_in_local=8,
                    converge_below=target_cost,
                    volume_weight=adaptive_loss_weight)
                amp0_bounds = np.array([0.1*amp0, 2*amp0])
                amp1_bounds = np.array([0.1*amp1, 2*amp1])
                ad_func_pars = {'adaptive_function': LearnerND_Minimizer,
                                'goal': lambda l: goal(l) or l.npoints >= max_iterations,
                                'bounds': [amp0_bounds, amp1_bounds],
                                'loss_per_simplex': loss,
                                'minimize': True,
                                'X0': np.array([np.linspace(*amp0_bounds, 10), 
                                                np.linspace(*amp1_bounds, 10)]).T }
            else:
                ad_func_pars = {'adaptive_function': nelder_mead,
                                'x0': [amp0, amp1],
                                'initial_step': initial_steps,
                                'no_improve_break': 8,
                                'no_improve_thr': target_cost/10,
                                'minimize': True,
                                'maxiter': max_iterations}
            self.ro_pulse_down_phi0(180)
            self.ro_pulse_down_phi1(0)
        else:
            if initial_steps is None:
                initial_steps = [10, 10, -0.1*amp0, -0.1*amp1]
            if use_adaptive_optimizer:
                goal = mk_min_threshold_goal_func(
                    max_pnts_beyond_threshold=2) 
                loss = mk_minimization_loss_func(
                    max_no_improve_in_local=8,
                    converge_below=target_cost,
                    volume_weight=adaptive_loss_weight)
                ph0_bounds = np.array([150, 210])
                ph1_bounds = np.array([0, 30])
                amp0_bounds = np.array([0.1*amp0, 2*amp0])
                amp1_bounds = np.array([0.1*amp1, 2*amp1])
                ad_func_pars = {'adaptive_function': LearnerND_Minimizer,
                                'goal': lambda l: goal(l) or l.npoints >= max_iterations,
                                'bounds': [ph0_bounds, ph1_bounds, 
                                            amp0_bounds, amp1_bounds],
                                'loss_per_simplex': loss,
                                'minimize': True,
                                'X0': np.array([np.linspace(*ph0_bounds, 10),
                                                np.linspace(*ph1_bounds, 10),
                                                np.linspace(*amp0_bounds, 10), 
                                                np.linspace(*amp1_bounds, 10)]).T }
            else:
                ad_func_pars = {'adaptive_function': nelder_mead,
                                'x0': [phi0, phi1, amp0, amp1],
                                'initial_step': initial_steps,
                                'no_improve_break': 8,
                                'no_improve_thr': target_cost/10,
                                'minimize': True,
                                'maxiter': max_iterations}

        nested_MC.set_adaptive_function_parameters(ad_func_pars)
        if use_adaptive_optimizer:
            nested_MC.set_optimization_method('adaptive')
        else:
            nested_MC.set_optimization_method('nelder_mead')

        optimizer_result = nested_MC.run(
            f"Depletion_tuneup_{self.name}_adaptive-{use_adaptive_optimizer}", 
            mode='adaptive')
        a = ma.OptimizationAnalysis(label='Depletion_tuneup')

        return a.optimization_result, optimizer_result

    def calibrate_ef_rabi(
            self,
            amps: list = np.linspace(-.8, .8, 18),
            recovery_pulse: bool = True,
            MC: Optional[MeasurementControl] = None,
            label: str = '',
            analyze=True,
            close_fig=True,
            prepare_for_timedomain=True,
            update=True
    ):
        """
        Calibrates the pi pulse of the ef/12 transition using
         a rabi oscillation of the ef/12 transition.

        Modulation frequency of the "ef" pulses is controlled through the
        `anharmonicity` parameter of the qubit object.
        Hint: the expected pi-pulse amplitude of the ef/12 transition is ~1/2
            the pi-pulse amplitude of the ge/01 transition.
        """
        a2 = self.measure_ef_rabi(
            amps=amps,
            recovery_pulse=recovery_pulse,
            MC=MC, label=label,
            analyze=analyze, close_fig=close_fig,
            prepare_for_timedomain=prepare_for_timedomain
        )
        if update:
            ef_pi_amp = a2.proc_data_dict['ef_pi_amp']
            self.mw_ef_amp(a2.proc_data_dict['ef_pi_amp'])

    ##########################################################################
    # calibrate_ functions (overrides for class Qubit)
    ##########################################################################

    def calibrate_motzoi(self,
                         MC: Optional[MeasurementControl] = None,
                         verbose=True,
                         update=True,
                         motzois=None,
                         disable_metadata = False):
        # USED_BY: inspire_dependency_graph.py,
        # USED_BY: device_dependency_graphs_v2.py,
        """
        Calibrates the DRAG coeffcieint value, named motzoi (after Felix Motzoi)
        for legacy reasons.

        For details see docstring of measure_motzoi method.
        """
        using_VSM = self.cfg_with_vsm()
        if using_VSM and motzois is None:
            motzois = gen_sweep_pts(start=0.1, stop=1.0, num=31)
        elif motzois is None:
            motzois = gen_sweep_pts(center=0, span=.3, num=31)

        # large range
        a = self.measure_motzoi(MC=MC, motzoi_amps=motzois, analyze=True,  disable_metadata = disable_metadata)
        opt_motzoi = a.get_intersect()[0]
        if opt_motzoi > max(motzois) or opt_motzoi < min(motzois):
            if verbose:
                print('optimal motzoi {:.3f} '.format(opt_motzoi) +
                      'outside of measured span, aborting')
            return False
        if update:
            if using_VSM:
                if verbose:
                    print('Setting motzoi to {:.3f}'.format(opt_motzoi))
                self.mw_vsm_D_amp(opt_motzoi)
            else:
                self.mw_motzoi(opt_motzoi)
        return opt_motzoi

    def calibrate_mixer_offsets_drive(
            self,
            mixer_channels=['G', 'D'],
            update: bool = True,
            ftarget=-110,
            maxiter=300
    ) -> bool:
        # USED_BY: device_dependency_graphs.py
        """
        Calibrates the mixer offset and updates the I and Q offsets in
        the qubit object.

        Args:
            mixer_channels (list):
                No use in no-VSM case
                With VSM specifies whether to calibrate offsets for both
                gaussuan 'G' and derivarive 'D' channel

            update (bool):
                should optimal values be set in the qubit object

            ftarget (float): power of the signal at the LO frequency
                for which the optimization is terminated
        """

        # turn relevant channels on

        using_VSM = self.cfg_with_vsm()
        MW_LutMan = self.instr_LutMan_MW.get_instr()
        AWG = MW_LutMan.AWG.get_instr()

        if using_VSM:
            if AWG.__class__.__name__ == 'QuTech_AWG_Module':
                chGI_par = AWG.parameters['ch1_offset']
                chGQ_par = AWG.parameters['ch2_offset']
                chDI_par = AWG.parameters['ch3_offset']
                chDQ_par = AWG.parameters['ch4_offset']

            else:
                # This part is AWG8 specific and wont work with a QWG
                awg_ch = self.mw_awg_ch()
                AWG.stop()
                AWG.set('sigouts_{}_on'.format(awg_ch - 1), 1)
                AWG.set('sigouts_{}_on'.format(awg_ch + 0), 1)
                AWG.set('sigouts_{}_on'.format(awg_ch + 1), 1)
                AWG.set('sigouts_{}_on'.format(awg_ch + 2), 1)

                chGI_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch - 1)]
                chGQ_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch + 0)]
                chDI_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch + 1)]
                chDQ_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch + 2)]
                # End of AWG8 specific part

            VSM = self.instr_VSM.get_instr()

            ch_in = self.mw_vsm_ch_in()
            # module 8 is hardcoded for mixer calibartions (signal hound)
            VSM.set('mod8_marker_source'.format(ch_in), 'int')
            VSM.set('mod8_ch{}_marker_state'.format(ch_in), 'on')

            # Calibrate Gaussian component mixer
            if 'G' in mixer_channels:
                VSM.set('mod8_ch{}_gaussian_amp'.format(ch_in), 1.0)
                VSM.set('mod8_ch{}_derivative_amp'.format(ch_in), 0.1)
                offset_I, offset_Q = cal_toolbox.mixer_carrier_cancellation(
                    SH=self.instr_SH.get_instr(),
                    source=self.instr_LO_mw.get_instr(),
                    MC=self.instr_MC.get_instr(),
                    chI_par=chGI_par, chQ_par=chGQ_par,
                    label='Mixer_offsets_drive_G' + self.msmt_suffix,
                    ftarget=ftarget, maxiter=maxiter)
                if update:
                    self.mw_mixer_offs_GI(offset_I)
                    self.mw_mixer_offs_GQ(offset_Q)
            if 'D' in mixer_channels:
                # Calibrate Derivative component mixer
                VSM.set('mod8_ch{}_gaussian_amp'.format(ch_in), 0.1)
                VSM.set('mod8_ch{}_derivative_amp'.format(ch_in), 1.0)

                offset_I, offset_Q = cal_toolbox.mixer_carrier_cancellation(
                    SH=self.instr_SH.get_instr(),
                    source=self.instr_LO_mw.get_instr(),
                    MC=self.instr_MC.get_instr(),
                    chI_par=chDI_par,
                    chQ_par=chDQ_par,
                    label='Mixer_offsets_drive_D' + self.msmt_suffix,
                    ftarget=ftarget, maxiter=maxiter)
                if update:
                    self.mw_mixer_offs_DI(offset_I)
                    self.mw_mixer_offs_DQ(offset_Q)

        else:
            if self._using_QWG():
                QWG_MW = self.instr_LutMan_MW.get_instr().AWG.get_instr()
                chI = self.instr_LutMan_MW.get_instr().channel_I()
                chQ = self.instr_LutMan_MW.get_instr().channel_Q()
                chI_par = QWG_MW.parameters['ch%s_offset' % chI]
                chQ_par = QWG_MW.parameters['ch%s_offset' % chQ]

                offset_I, offset_Q = cal_toolbox.mixer_carrier_cancellation(
                    SH=self.instr_SH.get_instr(),
                    source=self.instr_LO_mw.get_instr(),
                    MC=self.instr_MC.get_instr(),
                    chI_par=chI_par,
                    chQ_par=chQ_par,
                    ftarget=ftarget, maxiter=maxiter)
                if update:
                    self.mw_mixer_offs_GI(offset_I)
                    self.mw_mixer_offs_GQ(offset_Q)

            else:
                awg_ch = self.mw_awg_ch()
                AWG.stop()
                AWG.set('sigouts_{}_on'.format(awg_ch - 1), 1)
                AWG.set('sigouts_{}_on'.format(awg_ch + 0), 1)
                chGI_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch - 1)]
                chGQ_par = AWG.parameters['sigouts_{}_offset'.format(awg_ch + 0)]
                offset_I, offset_Q = cal_toolbox.mixer_carrier_cancellation(
                    SH=self.instr_SH.get_instr(),
                    source=self.instr_LO_mw.get_instr(),
                    MC=self.instr_MC.get_instr(),
                    chI_par=chGI_par, chQ_par=chGQ_par,
                    label='Mixer_offsets_drive' + self.msmt_suffix,
                    ftarget=ftarget, maxiter=maxiter)
                if update:
                    self.mw_mixer_offs_GI(offset_I)
                    self.mw_mixer_offs_GQ(offset_Q)

        return True

    def calibrate_optimal_weights(
            self,
            MC: Optional[MeasurementControl] = None,
            verify: bool = True,
            analyze: bool = True,
            update: bool = True,
            no_figs: bool = False,
            optimal_IQ: bool = False,
            measure_transients_CCL_switched: bool = False,
            prepare: bool = True,
            disable_metadata: bool = True,
            nr_shots_per_case: int = 2 ** 13,
            post_select: bool = False,
            averages: int = 2 ** 15,
            post_select_threshold: float = None,
            depletion_analysis: bool = False,
            depletion_optimization_window = None
    ) -> bool:
        """
        Measures readout transients for the qubit in ground and excited state to indicate
        at what times the transients differ. Based on the transients calculates weights
        that are used to  weigh measuremet traces to maximize the SNR.

        Args:
            optimal_IQ (bool):
                if set to True sets both the I and Q weights of the optimal
                weight functions for the verification experiment.
                A good sanity check is that when using optimal IQ one expects
                to see no signal in the  Q quadrature of the verification
                SSRO experiment.
            verify (bool):
                indicates whether to run measure_ssro at the end of the routine
                to find the new SNR and readout fidelities with optimized weights

            update (bool):
                specifies whether to update the weights in the qubit object
        """
        log.info('Calibrating optimal weights for {}'.format(self.name))
        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare:
            self.prepare_for_timedomain()

        # Ensure that enough averages are used to get accurate weights
        old_avg = self.ro_acq_averages()

        self.ro_acq_averages(averages)
        if measure_transients_CCL_switched:
            transients = self.measure_transients_CCL_switched(MC=MC,
                                                              analyze=analyze,
                                                              depletion_analysis=False)
        else:
            if depletion_analysis:
                a, transients = self.measure_transients(MC=MC, analyze=analyze,
                                                     depletion_analysis=depletion_analysis,
                                                     disable_metadata=disable_metadata,
                                                     depletion_optimization_window = depletion_optimization_window)
            else:
                transients = self.measure_transients(MC=MC, analyze=analyze,
                                                     depletion_analysis=depletion_analysis,
                                                     disable_metadata=disable_metadata)


        if analyze and depletion_analysis == False:
            ma.Input_average_analysis(IF=self.ro_freq_mod())

        self.ro_acq_averages(old_avg)
        # deskewing the input signal

        # Calculate optimal weights
        optimized_weights_I = (transients[1][0] - transients[0][0])
        optimized_weights_Q = (transients[1][1] - transients[0][1])
        # joint rescaling to +/-1 Volt
        maxI = np.max(np.abs(optimized_weights_I))
        maxQ = np.max(np.abs(optimized_weights_Q))
        # fixme: deviding the weight functions by four to not have overflow in
        # thresholding of the UHFQC
        weight_scale_factor = 1. / (4 * np.max([maxI, maxQ]))
        optimized_weights_I = np.array(weight_scale_factor * optimized_weights_I)
        optimized_weights_Q = np.array(weight_scale_factor * optimized_weights_Q)

        if update:
            self.ro_acq_weight_func_I(optimized_weights_I)
            self.ro_acq_weight_func_Q(optimized_weights_Q)
            if optimal_IQ:
                self.ro_acq_weight_type('optimal IQ')
            else:
                self.ro_acq_weight_type('optimal')
            if verify:
                self._prep_ro_integration_weights()
                self._prep_ro_instantiate_detectors()
                ssro_dict = self.measure_ssro(
                    no_figs=no_figs, update=update,
                    prepare=True, disable_metadata=disable_metadata,
                    nr_shots_per_case=nr_shots_per_case,
                    post_select=post_select,
                    post_select_threshold=post_select_threshold)
                return ssro_dict
        if verify:
            warnings.warn('Not verifying as settings were not updated.')

        if depletion_analysis:
            return a
        else:
            return True

    ##########################################################################
    # measure_ functions (overrides for class Qubit)
    # NB: functions closely related to overrides are also also included here
    ##########################################################################

    def measure_heterodyne_spectroscopy(
            self,
            freqs, MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            label=''
    ):
        # USED_BY: device_dependency_graphs.py (via find_resonator_frequency)
        """
        Measures a transmission through the feedline as a function of frequency.
        Usually used to find and characterize the resonators in routines such as
        find_resonators or find_resonator_frequency.

        Args:
            freqs (array):
                list of frequencies to sweep over

            analyze (bool):
                indicates whether to perform a hanger model
                fit to the data

            label (str):
                suffix to append to the measurement label
        """

        self.prepare_for_continuous_wave()

        if MC is None:
            MC = self.instr_MC.get_instr()

        # NB: the code replaced by this call contained an extra parameter "acq_length=self.ro_acq_integration_length()"
        # to UHFQC.spec_mode_on(), but that function no longer uses that parameter
        self.hal_acq_spec_mode_on()

        p = sqo.CW_RO_sequence(qubit_idx=self.cfg_qubit_nr(),
                               platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        MC.set_sweep_function(swf.Heterodyne_Frequency_Sweep_simple(
            MW_LO_source=self.instr_LO_ro.get_instr(),
            IF=self.ro_freq_mod()))
        MC.set_sweep_points(freqs)

        self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='Resonator_scan' + self.msmt_suffix + label)

        self.hal_acq_spec_mode_off()

        if analyze:
            ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)

    def measure_resonator_power(
            self,
            freqs,
            powers,
            MC: Optional[MeasurementControl] = None,
            analyze: bool = True,
            close_fig: bool = True,
            label: str = ''
    ):
        """
        Measures the readout resonator with UHFQC as a function of the pulse power.
        The pulse power is controlled by changing the amplitude of the UHFQC-generated
        waveform.

        Args:
            freqs (array):
                list of freqencies to sweep over

            powers (array):
                powers of the readout pulse to sweep over. The power is adjusted
                by changing the amplitude of the UHFQC output channels. Thereby
                the range of powers is limited by the dynamic range of mixers.
        """
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()

        p = sqo.CW_RO_sequence(qubit_idx=self.cfg_qubit_nr(), platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        MC.set_sweep_function(swf.Heterodyne_Frequency_Sweep_simple(
            MW_LO_source=self.instr_LO_ro.get_instr(),
            IF=self.ro_freq_mod()))
        MC.set_sweep_points(freqs)

        ro_lm = self.instr_LutMan_RO.get_instr()
        m_amp_par = ro_lm.parameters[
            'M_amp_R{}'.format(self.cfg_qubit_nr())]
        s2 = swf.lutman_par_dB_attenuation_UHFQC_dig_trig(
            LutMan=ro_lm, LutMan_parameter=m_amp_par)
        MC.set_sweep_function_2D(s2)
        MC.set_sweep_points_2D(powers)
        self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='Resonator_power_scan' + self.msmt_suffix + label, mode='2D')

        if analyze:
            ma.TwoD_Analysis(label='Resonator_power_scan',
                             close_fig=close_fig, normalize=True)

    def measure_ssro(
            self,
            MC: Optional[MeasurementControl] = None,
            nr_shots_per_case: int = 2 ** 13,  # 8192
            cases=('off', 'on'),
            prepare: bool = True,
            no_figs: bool = False,
            post_select: bool = False,
            post_select_threshold: float = None,
            nr_flux_dance: float = None,
            wait_time: float = None,
            update: bool = True,
            SNR_detector: bool = False,
            shots_per_meas: int = 2 ** 16,
            vary_residual_excitation: bool = True,
            disable_metadata: bool = True,
            label: str = ''
    ):
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs

        """
        Performs a number of single shot measurements with qubit in ground and excited state
        to extract the SNR and readout fidelities.

        Args:
            analyze (bool):
                should the analysis be executed

            nr_shots_per_case (int):
                total number of measurements in qubit ground and excited state

            cases:
                currently unused

            update_threshold (bool):
                indicating whether to update a threshold according
                to which the shot is classified as ground or excited state.

            prepare (bool):
                should the prepare_for_timedomain be executed?

            SNR_detector (bool):
                the function will return a dictionary suitable, making this function
                easier to use as a detector in the nested measurement

            shots_per_meas (int):
                number of single shot measurements per single
                acquisition with UHFQC

            vary_residual_excitation (bool):
                if False, uses the last known values of residual excitation
                and measurement induced relaxation and keeps these fixed.
            ...
        """

        # off and on, not including post selection init measurements yet
        nr_shots = nr_shots_per_case * 2

        old_RO_digit = self.ro_acq_digitized()
        self.ro_acq_digitized(False)

        if MC is None:
            MC = self.instr_MC.get_instr()

        if prepare:
            self.prepare_for_timedomain()

        # This snippet causes 0.08 s of overhead but is dangerous to bypass
        p = sqo.off_on(
            qubit_idx=self.cfg_qubit_nr(), pulse_comb='off_on',
            initialize=post_select,
            platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)

        # digitization setting is reset here but the detector still uses
        # the disabled setting that was set above
        self.ro_acq_digitized(old_RO_digit)

        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='Shot', unit='#',
                             upload=prepare)

        # save and change settings
        # plotting really slows down SSRO (16k shots plotting is slow)
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)

        MC.soft_avg(1)  # don't want to average single shots. FIXME changes state
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        d = self.int_log_det
        d.nr_shots = np.min([shots_per_meas, nr_shots])
        MC.set_detector_function(d)
        MC.run('SSRO_{}{}'.format(label, self.msmt_suffix),
               disable_snapshot_metadata=disable_metadata)

        # restore settings
        MC.live_plot_enabled(old_plot_setting)

        ######################################################################
        # SSRO Analysis
        ######################################################################
        if post_select_threshold == None:
            post_select_threshold = self.ro_acq_threshold()

        options_dict = {'post_select': post_select,
                        'nr_samples': 2 + 2 * post_select,
                        'post_select_threshold': post_select_threshold,
                        'predict_qubit_temp': True,
                        'qubit_freq': self.freq_qubit()}
        if not vary_residual_excitation:
            options_dict.update(
                {'fixed_p10': self.res_exc,
                 'fixed_p01': self.mmt_rel})

        a = ma2.ra.Singleshot_Readout_Analysis(
            options_dict=options_dict,
            extract_only=no_figs)

        ######################################################################
        # Update parameters in the qubit object based on the analysis
        ######################################################################
        if update:
            self.res_exc = a.proc_data_dict['quantities_of_interest']['residual_excitation']
            self.mmt_rel = a.proc_data_dict['quantities_of_interest']['relaxation_events']
            # UHFQC threshold is wrong, the magic number is a
            #  dirty hack. This works. we don't know why.
            magic_scale_factor = 1  # 0.655
            self.ro_acq_threshold(
                a.proc_data_dict['threshold_raw'] *
                magic_scale_factor)

            self.F_ssro(a.proc_data_dict['F_assignment_raw'])
            self.F_discr(a.proc_data_dict['F_discr'])
            self.ro_rel_events(a.proc_data_dict['quantities_of_interest']['relaxation_events'])
            self.ro_res_ext(a.proc_data_dict['quantities_of_interest']['residual_excitation'])

            warnings.warn("FIXME rotation angle could not be set")
            # self.ro_acq_rotated_SSB_rotation_angle(a.theta)

        return {'SNR': a.qoi['SNR'],
                'F_d': a.qoi['F_d'],
                'F_a': a.qoi['F_a'],
                'relaxation': a.proc_data_dict['relaxation_events'],
                'excitation': a.proc_data_dict['residual_excitation']}

    def measure_ssro_after_fluxing(
            self, 
            MC=None,
            nr_shots_per_case: int = 2**13,  # 8192
            cases=('off', 'on'),
            prepare: bool = True, 
            no_figs: bool = False,
            post_select: bool = False,
            post_select_threshold: float = None,
            nr_flux_after_init: float=None,
            flux_cw_after_init: Union[str, List[str]]=None,
            fluxed_qubit: str=None,
            wait_time_after_flux: float=0,
            update: bool = True,
            SNR_detector: bool = False,
            shots_per_meas: int = 2**16,
            vary_residual_excitation: bool = True,
            disable_metadata: bool = False, 
            label: str = ''
            ):
        """
        Performs a number of single shot measurements with qubit in ground and excited state
        to extract the SNR and readout fidelities.

        Args:
            analyze (bool):
                should the analysis be executed

            nr_shots_per_case (int):
                total number of measurements in qubit ground and excited state

            cases:
                currently unused

            update_threshold (bool):
                indicating whether to update a threshold according
                to which the shot is classified as ground or excited state.

            prepare (bool):
                should the prepare_for_timedomain be executed?

            SNR_detector (bool):
                the function will return a dictionary suitable, making this function
                easier to use as a detector in the nested measurement

            shots_per_meas (int):
                number of single shot measurements per single
                acquisition with UHFQC

            vary_residual_excitation (bool):
                if False, uses the last known values of residual excitation
                and measurement induced relaxation and keeps these fixed.
            ...
        """

        # off and on, not including post selection init measurements yet
        nr_shots = 2 * nr_shots_per_case

        old_RO_digit = self.ro_acq_digitized()
        self.ro_acq_digitized(False)

        if MC is None:
            MC = self.instr_MC.get_instr()

        # plotting really slows down SSRO (16k shots plotting is slow)
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)
        if prepare:
            self.prepare_for_timedomain()

        # This snippet causes 0.08 s of overhead but is dangerous to bypass
        p = sqo.off_on(
            qubit_idx=self.cfg_qubit_nr(), pulse_comb='off_on',
            nr_flux_after_init=nr_flux_after_init,
            flux_cw_after_init=flux_cw_after_init, 
            wait_time_after_flux=wait_time_after_flux, 
            fluxed_qubit_idx=self.find_instrument(fluxed_qubit).cfg_qubit_nr() \
                                if fluxed_qubit else None,
            initialize=post_select,
            platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)

        # digitization setting is reset here but the detector still uses
        # the disabled setting that was set above
        self.ro_acq_digitized(old_RO_digit)

        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='Shot', unit='#',
                             upload=prepare)
        MC.soft_avg(1)  # don't want to average single shots
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        d = self.int_log_det
        d.nr_shots = np.min([shots_per_meas, nr_shots])
        MC.set_detector_function(d)

        MC.run('SSRO_{}{}'.format(label, self.msmt_suffix),
               disable_snapshot_metadata=disable_metadata)
        MC.live_plot_enabled(old_plot_setting)

        ######################################################################
        # SSRO Analysis
        ######################################################################
        if post_select_threshold == None:
            post_select_threshold = self.ro_acq_threshold()

        options_dict = {'post_select': post_select,
                        'nr_samples': 2+2*post_select,
                        'post_select_threshold': post_select_threshold,
                        'predict_qubit_temp': True,
                        'qubit_freq': self.freq_qubit()}
        if not vary_residual_excitation:
            options_dict.update(
                {'fixed_p10': self.res_exc,
                 'fixed_p01': self.mmt_rel})

        a = ma2.ra.Singleshot_Readout_Analysis(
            options_dict=options_dict,
            extract_only=no_figs)

        ######################################################################
        # Update parameters in the qubit object based on the analysis
        ######################################################################
        if update:
            self.res_exc = a.proc_data_dict['quantities_of_interest']['residual_excitation']
            self.mmt_rel = a.proc_data_dict['quantities_of_interest']['relaxation_events']
            # UHFQC threshold is wrong, the magic number is a
            #  dirty hack. This works. we don't know why.
            magic_scale_factor = 1  # 0.655
            self.ro_acq_threshold(
                a.proc_data_dict['threshold_raw'] *
                magic_scale_factor)

            self.F_ssro(a.proc_data_dict['F_assignment_raw'])
            self.F_discr(a.proc_data_dict['F_discr'])
            self.ro_rel_events(
                a.proc_data_dict['quantities_of_interest']['relaxation_events'])
            self.ro_res_ext(
                a.proc_data_dict['quantities_of_interest']['residual_excitation'])

            log.warning("FIXME rotation angle could not be set")
            # self.ro_acq_rotated_SSB_rotation_angle(a.theta)

        return {'SNR': a.qoi['SNR'],
                'F_d': a.qoi['F_d'],
                'F_a': a.qoi['F_a'],
                'relaxation': a.proc_data_dict['relaxation_events'],
                'excitation': a.proc_data_dict['residual_excitation']}

    def measure_spectroscopy(
            self,
            freqs,
            mode='pulsed_marked',
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            label='',
            prepare_for_continuous_wave=True
    ):
        """
        Performs a two-tone spectroscopy experiment where one tone is kept
        fixed at the resonator readout frequency and another frequency is swept.

        args:
            freqs (array) : Frequency range you want to sweep
            mode  (string): 'CW' - Continuous wave
                            'pulsed_marked' - pulsed using trigger input of
                                              spec source
                            'pulsed_mixer' - pulsed using AWG and mixer
            analyze: indicates whether to look for the peak in the data
                and perform a fit
            label: suffix to append to the measurement label

        This experiment can be performed in three different modes
            Continuous wave (CW)
            Pulsed, marker modulated
            Pulsed, mixer modulated

        The mode argument selects which mode is being used and redirects the
        arguments to the appropriate method.
        """
        if mode == 'CW':
            self._measure_spectroscopy_CW(
                freqs=freqs, MC=MC,
                analyze=analyze, close_fig=close_fig,
                label=label,
                prepare_for_continuous_wave=prepare_for_continuous_wave
            )
        elif mode == 'pulsed_marked':
            self._measure_spectroscopy_pulsed_marked(
                freqs=freqs, MC=MC,
                analyze=analyze, close_fig=close_fig,
                label=label,
                prepare_for_continuous_wave=prepare_for_continuous_wave
            )
        elif mode == 'pulsed_mixer':
            self._measure_spectroscopy_pulsed_mixer(
                freqs=freqs, MC=MC,
                analyze=analyze, close_fig=close_fig,
                label=label,
                prepare_for_timedomain=prepare_for_continuous_wave
            )
        else:
            logging.error(f'Mode {mode} not recognized. Available modes: "CW", "pulsed_marked", "pulsed_mixer"')

    def measure_flux_frequency_timedomain(
        self,
        amplitude: float = None,
        times: list = np.arange(20e-9, 40e-9, 1/2.4e9),
        wait_time_flux: int = 0,
        disable_metadata: bool = False,
        analyze: bool = True,
        prepare_for_timedomain: bool = True,
        ):
        """
        Performs a cryoscope experiment to measure frequency
        detuning for a given flux pulse amplitude.
        Args:
            Times: 
                Flux pulse durations used for cryoscope trace.
            Amplitudes: 
                Amplitude of flux pulse used for cryoscope trace.
        Note on analysis: The frequency is calculated based on 
        a FFT of the cryoscope trace. This means the frequency
        resolution of this measurement will be given by the duration
        of the cryoscope trace. To minimize the duration of this 
        measurement we obtain the center frequency of the FFT by
        fitting it to a Lorentzian, which circumvents the frequency
        sampling.
        """
        assert self.ro_acq_weight_type()=='optimal'
        MC = self.instr_MC.get_instr()
        nested_MC = self.instr_nested_MC.get_instr()
        fl_lutman = self.instr_LutMan_Flux.get_instr()
        if amplitude:
            fl_lutman.sq_amp(amplitude)
        out_voltage = fl_lutman.sq_amp()*\
            fl_lutman.cfg_awg_channel_amplitude()*\
            fl_lutman.cfg_awg_channel_range()/2 # +/- 2.5V, else, 5Vpp
        if prepare_for_timedomain:
            self.prepare_for_timedomain()
            fl_lutman.load_waveforms_onto_AWG_lookuptable()
        p = mqo.Cryoscope(
            qubit_idxs=[self.cfg_qubit_nr()],
            flux_cw="fl_cw_06",
            wait_time_flux=wait_time_flux,
            platf_cfg=self.cfg_openql_platform_fn(),
            cc=self.instr_CC.get_instr().name,
            double_projections=False,
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)
        self.instr_CC.get_instr().start()
        sw_function = swf.FLsweep(fl_lutman, fl_lutman.sq_length,
                                  waveform_name="square")
        MC.set_sweep_function(sw_function)
        MC.set_sweep_points(times)
        values_per_point = 2
        values_per_point_suffex = ["cos", "sin"]
        d = self.get_int_avg_det(
            values_per_point=values_per_point,
            values_per_point_suffex=values_per_point_suffex,
            single_int_avg=True,
            always_prepare=False
        )
        MC.set_detector_function(d)
        label = f'Voltage_to_frequency_{out_voltage:.2f}V_{self.name}'
        MC.run(label,disable_snapshot_metadata=disable_metadata)
        # Run analysis
        if analyze:
            a = ma2.cv2.Time_frequency_analysis(
                label='Voltage_to_frequency')
            return a
    
    def calibrate_flux_arc(
        self,
        Times: list = np.arange(20e-9, 40e-9, 1/2.4e9),
        Amplitudes: list = [-0.4, -0.35, -0.3, 0.3, 0.35, 0.4],
        update: bool = True,
        disable_metadata: bool = False,
        prepare_for_timedomain: bool = True):
        """
        Calibrates the polynomial coeficients for flux (voltage) 
        to frequency conversion. Does so by measuring cryoscope traces
        at different amplitudes.
        Args:
            Times: 
                Flux pulse durations used to measure each
                cryoscope trace.
            Amplitudes: 
                DAC amplitudes of flux pulse used for each
                cryoscope trace.
        """
        assert self.ro_acq_weight_type()=='optimal'
        nested_MC = self.instr_nested_MC.get_instr()
        fl_lutman = self.instr_LutMan_Flux.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()
            fl_lutman.load_waveforms_onto_AWG_lookuptable()
        sw_function = swf.FLsweep(fl_lutman, fl_lutman.sq_amp,
                          waveform_name="square")
        nested_MC.set_sweep_function(sw_function)
        nested_MC.set_sweep_points(Amplitudes)
        def wrapper():
            a = self.measure_flux_frequency_timedomain(
                times = Times,
                disable_metadata=True,
                prepare_for_timedomain=False)
            return {'detuning':a.proc_data_dict['detuning']}
        d = det.Function_Detector(
            wrapper,
            result_keys=['detuning'],
            value_names=['detuning'],
            value_units=['Hz'])
        nested_MC.set_detector_function(d)
        label = f'Voltage_frequency_arc_{self.name}'
        nested_MC.run(label, disable_snapshot_metadata=disable_metadata)
        a = ma2.cv2.Flux_arc_analysis(label='Voltage_frequency_arc',
                    channel_amp=fl_lutman.cfg_awg_channel_amplitude(),
                    channel_range=fl_lutman.cfg_awg_channel_range())
        # Update detuning polynomial coeficients
        if update:
            p_coefs = a.qoi['P_coefs']
            fl_lutman.q_polycoeffs_freq_01_det(p_coefs)
        return a

# Adding measurement butterfly from pagani detached. RDC 16-02-2023

    def measure_msmt_butterfly(
            self,
            prepare_for_timedomain: bool = True,
            calibrate_optimal_weights: bool = False,
            nr_max_acq: int = 2**17,
            disable_metadata: bool = False,
            f_state: bool = False,
            no_figs: bool = False,
            opt_for = None,
            depletion_analysis: bool = False, 
            depletion_optimization_window = None):
        
        # ensure readout settings are correct
        assert self.ro_acq_weight_type() != 'optimal'
        assert self.ro_acq_digitized() == False

        if calibrate_optimal_weights:
            r = self.calibrate_optimal_weights(
                prepare=prepare_for_timedomain,
                verify=False, 
                optimal_IQ=True,
                disable_metadata=disable_metadata,
                depletion_analysis = depletion_analysis,
                depletion_optimization_window = depletion_optimization_window)

        if prepare_for_timedomain and calibrate_optimal_weights == False:
            self.prepare_for_timedomain()

        d = self.int_log_det
        # the msmt butterfly sequence has 3 measurements per state,
        # therefore we need to make sure the number of shots is a multiple of that
        uhfqc_max_avg = min(max(2**10, nr_max_acq), 2**20)

        if f_state:
            nr_measurements = 12
        else:
            nr_measurements = 8

        nr_shots = int((uhfqc_max_avg//nr_measurements) * nr_measurements)
        d.nr_shots = nr_shots
        p = sqo.butterfly(
            f_state = f_state,
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn()
        )
        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr()
        )
        MC = self.instr_MC.get_instr()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        print(nr_shots)
        MC.set_detector_function(d)
        MC.run(
            f"Measurement_butterfly_{self.name}_{f_state}",
            disable_snapshot_metadata=disable_metadata
        )
        a = ma2.ra.measurement_butterfly_analysis(
            qubit=self.name,
            label='butterfly',
            f_state=f_state,
            extract_only=no_figs)

        # calculate the cost function
        c = {}
        if opt_for == 'fidelity':
            c['ro_cost'] = 0.1 * r['depletion_cost'] + (3 - (a.qoi['Fidelity'] + a.qoi['p00_0'] + a.qoi['p11_1']))
        if opt_for == 'depletion':
            c['ro_cost'] = 1 * r['depletion_cost'] + 0.1 * (3 - (a.qoi['Fidelity'] + a.qoi['p00_0'] + a.qoi['p11_1']))
        if opt_for == 'total':
            c['ro_cost'] = 10 * r['depletion_cost'] + 10 * (1 - a.qoi['Fidelity']) + 2 - (a.qoi['p00_0'] + a.qoi['p11_1'])

        print('Important values:')
        print('- Depletion Cost: {}'.format(r['depletion_cost']))
        print('- Assignment Fidelity: {}%'.format(np.round(a.qoi['Fidelity'] * 100, 2)))
        print('- QND_g: {}%'.format(np.round(a.qoi['p00_0'] * 100, 2)))
        print('- QND_e: {}%'.format(np.round(a.qoi['p11_1'] * 100, 2)))
        print('- Readout Pulse Cost: {}'.format(c['ro_cost']))

        return c

###################################

    def measure_transients(
            self,
            MC: Optional[MeasurementControl] = None,
            analyze: bool = True,
            cases=('off', 'on'),
            prepare: bool = True,
            depletion_analysis: bool = True,
            depletion_analysis_plot: bool = True,
            depletion_optimization_window=None,
            disable_metadata: bool = False,
            plot_max_time=None,
            averages: int=2**15
    ):
        # docstring from parent class
        if MC is None:
            MC = self.instr_MC.get_instr()
        if plot_max_time is None:
            plot_max_time = self.ro_acq_integration_length() + 1000e-9

        # store the original averaging settings so that we can restore them at the end.
        old_avg = self.ro_acq_averages()
        self.ro_acq_averages(averages)

        if prepare:
            self.prepare_for_timedomain()

            # off/on switching is achieved by turning the MW source on and
            # off as this is much faster than recompiling/uploading
            p = sqo.off_on(
                qubit_idx=self.cfg_qubit_nr(), pulse_comb='on',
                initialize=False,
                platf_cfg=self.cfg_openql_platform_fn())
            self.instr_CC.get_instr().eqasm_program(p.filename)
        else:
            p = None  # object needs to exist for the openql_sweep to work

        transients = []
        for i, pulse_comb in enumerate(cases):
            if 'off' in pulse_comb.lower():
                self.instr_LO_mw.get_instr().off()
            elif 'on' in pulse_comb.lower():
                self.instr_LO_mw.get_instr().on()
            else:
                raise ValueError(f"pulse_comb {pulse_comb} not understood: Only 'on' and 'off' allowed.")

            s = swf.OpenQL_Sweep(openql_program=p,
                                 CCL=self.instr_CC.get_instr(),
                                 parameter_name='Transient time', unit='s',
                                 upload=prepare)
            MC.set_sweep_function(s)

            if 'UHFQC' in self.instr_acquisition():
                sampling_rate = 1.8e9
            else:
                raise NotImplementedError()
            MC.set_sweep_points(
                np.arange(self.input_average_detector.nr_samples) /
                sampling_rate)
            MC.set_detector_function(self.input_average_detector)
            data = MC.run(
                'Measure_transients{}_{}'.format(self.msmt_suffix, i),
                disable_snapshot_metadata=disable_metadata)

            dset = data['dset']
            transients.append(dset.T[1:])
            if analyze:
                ma.MeasurementAnalysis()
        
        # restore initial averaging settings.
        self.ro_acq_averages(old_avg)
        
        if depletion_analysis:
            print('Sweeping parameters:')
            print(r"- amp0 = {}".format(self.ro_pulse_up_amp_p0()))
            print(r"- amp1 = {}".format(self.ro_pulse_up_amp_p1()))
            print(r"- amp2 = {}".format(self.ro_pulse_down_amp0()))
            print(r"- amp3 = {}".format(self.ro_pulse_down_amp1()))
            print(r"- phi2 = {}".format(self.ro_pulse_down_phi0()))
            print(r"- phi3 = {}".format(self.ro_pulse_down_phi1()))

            a = ma.Input_average_analysis(
                IF=self.ro_freq_mod(),
                optimization_window=depletion_optimization_window,
                plot=depletion_analysis_plot,
                plot_max_time=plot_max_time)
            return a, [np.array(t, dtype=np.float64) for t in transients] # before it was only a 
        else:
            return [np.array(t, dtype=np.float64) for t in transients]

    def measure_rabi(
            self,
            MC: Optional[MeasurementControl] = None,
            amps=np.linspace(0, 1, 31),
            analyze=True,
            close_fig=True,
            real_imag=True,
            prepare_for_timedomain=True,
            all_modules=False
    ):
        """
        Perform a Rabi experiment in which amplitude of the MW pulse is sweeped
        while the drive frequency and pulse duration is kept fixed

        Args:
            amps (array):
                range of amplitudes to sweep. If cfg_with_vsm()==True pulse amplitude
                is adjusted by sweeping the attenuation of the relevant gaussian VSM channel,
                in max range (0.1 to 1.0).
                If cfg_with_vsm()==False adjusts the channel amplitude of the AWG in range (0 to 1).

        Relevant parameters:
            mw_amp180 (float):
                amplitude of the waveform corresponding to pi pulse (from 0 to 1)

            mw_channel_amp (float):
                AWG channel amplitude (digitally scaling the waveform; from 0 to 1)
        """

        if self.cfg_with_vsm():
            self.measure_rabi_vsm(
                MC,
                amps,
                analyze,
                close_fig,
                real_imag,
                prepare_for_timedomain,
                all_modules
            )
        else:
            self.measure_rabi_channel_amp(
                MC,
                amps,
                analyze,
                close_fig,
                real_imag,
                prepare_for_timedomain,
            )

    def measure_rabi_ramzz(
            self,
            measurement_qubit,
            ramzz_wait_time_ns,
            MC: Optional[MeasurementControl] = None,
            amps=np.linspace(0, 1, 31),
            analyze=True,
            close_fig=True,
            real_imag=True,
            prepare_for_timedomain=True,
            all_modules=False
    ):
        """
        Perform a Rabi experiment in which amplitude of the MW pulse is sweeped
        while the drive frequency and pulse duration is kept fixed

        Args:
            amps (array):
                range of amplitudes to sweep. If cfg_with_vsm()==True pulse amplitude
                is adjusted by sweeping the attenuation of the relevant gaussian VSM channel,
                in max range (0.1 to 1.0).
                If cfg_with_vsm()==False adjusts the channel amplitude of the AWG in range (0 to 1).

        Relevant parameters:
            mw_amp180 (float):
                amplitude of the waveform corresponding to pi pulse (from 0 to 1)

            mw_channel_amp (float):
                AWG channel amplitude (digitally scaling the waveform; from 0 to 1)
        """

        if self.cfg_with_vsm():
            self.measure_rabi_vsm(
                MC,
                amps,
                analyze,
                close_fig,
                real_imag,
                prepare_for_timedomain,
                all_modules
            )
        else:
            self.measure_rabi_channel_amp_ramzz(
                measurement_qubit,
                ramzz_wait_time_ns,
                MC,
                amps,
                analyze,
                close_fig,
                real_imag,
                prepare_for_timedomain
            )

    def measure_rabi_vsm(
            self,
            MC: Optional[MeasurementControl] = None,
            amps=np.linspace(0.1, 1.0, 31),
            analyze=True,
            close_fig=True,
            real_imag=True,
            prepare_for_timedomain=True,
            all_modules=False
    ):
        """
        Perform a Rabi experiment in which amplitude of the MW pulse is sweeped
        while the drive frequency and pulse duration is kept fixed

        Args:
            amps (array):
                range of amplitudes to sweep. Pulse amplitude is adjusted by sweeping
                the attenuation of the relevant gaussian VSM channel,
                in max range (0.1 to 1.0).
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        p = sqo.off_on(
            qubit_idx=self.cfg_qubit_nr(), pulse_comb='on',
            initialize=False,
            platf_cfg=self.cfg_openql_platform_fn())

        VSM = self.instr_VSM.get_instr()
        mod_out = self.mw_vsm_mod_out()
        ch_in = self.mw_vsm_ch_in()
        if all_modules:
            mod_sweep = []
            for i in range(8):
                VSM.set('mod{}_ch{}_marker_state'.format(i + 1, ch_in), 'on')
                G_par = VSM.parameters['mod{}_ch{}_gaussian_amp'.format(i + 1, ch_in)]
                D_par = VSM.parameters['mod{}_ch{}_derivative_amp'.format(i + 1, ch_in)]
                mod_sweep.append(swf.two_par_joint_sweep(
                    G_par, D_par, preserve_ratio=False))
            s = swf.multi_sweep_function(sweep_functions=mod_sweep,
                                         retrieve_value=True)
        else:
            G_par = VSM.parameters['mod{}_ch{}_gaussian_amp'.format(mod_out, ch_in)]
            D_par = VSM.parameters['mod{}_ch{}_derivative_amp'.format(mod_out, ch_in)]

            s = swf.two_par_joint_sweep(G_par, D_par, preserve_ratio=False,
                                        retrieve_value=True, instr=VSM)

        self.instr_CC.get_instr().eqasm_program(p.filename)

        MC.set_sweep_function(s)
        MC.set_sweep_points(amps)
        #  real_imag is acutally not polar and as such works for opt weights
        self.int_avg_det_single._set_real_imag(real_imag)  # FIXME: changes state
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='rabi_' + self.msmt_suffix)
        ma.Rabi_Analysis(label='rabi_')
        return True

    def measure_rabi_channel_amp(
            self,
            MC: Optional[MeasurementControl] = None,
            amps=np.linspace(0, 1, 31),
            analyze=True,
            close_fig=True,
            real_imag=True,
            prepare_for_timedomain=True
    ):
        """
        Perform a Rabi experiment in which amplitude of the MW pulse is sweeped
        while the drive frequency and pulse duration is kept fixed

        Args:
            amps (array):
                range of amplitudes to sweep. Amplitude is adjusted via the channel
                amplitude of the AWG, in max range (0 to 1).
        """

        MW_LutMan = self.instr_LutMan_MW.get_instr()

        if MC is None:
            MC = self.instr_MC.get_instr()

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        p = sqo.off_on(
            qubit_idx=self.cfg_qubit_nr(), pulse_comb='on',
            initialize=False,
            platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)

        s = MW_LutMan.channel_amp
        MC.set_sweep_function(s)
        MC.set_sweep_points(amps)
        # real_imag is actually not polar and as such works for opt weights
        self.int_avg_det_single._set_real_imag(real_imag)  # FIXME: changes state
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='rabi_' + self.msmt_suffix)

        ma.Rabi_Analysis(label='rabi_')
        return True
    
    def measure_rabi_channel_amp_ramzz(
            self,
            measurement_qubit,
            ramzz_wait_time_ns,
            MC: Optional[MeasurementControl] = None,
            amps=np.linspace(0, 1, 31),
            analyze=True,
            close_fig=True,
            real_imag=True,
            prepare_for_timedomain=True
    ):
        """
        Perform a Rabi experiment in which amplitude of the MW pulse is sweeped
        while the drive frequency and pulse duration is kept fixed

        Args:
            amps (array):
                range of amplitudes to sweep. Amplitude is adjusted via the channel
                amplitude of the AWG, in max range (0 to 1).
        """

        MW_LutMan = self.instr_LutMan_MW.get_instr()

        if MC is None:
            MC = self.instr_MC.get_instr()

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
            measurement_qubit.prepare_for_timedomain()

        p = sqo.off_on_ramzz(
            qubit_idx=self.cfg_qubit_nr(),
            measured_qubit_idx = measurement_qubit.cfg_qubit_nr(),
            ramzz_wait_time_ns = ramzz_wait_time_ns,
            pulse_comb='on',
            initialize=False,
            platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)

        s = MW_LutMan.channel_amp
        MC.set_sweep_function(s)
        MC.set_sweep_points(amps)
        # real_imag is actually not polar and as such works for opt weights
        measurement_qubit.int_avg_det_single._set_real_imag(real_imag)  # FIXME: changes state
        MC.set_detector_function(measurement_qubit.int_avg_det_single)
        MC.run(name='rabi_' + self.msmt_suffix)

        ma.Rabi_Analysis(label='rabi_')
        return True

    def measure_rabi_channel_amp_ramzz_measurement(self, meas_qubit,
                                                   ramzz_wait_time, MC=None,
                                                   amps=np.linspace(0, 1, 31),
                                                   analyze=True, close_fig=True,
                                                   real_imag=True,
                                                   prepare_for_timedomain=True):
        """
        Perform a Rabi experiment in which amplitude of the MW pulse is sweeped
        while the drive frequency and pulse duration is kept fixed
        Args:
            meas_qubit (ccl transmon):
                qubit used to read out self.
            ramzz_wait_time (float):
                wait time in-between ramsey pi/2 pulses.
            amps (array):
                range of amplitudes to sweep. Amplitude is adjusted via the channel
                amplitude of the AWG, in max range (0 to 1).
        """

        MW_LutMan = self.instr_LutMan_MW.get_instr()

        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()
            meas_qubit.prepare_for_timedomain()
        p = sqo.off_on_ramzz_measurement(
            inv_qubit_idx=self.cfg_qubit_nr(),
            meas_qubit_idx=meas_qubit.cfg_qubit_nr(),
            pulse_comb='on',
            initialize=False,
            platf_cfg=self.cfg_openql_platform_fn(),
            ramzz_wait_time_ns=int(ramzz_wait_time*1e9))
        self.instr_CC.get_instr().eqasm_program(p.filename)

        s = MW_LutMan.channel_amp
        MC.set_sweep_function(s)
        MC.set_sweep_points(amps)

        # real_imag is acutally not polar and as such works for opt weights
        self.int_avg_det_single._set_real_imag(real_imag)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='rabi_'+self.name+'_ramzz_'+meas_qubit.name)
        ma.Rabi_Analysis(label='rabi_')
        return True

    def measure_depletion_allxy(self, MC=None,
                                analyze=True, close_fig=True,
                                prepare_for_timedomain=True,
                                label=''):
        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        p = sqo.depletion_AllXY(qubit_idx=self.cfg_qubit_nr(),
                      platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(21*2*3))
        MC.set_detector_function(d)
        MC.run('Depletion_AllXY'+self.msmt_suffix+label)
        ma2.mra.Depletion_AllXY_analysis(self.name, label='Depletion')

    def measure_allxy(
            self,
            MC: Optional[MeasurementControl] = None,
            label: str = '',
            analyze=True,
            close_fig=True,
            prepare_for_timedomain=True,
            disable_metadata = False
    ) -> float:
        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        p = sqo.AllXY(qubit_idx=self.cfg_qubit_nr(), double_points=True,
                      platf_cfg=self.cfg_openql_platform_fn())

        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(42))
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('AllXY' + label + self.msmt_suffix, disable_snapshot_metadata = disable_metadata)

        if analyze:
            a = ma.AllXY_Analysis(close_main_fig=close_fig)
            return a.deviation_total

    def allxy_GBT(  # FIXME: prefix with "measure_"
            self,
            MC: Optional[MeasurementControl] = None,
            label: str = '',
            analyze=True,
            close_fig=True,
            prepare_for_timedomain=True,
            termination_opt=0.02):
        # USED_BY: inspire_dependency_graph.py,
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs
        '''
        This function is the same as measure AllXY, but with a termination limit
        This termination limit is as a system metric to evalulate the calibration
        by GBT if good or not.
        '''
        old_avg = self.ro_soft_avg()
        self.ro_soft_avg(4)

        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        p = sqo.AllXY(qubit_idx=self.cfg_qubit_nr(), double_points=True,
                      platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())

        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(42))
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('AllXY' + label + self.msmt_suffix)

        self.ro_soft_avg(old_avg)
        a = ma.AllXY_Analysis(close_main_fig=close_fig)
        if a.deviation_total > termination_opt:
            return False
        else:
            return True

    def measure_T1(
            self,
            times=None,
            update=True,
            nr_cz_instead_of_idle_time: list = None,
            qb_cz_instead_of_idle_time: str = None,
            nr_flux_dance: float = None,
            wait_time_after_flux_dance: float = 0,
            prepare_for_timedomain=True,
            close_fig=True,
            analyze=True,
            MC: Optional[MeasurementControl] = None,
            disable_metadata: bool = False,
            auto = True
    ):
        # USED_BY: inspire_dependency_graph.py,
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs
        # FIXME: split into basic T1 and T1 with flux dance
        """
        N.B. this is a good example for a generic timedomain experiment using the HAL_Transmon.
        """

        if times is not None and nr_cz_instead_of_idle_time is not None:
            raise ValueError("Either idle time or CZ mode must be chosen!")

        if nr_cz_instead_of_idle_time is not None and not qb_cz_instead_of_idle_time:
            raise ValueError("If CZ instead of idle time should be used, qubit to apply CZ to must be given!")

        if qb_cz_instead_of_idle_time:
            qb_cz_idx = self.find_instrument(qb_cz_instead_of_idle_time).cfg_qubit_nr()

        if MC is None:
            MC = self.instr_MC.get_instr()

        if times is None:
            if nr_cz_instead_of_idle_time is not None:
                # convert given numbers of CZs into time
                # NOTE: CZ time hardcoded to 40ns!
                times = np.array(nr_cz_instead_of_idle_time) * 40e-9
            else:
                # default timing: 4 x current T1
                times = np.linspace(0, self.T1() * 4, 31)

        if nr_cz_instead_of_idle_time is not None:
            # define time for calibration points at sufficiently distant times
            dt = 10 * 40e-9  # (times[-1] - times[-2])/2
        else:
            # append the calibration points, times are for location in plot
            dt = times[1] - times[0]

        times = np.concatenate([times, (times[-1] + 1 * dt,
                                        times[-1] + 2 * dt,
                                        times[-1] + 3 * dt,
                                        times[-1] + 4 * dt)])

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        p = sqo.T1(
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn(),
            times=times,
            nr_cz_instead_of_idle_time=nr_cz_instead_of_idle_time,
            qb_cz_idx=qb_cz_idx if qb_cz_instead_of_idle_time else None,
            nr_flux_dance=nr_flux_dance,
            wait_time_after_flux_dance=wait_time_after_flux_dance
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            parameter_name='Time',
            unit='s',
            CCL=self.instr_CC.get_instr()
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('T1' + self.msmt_suffix, disable_snapshot_metadata = disable_metadata)

        if analyze:
            a = ma.T1_Analysis(auto=auto, close_fig=True)
            if update:
                self.T1(a.T1)
            return a.T1

    def measure_T1_ramzz(
            self,
            measurement_qubit,
            ramzz_wait_time_ns,
            times=None,
            update=True,
            nr_flux_dance: float = None,
            prepare_for_timedomain=True,
            close_fig=True,
            analyze=True,
            MC: Optional[MeasurementControl] = None,
    ):
        # USED_BY: inspire_dependency_graph.py,
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs
        # FIXME: split into basic T1 and T1 with flux dance
        """
        N.B. this is a good example for a generic timedomain experiment using the HAL_Transmon.
        """

        if MC is None:
            MC = self.instr_MC.get_instr()

        if times is None:
            times = np.linspace(0, self.T1() * 4, 31)

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]

        times = np.concatenate([times, (times[-1] + 1 * dt,
                                        times[-1] + 2 * dt,
                                        times[-1] + 3 * dt,
                                        times[-1] + 4 * dt)])

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
            measurement_qubit.prepare_for_timedomain()

        p = sqo.T1_ramzz(
            qubit_idx=self.cfg_qubit_nr(),
            measured_qubit_idx = measurement_qubit.cfg_qubit_nr(),
            ramzz_wait_time_ns = ramzz_wait_time_ns,
            platf_cfg=self.cfg_openql_platform_fn(),
            times=times,
            nr_flux_dance=nr_flux_dance,
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            parameter_name='Time',
            unit='s',
            CCL=self.instr_CC.get_instr()
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        d = measurement_qubit.int_avg_det
        MC.set_detector_function(d)
        MC.run('T1' + self.msmt_suffix)

        if analyze:
            a = ma.T1_Analysis(auto=True, close_fig=True)
            if update:
                self.T1(a.T1)
            return a.T1

    def measure_T1_2nd_excited_state(
            self,
            times=None,
            MC: Optional[MeasurementControl] = None,
            analyze=True, close_fig=True, update=True,
            prepare_for_timedomain=True):
        """
        Performs a T1 experiment on the 2nd excited state.

        Note: changes pulses on instr_LutMan_MW
        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            times = np.linspace(0, self.T1() * 4, 31)

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        # Load pulses to the ef transition
        mw_lutman = self.instr_LutMan_MW.get_instr()
        mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()

        p = sqo.T1_second_excited_state(
            times,
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn()
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            parameter_name='Time',
            unit='s',
            CCL=self.instr_CC.get_instr()
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(p.sweep_points)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('T1_2nd_exc_state_' + self.msmt_suffix)

        a = ma.T1_Analysis(auto=True, close_fig=True)
        return a.T1

    def measure_ramsey(
            self,
            times=None,
            MC: Optional[MeasurementControl] = None,
            artificial_detuning: float = None,
            freq_qubit: float = None,
            label: str = '',
            prepare_for_timedomain=True,
            analyze=True,
            close_fig=True,
            update=True,
            detector=False,
            double_fit=False,
            test_beating=True, 
            disable_metadata = False
    ):
        # USED_BY: inspire_dependency_graph.py,
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs

        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            # funny default is because there is no real time sideband modulation
            stepsize = max((self.T2_star() * 4 / 61) // (abs(self.cfg_cycle_time()))
                           * abs(self.cfg_cycle_time()), 40e-9)
            times = np.arange(0, self.T2_star() * 4, stepsize)

        if artificial_detuning is None:
            # artificial_detuning = 0
            # raise ImplementationError("Artificial detuning does not work, currently uses real detuning")
            # artificial_detuning = 3/times[-1]
            artificial_detuning = 5 / times[-1]

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1] + 1 * dt,
                                 times[-1] + 2 * dt,
                                 times[-1] + 3 * dt,
                                 times[-1] + 4 * dt)])

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        # adding 'artificial' detuning by detuning the qubit LO
        if freq_qubit is None:
            freq_qubit = self.freq_qubit()
        # FIXME: this should have no effect if artificial detuning = 0. This is a bug,
        # this is real detuning, not artificial detuning
        old_frequency = self.instr_LO_mw.get_instr().get('frequency')
        self.instr_LO_mw.get_instr().set(
            'frequency', freq_qubit -
                         self.mw_freq_mod.get() + artificial_detuning)

        p = sqo.Ramsey(
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn(),
            times=times
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name='Time',
            unit='s'
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('Ramsey' + label + self.msmt_suffix, disable_snapshot_metadata = disable_metadata)

        # Restore old frequency value
        self.instr_LO_mw.get_instr().set('frequency', old_frequency)

        if analyze:
            a = ma.Ramsey_Analysis(
                auto=True,
                close_fig=True,
                freq_qubit=freq_qubit,
                artificial_detuning=artificial_detuning
            )
            if test_beating and a.fit_res.chisqr > 0.4:
                logging.warning('Found double frequency in Ramsey: large '
                                'deviation found in single frequency fit.'
                                'Trying double frequency fit.')
                double_fit = True
            if update:
                self.T2_star(a.T2_star['T2_star'])
            if double_fit:
                b = ma.DoubleFrequency()
                res = {
                    'T2star1': b.tau1,
                    'T2star2': b.tau2,
                    'frequency1': b.f1,
                    'frequency2': b.f2
                }
                return res

            else:
                res = {
                    'T2star': a.T2_star['T2_star'],
                    'frequency': a.qubit_frequency,
                }
                return res

    def measure_ramsey_ramzz(
            self,
            measurement_qubit,
            ramzz_wait_time_ns,
            times=None,
            MC: Optional[MeasurementControl] = None,
            artificial_detuning: float = None,
            freq_qubit: float = None,
            label: str = '',
            prepare_for_timedomain=True,
            analyze=True,
            close_fig=True,
            update=True,
            detector=False,
            double_fit=False,
            test_beating=True
    ):
        # USED_BY: inspire_dependency_graph.py,
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs

        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            # funny default is because there is no real time sideband modulation
            stepsize = max((self.T2_star() * 4 / 61) // (abs(self.cfg_cycle_time()))
                           * abs(self.cfg_cycle_time()), 40e-9)
            times = np.arange(0, self.T2_star() * 4, stepsize)

        if artificial_detuning is None:
            # artificial_detuning = 0
            # raise ImplementationError("Artificial detuning does not work, currently uses real detuning")
            # artificial_detuning = 3/times[-1]
            artificial_detuning = 5 / times[-1]

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1] + 1 * dt,
                                 times[-1] + 2 * dt,
                                 times[-1] + 3 * dt,
                                 times[-1] + 4 * dt)])

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
            measurement_qubit.prepare_for_timedomain()

        # adding 'artificial' detuning by detuning the qubit LO
        if freq_qubit is None:
            freq_qubit = self.freq_qubit()
        # FIXME: this should have no effect if artificial detuning = 0. This is a bug,
        # this is real detuning, not artificial detuning
        old_frequency = self.instr_LO_mw.get_instr().get('frequency')
        self.instr_LO_mw.get_instr().set(
            'frequency', freq_qubit -
                         self.mw_freq_mod.get() + artificial_detuning)

        p = sqo.Ramsey_ramzz(
            qubit_idx=self.cfg_qubit_nr(),
            measured_qubit_idx = measurement_qubit.cfg_qubit_nr(),
            ramzz_wait_time_ns = ramzz_wait_time_ns,
            platf_cfg=self.cfg_openql_platform_fn(),
            times=times
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name='Time',
            unit='s'
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        d = measurement_qubit.int_avg_det
        MC.set_detector_function(d)
        MC.run('Ramsey' + label + self.msmt_suffix)

        # Restore old frequency value
        self.instr_LO_mw.get_instr().set('frequency', old_frequency)

        if analyze:
            a = ma.Ramsey_Analysis(
                auto=True,
                close_fig=True,
                freq_qubit=freq_qubit,
                artificial_detuning=artificial_detuning
            )
            if test_beating and a.fit_res.chisqr > 0.4:
                logging.warning('Found double frequency in Ramsey: large '
                                'deviation found in single frequency fit.'
                                'Trying double frequency fit.')
                double_fit = True
            if update:
                self.T2_star(a.T2_star['T2_star'])
            if double_fit:
                b = ma.DoubleFrequency()
                res = {
                    'T2star1': b.tau1,
                    'T2star2': b.tau2,
                    'frequency1': b.f1,
                    'frequency2': b.f2
                }
                return res

            else:
                res = {
                    'T2star': a.T2_star['T2_star'],
                    'frequency': a.qubit_frequency,
                }
                return res

    def measure_complex_ramsey(
            self,
            times=None,
            MC: Optional[MeasurementControl] = None,
            freq_qubit: float = None,
            label: str = '',
            prepare_for_timedomain=True,
            analyze=True, close_fig=True, update=True,
            detector=False,
            double_fit=False,
            test_beating=True
    ):
        if MC is None:
            MC = self.instr_MC.get_instr()

        # readout must use IQ data
        old_ro_type = self.ro_acq_weight_type()
        self.ro_acq_weight_type('optimal IQ')

        # default timing
        if times is None:
            # funny default is because there is no real time sideband
            # modulation
            stepsize = max((self.T2_star() * 4 / 61) // (abs(self.cfg_cycle_time()))
                           * abs(self.cfg_cycle_time()), 40e-9)
            times = np.arange(0, self.T2_star() * 4, stepsize)

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        times = np.concatenate([np.repeat(times, 2),
                                (times[-1] + 1 * dt,
                                 times[-1] + 2 * dt,
                                 times[-1] + 3 * dt,
                                 times[-1] + 4 * dt)])

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        # adding 'artificial' detuning by detuning the qubit LO
        if freq_qubit is None:
            freq_qubit = self.freq_qubit()
        # # this should have no effect if artificial detuning = 0. This is a bug,
        # This is real detuning, not artificial detuning

        p = sqo.complex_Ramsey(times, qubit_idx=self.cfg_qubit_nr(),
                               platf_cfg=self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr(),
                             parameter_name='Time', unit='s')
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)

        d = self.int_avg_det
        MC.set_detector_function(d)

        MC.run('complex_Ramsey' + label + self.msmt_suffix)
        self.ro_acq_weight_type(old_ro_type)

        if analyze:
            a = ma2.ComplexRamseyAnalysis(label='complex_Ramsey', close_figs=True)
            if update:
                fit_res = a.fit_dicts['exp_fit']['fit_res']
                fit_frequency = fit_res.params['frequency'].value
                freq_qubit = self.freq_qubit()
                self.freq_qubit(freq_qubit + fit_frequency)
            # if test_beating and a.fit_res.chisqr > 0.4:
            #     logging.warning('Found double frequency in Ramsey: large '
            #                     'deviation found in single frequency fit.'
            #                     'Trying double frequency fit.')
            #     double_fit = True
            # if update:
            #     self.T2_star(a.T2_star['T2_star'])
            # if double_fit:
            #     b = ma.DoubleFrequency()
            #     res = {
            #         'T2star1': b.tau1,
            #         'T2star2': b.tau2,
            #         'frequency1': b.f1,
            #         'frequency2': b.f2
            #     }
            #     return res

            # else:
            #     res = {
            #         'T2star': a.T2_star['T2_star'],
            #         'frequency': a.qubit_frequency,
            #     }
            #     return res

    def measure_echo(
            self,
            times=None,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            update=True,
            label: str = '',
            prepare_for_timedomain=True, 
            disable_metadata = False
    ):
        # USED_BY: inspire_dependency_graph.py,
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs
        """
        Note: changes pulses on instr_LutMan_MW

        Args:
            times:
            MC:
            analyze:
            close_fig:
            update:
            label:
            prepare_for_timedomain:

        Returns:

        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            # Old formulation of the time vector
            ## funny default is because there is no real time sideband
            ## modulation
            #stepsize = max((self.T2_echo() * 2 / 61) // (abs(self.cfg_cycle_time()))
            #               * abs(self.cfg_cycle_time()), 20e-9)
            #times = np.arange(0, self.T2_echo() * 4, stepsize * 2)

            # New version by LDC. 022/09/13
            # I want all T2echo experiments to have the same number of time values.
            numpts=51
            stepsize = max((self.T2_echo() * 4 / (numpts-1)) // 40e-9, 1) * 40.0e-9
            times = np.arange(0, numpts*stepsize, stepsize)

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1] + 1 * dt,
                                 times[-1] + 2 * dt,
                                 times[-1] + 3 * dt,
                                 times[-1] + 4 * dt)])

        # Checking if pulses are on 20 ns grid
        if not all([np.round(t * 1e9) % (2 * self.cfg_cycle_time() * 1e9) == 0 for t in times]):
            raise ValueError('timesteps must be multiples of 40 ns')

        # Checking if pulses are locked to the pulse modulation
        mw_lutman = self.instr_LutMan_MW.get_instr()
        if not all([np.round(t / 1 * 1e9) % (2 / self.mw_freq_mod.get() * 1e9) == 0 for t in times]) and \
                mw_lutman.cfg_sideband_mode() != 'real-time':
            raise ValueError('timesteps must be multiples of 2 modulation periods')

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        mw_lutman.load_phase_pulses_to_AWG_lookuptable()

        p = sqo.echo(
            times,
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn()
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name="Time",
            unit="s"
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('echo' + label + self.msmt_suffix, disable_snapshot_metadata = disable_metadata)

        if analyze:
            # N.B. v1.5 analysis
            a = ma.Echo_analysis_V15(label='echo', auto=True, close_fig=True)
            if update:
                self.T2_echo(a.fit_res.params['tau'].value)
            return a

    def measure_echo_ramzz(self,
                           measurement_qubit,
                           ramzz_wait_time_ns,
                           times=None, MC=None,
                           analyze=True, close_fig=True, update=True,
                           label: str = '', prepare_for_timedomain=True):
        # docstring from parent class
        # N.B. this is a good example for a generic timedomain experiment using
        # the CCL transmon.
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            # funny default is because there is no real time sideband
            # modulation
            stepsize = max((self.T2_echo()*2/61)//(abs(self.cfg_cycle_time()))
                            * abs(self.cfg_cycle_time()), 20e-9)
            times = np.arange(0, self.T2_echo()*4, stepsize*2)

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1]+1*dt,
                                    times[-1]+2*dt,
                                    times[-1]+3*dt,
                                    times[-1]+4*dt)])

        mw_lutman = self.instr_LutMan_MW.get_instr()
        # # Checking if pulses are on 20 ns grid
        if not all([np.round(t*1e9) % (2*self.cfg_cycle_time()*1e9) == 0 for
                    t in times]):
            raise ValueError('timesteps must be multiples of 40e-9')

        # # Checking if pulses are locked to the pulse modulation
        if not all([np.round(t/1*1e9) % (2/self.mw_freq_mod.get()*1e9) == 0 for t in times]) and\
            mw_lutman.cfg_sideband_mode() != 'real-time':
            raise ValueError(
                'timesteps must be multiples of 2 modulation periods')

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
            measurement_qubit.prepare_for_timedomain()

        mw_lutman.load_phase_pulses_to_AWG_lookuptable()
        p = sqo.echo_ramzz(times,
                            qubit_idx=self.cfg_qubit_nr(),
                            measurement_qubit_idx=measurement_qubit.cfg_qubit_nr(),
                            ramzz_wait_time_ns=ramzz_wait_time_ns,
                            platf_cfg=self.cfg_openql_platform_fn())

        s = swf.OpenQL_Sweep(openql_program=p,
                                CCL=self.instr_CC.get_instr(),
                                parameter_name="Time", unit="s")
        d = measurement_qubit.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('echo_'+label+self.name+'_ramzz_'+measurement_qubit.name)
        if analyze:
            # N.B. v1.5 analysis
            a = ma.Echo_analysis_V15(label='echo', auto=True, close_fig=True)
            if update:
                self.T2_echo(a.fit_res.params['tau'].value)
            return a


    def measure_restless_ramsey(
            self,
            amount_of_repetitions=None,
            time=None,
            amount_of_shots=2**20,
            MC: Optional[MeasurementControl] = None,
            prepare_for_timedomain=True
    ):
        label = f"Restless_Ramsey_N={amount_of_repetitions}_tau={time}"

        if MC is None:
            MC = self.instr_MC.get_instr()

        old_weight_type = self.ro_acq_weight_type()
        old_digitized = self.ro_acq_digitized()
        self.ro_acq_weight_type('optimal')
        self.ro_acq_digitized(False)

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        else:
            self.prepare_readout()
        MC.soft_avg(1)

        p = sqo.Restless_Ramsey(
            time=time,
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn()
        )
        s = swf.OpenQL_Sweep(openql_program=p, CCL=self.instr_CC.get_instr())
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(amount_of_repetitions * amount_of_shots))

        d = self.int_log_det
        d.nr_shots = amount_of_shots # int(4094/nr_repetitions) * nr_repetitions
        MC.set_detector_function(d)
        MC.run(label + self.msmt_suffix)
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

    def measure_flipping(
            self,
            number_of_flips=np.arange(0, 61, 2),
            equator=True,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            update=False,
            ax='x',
            angle='180',
            disable_metadata = False):
        """
        Measurement for fine-tuning of the pi and pi/2 pulse amplitudes. Executes sequence
        pi (repeated N-times) - pi/2 - measure
        with variable number N. In this way the error in the amplitude of the MW pi pulse
        accumulate allowing for fine tuning. Alternatively N repetitions of the pi pulse
        can be replaced by 2N repetitions of the pi/2-pulse

        Args:
            number_of_flips (array):
                number of pi pulses to apply. It is recommended to use only even numbers,
                since then the expected signal has a sine shape. Otherwise it has -1^N * sin shape
                which will not be correctly analyzed.

            equator (bool);
                specify whether to apply the final pi/2 pulse. Setting to False makes the sequence
                first-order insensitive to pi-pulse amplitude errors.

            ax (str {'x', 'y'}):
                axis arour which the pi pulses are to be performed. Possible values 'x' or 'y'

            angle (str {'90', '180'}):
                specifies whether to apply pi or pi/2 pulses. Possible values: '180' or '90'

            update (bool):
                specifies whether to update parameter controlling MW pulse amplitude.
                This parameter is mw_vsm_G_amp in VSM case or mw_channel_amp in no-VSM case.
                Update is performed only if change by more than 0.2% (0.36 deg) is needed.
        """

        if MC is None:
            MC = self.instr_MC.get_instr()

        # allow flipping only with pi/2 or pi, and x or y pulses
        assert angle in ['90', '180']
        assert ax.lower() in ['x', 'y']

        # append the calibration points, times are for location in plot
        nf = np.array(number_of_flips)
        dn = nf[1] - nf[0]
        nf = np.concatenate([nf,
                             (nf[-1] + 1 * dn,
                              nf[-1] + 2 * dn,
                              nf[-1] + 3 * dn,
                              nf[-1] + 4 * dn)])

        self.prepare_for_timedomain()

        p = sqo.flipping(
            number_of_flips=nf,
            equator=equator,
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn(),
            ax=ax.lower(),
            angle=angle
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            unit='#',
            CCL=self.instr_CC.get_instr()
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(nf)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('flipping_' + ax + angle + self.msmt_suffix, disable_snapshot_metadata = disable_metadata)

        if analyze:
            a = ma2.FlippingAnalysis(options_dict={'scan_label': 'flipping'})

            if update:
                # choose scale factor based on simple goodness-of-fit comparison
                # This method gives priority to the line fit:
                # the cos fit will only be chosen if its chi^2 relative to the
                # chi^2 of the line fit is at least 10% smaller
                scale_factor = a.get_scale_factor()

                # for debugging purposes
                print(scale_factor)

                if abs(scale_factor - 1) < 0.2e-3:
                    print('Pulse amplitude accurate within 0.02%. Amplitude not updated.')
                    return a

                if angle == '180':
                    if self.cfg_with_vsm():
                        amp_old = self.mw_vsm_G_amp()
                        self.mw_vsm_G_amp(scale_factor * amp_old)
                    else:
                        amp_old = self.mw_channel_amp()
                        self.mw_channel_amp(scale_factor * amp_old)
                elif angle == '90':
                    amp_old = self.mw_amp90_scale()
                    self.mw_amp90_scale(scale_factor * amp_old)

                print('Pulse amplitude for {}-{} pulse changed from {:.3f} to {:.3f}'.format(
                    ax, angle, amp_old, scale_factor * amp_old))

        return a

    def flipping_GBT(
            self, 
            nr_sequence: int = 7,                # max number of flipping iterations
            number_of_flips=np.arange(0, 31, 2), # specifies the number of pi pulses at each step
            eps=0.0005,
            disable_metadata = False):                           # specifies the GBT threshold
        # FIXME: prefix with "measure_"
        # USED_BY: inspire_dependency_graph.py,
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs.py
        '''
        This function is to measure flipping sequence for whatever nr_of times
        a function needs to be run to calibrate the Pi and Pi/2 Pulse.
        Right now this method will always return true no matter what
        Later we can add a condition as a check.
        '''

        ###############################################
        ###############################################
        # Monitor key temperatures of interest
        # ADDED BY LDC.  THIS IS A KLUGE!
        # CAREFUL, thsi is Quantum-Inspire specific!!!
        # thisTWPA1=self.find_instrument('TWPA_pump_1')
        # thisTWPA2=self.find_instrument('TWPA_pump_2')
        # #thisVSM=self.find_instrument('VSM')
        # TempTWPA1=thisTWPA1.temperature()
        # TempTWPA2=thisTWPA2.temperature()
        #TempVSM=thisVSM.temperature_avg()
        # for diagnostics only
        # print('Key temperatures (degC):')
        # print('='*35)
        # print(f'TWPA_Pump_1:\t{float(TempTWPA1):0.2f}')
        # print(f'TWPA_Pump_2:\t{float(TempTWPA2):0.2f}')
        # #print(f'VSM:\t\t{float(TempVSM):0.2f}')
        # print('='*35)
        ###############################################
        ###############################################

        for i in range(nr_sequence):
            a = self.measure_flipping(update=True, number_of_flips=number_of_flips, disable_metadata = disable_metadata)
            scale_factor = a.get_scale_factor()
            if abs(1 - scale_factor) <= eps:
                return True
        else:
            return False

    def measure_motzoi(
            self,
            motzoi_amps=None,
            prepare_for_timedomain: bool = True,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            disable_metadata = False
    ):
        # USED_BY: device_dependency_graphs.py (via calibrate_motzoi)
        """
        Sweeps the amplitude of the DRAG coefficients looking for leakage reduction
        and optimal correction for the phase error due to stark shift resulting
        from transition to higher qubit states. In this measurement the two-pulse
        sequence are applied:
        X180-Y90 and Y180-X90 and the amplitude of the gaussian-derivative component
        of the MW pulse is sweeped. When the DRAG coefficient is adjusted correctly
        the two sequences yield the same result.

        Refs:
        Motzoi PRL 103, 110501 (2009)
        Chow PRA 82, 040305(R) (2010)
        Lucero PRA 82, 042339 (2010)

        Args:
            motzoi_amps (array):
                DRAG coefficients to sweep over. In VSM case the amplitude
                is adjusted by varying attenuation of the derivative channel for the
                relevant module. In no-VSM the DRAG parameter is adjusted by reloading
                of the waveform on the AWG.

        Returns:
            float:
                value of the DRAG parameter for which the two sequences yield the same result
                error is mimimized.
        """
        using_VSM = self.cfg_with_vsm()
        MW_LutMan = self.instr_LutMan_MW.get_instr()

        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        p = sqo.motzoi_XY(
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn()
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)

        # determine swf_func and motzoi_amps
        if using_VSM:
            VSM = self.instr_VSM.get_instr()
            if motzoi_amps is None:
                motzoi_amps = np.linspace(0.1, 1.0, 31)
            mod_out = self.mw_vsm_mod_out()
            ch_in = self.mw_vsm_ch_in()
            D_par = VSM.parameters['mod{}_ch{}_derivative_amp'.format(mod_out, ch_in)]
            swf_func = wrap_par_to_swf(D_par, retrieve_value=True)
        else:
            if motzoi_amps is None:
                motzoi_amps = np.linspace(-.3, .3, 31)
            if self._using_QWG():
                swf_func = swf.QWG_lutman_par(
                    LutMan=MW_LutMan,
                    LutMan_parameter=MW_LutMan.mw_motzoi
                )
            else:
                swf_func = swf.lutman_par(
                    LutMan=MW_LutMan,
                    LutMan_parameter=MW_LutMan.mw_motzoi
                )

        MC.set_sweep_function(swf_func)
        MC.set_sweep_points(motzoi_amps)
        d = self.get_int_avg_det(
            single_int_avg=True,
            values_per_point=2,
            values_per_point_suffex=['yX', 'xY'],
            always_prepare=True
        )
        MC.set_detector_function(d)
        MC.run('Motzoi_XY' + self.msmt_suffix, disable_snapshot_metadata = disable_metadata)

        if analyze:
            if self.ro_acq_weight_type() == 'optimal':
                a = ma2.Intersect_Analysis(
                    options_dict={'ch_idx_A': 0,
                                  'ch_idx_B': 1},
                    normalized_probability=True)
            else:
                # if statement required if 2 channels readout
                logging.warning(
                    'It is recommended to do this with optimal weights')
                a = ma2.Intersect_Analysis(
                    options_dict={'ch_idx_A': 0,
                                  'ch_idx_B': 1},
                    normalized_probability=False)
            return a

    def measure_rabi_mw_crosstalk(self, MC=None, amps=np.linspace(0, 1, 31),
                             cross_driving_qubit=None,
                             analyze=True, close_fig=True, real_imag=True,
                             disable_metadata = False, 
                             prepare_for_timedomain=True):
        """
        Perform a Rabi experiment in which amplitude of the MW pulse is sweeped
        while the drive frequency and pulse duration is kept fixed

        Args:
            amps (array):
                range of amplitudes to sweep. Amplitude is adjusted via the channel
                amplitude of the AWG, in max range (0 to 1).
        """

        if cross_driving_qubit is not None:
            MW_LutMan = self.find_instrument(cross_driving_qubit).instr_LutMan_MW.get_instr()
            qubi_cd_idx = self.find_instrument(cross_driving_qubit).cfg_qubit_nr()
            self.find_instrument(cross_driving_qubit)._prep_td_sources()
            self.find_instrument(cross_driving_qubit)._prep_mw_pulses()

        else:
            MW_LutMan = self.instr_LutMan_MW.get_instr()

        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        
        p = sqo.off_on_mw_crosstalk(
            qubit_idx=self.cfg_qubit_nr(), pulse_comb='on',
            initialize=False,
            cross_driving_qubit=qubi_cd_idx if cross_driving_qubit else None,
            platf_cfg=self.cfg_openql_platform_fn())
        self.instr_CC.get_instr().eqasm_program(p.filename)

        s = MW_LutMan.channel_amp
        print(s)
        MC.set_sweep_function(s)
        MC.set_sweep_points(amps)
        # real_imag is acutally not polar and as such works for opt weights
        self.int_avg_det_single._set_real_imag(real_imag)
        MC.set_detector_function(self.int_avg_det_single)

        label = f'_drive_{cross_driving_qubit}' if cross_driving_qubit else '' 
        MC.run(name=f'rabi'+self.msmt_suffix+label,
               disable_snapshot_metadata=disable_metadata)
        a = None
        try:
            a = ma.Rabi_Analysis(label='rabi_')
        except Exception as e:
            warnings.warn("Failed to fit Rabi for the cross-driving case.")

        if a:
            return a

    def measure_mw_crosstalk(self, MC=None, amps=np.linspace(0, 1, 121),
                 cross_driving_qb=None,disable_metadata = False,
                 analyze=True, close_fig=True, real_imag=True,
                 prepare_for_timedomain=True):
        """
        Measure MW crosstalk matrix by measuring two Rabi experiments: 
        1. a0 : standand rabi (drive the qubit qj through its dedicated drive line Dj) 
        2. a1 : cross-drive rabi (drive the qubit qj through another drive line (Di)
         at the freq of the qj) 
        Args:
            amps (array):
                range of amplitudes to sweep. If cfg_with_vsm()==True pulse amplitude
                is adjusted by sweeping the attenuation of the relevant gaussian VSM channel,
                in max range (0.1 to 1.0).
                If cfg_with_vsm()==False adjusts the channel amplitude of the AWG in range (0 to 1).

            cross_driving_qubit is qubit qi with its drive line Di.
        Relevant parameters:
            mw_amp180 (float):
                amplitude of the waveform corresponding to pi pulse (from 0 to 1)

            mw_channel_amp (float):
                AWG channel amplitude (digitally scaling the waveform; form 0 to 1)
        """

        try:    
            freq_qj = self.freq_qubit() # set qi to this qubit freq of qubit j
            cross_driving_qubit = None
            amps=np.linspace(0, 0.1, 51)
            a0 = self.measure_rabi_mw_crosstalk(MC, amps,cross_driving_qubit,
                                          analyze, close_fig, real_imag,disable_metadata,
                                          prepare_for_timedomain)

            cross_driving_qubit = cross_driving_qb
            qi = self.find_instrument(cross_driving_qubit)
            freq_qi = qi.freq_qubit()
            qi.freq_qubit(freq_qj)
            amps=np.linspace(0, 1, 121)
            prepare_for_timedomain = False
            a1 = self.measure_rabi_mw_crosstalk(MC, amps,cross_driving_qubit,
                                          analyze, close_fig, real_imag,disable_metadata,
                                          prepare_for_timedomain)
            ## set back the right parameters. 
            qi.freq_qubit(freq_qi)
        except:
            print_exception()
            qi.freq_qubit(freq_qi)
            raise Exception('Experiment failed')

        try:
            pi_ajj = abs(a0.fit_result.params['period'].value) / 2
            pi_aji = abs(a1.fit_result.params['period'].value) / 2

            mw_isolation = 20*np.log10(pi_aji/pi_ajj)

            return mw_isolation
        except:
            mw_isolation = 80

    ##########################################################################
    # measure_ functions (HAL_Transmon specific, not present in parent class Qubit)
    ##########################################################################

    def measure_photon_number_splitting(
            self,
            freqs,
            powers,
            MC: Optional[MeasurementControl] = None,
            analyze: bool = True,
            close_fig: bool = True
    ):
        """
        Measures the CW qubit spectroscopy as a function of the RO pulse power
        to find a photon splitting.

        Refs:
        Schuster Nature 445, 515–518 (2007)
            (note that in the paper RO resonator has lower frequency than the qubit)

        Args:
            freqs (array):
                list of freqencies to sweep over

            powers (array):
                powers of the readout pulse to sweep over. The power is adjusted
                by changing the amplitude of the UHFQC output channels. Thereby
                the range of powers is limited by the dynamic range of mixers.
        """

        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()

        p = sqo.CW_RO_sequence(
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn()
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        spec_source = self.instr_spec_source.get_instr()
        spec_source.on()

        MC.set_sweep_function(spec_source.frequency)
        MC.set_sweep_points(freqs)

        ro_lm = self.instr_LutMan_RO.get_instr()
        m_amp_par = ro_lm.parameters['M_amp_R{}'.format(self.cfg_qubit_nr())]
        s2 = swf.lutman_par_dB_attenuation_UHFQC_dig_trig(
            LutMan=ro_lm, LutMan_parameter=m_amp_par)
        MC.set_sweep_function_2D(s2)
        MC.set_sweep_points_2D(powers)
        self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
        MC.set_detector_function(self.int_avg_det_single)
        label = 'Photon_number_splitting'
        MC.run(name=label + self.msmt_suffix, mode='2D')

        spec_source.off()

        if analyze:
            ma.TwoD_Analysis(label=label,
                             close_fig=close_fig, normalize=True)

    def measure_resonator_frequency_dac_scan(
            self,
            freqs,
            dac_values,
            MC: Optional[MeasurementControl] = None,
            analyze: bool = True,
            close_fig: bool = True,
            fluxChan=None,
            label=''
    ):
        """
        Performs the resonator spectroscopy as a function of the current applied
        to the flux bias line.

        Args:
            freqs (array):
                list of freqencies to sweep over

            dac_values (array):
                list of the DAC values (current values) to sweep over

            fluxChan (str):
                channel of the instrument controlling the flux to sweep. By default
                the channel used is specified by self.fl_dc_ch.

            analyze (bool):
                indicates whether to generate colormaps of the measured data

            label (str):
                suffix to append to the measurement label

        Relevant qubit parameters:
            instr_FluxCtrl (str):
                instrument controlling the current bias

            fluxChan (str):
                channel of the flux control instrument corresponding to the qubit
        """
        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()

        p = sqo.CW_RO_sequence(
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn()
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        MC.set_sweep_function(swf.Heterodyne_Frequency_Sweep_simple(
            MW_LO_source=self.instr_LO_ro.get_instr(),
            IF=self.ro_freq_mod()))
        MC.set_sweep_points(freqs)

        dac_par = self.hal_flux_get_parameters(fluxChan)
        # FIXME: the original code below ignores flucChan on ivvi, but not on SPI. This is probably a bug
        # if 'ivvi' in self.instr_FluxCtrl().lower():
        #     IVVI = self.instr_FluxCtrl.get_instr()
        #     dac_par = IVVI.parameters['dac{}'.format(self.fl_dc_ch())]
        # else:
        #     # Assume the flux is controlled using an SPI rack
        #     fluxcontrol = self.instr_FluxCtrl.get_instr()
        #     if fluxChan == None:
        #         dac_par = fluxcontrol.parameters[(self.fl_dc_ch())]
        #     else:
        #         dac_par = fluxcontrol.parameters[(fluxChan)]

        MC.set_sweep_function_2D(dac_par)
        MC.set_sweep_points_2D(dac_values)
        self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='Resonator_dac_scan' + self.msmt_suffix + label, mode='2D')

        if analyze:
            ma.TwoD_Analysis(label='Resonator_dac_scan', close_fig=close_fig)

    def measure_qubit_frequency_dac_scan(
            self, freqs,
            dac_values,
            mode='pulsed_marked',
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            fluxChan=None,
            close_fig=True,
            nested_resonator_calibration=False,
            nested_resonator_calibration_use_min=False,
            resonator_freqs=None,
            trigger_idx=None
    ):
        """
        Performs the qubit spectroscopy while changing the current applied
        to the flux bias line.

        Args:
            freqs (array):
                MW drive frequencies to sweep over

            dac_values (array):
                values of the current to sweep over

            mode (str {'pulsed_mixer', 'CW', 'pulsed_marked'}):
                specifies the spectroscopy mode (cf. measure_spectroscopy method)

            fluxChan (str):
                Fluxchannel that is varied. Defaults to self.fl_dc_ch

            nested_resonator_calibration (bool):
                specifies whether to track the RO resonator
                frequency (which itself is flux-dependent)

            nested_resonator_calibration_use_min (bool):
                specifies whether to use the resonance
                minimum in the nested routine

            resonator_freqs (array):
                manual specifications of the frequencies over in which to
                search for RO resonator in the nested routine

            analyze (bool):
                indicates whether to generate colormaps of the measured data

            label (str):
                suffix to append to the measurement label

        Relevant qubit parameters:
            instr_FluxCtrl (str):
                instrument controlling the current bias

            fluxChan (str):
                channel of the flux control instrument corresponding to the qubit
        """

        if mode == 'pulsed_mixer':
            old_channel_amp = self.mw_channel_amp()
            self.mw_channel_amp(1)
            self.prepare_for_timedomain()
            self.mw_channel_amp(old_channel_amp)
        elif mode == 'CW' or mode == 'pulsed_marked':
            self.prepare_for_continuous_wave()
        else:
            logging.error('Mode {} not recognized'.format(mode))
        if MC is None:
            MC = self.instr_MC.get_instr()
        if trigger_idx is None:
            trigger_idx = self.cfg_qubit_nr()

        CC = self.instr_CC.get_instr()
        if mode == 'pulsed_marked':
            p = sqo.pulsed_spec_seq_marked(
                qubit_idx=self.cfg_qubit_nr(),
                spec_pulse_length=self.spec_pulse_length(),
                platf_cfg=self.cfg_openql_platform_fn(),
                trigger_idx=trigger_idx
            )
        else:
            p = sqo.pulsed_spec_seq(
                qubit_idx=self.cfg_qubit_nr(),
                spec_pulse_length=self.spec_pulse_length(),
                platf_cfg=self.cfg_openql_platform_fn()
            )
        CC.eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        dac_par = self.hal_flux_get_parameters(fluxChan)

        if mode == 'pulsed_mixer':
            spec_source = self.instr_spec_source_2.get_instr()
            spec_source.on()
        else:
            spec_source = self.instr_spec_source.get_instr()
            spec_source.on()
            # if mode == 'pulsed_marked':
            #     spec_source.pulsemod_state('On')

        MC.set_sweep_function(spec_source.frequency)
        MC.set_sweep_points(freqs)
        if nested_resonator_calibration:
            res_updating_dac_par = swf.Nested_resonator_tracker(
                qubit=self,
                nested_MC=self.instr_nested_MC.get_instr(),
                freqs=resonator_freqs,
                par=dac_par,
                use_min=nested_resonator_calibration_use_min,
                reload_sequence=True,
                sequence_file=p,
                cc=CC
            )
            MC.set_sweep_function_2D(res_updating_dac_par)
        else:
            MC.set_sweep_function_2D(dac_par)
        MC.set_sweep_points_2D(dac_values)
        self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
        self.int_avg_det_single.always_prepare = True
        MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='Qubit_dac_scan' + self.msmt_suffix, mode='2D')

        if analyze:
            return ma.TwoD_Analysis(
                label='Qubit_dac_scan',
                close_fig=close_fig
            )

    def measure_qubit_frequency_dac_scan_ramzz(
            self, freqs,
            dac_values,
            measurement_qubit,
            ramzz_wait_time_ns,
            mode='pulsed_marked',
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            fluxChan=None,
            close_fig=True,
            nested_resonator_calibration=False,
            nested_resonator_calibration_use_min=False,
            resonator_freqs=None,
            trigger_idx=None
    ):
        """
        Performs the qubit spectroscopy while changing the current applied
        to the flux bias line.

        Args:
            freqs (array):
                MW drive frequencies to sweep over

            dac_values (array):
                values of the current to sweep over

            mode (str {'pulsed_mixer', 'CW', 'pulsed_marked'}):
                specifies the spectroscopy mode (cf. measure_spectroscopy method)

            fluxChan (str):
                Fluxchannel that is varied. Defaults to self.fl_dc_ch

            nested_resonator_calibration (bool):
                specifies whether to track the RO resonator
                frequency (which itself is flux-dependent)

            nested_resonator_calibration_use_min (bool):
                specifies whether to use the resonance
                minimum in the nested routine

            resonator_freqs (array):
                manual specifications of the frequencies over in which to
                search for RO resonator in the nested routine

            analyze (bool):
                indicates whether to generate colormaps of the measured data

            label (str):
                suffix to append to the measurement label

        Relevant qubit parameters:
            instr_FluxCtrl (str):
                instrument controlling the current bias

            fluxChan (str):
                channel of the flux control instrument corresponding to the qubit
        """

        if mode == 'pulsed_mixer':
            old_channel_amp = self.mw_channel_amp()
            self.mw_channel_amp(1)
            self.prepare_for_timedomain()
            self.mw_channel_amp(old_channel_amp)
        elif mode == 'CW' or mode == 'pulsed_marked':
            self.prepare_for_continuous_wave()
            measurement_qubit.prepare_for_timedomain()
        else:
            logging.error('Mode {} not recognized'.format(mode))
        if MC is None:
            MC = self.instr_MC.get_instr()
        if trigger_idx is None:
            trigger_idx = self.cfg_qubit_nr()

        CC = self.instr_CC.get_instr()
        if mode == 'pulsed_marked':
            p = sqo.pulsed_spec_seq_marked(
                qubit_idx=self.cfg_qubit_nr(),
                spec_pulse_length=self.spec_pulse_length(),
                platf_cfg=self.cfg_openql_platform_fn(),
                trigger_idx=trigger_idx
            )
        else:
            p = sqo.pulsed_spec_seq_ramzz(
                qubit_idx=self.cfg_qubit_nr(),
                measured_qubit_idx = measurement_qubit.cfg_qubit_nr(),
                ramzz_wait_time_ns = ramzz_wait_time_ns,
                spec_pulse_length=self.spec_pulse_length(),
                platf_cfg=self.cfg_openql_platform_fn()
            )
        CC.eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        dac_par = self.hal_flux_get_parameters(fluxChan)

        if mode == 'pulsed_mixer':
            spec_source = self.instr_spec_source_2.get_instr()
            spec_source.on()
        else:
            spec_source = self.instr_spec_source.get_instr()
            spec_source.on()
            # if mode == 'pulsed_marked':
            #     spec_source.pulsemod_state('On')

        MC.set_sweep_function(spec_source.frequency)
        MC.set_sweep_points(freqs)
        if nested_resonator_calibration:
            res_updating_dac_par = swf.Nested_resonator_tracker(
                qubit=self,
                nested_MC=self.instr_nested_MC.get_instr(),
                freqs=resonator_freqs,
                par=dac_par,
                use_min=nested_resonator_calibration_use_min,
                reload_sequence=True,
                sequence_file=p,
                cc=CC
            )
            MC.set_sweep_function_2D(res_updating_dac_par)
        else:
            MC.set_sweep_function_2D(dac_par)
        MC.set_sweep_points_2D(dac_values)
        measurement_qubit.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
        measurement_qubit.int_avg_det_single.always_prepare = True
        MC.set_detector_function(measurement_qubit.int_avg_det_single)
        MC.run(name='Qubit_dac_scan' + self.msmt_suffix, mode='2D')

        if analyze:
            return ma.TwoD_Analysis(
                label='Qubit_dac_scan',
                close_fig=close_fig
            )

    def _measure_spectroscopy_CW(
            self,
            freqs,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            label='',
            prepare_for_continuous_wave=True):
        """
        Does a CW spectroscopy experiment by sweeping the frequency of a
        microwave source.

        Relevant qubit parameters:
            instr_spec_source (RohdeSchwarz_SGS100A):
                instrument used to apply CW excitation

            spec_pow (float):
                power of the MW excitation at the output of the spec_source (dBm)
                FIXME: parameter disappeared, and power not set

            label (str):
                suffix to append to the measurement label
        """
        if prepare_for_continuous_wave:
            self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()

        self.hal_acq_spec_mode_on()

        p = sqo.pulsed_spec_seq(
            qubit_idx=self.cfg_qubit_nr(),
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn()
        )

        self.instr_CC.get_instr().eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        spec_source = self.instr_spec_source.get_instr()
        spec_source.on()
        # Set marker mode off for CW:
        if not spec_source.get_idn()['model'] == 'E8257D':  # FIXME: HW dependency on old HP/Keysight model
            spec_source.pulsemod_state('Off')

        MC.set_sweep_function(spec_source.frequency)
        MC.set_sweep_points(freqs)
        if self.cfg_spec_mode():
            print('Enter loop')
            MC.set_detector_function(self.UHFQC_spec_det)
        else:
            self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
            MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='CW_spectroscopy' + self.msmt_suffix + label)

        self.hal_acq_spec_mode_off()

        if analyze:
            ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)

    def measure_spectroscopy_CW_ramzz(
            self,
            freqs,
            measurement_qubit,
            ramzz_wait_time_ns,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            label='',
            prepare_for_continuous_wave=True):
        """
        Does a CW spectroscopy experiment by sweeping the frequency of a
        microwave source.

        Relevant qubit parameters:
            instr_spec_source (RohdeSchwarz_SGS100A):
                instrument used to apply CW excitation

            spec_pow (float):
                power of the MW excitation at the output of the spec_source (dBm)
                FIXME: parameter disappeared, and power not set

            label (str):
                suffix to append to the measurement label
        """
        if prepare_for_continuous_wave:
            self.prepare_for_continuous_wave()
            measurement_qubit.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        self.hal_acq_spec_mode_on()

        p = sqo.pulsed_spec_seq_ramzz(
            qubit_idx=self.cfg_qubit_nr(),
            measured_qubit_idx = measurement_qubit.cfg_qubit_nr(),
            ramzz_wait_time_ns = ramzz_wait_time_ns,
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn()
        )

        self.instr_CC.get_instr().eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        spec_source = self.instr_spec_source.get_instr()
        spec_source.on()
        # Set marker mode off for CW:
        if not spec_source.get_idn()['model'] == 'E8257D':  # FIXME: HW dependency on old HP/Keysight model
            spec_source.pulsemod_state('Off')

        MC.set_sweep_function(spec_source.frequency)
        MC.set_sweep_points(freqs)
        if self.cfg_spec_mode():
            print('Enter loop')
            MC.set_detector_function(measurement_qubit.UHFQC_spec_det)
        else:
            measurement_qubit.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
            MC.set_detector_function(measurement_qubit.int_avg_det_single)
        MC.run(name='CW_spectroscopy' + self.msmt_suffix + label)

        self.hal_acq_spec_mode_off()

        if analyze:
            ma.Homodyne_Analysis(label=self.msmt_suffix, close_fig=close_fig)

    def _measure_spectroscopy_pulsed_marked(
            self,
            freqs,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            label='',
            prepare_for_continuous_wave=True,
            trigger_idx=None
    ):
        """
        Performs a spectroscopy experiment by triggering the spectroscopy source
        with a CCLight trigger.
        """

        if prepare_for_continuous_wave:
            self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()

        self.hal_acq_spec_mode_on()

        wait_time_ns = self.spec_wait_time() * 1e9

        if trigger_idx is None:
            trigger_idx = self.cfg_qubit_nr()

        CC = self.instr_CC.get_instr()
        p = sqo.pulsed_spec_seq_marked(
            qubit_idx=self.cfg_qubit_nr(),
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn(),
            cc=self.instr_CC(),  # FIXME: add ".get_instr()"
            trigger_idx=trigger_idx if (CC.name.upper() == 'CCL' or CC.name.upper() == 'CC') else 15,
            # FIXME: CCL is deprecated
            wait_time_ns=wait_time_ns)

        CC.eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        spec_source = self.instr_spec_source.get_instr()
        spec_source.on()
        # Set marker mode off for CW:
        spec_source.pulsemod_state('On')

        MC.set_sweep_function(spec_source.frequency)
        MC.set_sweep_points(freqs)
        if self.cfg_spec_mode():
            MC.set_detector_function(self.UHFQC_spec_det)
        else:
            self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
            MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='pulsed_marker_spectroscopy' + self.msmt_suffix + label)

        self.hal_acq_spec_mode_off()

        if analyze:
            ma.Qubit_Spectroscopy_Analysis(
                label=self.msmt_suffix,
                close_fig=close_fig,
                qb_name=self.name
            )

    def _measure_spectroscopy_pulsed_mixer(
            self,
            freqs,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            label='',
            prepare_for_timedomain=True
    ):
        """
        Performs pulsed spectroscopy by modulating a cw pulse with a square
        which is generated by an AWG. Uses the self.mw_LO as spec source, as
        that usually is the LO of the AWG/QWG mixer.

        Is considered as a time domain experiment as it utilizes the AWG

        Relevant parameters:
            spec_pow (float):
                power of the LO fed into the mixer

            spec_amp (float):
                amplitude of the square waveform used to generate
                microwave tone

            spec_pulse_length (float):
                length of the spectroscopy pulse. The length is
                controlled by the qisa file, which indicates how many 20 ns long
                square pulses should be triggered back-to-back
        """

        if MC is None:
            MC = self.instr_MC.get_instr()

        self.hal_acq_spec_mode_on()

        # Save current value of mw_channel_amp to make this measurement
        # independent of the value.
        old_channel_amp = self.mw_channel_amp()
        self.mw_channel_amp(1)

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        p = sqo.pulsed_spec_seq(
            qubit_idx=self.cfg_qubit_nr(),
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn())

        self.instr_CC.get_instr().eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        spec_source = self.instr_spec_source_2.get_instr()
        # spec_source.on()
        # Set marker mode off for mixer CW:

        MC.set_sweep_function(spec_source.frequency)
        MC.set_sweep_points(freqs)

        if self.cfg_spec_mode():
            print('Enter loop')
            MC.set_detector_function(self.UHFQC_spec_det)
        else:
            self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
            MC.set_detector_function(self.int_avg_det_single)

        # d = self.int_avg_det
        # MC.set_detector_function(d)
        MC.run(name='pulsed_mixer_spectroscopy' + self.msmt_suffix + label)

        self.mw_channel_amp(old_channel_amp)

        self.hal_acq_spec_mode_off()

        if analyze:
            ma.Qubit_Spectroscopy_Analysis(
                label=self.msmt_suffix,
                close_fig=close_fig,
                qb_name=self.name
            )

    def measure_anharmonicity(
            self,
            freqs_01=None,
            freqs_12=None,
            f_01_power=None,
            f_12_power=None,
            MC: Optional[MeasurementControl] = None,
            spec_source_2=None,
            mode='pulsed_marked',
            step_size: int = 1e6
    ):
        """
        Measures the qubit spectroscopy as a function of frequency of the two
        driving tones. The qubit transitions are observed when frequency of one
        drive matches the qubit frequency, or when sum of frequencies matches
        energy difference between ground and second excited state. Consequently
        frequency of 01 and 12  transitions can be extracted simultaneously
        yoielding anharmonicity measurement.

        Typically a good guess for the 12 transition frequencies is
        f01 + alpha where alpha is the anharmonicity and typically ~ -300 MHz

        Args:
            freqs_01: frequencies of the first qubit drive
            freqs_12: frequencies of the second qubit drive
            f_01_power: power of the first qubit drive. By default the power
                is set to self.spec_pow
            f_12_power: power of the second qubit drive. By default the power
                is set to self.spec_pow. Likely it needs to be increased
                by 10-20 dB to yield meaningful result
            spec_source_2: instrument used to apply second MW drive.
                By default instrument specified by self.instr_spec_source_2 is used
            mode (str):
                if pulsed_marked uses pulsed spectroscopy sequence assuming
                that the sources are pulsed using a marker.
                Otherwise, uses CW spectroscopy.
        """
        # f_anharmonicity = np.mean(freqs_01) - np.mean(freqs_12)
        # if f_01_power == None:
        #     f_01_power = self.spec_pow()
        # if f_12_power == None:
        #     f_12_power = f_01_power+20
        if freqs_01 is None:
            freqs_01 = self.freq_qubit() + np.arange(-20e6, 20.1e6, step_size)
        if freqs_12 is None:
            freqs_12 = self.freq_qubit() + self.anharmonicity() + \
                       np.arange(-20e6, 20.1e6, 1e6)
        f_anharmonicity = np.mean(freqs_01) - np.mean(freqs_12)
        if f_01_power == None:
            f_01_power = self.spec_pow()
        if f_12_power == None:
            f_12_power = f_01_power + 5
        print('f_anharmonicity estimation', f_anharmonicity)
        print('f_12 estimations', np.mean(freqs_12))

        if mode == 'pulsed_marked':
            p = sqo.pulsed_spec_seq_marked(
                qubit_idx=self.cfg_qubit_nr(),
                spec_pulse_length=self.spec_pulse_length(),
                platf_cfg=self.cfg_openql_platform_fn(),
                trigger_idx=0,
                trigger_idx_2=9
            )
        else:
            p = sqo.pulsed_spec_seq(
                qubit_idx=self.cfg_qubit_nr(),
                spec_pulse_length=self.spec_pulse_length(),
                platf_cfg=self.cfg_openql_platform_fn()
            )
        self.instr_CC.get_instr().eqasm_program(p.filename)

        if MC is None:
            MC = self.instr_MC.get_instr()

        self.prepare_for_continuous_wave()
        self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state

        spec_source = self.hal_spec_source_on(f_01_power, mode == 'pulsed_marked')

        # FIXME:WIP: also handle spec_source_2
        if spec_source_2 is None:
            spec_source_2 = self.instr_spec_source_2.get_instr()
        spec_source_2.on()
        if mode == 'pulsed_marked':
            spec_source_2.pulsemod_state('On')
        else:
            spec_source_2.pulsemod_state('Off')
        spec_source_2.power(f_12_power)

        MC.set_sweep_function(wrap_par_to_swf(
            spec_source.frequency, retrieve_value=True))
        MC.set_sweep_points(freqs_01)
        MC.set_sweep_function_2D(wrap_par_to_swf(
            spec_source_2.frequency, retrieve_value=True))
        MC.set_sweep_points_2D(freqs_12)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run_2D(name='Two_tone_' + self.msmt_suffix)

        ma.TwoD_Analysis(auto=True)

        self.hal_spec_source_off()
        spec_source_2.off()

        ma.Three_Tone_Spectroscopy_Analysis(
            label='Two_tone',
            f01=np.mean(freqs_01),
            f12=np.mean(freqs_12)
        )

    def measure_anharmonicity_GBT(
            self,
            freqs_01=None,
            freqs_12=None,
            f_01_power=None,
            f_12_power=None,
            MC: Optional[MeasurementControl] = None,
            spec_source_2=None,
            mode='pulsed_marked'
    ):
        """
        Measures the qubit spectroscopy as a function of frequency of the two
        driving tones. The qubit transitions are observed when frequency of one
        drive matches the qubit frequency, or when sum of frequencies matches
        energy difference between ground and second excited state. Consequently
        frequency of 01 and 12  transitions can be extracted simultaneously
        yoielding anharmonicity measurement.

        Typically a good guess for the 12 transition frequencies is
        f01 + alpha where alpha is the anharmonicity and typically ~ -300 MHz

        Args:
            freqs_01: frequencies of the first qubit drive
            freqs_12: frequencies of the second qubit drive
            f_01_power: power of the first qubit drive. By default the power
                is set to self.spec_pow
            f_12_power: power of the second qubit drive. By default the power
                is set to self.spec_pow. Likely it needs to be increased
                by 10-20 dB to yield meaningful result
            spec_source_2: instrument used to apply second MW drive.
                By default instrument specified by self.instr_spec_source_2 is used
            mode (str):
                if pulsed_marked uses pulsed spectroscopy sequence assuming
                that the sources are pulsed using a marker.
                Otherwise, uses CW spectroscopy.
        """
        if freqs_01 is None:
            freqs_01 = self.freq_qubit() + np.arange(-30e6, 30.1e6, 0.5e6)
        if freqs_12 is None:
            freqs_12 = self.freq_qubit() + self.anharmonicity() + \
                       np.arange(-30e6, 30.1e6, 0.5e6)
        f_anharmonicity = np.mean(freqs_01) - np.mean(freqs_12)
        if f_01_power == None:
            f_01_power = self.spec_pow()
        if f_12_power == None:
            f_12_power = f_01_power + 20

        print('f_anharmonicity estimation', f_anharmonicity)
        print('f_12 estimations', np.mean(freqs_12))

        p = sqo.pulsed_spec_seq_marked(
            qubit_idx=self.cfg_qubit_nr(),
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn(),
            trigger_idx=0)
        self.instr_CC.get_instr().eqasm_program(p.filename)

        if MC is None:
            MC = self.instr_MC.get_instr()

        # save parameter
        old_spec_pow = self.spec_pow()  # FIXME: changed by prepare_for_continuous_wave

        self.prepare_for_continuous_wave()
        self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state

        spec_source = self.hal_spec_source_on(f_01_power, mode == 'pulsed_marked')

        # configure spec_source
        if spec_source_2 is None:
            spec_source_2 = self.instr_spec_source_2.get_instr()
        spec_source_2.on()
        if mode == 'pulsed_marked':
            spec_source_2.pulsemod_state('On')
        else:
            spec_source_2.pulsemod_state('Off')
        spec_source_2.power(f_12_power)

        MC.set_sweep_function(wrap_par_to_swf(spec_source.frequency, retrieve_value=True))
        MC.set_sweep_points(freqs_01)
        MC.set_sweep_function_2D(wrap_par_to_swf(spec_source_2.frequency, retrieve_value=True))
        MC.set_sweep_points_2D(freqs_12)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run_2D(name='Two_tone_' + self.msmt_suffix)
        ma.TwoD_Analysis(auto=True)

        self.hal_spec_source_off()
        spec_source_2.off()

        # restore parameter
        self.spec_pow(old_spec_pow)

        # if analyze:
        #     a = ma.Three_Tone_Spectroscopy_Analysis(label='Two_tone',  f01=np.mean(freqs_01), f12=np.mean(freqs_12))
        #     if update:
        #         self.anharmonicity(a.anharm)
        #     return a.T1

        ma_obj = ma.Three_Tone_Spectroscopy_Analysis_test(
            label='Two_tone',
            f01=np.mean(freqs_01),
            f12=np.mean(freqs_12)
        )
        rel_change = (abs(self.anharmonicity()) - ma_obj.Anharm_dict['anharmonicity']) / self.anharmonicity()
        threshold_for_change = 0.1
        if np.abs(rel_change) > threshold_for_change:
            return False
        else:
            return True

    def measure_photon_nr_splitting_from_bus(
            self,
            f_bus,
            freqs_01=None,
            powers=np.arange(-10, 10, 1),
            MC: Optional[MeasurementControl] = None,
            spec_source_2=None
    ):
        """
        Measures photon splitting of the qubit due to photons in the bus resonators.
        Specifically it is a CW qubit pectroscopy with the second  variable-power CW tone
        applied at frequency f_bus.

        Refs:
        Schuster Nature 445, 515–518 (2007)
            (note that in the paper RO resonator has lower frequency than the qubit)

        Args:
            f_bus: bus frequency at which variable-power CW tone is applied
            freqs_01: range of frequencies of the CW qubit MW drive. If not specified
            range -60 MHz to +5 MHz around freq_qubit fill be used.
            powers: sweeped powers of the bus CW drive.
            spec_source_2: sepcifies instrument used to apply bus MW drive. By default
                instr_spec_source_2 is used.
        """
        if freqs_01 is None:
            freqs_01 = np.arange(self.freq_qubit() - 60e6,
                                 self.freq_qubit() + 5e6, 0.7e6)

        self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()

        if spec_source_2 is None:
            spec_source_2 = self.instr_spec_source_2.get_instr()

        p = sqo.pulsed_spec_seq(
            qubit_idx=self.cfg_qubit_nr(),
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn()
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)

        self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state

        spec_source = self.instr_spec_source.get_instr()
        spec_source.on()
        spec_source.power(self.spec_pow())
        # FIXME: does not touch pulsed mode, which is touched in other functions

        spec_source_2.on()
        spec_source_2.frequency(f_bus)

        MC.set_sweep_function(wrap_par_to_swf(spec_source.frequency, retrieve_value=True))
        MC.set_sweep_points(freqs_01)
        MC.set_sweep_function_2D(wrap_par_to_swf(spec_source_2.power, retrieve_value=True))
        MC.set_sweep_points_2D(powers)
        MC.set_detector_function(self.int_avg_det_single)
        MC.run_2D(name='Photon_nr_splitting' + self.msmt_suffix)

        ma.TwoD_Analysis(auto=True)
        spec_source.off()
        spec_source_2.off()

    @deprecated(version='0.4', reason="broken")
    def measure_ssro_vs_frequency_amplitude(
            self, freqs=None, amps_rel=np.linspace(0, 1, 11),
            nr_shots=4092 * 4, nested_MC: Optional[MeasurementControl] = None, analyze=True,
            use_optimal_weights=False, label='SSRO_freq_amp_sweep'):
        """
        Measures SNR and readout fidelities as a function of the readout pulse amplitude
        and frequency. Resonator depletion pulses are automatically scaled.
        Weights are not optimized - routine is intended to be used with SSB weights.

        Args:
            freqs (array):
                readout freqencies to loop over

            amps_rel (array):
                readout pulse amplitudes to loop over. Value of 1 indicates
                amplitude currently specified in the qubit object.

            nr_shots (int):
                total number of measurements in qubit ground and excited state
        """
        warnings.warn('FIXME: Does not make use of the SSRO detector')

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()
        if freqs is None:
            freqs = np.linspace(self.ro_freq() - 4e6, self.ro_freq() + 2e6, 11)

        self.prepare_for_timedomain()
        RO_lutman = self.instr_LutMan_RO.get_instr()

        old_ro_prepare_state = self.cfg_prepare_ro_awg()
        self.ro_acq_digitized(False)  # FIXME: changes state
        self.cfg_prepare_ro_awg(False)  # FIXME: changes state (old_ro_prepare_state above is unused)

        sweep_function = swf.lutman_par_depletion_pulse_global_scaling(
            LutMan=RO_lutman,
            resonator_numbers=[self.cfg_qubit_nr()],
            optimization_M_amps=[self.ro_pulse_amp()],
            optimization_M_amp_down0s=[self.ro_pulse_down_amp0()],
            optimization_M_amp_down1s=[self.ro_pulse_down_amp1()],
            upload=True
        )
        # FIXME: code missing here (already gone in GIT tag "v0.2")

    def measure_ssro_vs_TWPA_frequency_power(
            self,
            pump_source,
            freqs,
            powers,
            nr_shots=4092 * 4,
            nested_MC: Optional[MeasurementControl] = None,
            analyze=True
    ):
        """
        Measures the SNR and readout fidelities as a function of the TWPA
            pump frequency and power.

        Args:
            pump_source (RohdeSchwarz_SGS100A):
                object controlling the MW source serving as TWPA pump

            freqs (array):
                TWPA pump frequencies to sweep over

            powers (array):
                list of TWPA pump powers to sweep over

            nr_shots (int):
                number of single-shot measurements used to estimate SNR
                and redout fidelities
        """
        warnings.warn('FIXME: Does not make use of the SSRO detector')

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        self.prepare_for_timedomain()
        RO_lutman = self.instr_LutMan_RO.get_instr()
        old_ro_prepare_state = self.cfg_prepare_ro_awg()
        self.ro_acq_digitized(False)
        self.cfg_prepare_ro_awg(False)

        d = det.Function_Detector(
            self.measure_ssro,
            msmt_kw={
                'nr_shots': nr_shots,
                'analyze': True, 'SNR_detector': True,
                'cal_residual_excitation': True,
                'prepare': False,
                'disable_metadata': True
            },
            result_keys=['SNR', 'F_d', 'F_a']
        )
        nested_MC.set_sweep_function(pump_source.frequency)
        nested_MC.set_sweep_points(freqs)
        nested_MC.set_detector_function(d)
        nested_MC.set_sweep_function_2D(pump_source.power)
        nested_MC.set_sweep_points_2D(powers)
        label = 'SSRO_freq_amp_sweep' + self.msmt_suffix
        nested_MC.run(label, mode='2D')

        self.cfg_prepare_ro_awg(old_ro_prepare_state)

        if analyze:
            ma.TwoD_Analysis(label=label, plot_all=True, auto=True)

    def measure_ssro_vs_pulse_length(
            self,
            lengths=np.arange(100e-9, 1501e-9, 100e-9),
            nr_shots=4092 * 4,
            nested_MC: Optional[MeasurementControl] = None,
            analyze=True,
            label_suffix: str = ''
    ):
        """
        Measures the SNR and readout fidelities as a function of the duration
            of the readout pulse. For each pulse duration transients are
            measured and optimal weights calculated.

        Args:
            lengths (array):
                durations of the readout pulse for which SNR is measured

            nr_shots (int):
                number of single-shot measurements used to estimate SNR
                and readout fidelities
        """
        warnings.warn('FIXME: Does not make use of the SSRO detector')

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()
        self.ro_acq_digitized(False)
        self.prepare_for_timedomain()
        RO_lutman = self.instr_LutMan_RO.get_instr()

        d = det.Function_Detector(
            self.calibrate_optimal_weights,
            msmt_kw={
                'analyze': True,
            },
            result_keys=['SNR', 'F_d', 'F_a', 'relaxation', 'excitation']
        )
        # sweep_function = swf.lutman_par_UHFQC_dig_trig(
        #     LutMan=RO_lutman,
        #     LutMan_parameter=RO_lutman['M_length_R{}'.format(
        #         self.cfg_qubit_nr())]
        # )
        # nested_MC.set_sweep_function(sweep_function)
        nested_MC.set_sweep_function(self.ro_pulse_length)
        nested_MC.set_sweep_points(lengths)
        nested_MC.set_detector_function(d)
        label = 'SSRO_length_sweep' + self.msmt_suffix + label_suffix
        nested_MC.run(label)

        if analyze:
            ma.MeasurementAnalysis(label=label, plot_all=False, auto=True)

    def measure_transients_CCL_switched(
            self,
            MC: Optional[MeasurementControl] = None,
            analyze: bool = True,
            cases=('off', 'on'),
            prepare: bool = True,
            depletion_analysis: bool = True,
            depletion_analysis_plot: bool = True,
            depletion_optimization_window=None
    ):
        if MC is None:
            MC = self.instr_MC.get_instr()

        self.prepare_for_timedomain()
        # off/on switching is achieved by turning the MW source on and
        # off as this is much faster than recompiling/uploading

        transients = []
        for i, pulse_comb in enumerate(cases):
            p = sqo.off_on(
                qubit_idx=self.cfg_qubit_nr(), pulse_comb=pulse_comb,
                initialize=False,
                platf_cfg=self.cfg_openql_platform_fn()
            )
            self.instr_CC.get_instr().eqasm_program(p.filename)

            s = swf.OpenQL_Sweep(
                openql_program=p,
                CCL=self.instr_CC.get_instr(),
                parameter_name='Transient time', unit='s',
                upload=prepare
            )
            MC.set_sweep_function(s)

            if 'UHFQC' in self.instr_acquisition():
                sampling_rate = 1.8e9
            else:
                raise NotImplementedError()

            MC.set_sweep_points(np.arange(self.input_average_detector.nr_samples) / sampling_rate)
            MC.set_detector_function(self.input_average_detector)
            data = MC.run('Measure_transients{}_{}'.format(self.msmt_suffix, i))
            dset = data['dset']
            transients.append(dset.T[1:])
            if analyze:
                ma.MeasurementAnalysis()
        if depletion_analysis:
            a = ma.Input_average_analysis(
                IF=self.ro_freq_mod(),
                optimization_window=depletion_optimization_window,
                plot=depletion_analysis_plot
            )
            return a
        else:
            return [np.array(t, dtype=np.float64) for t in transients]

    def measure_RO_QND(
            self,
            prepare_for_timedomain: bool = False,
            calibrate_optimal_weights: bool = False,
            ):
        # ensure readout settings are correct
        old_ro_type = self.ro_acq_weight_type()
        old_acq_type = self.ro_acq_digitized()

        if calibrate_optimal_weights:
            self.calibrate_optimal_weights(prepare=False)

        self.ro_acq_digitized(False)
        self.ro_acq_weight_type('optimal IQ')
        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        else:
            # we always need to prepare at least readout
            self.prepare_readout()

        d = self.int_log_det
        # the QND sequence has 5 measurements,
        # therefore we need to make sure the number of shots is a multiple of that
        uhfqc_max_avg = 2**17
        d.nr_shots = int(uhfqc_max_avg/5) * 5
        p = sqo.RO_QND_sequence(q_idx = self.cfg_qubit_nr(),
                                platf_cfg = self.cfg_openql_platform_fn())
        s = swf.OpenQL_Sweep(openql_program=p,
                             CCL=self.instr_CC.get_instr())
        MC = self.instr_MC.get_instr()
        MC.soft_avg(1)
        MC.live_plot_enabled(False)
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(int(uhfqc_max_avg/5)*5))
        MC.set_detector_function(d)
        MC.run(f"RO_QND_measurement_{self.name}")
        self.ro_acq_weight_type(old_ro_type)
        self.ro_acq_digitized(old_acq_type)

        a = ma2.mra.measurement_QND_analysis(qubit=self.name, label='QND')
        return a.quantities_of_interest

    def calibrate_RO_QND(
            self,
            amps: list,
            calibrate_optimal_weights: bool = False
            ):
        s = self.ro_pulse_amp
        d = det.Function_Detector(self.measure_RO_QND,
                                  result_keys=['P_QND', 'P_QNDp'],
                                  value_names=['P_QND', 'P_QNDp'],
                                  value_units=['a.u.', 'a.u.'],
                                  msmt_kw={'calibrate_optimal_weights': calibrate_optimal_weights}
                                  )
        nested_MC = self.instr_nested_MC.get_instr()
        nested_MC.set_detector_function(d)
        nested_MC.set_sweep_function(s)
        nested_MC.set_sweep_points(amps)
        nested_MC.run(f"RO_QND_sweep_{self.name}")

    def measure_dispersive_shift_pulsed(
            self, freqs=None,
            MC: Optional[MeasurementControl] = None,
            analyze: bool = True,
            prepare: bool = True,
            Pulse_comb: list=['off', 'on']
    ):
        # USED_BY: device_dependency_graphs_v2.py,
        # USED_BY: device_dependency_graphs

        """
        Measures the RO resonator spectroscopy with the qubit in ground and excited state.
        Specifically, performs two experiments. Applies sequence:
        - initialize qubit in ground state (  wait)
        - (only in the second experiment) apply a (previously tuned up) pi pulse
        - apply readout pulse and measure
        This sequence is repeated while sweeping ro_freq.

        Args:
            freqs (array):
                sweeped range of ro_freq
        """

        # docstring from parent class
        if MC is None:
            MC = self.instr_MC.get_instr()

        if freqs is None:
            if self.freq_res() is None:
                raise ValueError("Qubit has no resonator frequency. Update freq_res parameter.")
            else:
                freqs = self.freq_res() + np.arange(-10e6, 5e6, .1e6)

        if 'optimal' in self.ro_acq_weight_type():
            raise NotImplementedError("Change readout demodulation to SSB.")

        self.prepare_for_timedomain()

        # off/on switching is achieved by turning the MW source on and
        # off as this is much faster than recompiling/uploading
        f_res = []
        for i, pulse_comb in enumerate(Pulse_comb):
            p = sqo.off_on(
                qubit_idx=self.cfg_qubit_nr(), pulse_comb=pulse_comb,
                initialize=False,
                platf_cfg=self.cfg_openql_platform_fn())
            self.instr_CC.get_instr().eqasm_program(p.filename)
            # CC gets started in the int_avg detector

            MC.set_sweep_function(swf.Heterodyne_Frequency_Sweep_simple(
                MW_LO_source=self.instr_LO_ro.get_instr(),
                IF=self.ro_freq_mod()))
            MC.set_sweep_points(freqs)

            self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
            MC.set_detector_function(self.int_avg_det_single)
            MC.run(name='Resonator_scan_' + pulse_comb + self.msmt_suffix)

            if analyze:
                ma.MeasurementAnalysis()
                a = ma.Homodyne_Analysis(
                    label=self.msmt_suffix, close_fig=True)
                # fit converts to Hz
                f_res.append(a.fit_results.params['f0'].value * 1e9)

        if analyze:
            a = ma2.Dispersive_shift_Analysis()
            self.dispersive_shift(a.qoi['dispersive_shift'])
            # Dispersive shift from 'hanger' fit
            # print('dispersive shift is {} MHz'.format((f_res[1]-f_res[0])*1e-6))
            # Dispersive shift from peak finder
            print('dispersive shift is {} MHz'.format(
                a.qoi['dispersive_shift'] * 1e-6))

            return True

    def measure_error_fraction(
            self,
            MC: Optional[MeasurementControl] = None,
            analyze: bool = True,
            nr_shots: int = 2048 * 4,
            sequence_type='echo',
            prepare: bool = True,
            feedback=False,
            depletion_time=None,
            net_gate='pi'
    ):
        """
        This performs a multi round experiment, the repetition rate is defined
        by the ro_duration which can be changed by regenerating the
        configuration file.
        The analysis counts single errors. The definition of an error is
        adapted automatically by choosing feedback or the net_gate.
        it requires high SNR single shot readout and a calibrated threshold.
        """

        self.ro_acq_digitized(True)  # FIXME: changes state

        if MC is None:
            MC = self.instr_MC.get_instr()

        # save and change parameters
        # plotting really slows down SSRO (16k shots plotting is slow)
        old_plot_setting = MC.live_plot_enabled()
        MC.live_plot_enabled(False)

        MC.soft_avg(1)  # don't want to average single shots # FIXME: changes state

        if prepare:
            self.prepare_for_timedomain()
            # off/on switching is achieved by turning the MW source on and
            # off as this is much faster than recompiling/uploading
            p = sqo.RTE(
                qubit_idx=self.cfg_qubit_nr(), sequence_type=sequence_type,
                platf_cfg=self.cfg_openql_platform_fn(), net_gate=net_gate,
                feedback=feedback)
            self.instr_CC.get_instr().eqasm_program(p.filename)
        else:
            p = None  # object needs to exist for the openql_sweep to work

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name='shot nr',
            unit='#',
            upload=prepare
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(np.arange(nr_shots))
        d = self.int_log_det
        MC.set_detector_function(d)

        exp_metadata = {'feedback': feedback, 'sequence_type': sequence_type,
                        'depletion_time': depletion_time, 'net_gate': net_gate}
        suffix = 'depletion_time_{}_ro_pulse_{}_feedback_{}_net_gate_{}'.format(
            depletion_time, self.ro_pulse_type(), feedback, net_gate)
        MC.run(
            'RTE_{}_{}'.format(self.msmt_suffix, suffix),
            exp_metadata=exp_metadata
        )

        # restore parameters
        MC.live_plot_enabled(old_plot_setting)

        if analyze:
            a = ma2.Single_Qubit_RoundsToEvent_Analysis(
                t_start=None,
                t_stop=None,
                options_dict={'typ_data_idx': 0,
                              'scan_label': 'RTE'},
                extract_only=True
            )
            return {'error fraction': a.proc_data_dict['frac_single']}


    def measure_msmt_induced_dephasing(
            self,
            MC: Optional[MeasurementControl] = None,
            sequence='ramsey',
            label: str = '',
            verbose: bool = True,
            analyze: bool = True,
            close_fig: bool = True,
            update: bool = True,
            cross_target_qubits: list = None,
            multi_qubit_platf_cfg=None,
            target_qubit_excited=False,
            extra_echo=False
    ):
        # Refs:
        # Schuster PRL 94, 123602 (2005)
        # Gambetta PRA 74, 042318 (2006)
        if MC is None:
            MC = self.instr_MC.get_instr()
        if cross_target_qubits is None:
            platf_cfg = self.cfg_openql_platform_fn()
        else:
            platf_cfg = multi_qubit_platf_cfg

        self.prepare_for_timedomain()
        self.instr_LutMan_MW.get_instr().load_phase_pulses_to_AWG_lookuptable()

        if cross_target_qubits is None:
            qubits = [self.cfg_qubit_nr()]
        else:
            qubits = []
            for cross_target_qubit in cross_target_qubits:
                qubits.append(cross_target_qubit.cfg_qubit_nr())
            qubits.append(self.cfg_qubit_nr())

        # angles = np.arange(0, 421, 20)
        angles = np.concatenate([np.arange(0, 101, 20), np.arange(140, 421, 20)])  # avoid CW15, issue

        # generate OpenQL program
        if sequence == 'ramsey':
            readout_pulse_length = self.ro_pulse_length()
            readout_pulse_length += self.ro_pulse_down_length0()
            readout_pulse_length += self.ro_pulse_down_length1()
            if extra_echo:
                wait_time = readout_pulse_length / 2 + 0e-9
            else:
                wait_time = 0

            p = mqo.Ramsey_msmt_induced_dephasing(  # FIXME: renamed to Msmt_induced_dephasing_ramsey and changed
                qubits=qubits,
                angles=angles,
                platf_cfg=platf_cfg,
                target_qubit_excited=target_qubit_excited,
                extra_echo=extra_echo,
                wait_time=wait_time
            )
        elif sequence == 'echo':
            readout_pulse_length = self.ro_pulse_length()
            readout_pulse_length += self.ro_pulse_down_length0()
            readout_pulse_length += self.ro_pulse_down_length1()
            if extra_echo:
                wait_time = readout_pulse_length / 2 + 20e-9
            else:
                wait_time = readout_pulse_length + 40e-9

            p = mqo.echo_msmt_induced_dephasing(  # FIXME: vanished
                qubits=qubits,
                angles=angles,
                platf_cfg=platf_cfg,
                wait_time=wait_time,
                target_qubit_excited=target_qubit_excited,
                extra_echo=extra_echo
            )
        else:
            raise ValueError('sequence must be set to ramsey or echo')

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name='angle',
            unit='degree'
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(angles)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run(sequence + label + self.msmt_suffix)

        if analyze:
            a = ma.Ramsey_Analysis(
                label=sequence,
                auto=True,
                close_fig=True,
                freq_qubit=self.freq_qubit(),
                artificial_detuning=0,  # FIXME
                phase_sweep_only=True
            )
            phase_deg = (a.fit_res.params['phase'].value) * 360 / (2 * np.pi) % 360
            res = {
                'coherence': a.fit_res.params['amplitude'].value,
                'phase': phase_deg,
            }
            if verbose:
                print('> ramsey analyse', res)
            return res
        # else:
        #    return {'coherence': -1,
        #            'phase' : -1}


    def measure_CPMG(
            self,
            times=None,
            orders=None,
            MC: Optional[MeasurementControl] = None,
            sweep='tau',
            analyze=True,
            close_fig=True,
            update=False,
            label: str = '',
            prepare_for_timedomain=True
    ):
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None and sweep == 'tau':
            # funny default is because there is no real time sideband
            # modulation
            stepsize = max((self.T2_echo() * 2 / 61) // (abs(self.cfg_cycle_time()))
                           * abs(self.cfg_cycle_time()), 20e-9)
            times = np.arange(0, self.T2_echo() * 4, stepsize * 2)

        if orders is None and sweep == 'tau':
            orders = 2
        if orders < 1 and sweep == 'tau':
            raise ValueError('Orders must be larger than 1')

        # append the calibration points, times are for location in plot
        if sweep == 'tau':
            dt = times[1] - times[0]
            times = np.concatenate([times,
                                    (times[-1] + 1 * dt,
                                     times[-1] + 2 * dt,
                                     times[-1] + 3 * dt,
                                     times[-1] + 4 * dt)])
        elif sweep == 'order':
            dn = orders[1] - orders[0]
            orders = np.concatenate([orders,
                                     (orders[-1] + 1 * dn,
                                      orders[-1] + 2 * dn,
                                      orders[-1] + 3 * dn,
                                      orders[-1] + 4 * dn)])
        # # Checking if pulses are on 20 ns grid
        if sweep == 'tau':
            if not all([np.round((t * 1e9) / (2 * orders)) % (self.cfg_cycle_time() * 1e9) == 0 for t in times]):
                raise ValueError('timesteps must be multiples of 40e-9')
        elif sweep == 'order':
            if not np.round(times / 2) % (self.cfg_cycle_time() * 1e9) == 0:
                raise ValueError('timesteps must be multiples of 40e-9')

        # # Checking if pulses are locked to the pulse modulation
        if sweep == 'tau':
            if not all([np.round(t / 1 * 1e9) % (2 / self.mw_freq_mod.get() * 1e9) == 0 for t in times]):
                raise ValueError('timesteps must be multiples of 2 modulation periods')

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        mw_lutman = self.instr_LutMan_MW.get_instr()
        mw_lutman.load_phase_pulses_to_AWG_lookuptable()

        # generate OpenQL program
        if sweep == 'tau':
            print(times)
            p = sqo.CPMG(
                times,
                orders,
                qubit_idx=self.cfg_qubit_nr(),
                platf_cfg=self.cfg_openql_platform_fn()
            )
            s = swf.OpenQL_Sweep(
                openql_program=p,
                CCL=self.instr_CC.get_instr(),
                parameter_name="Time",
                unit="s"
            )
        elif sweep == 'order':
            p = sqo.CPMG_SO(
                times,
                orders,
                qubit_idx=self.cfg_qubit_nr(),
                platf_cfg=self.cfg_openql_platform_fn()
            )
            s = swf.OpenQL_Sweep(
                openql_program=p,
                CCL=self.instr_CC.get_instr(),
                parameter_name="Order", unit=""
            )

        d = self.int_avg_det
        MC.set_sweep_function(s)
        if sweep == 'tau':
            MC.set_sweep_points(times)
        elif sweep == 'order':
            MC.set_sweep_points(orders)
        MC.set_detector_function(d)
        if sweep == 'tau':
            msmt_title = 'CPMG_order_' + str(orders) + label + self.msmt_suffix
        elif sweep == 'order':
            msmt_title = 'CPMG_tauN_' + str(times) + label + self.msmt_suffix
        MC.run(msmt_title)

        if analyze:
            # N.B. v1.5 analysis
            if sweep == 'tau':
                a = ma.Echo_analysis_V15(label='CPMG', auto=True, close_fig=True)
                if update:
                    self.T2_echo(a.fit_res.params['tau'].value)
            elif sweep == 'order':
                a = ma2.Single_Qubit_TimeDomainAnalysis(label='CPMG', auto=True, close_fig=True)

            return a


    def measure_spin_locking_simple(
            self,
            times=None,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            update=True,
            label: str = '',
            prepare_for_timedomain=True,
            tomo=False,
            mw_gate_duration: float = 40e-9
    ):
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            # funny default is because there is no real time sideband modulation
            stepsize = max((self.T2_echo() * 2 / 61) // (abs(self.cfg_cycle_time()))
                           * abs(self.cfg_cycle_time()), 20e-9)
            times = np.arange(0, self.T2_echo() * 4, stepsize * 2)

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        if tomo:
            times = np.concatenate([np.repeat(times, 2),
                                    (times[-1] + 1 * dt,
                                     times[-1] + 2 * dt,
                                     times[-1] + 3 * dt,
                                     times[-1] + 4 * dt,
                                     times[-1] + 5 * dt,
                                     times[-1] + 6 * dt)])
        else:
            times = np.concatenate([times,
                                    (times[-1] + 1 * dt,
                                     times[-1] + 2 * dt,
                                     times[-1] + 3 * dt,
                                     times[-1] + 4 * dt)])

        # # Checking if pulses are on 20 ns grid
        if not all([np.round(t * 1e9) % (self.cfg_cycle_time() * 1e9) == 0 for t in times]):
            raise ValueError('timesteps must be multiples of 20e-9')

        # # Checking if pulses are locked to the pulse modulation
        if not all([np.round(t / 1 * 1e9) % (2 / self.mw_freq_mod.get() * 1e9) == 0 for t in times]):
            raise ValueError('timesteps must be multiples of 2 modulation periods')

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        mw_lutman = self.instr_LutMan_MW.get_instr()
        mw_lutman.load_square_waves_to_AWG_lookuptable()

        p = sqo.spin_lock_simple(
            times,
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn(),
            tomo=tomo,
            mw_gate_duration=mw_gate_duration
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name="Time",
            unit="s"
        )
        d = self.int_avg_det
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        MC.set_detector_function(d)
        MC.run('spin_lock_simple' + label + self.msmt_suffix)

        if analyze:
            a = ma.T1_Analysis(label='spin_lock_simple', auto=True, close_fig=True)
            return a


    def measure_spin_locking_echo(
            self,
            times=None,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            update=True,
            label: str = '',
            prepare_for_timedomain=True
    ):
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            # funny default is because there is no real time sideband modulation
            stepsize = max((self.T2_echo() * 2 / 61) // (abs(self.cfg_cycle_time()))
                           * abs(self.cfg_cycle_time()), 20e-9)
            times = np.arange(0, self.T2_echo() * 4, stepsize * 2)

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        times = np.concatenate([times,
                                (times[-1] + 1 * dt,
                                 times[-1] + 2 * dt,
                                 times[-1] + 3 * dt,
                                 times[-1] + 4 * dt)])

        # # Checking if pulses are on 20 ns grid
        if not all([np.round(t * 1e9) % (self.cfg_cycle_time() * 1e9) == 0 for t in times]):
            raise ValueError('timesteps must be multiples of 20e-9')

        # # Checking if pulses are locked to the pulse modulation
        if not all([np.round(t / 1 * 1e9) % (2 / self.mw_freq_mod.get() * 1e9) == 0 for t in times]):
            raise ValueError('timesteps must be multiples of 2 modulation periods')

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        mw_lutman = self.instr_LutMan_MW.get_instr()
        mw_lutman.load_square_waves_to_AWG_lookuptable()

        p = sqo.spin_lock_echo(
            times,
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn()
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name="Time",
            unit="s"
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('spin_lock_echo' + label + self.msmt_suffix)

        if analyze:
            a = ma.T1_Analysis(label='spin_lock_echo', auto=True, close_fig=True)
            return a


    def measure_rabi_frequency(
            self,
            times=None,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            update=True,
            label: str = '',
            prepare_for_timedomain=True,
            tomo=False,
            mw_gate_duration: float = 40e-9
    ):
        if MC is None:
            MC = self.instr_MC.get_instr()

        # default timing
        if times is None:
            # funny default is because there is no real time sideband
            # modulation
            stepsize = max((self.T2_echo() * 2 / 61) // (abs(self.cfg_cycle_time()))
                           * abs(self.cfg_cycle_time()), 160e-9)
            times = np.arange(0, self.T2_echo() * 4, stepsize * 2)

        # append the calibration points, times are for location in plot
        dt = times[1] - times[0]
        if tomo:
            times = np.concatenate([np.repeat(times, 2),
                                    (times[-1] + 1 * dt,
                                     times[-1] + 2 * dt,
                                     times[-1] + 3 * dt,
                                     times[-1] + 4 * dt,
                                     times[-1] + 5 * dt,
                                     times[-1] + 6 * dt)])
        else:
            times = np.concatenate([times,
                                    (times[-1] + 1 * dt,
                                     times[-1] + 2 * dt,
                                     times[-1] + 3 * dt,
                                     times[-1] + 4 * dt)])

        # # # Checking if pulses are on 20 ns grid
        # if not all([np.round(t*1e9) % (self.cfg_cycle_time()*1e9) == 0 for
        #             t in times]):
        #     raise ValueError('timesteps must be multiples of 40e-9')

        # # # Checking if pulses are locked to the pulse modulation
        # if not all([np.round(t/1*1e9) % (2/self.mw_freq_mod.get()*1e9)
        #             == 0 for t in times]):
        #     raise ValueError(
        #         'timesteps must be multiples of 2 modulation periods')

        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        mw_lutman = self.instr_LutMan_MW.get_instr()
        mw_lutman.load_square_waves_to_AWG_lookuptable()

        p = sqo.rabi_frequency(
            times,
            qubit_idx=self.cfg_qubit_nr(),
            platf_cfg=self.cfg_openql_platform_fn(),
            mw_gate_duration=mw_gate_duration,
            tomo=tomo
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            CCL=self.instr_CC.get_instr(),
            parameter_name="Time", unit="s"
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(times)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('rabi_frequency' + label + self.msmt_suffix)

        if analyze:
            a = ma.Echo_analysis_V15(label='rabi_frequency', auto=True, close_fig=True)
            return a


    def measure_single_qubit_randomized_benchmarking(
            self,
            nr_cliffords=2 ** np.arange(12),
            nr_seeds=100,
            MC: Optional[MeasurementControl] = None,
            recompile: bool = 'as needed',
            prepare_for_timedomain: bool = True,
            ignore_f_cal_pts: bool = False,
            compile_only: bool = False,
            rb_tasks=None,
            disable_metadata = False):
        # USED_BY: inspire_dependency_graph.py,
        """
        Measures randomized benchmarking decay including second excited state
        population.

        For this it:
            - stores single shots using SSB weights (int. logging)
            - uploads a pulse driving the ef/12 transition (should be calibr.)
            - performs RB both with and without an extra pi-pulse
            - Includes calibration poitns for 0, 1, and 2 (g,e, and f)
            - analysis extracts fidelity and leakage/seepage

        Refs:
        Knill PRA 77, 012307 (2008)
        Wood PRA 97, 032306 (2018)

        Args:
            nr_cliffords (array):
                list of lengths of the clifford gate sequences

            nr_seeds (int):
                number of random sequences for each sequence length

            recompile (bool, str {'as needed'}):
                indicate whether to regenerate the sequences of clifford gates.
                By default it checks whether the needed sequences were already
                generated since the most recent change of OpenQL file
                specified in self.cfg_openql_platform_fn
        """

        # because only 1 seed is uploaded each time
        if MC is None:
            MC = self.instr_MC.get_instr()

        # Settings that have to be changed....
        old_weight_type = self.ro_acq_weight_type()
        old_digitized = self.ro_acq_digitized()
        self.ro_acq_weight_type('optimal IQ')
        self.ro_acq_digitized(False)

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        else:
            self.prepare_readout()

        MC.soft_avg(1)  # FIXME: changes state

        # restore settings
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

        # Load pulses to the ef transition
        mw_lutman = self.instr_LutMan_MW.get_instr()
        mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable()

        net_cliffords = [0, 3]  # always measure double sided

        def send_rb_tasks(pool_):
            tasks_inputs = []
            for i in range(nr_seeds):
                task_dict = dict(
                    qubits=[self.cfg_qubit_nr()],
                    nr_cliffords=nr_cliffords,
                    net_cliffords=net_cliffords,  # always measure double sided
                    nr_seeds=1,
                    platf_cfg=self.cfg_openql_platform_fn(),
                    program_name='RB_s{}_ncl{}_net{}_{}'.format(i, nr_cliffords, net_cliffords, self.name),
                    recompile=recompile
                )
                tasks_inputs.append(task_dict)
            # pool.starmap_async can be used for positional arguments
            # but we are using a wrapper
            rb_tasks = pool_.map_async(cl_oql.parallel_friendly_rb, tasks_inputs)

            return rb_tasks

        if compile_only:
            raise NotImplementedError
            # FIXME: code below contains errors:
            # assert pool is not None
            # rb_tasks = send_rb_tasks(pool)
            # return rb_tasks

        if rb_tasks is None:
            # Using `with ...:` makes sure the other processes will be terminated
            # avoid starting too mane processes,
            # nr_processes = None will start as many as the PC can handle
            nr_processes = None if recompile else 1
            with multiprocessing.Pool(nr_processes) as pool:
                rb_tasks = send_rb_tasks(pool)
                cl_oql.wait_for_rb_tasks(rb_tasks)

        print(rb_tasks)
        programs_filenames = rb_tasks.get()

        counter_param = ManualParameter('name_ctr', initial_value=0)
        prepare_function_kwargs = {
            'counter_param': counter_param,
            'programs_filenames': programs_filenames,
            'CC': self.instr_CC.get_instr()}

        # to include calibration points
        sweep_points = np.append(
            # repeat twice because of net clifford being 0 and 3
            np.repeat(nr_cliffords, 2),
            [nr_cliffords[-1] + .5] * 2 + [nr_cliffords[-1] + 1.5] * 2 +
            [nr_cliffords[-1] + 2.5] * 2,
        )

        s = swf.None_Sweep(parameter_name='Number of Cliffords', unit='#')
        MC.set_sweep_function(s)
        reps_per_seed = 4094 // len(sweep_points)
        MC.set_sweep_points(np.tile(sweep_points, reps_per_seed * nr_seeds))
        d = self.int_log_det
        d.prepare_function = load_range_of_oql_programs_from_filenames
        d.prepare_function_kwargs = prepare_function_kwargs
        d.nr_shots = reps_per_seed * len(sweep_points)
        MC.set_detector_function(d)
        MC.run('RB_{}seeds'.format(nr_seeds) + self.msmt_suffix, exp_metadata={'bins': sweep_points},
                disable_snapshot_metadata = disable_metadata)

        a = ma2.RandomizedBenchmarking_SingleQubit_Analysis(
            label='RB_',
            rates_I_quad_ch_idx=0,
            cal_pnts_in_dset=np.repeat(["0", "1", "2"], 2)
        )
        
        for key in a.proc_data_dict['quantities_of_interest'].keys():
            if 'eps_simple_lin_trans' in key:
                self.F_RB((1-a.proc_data_dict['quantities_of_interest'][key].n)**(1/1.875))
        
        return True


    def measure_randomized_benchmarking_old(
            self,
            nr_cliffords=2 ** np.arange(12),
            nr_seeds=100,
            double_curves=False,
            MC: Optional[MeasurementControl] = None,
            analyze=True,
            close_fig=True,
            verbose: bool = True,
            upload=True,
            update=True
    ):
        # USED_BY: device_dependency_graphs_v2.py,

        # Old version not including two-state calibration points and logging
        # detector.
        # Adding calibration points
        if double_curves:
            nr_cliffords = np.repeat(nr_cliffords, 2)
        nr_cliffords = np.append(
            nr_cliffords, [nr_cliffords[-1] + .5] * 2 + [nr_cliffords[-1] + 1.5] * 2)
        self.prepare_for_timedomain()
        if MC is None:
            MC = self.instr_MC.get_instr()

        MC.soft_avg(nr_seeds)  # FIXME: changes state
        counter_param = ManualParameter('name_ctr', initial_value=0)
        programs = []
        if verbose:
            print('Generating {} RB programs'.format(nr_seeds))
        t0 = time.time()
        for i in range(nr_seeds):
            p = sqo.randomized_benchmarking(
                qubit_idx=self.cfg_qubit_nr(),
                nr_cliffords=nr_cliffords,
                platf_cfg=self.cfg_openql_platform_fn(),
                nr_seeds=1, program_name='RB_{}'.format(i),
                double_curves=double_curves
            )
            programs.append(p)
        if verbose:
            print('Succesfully generated {} RB programs in {:.1f}s'.format(nr_seeds, time.time() - t0))

        prepare_function_kwargs = {
            'counter_param': counter_param,
            'programs': programs,
            'CC': self.instr_CC.get_instr()
        }

        s = swf.None_Sweep()
        s.parameter_name = 'Number of Cliffords'
        s.unit = '#'
        MC.set_sweep_function(s)
        MC.set_sweep_points(nr_cliffords)
        d = self.int_avg_det
        d.prepare_function = load_range_of_oql_programs
        d.prepare_function_kwargs = prepare_function_kwargs
        d.nr_averages = 128
        MC.set_detector_function(d)
        MC.run('RB_{}seeds'.format(nr_seeds) + self.msmt_suffix)

        if double_curves:
            a = ma.RB_double_curve_Analysis(
                T1=self.T1(),
                pulse_delay=self.mw_gauss_width.get() * 4
            )
        else:
            a = ma.RandomizedBenchmarking_Analysis(
                close_main_fig=close_fig, T1=self.T1(),
                pulse_delay=self.mw_gauss_width.get() * 4
            )
        if update:
            self.F_RB(a.fit_res.params['fidelity_per_Clifford'].value)
        return a.fit_res.params['fidelity_per_Clifford'].value


    def measure_ef_rabi_2D(
            self,
            amps: list = np.linspace(0, .8, 18),
            anharmonicity: list = np.arange(-275e6, -326e6, -5e6),
            recovery_pulse: bool = True,
            MC: Optional[MeasurementControl] = None,
            label: str = '',
            analyze=True,
            close_fig=True,
            prepare_for_timedomain=True,
            disable_metadata = False):
        """
        Measures a rabi oscillation of the ef/12 transition.

        Modulation frequency of the "ef" pulses is controlled through the
        `anharmonicity` parameter of the qubit object.
        Hint: the expected pi-pulse amplitude of the ef/12 transition is ~1/2
            the pi-pulse amplitude of the ge/01 transition.
        """
        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        mw_lutman = self.instr_LutMan_MW.get_instr()
        mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable(amps=amps)

        p = sqo.ef_rabi_seq(
            self.cfg_qubit_nr(),
            amps=amps, recovery_pulse=recovery_pulse,
            platf_cfg=self.cfg_openql_platform_fn(),
            add_cal_points=False
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            parameter_name='Pulse amp',
            unit='dac',
            CCL=self.instr_CC.get_instr()
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(p.sweep_points)
        MC.set_sweep_function_2D(swf.anharmonicity_sweep(qubit=self, amps=amps))
        MC.set_sweep_points_2D(anharmonicity)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('ef_rabi_2D' + label + self.msmt_suffix, mode='2D', disable_snapshot_metadata = disable_metadata)

        if analyze:
            a = ma.TwoD_Analysis()
            return a


    def measure_ef_rabi(
            self,
            amps: list = np.linspace(0, .8, 18),
            recovery_pulse: bool = True,
            MC: Optional[MeasurementControl] = None,
            label: str = '',
            analyze=True,
            close_fig=True,
            prepare_for_timedomain=True,
            disable_metadata = False
    ):
        """
        Measures a rabi oscillation of the ef/12 transition.

        Modulation frequency of the "ef" pulses is controlled through the
        `anharmonicity` parameter of the qubit object.
        Hint: the expected pi-pulse amplitude of the ef/12 transition is ~1/2
            the pi-pulse amplitude of the ge/01 transition.
        """
        if MC is None:
            MC = self.instr_MC.get_instr()
        if prepare_for_timedomain:
            self.prepare_for_timedomain()

        mw_lutman = self.instr_LutMan_MW.get_instr()
        mw_lutman.load_ef_rabi_pulses_to_AWG_lookuptable(amps=amps)

        p = sqo.ef_rabi_seq(
            self.cfg_qubit_nr(),
            amps=amps, recovery_pulse=recovery_pulse,
            platf_cfg=self.cfg_openql_platform_fn(),
            add_cal_points=True
        )

        s = swf.OpenQL_Sweep(
            openql_program=p,
            parameter_name='Pulse amp',
            unit='dac',
            CCL=self.instr_CC.get_instr()
        )
        MC.set_sweep_function(s)
        MC.set_sweep_points(p.sweep_points)
        d = self.int_avg_det
        MC.set_detector_function(d)
        MC.run('ef_rabi' + label + self.msmt_suffix, disable_snapshot_metadata = disable_metadata)

        if analyze:
            a2 = ma2.EFRabiAnalysis(close_figs=True, label='ef_rabi')
            # if update:
            #     ef_pi_amp = a2.proc_data_dict['ef_pi_amp']
            #     self.ef_amp180(a2.proc_data_dict['ef_pi_amp'])
            return a2


    def measure_gst_1Q(
            self,
            shots_per_meas: int,
            maxL: int = 256,
            MC: Optional[MeasurementControl] = None,
            recompile='as needed',
            prepare_for_timedomain: bool = True
    ):
        """
        Performs single qubit Gate Set Tomography experiment of the StdXYI gateset.

        Requires optimal weights and a calibrated digitized readout.

        Args:
            shots_per_meas (int):
            maxL (int)          : specifies the maximum germ length,
                                  must be power of 2.
            lite_germs(bool)    : if True uses "lite" germs


        """
        if MC is None:
            MC = self.instr_MC.get_instr()

        ########################################
        # Readout settings that have to be set #
        ########################################

        old_weight_type = self.ro_acq_weight_type()
        old_digitized = self.ro_acq_digitized()
        self.ro_acq_weight_type('optimal')
        self.ro_acq_digitized(True)

        if prepare_for_timedomain:
            self.prepare_for_timedomain()
        else:
            self.prepare_readout()

        MC.soft_avg(1)  # FIXME: changes state

        # restore the settings
        self.ro_acq_weight_type(old_weight_type)
        self.ro_acq_digitized(old_digitized)

        ########################################
        # Readout settings that have to be set #
        ########################################

        programs, exp_list_fn = pygsti_oql.single_qubit_gst(
            q0=self.cfg_qubit_nr(),
            maxL=maxL,
            platf_cfg=self.cfg_openql_platform_fn(),
            recompile=recompile
        )

        counter_param = ManualParameter('name_ctr', initial_value=0)

        s = swf.OpenQL_Sweep(openql_program=programs[0], CCL=self.instr_CC.get_instr())
        d = self.int_log_det

        # poor man's GST contains 731 distinct gatestrings

        sweep_points = np.concatenate([p.sweep_points for p in programs])
        nr_of_meas = len(sweep_points)
        print('nr_of_meas:', nr_of_meas)

        prepare_function_kwargs = {
            'counter_param': counter_param,
            'programs': programs,
            'CC': self.instr_CC.get_instr(),
            'detector': d}
        # hacky as heck
        d.prepare_function_kwargs = prepare_function_kwargs
        d.prepare_function = oqh.load_range_of_oql_programs_varying_nr_shots

        shots = np.tile(sweep_points, shots_per_meas)

        MC.soft_avg(1)  # FIXME: changes state
        MC.set_sweep_function(s)
        MC.set_sweep_points(shots)
        MC.set_detector_function(d)
        MC.run('Single_qubit_GST_L{}_{}'.format(maxL, self.msmt_suffix),
               exp_metadata={'bins': sweep_points,
                             'gst_exp_list_filename': exp_list_fn})
        a = ma2.GST_SingleQubit_DataExtraction(label='Single_qubit_GST')
        return a


    def measure_flux_arc_tracked_spectroscopy(
            self,
            dac_values=None,
            polycoeffs=None,
            MC: Optional[MeasurementControl] = None,
            nested_MC: Optional[MeasurementControl] = None,
            fluxChan=None
    ):
        """
        Creates a qubit DAC arc by fitting a polynomial function through qubit
        frequencies obtained by spectroscopy.

        If polycoeffs is given, it will predict the first frequencies to
        measure by from this estimate. If not, it will use a wider range in
        spectroscopy for the first to values to ensure a peak in spectroscopy
        is found.

        It will fit a 2nd degree polynomial each time qubit spectroscopy is
        performed, and all measured qubit frequencies to construct a new
        polynomial after each spectroscopy measurement.

        Args:
            dac_values (array):
                DAC values that are to be probed, which control the flux bias

            polycoeffs (array):
                initial coefficients of a second order polynomial. Used
                for predicting the qubit frequencies in the arc.

            MC (MeasurementControl):
                main MC that varies the DAC current

            nested_MC (MeasurementControl):
                MC that will measure spectroscopy for each current.
                Is used inside the composite detector

            fluxChan (str):
                Fluxchannel that is varied. Defaults to self.fl_dc_ch
        """

        if dac_values is None:
            if self.fl_dc_I0() is None:
                dac_values = np.linspace(-5e-3, 5e-3, 11)
            else:
                dac_values_1 = np.linspace(self.fl_dc_I0(),
                                           self.fl_dc_I0() + 3e-3,
                                           11)
                dac_values_2 = np.linspace(self.fl_dc_I0() + 3e-3,
                                           self.fl_dc_I0() + 5e-3,
                                           6)
                dac_values_ = np.linspace(self.fl_dc_I0(),
                                          self.fl_dc_I0() - 5e-3,
                                          11)

        dac_values = np.concatenate([dac_values_1, dac_values_2])

        if MC is None:
            MC = self.instr_MC.get_instr()

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        fluxcontrol = self.instr_FluxCtrl.get_instr()
        if fluxChan is None:
            dac_par = fluxcontrol.parameters[(self.fl_dc_ch())]
        else:
            dac_par = fluxcontrol.parameters[(fluxChan)]

        if polycoeffs is None:
            polycoeffs = self.fl_dc_polycoeff()

        d = cdf.Tracked_Qubit_Spectroscopy(
            qubit=self,
            nested_MC=nested_MC,
            qubit_initial_frequency=self.freq_qubit(),
            resonator_initial_frequency=self.freq_res(),
            sweep_points=dac_values,
            polycoeffs=polycoeffs
        )

        MC.set_sweep_function(dac_par)
        MC.set_sweep_points(dac_values)
        MC.set_detector_function(d)
        MC.run(name='Tracked_Spectroscopy')


    def measure_msmt_induced_dephasing_sweeping_amps(
            self,
            amps_rel=None,
            nested_MC: Optional[MeasurementControl] = None,
            cross_target_qubits=None,
            multi_qubit_platf_cfg=None,
            analyze=False,
            verbose: bool = True,
            sequence='ramsey',
            target_qubit_excited=False,
            extra_echo=False
    ):
        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        if cross_target_qubits is None or (len(cross_target_qubits) == 1 and self.name == cross_target_qubits[0]):
            cross_target_qubits = None

        if cross_target_qubits is None:
            # Only measure on a single Qubit
            cfg_qubit_nrs = [self.cfg_qubit_nr()]
            optimization_M_amps = [self.ro_pulse_amp()]
            optimization_M_amp_down0s = [self.ro_pulse_down_amp0()]
            optimization_M_amp_down1s = [self.ro_pulse_down_amp1()]
            readout_pulse_length = self.ro_pulse_length()
            readout_pulse_length += self.ro_pulse_down_length0()
            readout_pulse_length += self.ro_pulse_down_length1()
            amps_rel = np.linspace(0, 0.5, 11) if amps_rel is None else amps_rel
        else:
            cfg_qubit_nrs = []
            optimization_M_amps = []
            optimization_M_amp_down0s = []
            optimization_M_amp_down1s = []
            readout_pulse_lengths = []
            for cross_target_qubit in cross_target_qubits:
                cfg_qubit_nrs.append(cross_target_qubit.cfg_qubit_nr())
                optimization_M_amps.append(cross_target_qubit.ro_pulse_amp())
                optimization_M_amp_down0s.append(cross_target_qubit.ro_pulse_down_amp0())
                optimization_M_amp_down1s.append(cross_target_qubit.ro_pulse_down_amp1())
                ro_len = cross_target_qubit.ro_pulse_length()
                ro_len += cross_target_qubit.ro_pulse_down_length0()
                ro_len += cross_target_qubit.ro_pulse_down_length1()
                readout_pulse_lengths.append(ro_len)

            readout_pulse_length = np.max(readout_pulse_lengths)

        RO_lutman = self.instr_LutMan_RO.get_instr()
        if sequence == 'ramsey':
            RO_lutman.set('M_final_delay_R{}'.format(self.cfg_qubit_nr()), 200e-9)
        elif sequence == 'echo':
            RO_lutman.set('M_final_delay_R{}'.format(self.cfg_qubit_nr()), 200e-9)  # +readout_pulse_length)
        else:
            raise NotImplementedError('dephasing sequence not recognized')

        RO_lutman.set('M_final_amp_R{}'.format(self.cfg_qubit_nr()), self.ro_pulse_amp())

        # save and change parameters
        waveform_name = 'up_down_down_final'  # FIXME: misnomer
        old_waveform_name = self.ro_pulse_type()
        self.ro_pulse_type(waveform_name)
        old_delay = self.ro_acq_delay()
        d = RO_lutman.get('M_final_delay_R{}'.format(self.cfg_qubit_nr()))  # FIXME: just set a few lines above
        self.ro_acq_delay(old_delay + readout_pulse_length + d)

        # self.ro_acq_integration_length(readout_pulse_length+100e-9)
        self.ro_acq_weight_type('SSB')  # FIXME: changes state
        self.prepare_for_timedomain()

        # save and change some more parameters
        old_ro_prepare_state = self.cfg_prepare_ro_awg()
        self.cfg_prepare_ro_awg(False)

        sweep_function = swf.lutman_par_depletion_pulse_global_scaling(
            LutMan=RO_lutman,
            resonator_numbers=cfg_qubit_nrs,
            optimization_M_amps=optimization_M_amps,
            optimization_M_amp_down0s=optimization_M_amp_down0s,
            optimization_M_amp_down1s=optimization_M_amp_down1s,
            upload=True
        )
        d = det.Function_Detector(
            self.measure_msmt_induced_dephasing,
            msmt_kw={
                'cross_target_qubits': cross_target_qubits,
                'multi_qubit_platf_cfg': multi_qubit_platf_cfg,
                'analyze': True,
                'sequence': sequence,
                'target_qubit_excited': target_qubit_excited,
                'extra_echo': extra_echo
            },
            result_keys=['coherence', 'phase']
        )

        nested_MC.set_sweep_function(sweep_function)
        nested_MC.set_sweep_points(amps_rel)
        nested_MC.set_detector_function(d)

        label = 'ro_amp_sweep_dephasing' + self.msmt_suffix
        nested_MC.run(label)

        # Restore qubit objects parameters to previous settings
        self.ro_pulse_type(old_waveform_name)
        self.ro_acq_delay(old_delay)
        self.cfg_prepare_ro_awg(old_ro_prepare_state)

        if analyze:
            res = ma.MeasurementAnalysis(
                label=label, plot_all=False, auto=True)
            return res


    def measure_SNR_sweeping_amps(
            self,
            amps_rel,
            nr_shots=2 * 4094,
            nested_MC: Optional[MeasurementControl] = None,
            analyze=True
    ):
        """
        Measures SNR and readout fidelities as a function of the readout pulse
        amplitude. Resonator depletion pulses are automatically scaled.
        Weights are not optimized - routine is intended to be used with SSB weights.

        Args:
            amps_rel (array):
                readout pulse amplitudes to loop over. Value of 1 indicates
                amplitude currently specified in the qubit object.

            nr_shots (int):
                total number of measurements in qubit ground and excited state
        """

        if nested_MC is None:
            nested_MC = self.instr_nested_MC.get_instr()

        self.prepare_for_timedomain()

        RO_lutman = self.instr_LutMan_RO.get_instr()

        # save and change parameters
        old_ro_prepare_state = self.cfg_prepare_ro_awg()
        self.cfg_prepare_ro_awg(False)

        sweep_function = swf.lutman_par_depletion_pulse_global_scaling(
            LutMan=RO_lutman,
            resonator_numbers=[self.cfg_qubit_nr()],
            optimization_M_amps=[self.ro_pulse_amp()],
            optimization_M_amp_down0s=[self.ro_pulse_down_amp0()],
            optimization_M_amp_down1s=[self.ro_pulse_down_amp1()],
            upload=True
        )
        d = det.Function_Detector(
            self.measure_ssro,
            msmt_kw={
                'nr_shots': nr_shots,
                'analyze': True, 'SNR_detector': True,
                'cal_residual_excitation': False,
            },
            result_keys=['SNR', 'F_d', 'F_a']
        )

        nested_MC.set_sweep_function(sweep_function)
        nested_MC.set_sweep_points(amps_rel)
        nested_MC.set_detector_function(d)
        label = 'ro_amp_sweep_SNR' + self.msmt_suffix
        nested_MC.run(label)

        # restore parameters
        self.cfg_prepare_ro_awg(old_ro_prepare_state)

        if analyze:
            ma.MeasurementAnalysis(label=label, plot_all=False, auto=True)


    def measure_quantum_efficiency(
            self,
            amps_rel=None,
            nr_shots=2 * 4094,
            analyze=True,
            verbose=True,
            dephasing_sequence='ramsey'
    ):
        # requires the cc light to have the readout time configured equal
        # to the measurement and depletion time + 60 ns buffer
        # it requires an optimized depletion pulse
        amps_rel = np.linspace(0, 0.5, 11) if amps_rel is None else amps_rel
        self.cfg_prepare_ro_awg(True)

        start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.measure_msmt_induced_dephasing_sweeping_amps(
            amps_rel=amps_rel,
            analyze=False,
            sequence=dephasing_sequence
        )
        readout_pulse_length = self.ro_pulse_length()
        readout_pulse_length += self.ro_pulse_down_length0()
        readout_pulse_length += self.ro_pulse_down_length1()
        # self.ro_acq_integration_length(readout_pulse_length+0e-9)

        self.ro_pulse_type('up_down_down')  # FIXME: changes state
        # setting acquisition weights to optimal
        self.ro_acq_weight_type('optimal')  # FIXME: changes state

        # calibrate residual excitation and relaxation at high power
        self.measure_ssro(
            cal_residual_excitation=True,
            SNR_detector=True,
            nr_shots=nr_shots,
            update_threshold=False
        )
        self.measure_SNR_sweeping_amps(
            amps_rel=amps_rel,
            analyze=False
        )

        end_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # set the pulse back to optimal depletion
        self.ro_pulse_type('up_down_down')

        if analyze:
            options_dict = {
                'individual_plots': True,
                'verbose': verbose,
            }
            qea = ma2.QuantumEfficiencyAnalysis(
                t_start=start_time,
                t_stop=end_time,
                use_sweeps=True,
                options_dict=options_dict,
                label_dephasing='_ro_amp_sweep_dephasing' + self.msmt_suffix,
                label_ssro='_ro_amp_sweep_SNR' + self.msmt_suffix)

            # qea.run_analysis()
            eta = qea.fit_dicts['eta']
            u_eta = qea.fit_dicts['u_eta']

            return {'eta': eta, 'u_eta': u_eta,
                    't_start': start_time, 't_stop': end_time}
        else:
            return {}

    ##########################################################################
    # other functions (HAL_Transmon specific)
    ##########################################################################

    def bus_frequency_flux_sweep(
            self,
            freqs,
            spec_source_bus,
            bus_power,
            dacs,
            dac_param,
            f01=None,
            label='',
            close_fig=True,
            analyze=True,
            MC: Optional[MeasurementControl] = None,
            prepare_for_continuous_wave=True
    ):
        """
        Drive the qubit and sit at the spectroscopy peak while the bus is driven with
        bus_spec_source. At the same time sweep dac channel specified by dac_param over
        set of values sepcifeid by dacs.

        Practical comments:
        - sweep flux bias of different (neighbour) qubit than the one measured
        - set spec_power of the first tone high (say, +15 dB relative to value optimal
                for sharp spectroscopy). This makes you less sensitive to flux crosstalk.

        Args:
            freqs (array):
                list of frequencies of the second drive tone (at bus frequency)

            spec_source_bus (RohdeSchwarz_SGS100A):
                rf source used for the second spectroscopy tone

            bus_power (float):
                power of the second spectroscopy tone

            dacs (array):
                valuses of current bias to measure

            dac_param (str):
                parameter corresponding to the sweeped current bias

            f_01 (flaot):
                frequency of 01 transition (default: self.freq_qubit())

            analyze (bool):
                indicates whether to look for peas in the data and perform a fit

            label (bool):
                suffix to append to the measurement label

            prepare_for_continuous_wave (bool):
                indicates whether to regenerate a waveform
                generating a readout tone and set all the instruments according
                to the parameters stored in the qubit object
        """
        if f01 == None:
            f01 = self.freq_qubit()

        if prepare_for_continuous_wave:
            self.prepare_for_continuous_wave()
        if MC is None:
            MC = self.instr_MC.get_instr()

        self.hal_acq_spec_mode_on()

        p = sqo.pulsed_spec_seq(
            qubit_idx=self.cfg_qubit_nr(),
            spec_pulse_length=self.spec_pulse_length(),
            platf_cfg=self.cfg_openql_platform_fn()
        )
        self.instr_CC.get_instr().eqasm_program(p.filename)
        # CC gets started in the int_avg detector

        spec_source = self.instr_spec_source.get_instr()
        spec_source.on()
        spec_source.frequency(f01)
        # spec_source.power(self.spec_pow())

        spec_source_bus.on()
        spec_source_bus.power(bus_power)

        MC.set_sweep_function(spec_source_bus.frequency)
        MC.set_sweep_points(freqs)

        MC.set_sweep_function_2D(dac_param)
        MC.set_sweep_points_2D(dacs)

        if self.cfg_spec_mode():
            print('Enter loop')
            MC.set_detector_function(self.UHFQC_spec_det)
        else:
            self.int_avg_det_single._set_real_imag(False)  # FIXME: changes state
            MC.set_detector_function(self.int_avg_det_single)
        MC.run(name='Bus_flux_sweep_' + self.msmt_suffix + label, mode='2D')

        spec_source_bus.off()
        # FIXME: spec_source not touched
        self.hal_acq_spec_mode_off()

        if analyze:
            ma.TwoD_Analysis(label=self.msmt_suffix, close_fig=close_fig)


    def check_qubit_spectroscopy(self, freqs=None, MC=None):
        """
        Check the qubit frequency with spectroscopy of 15 points.

        Uses both the peak finder and the lorentzian fit to determine the
        outcome of the check:
        - Peak finder: if no peak is found, there is only noise. Will
                       definitely need recalibration.
        - Fitting: if a peak is found, will do normal spectroscopy fitting
                   and determine deviation from what it thinks the qubit
                   frequency is
        """
        if freqs is None:
            freq_center = self.freq_qubit()
            freq_span = 10e6
            freqs = np.linspace(freq_center - freq_span / 2,
                                freq_center + freq_span / 2,
                                15)
        self.measure_spectroscopy(MC=MC, freqs=freqs)

        label = 'spec'
        a = ma.Qubit_Spectroscopy_Analysis(
            label=label,
            close_fig=True,
            qb_name=self.name
        )

        freq_peak = a.peaks['peak']
        if freq_peak is None:
            result = 1.0
        else:
            freq = a.fitted_freq
            result = np.abs(self.freq_qubit() - freq) / self.freq_qubit()
        return result


    def check_rabi(self, MC: Optional[MeasurementControl] = None, amps=None):
        """
        Takes 5 equidistantly space points: 3 before channel amp, one at
        channel amp and one after. Compares them with the expected Rabi curve
        and returns a value in [0,1] to show the quality of the calibration
        """
        if amps is None:
            amps = np.linspace(0, 4 / 3 * self.mw_channel_amp(), 5)

        amp = self.measure_rabi(MC=MC, amps=amps, analyze=False)
        old_amp = self.mw_channel_amp()
        return np.abs(amp - old_amp)


    def check_ramsey(self, MC: Optional[MeasurementControl] = None, times=None, artificial_detuning=None):
        # USED_BY: device_dependency_graphs.py

        if artificial_detuning is None:
            artificial_detuning = 0.1e6

        if times is None:
            times = np.linspace(0, 0.5 / artificial_detuning, 6)

        a = self.measure_ramsey(times=times, MC=MC,
                                artificial_detuning=artificial_detuning)
        freq = a['frequency']
        check_result = (freq - self.freq_qubit()) / freq
        return check_result


    def create_ssro_detector(
            self,
            calibrate_optimal_weights: bool = False,
            prepare_function=None,
            prepare_function_kwargs: dict = None,
            ssro_kwargs: dict = None
    ):
        """
        Wraps measure_ssro using the Function Detector.

        Args:
            calibrate_optimal_weights
        """
        if ssro_kwargs is None:
            ssro_kwargs = {
                'nr_shots_per_case': 8192,
                'analyze': True,
                'prepare': False,
                'disable_metadata': True
            }

        if not calibrate_optimal_weights:
            d = det.Function_Detector(
                self.measure_ssro,
                msmt_kw=ssro_kwargs,
                result_keys=['SNR', 'F_d', 'F_a'],
                prepare_function=prepare_function,
                prepare_function_kwargs=prepare_function_kwargs,
                always_prepare=True)
        else:
            d = det.Function_Detector(
                self.calibrate_optimal_weights,
                msmt_kw=ssro_kwargs,
                result_keys=['SNR', 'F_d', 'F_a'],
                prepare_function=prepare_function,
                prepare_function_kwargs=prepare_function_kwargs,
                always_prepare=True)
        return d


    def calc_current_to_freq(self, curr: float):
        """
        Converts DC current to frequency in Hz for a qubit

        Args:
            curr (float):
                current in A
        """
        polycoeffs = self.fl_dc_polycoeff()

        return np.polyval(polycoeffs, curr)


    def calc_freq_to_current(self, freq, kind='root_parabola', **kw):
        """
        Find the amplitude that corresponds to a given frequency, by
        numerically inverting the fit.

        Args:
            freq (float, array):
                The frequency or set of frequencies.

            **kw : get passed on to methods that implement the different "kind"
                of calculations.
        """

        return ct.freq_to_amp_root_parabola(
            freq=freq,
            poly_coeffs=self.fl_dc_polycoeff(),
            **kw)


    def set_target_freqency(
            self, target_frequency=6e9,
            sweetspot_current=None,
            sweetspot_frequency=None,
            phi0=30e-3,
            Ec=270e6,
            span_res=30e6,
            span_q=0.5e9,
            step_q=1e6,
            step_res=0.5e6,
            I_correct=0.1e-3,
            accuracy=0.1e9,
            fine_tuning=False
    ):
        """
        Fluxing a qubit to a targeted frequency based on an estimation using the fluxarc.

        Args: target_frequency (float)
                  frequency at which you want to bias the qubit in Hz

              sweetspot_current (float)
                  current at sweetspot frequency in A
              sweetspot_frequency (float)
                  qubit frequency at sweetspot in Hz
              phi0 (float)
                  value of phi0 (length of fluxarc) in A
              Ec (float)
                  Value of Ec in Hz (estimated as 270 MHz)
        """

        # if target_frequency is None:
        #   if self.name
        if sweetspot_current is None:
            sweetspot_current = self.fl_dc_I0()
        if sweetspot_frequency is None:
            sweetspot_frequency = self.freq_max()
        I = phi0 / np.pi * np.arccos(((target_frequency + Ec) / (sweetspot_frequency + Ec)) ** 2) + sweetspot_current
        print('Baised current at target is {}'.format(I))
        fluxcurrent = self.instr_FluxCtrl.get_instr()
        fluxcurrent.set(self.fl_dc_ch(), I)
        center_res = self.freq_res()
        center_q = target_frequency
        if fine_tuning is False:
            res = self.find_resonator_frequency(freqs=np.arange(-span_res / 2, span_res / 2, step_res) + center_res,
                                                update=True)
            if res == self.freq_res():
                print(self.freq_res())
            else:
                res2 = self.find_resonator_frequency(freqs=np.arange(-span_res, span_res, step_res) + center_res,
                                                     update=True)
                if res2 == self.freq_res():
                    print(self.freqs(res))
                else:
                    raise ValueError('Resonator {} cannot be found at target frequency'.format(self.name))
            f = self.find_frequency(freqs=np.arange(-span_q / 2, span_q / 2, step_q) + center_q, update=True)
            if f:
                print('Qubit frequency at target is {}'.format(self.freq_qubit()))
            else:
                f2 = self.find_frequency(freqs=np.arange(-span_q, span_q, step_q) + center_q)
                if f2 == True:
                    print('Qubit frequency at target is {}'.format(self.freq_qubit()))
                else:
                    raise ValueError('Qubit {} cannot be found at target frequency'.format(self.name))
        else:
            while abs(self.freq_qubit() - target_frequency) > accuracy:
                if self.freq_qubit() - target_frequency > 0:
                    I = I + I_correct
                else:
                    I = I - I_correct
                print(I)
                fluxcurrent.set(self.fl_dc_ch(), I)
                self.find_resonator_frequency(freqs=np.arange(-span_res / 2, span_res / 2, step_res) + center_res)
                self.find_frequency(freqs=np.arange(-span_q / 5, span_q / 5, step_q) + center_q)
                return True

    ##########################################################################
    # HAL_ShimSQ overrides, for stuff not relating to hardware (e.g. LutMan)
    ##########################################################################

    def _prep_mw_pulses(self):
        """
        Configure MW_Lutman parameters and upload waveforms
        """
        # NB: the original code of this function was split in the part still present here, which sets pulse
        # attributes on the MW_Lutman, and the part in HAL_ShimSQ::_prep_mw_pulses, that handles hardware specific
        # functionality.
        # FIXME: this is a first step towards a real abstraction layer


        # 1. Gets instruments and prepares cases
        MW_LutMan = self.instr_LutMan_MW.get_instr()

        # 2. Prepares parameters for waveforms (except pi-pulse amp 'mw_amp180', which depends on VSM usage)
        MW_LutMan.channel_amp(self.mw_channel_amp())
        # FIXME: it looks like the semantics of channel_amp vs. mw_amp180 depend on the connected instruments, and
        #  that the calibration routines have awareness of these details. Also, many (time domain) routines don't touch
        #  mw_channel_amp, and thus depend on each other (in undocumented but probably intended ways)
        MW_LutMan.mw_amp90_scale(self.mw_amp90_scale())
        MW_LutMan.mw_gauss_width(self.mw_gauss_width())
        MW_LutMan.mw_motzoi(self.mw_motzoi())
        MW_LutMan.mw_modulation(self.mw_freq_mod())
        MW_LutMan.spec_amp(self.spec_amp())

        MW_LutMan.channel_range(self.mw_channel_range())  # FIXME: assumes AWG8_MW_LutMan

        # used for ef pulsing
        MW_LutMan.mw_ef_amp180(self.mw_ef_amp())

        if MW_LutMan.cfg_sideband_mode() != 'real-time':
            MW_LutMan.mw_ef_modulation(MW_LutMan.mw_modulation() + self.anharmonicity())
        else:
            MW_LutMan.mw_ef_modulation(self.anharmonicity())

        super()._prep_mw_pulses()
        # FIXME: hardware handling moved to HAL_ShimSQ::_prep_mw_pulses()


    def _prep_ro_pulse(self, upload=True, CW=False):
        # FIXME: move LutMan support here from HAL_ShimSQ
        super()._prep_ro_pulse(upload, CW)

    def _prep_ro_integration_weights(self):
        # FIXME: move LutMan support here from HAL_ShimSQ
        super()._prep_ro_integration_weights()