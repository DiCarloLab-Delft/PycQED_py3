import numpy as np
from pycqed.analysis import measurement_analysis as ma
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import detector_functions as det

MC_instr = None
VNA_instr = None


def acquire_single_linear_frequency_span(file_name, start_freq=None,
                                         stop_freq=None, center_freq=None,
                                         span=None, nr_avg=1, sweep_mode='auto',
                                         nbr_points=101, power=-20,
                                         bandwidth=100, measure='S21'):
    """
    Acquires a single trace from the VNA.
    Inputs:
            file_name (str), name of the output file.
            start_freq (float), starting frequency of the trace.
            stop_freq (float), stoping frequency of the trace.
            center_freq (float), central frequency of the trace.
            span (float), span of the trace.
            nbr_points (int), Number of points within the trace.
            power (float), power in dBm.
            bandwidth (float), bandwidth in Hz.
            measure (str), scattering parameter to measure (ie. 'S21').
    Output:
            NONE

    Beware that start/stop and center/span are just two ways of configuring the
    same thing.
    """
    # set VNA sweep function
    if start_freq != None and stop_freq != None:
        MC_instr.set_sweep_function(swf.ZNB_VNA_sweep(VNA_instr,
                                                      start_freq=start_freq,
                                                      stop_freq=stop_freq,
                                                      npts=nbr_points,
                                                      force_reset=True))
    elif center_freq != None and span != None:
        MC_instr.set_sweep_function(swf.ZNB_VNA_sweep(VNA_instr,
                                                      center_freq=center_freq,
                                                      span=span,
                                                      npts=nbr_points,
                                                      force_reset=True))

    # set VNA detector function
    MC_instr.set_detector_function(det.ZNB_VNA_detector(VNA_instr))

    # VNA settings
    # VNA_instr.average_state('off')
    VNA_instr.bandwidth(bandwidth)

    # hack to measure S parameters different from S21
    str_to_write = "calc:par:meas 'trc1', '%s'" % measure
    print(str_to_write)
    VNA_instr.visa_handle.write(str_to_write)

    VNA_instr.avg(nr_avg)
    VNA_instr.number_sweeps_all(nr_avg)
    VNA_instr.average_mode(sweep_mode)

    VNA_instr.power(power)
    VNA_instr.timeout(600)

    MC_instr.run(name=file_name)
    # ma.Homodyne_Analysis(auto=True, label=file_name, fitting_model='hanger')
    # ma.VNA_Analysis(auto=True, label=file_name)


def acquire_current_trace(file_name):
    """
    Acquires the trace currently displayed on VNA.
    Inputs:
            file_name (str), name of the output file.
    Output:
            NONE
    """
    # get values from VNA
    start_freq = VNA_instr.start_frequency()
    stop_freq = VNA_instr.stop_frequency()
    nbr_points = VNA_instr.npts()
    power = VNA_instr.power()
    bandwidth = VNA_instr.bandwidth()

    current_sweep_time = VNA_instr.sweep_time()
    print(current_sweep_time)

    acquire_single_linear_frequency_span(file_name, start_freq=start_freq,
                                         stop_freq=stop_freq,
                                         nbr_points=nbr_points,
                                         power=power, bandwidth=bandwidth)


def acquire_linear_frequency_span_vs_power(file_name, start_freq=None,
                                     stop_freq=None, center_freq=None,
                                     start_power=None, stop_power=None,
                                     step_power=2,
                                     span=None, nbr_points=101,
                                     bandwidth=100, measure='S21'):
    """
    Acquires a single trace from the VNA.
    Inputs:
            file_name (str), name of the output file.
            start_freq (float), starting frequency of the trace.
            stop_freq (float), stoping frequency of the trace.
            center_freq (float), central frequency of the trace.
            span (float), span of the trace.
            nbr_points (int), Number of points within the trace.
            start_power, stop_power, step_power (float), power range in dBm.
            bandwidth (float), bandwidth in Hz.
            measure (str), scattering parameter to measure (ie. 'S21').
    Output:
            NONE

    Beware that start/stop and center/span are just two ways of configuring the
    same thing.
    """
    # set VNA sweep function
    if start_freq != None and stop_freq != None:
        swf_fct_1D = swf.ZNB_VNA_sweep(VNA_instr,
                                       start_freq=start_freq,
                                       stop_freq=stop_freq,
                                       npts=nbr_points,
                                       force_reset=True)
        MC_instr.set_sweep_function(swf_fct_1D)
    elif center_freq != None and span != None:
        swf_fct_1D = swf.ZNB_VNA_sweep(VNA_instr,
                                       center_freq=center_freq,
                                       span=span,
                                       npts=nbr_points,
                                       force_reset=True)
        MC_instr.set_sweep_function(swf_fct_1D)

    if start_power != None and stop_power != None:
        # it prepares the sweep_points, such that it does not complain.
        swf_fct_1D.prepare()
        MC_instr.set_sweep_points(swf_fct_1D.sweep_points)
        MC_instr.set_sweep_function_2D(VNA_instr.power)
        MC_instr.set_sweep_points_2D(np.arange(start_power,
                                               stop_power+step_power/2.,
                                               step_power))
    else:
        raise ValueError('Need to define power range.')

    # set VNA detector function
    MC_instr.set_detector_function(det.ZNB_VNA_detector(VNA_instr))

    # VNA settings
    VNA_instr.average_state('off')
    VNA_instr.bandwidth(bandwidth)

    # hack to measure S parameters different from S21
    str_to_write = "calc:par:meas 'trc1', '%s'" % measure
    print(str_to_write)
    VNA_instr.visa_handle.write(str_to_write)
    VNA_instr.timeout(600)

    MC_instr.run(name=file_name, mode='2D')
#     ma.Homodyne_Analysis(auto=True, label=file_name, fitting_model='hanger')
    ma.TwoD_Analysis(auto=True, label=file_name)

def acquire_2D_linear_frequency_span_vs_param(file_name, start_freq=None,
                                     stop_freq=None, center_freq=None,
                                     parameter=None, sweep_vector=None,
                                     span=None, nbr_points=101, power=-20,
                                     bandwidth=100, measure='S21'):
    """
    Acquires a single trace from the VNA.
    Inputs:
            file_name (str), name of the output file.
            start_freq (float), starting frequency of the trace.
            stop_freq (float), stoping frequency of the trace.
            center_freq (float), central frequency of the trace.
            span (float), span of the trace.
            nbr_points (int), Number of points within the trace.
            power (float), power in dBm.
            bandwidth (float), bandwidth in Hz.
            measure (str), scattering parameter to measure (ie. 'S21').
    Output:
            NONE

    Beware that start/stop and center/span are just two ways of configuring the
    same thing.
    """
    # set VNA sweep function
    if start_freq != None and stop_freq != None:
        swf_fct_1D = swf.ZNB_VNA_sweep(VNA_instr,
                                       start_freq=start_freq,
                                       stop_freq=stop_freq,
                                       npts=nbr_points,
                                       force_reset=True)
        MC_instr.set_sweep_function(swf_fct_1D)
    elif center_freq != None and span != None:
        swf_fct_1D = swf.ZNB_VNA_sweep(VNA_instr,
                                       center_freq=center_freq,
                                       span=span,
                                       npts=nbr_points,
                                       force_reset=True)
        MC_instr.set_sweep_function(swf_fct_1D)

    if parameter is not None and sweep_vector is not None:
        # it prepares the sweep_points, such that it does not complain.
        swf_fct_1D.prepare()
        MC_instr.set_sweep_points(swf_fct_1D.sweep_points)
        MC_instr.set_sweep_function_2D(parameter)
        MC_instr.set_sweep_points_2D(sweep_vector)
    else:
        raise ValueError('Need to define parameter and its range.')

    # set VNA detector function
    MC_instr.set_detector_function(det.ZNB_VNA_detector(VNA_instr))

    # VNA settings
    VNA_instr.power(power)
    VNA_instr.average_state('off')
    VNA_instr.bandwidth(bandwidth)

    # hack to measure S parameters different from S21
    str_to_write = "calc:par:meas 'trc1', '%s'" % measure
    print(str_to_write)
    VNA_instr.visa_handle.write(str_to_write)
    VNA_instr.timeout(600)

    MC_instr.run(name=file_name, mode='2D')
#     ma.Homodyne_Analysis(auto=True, label=file_name, fitting_model='hanger')
    ma.TwoD_Analysis(auto=True, label=file_name)


def acquire_disjoint_frequency_traces(file_name, list_freq_ranges,
                                      power=-20,
                                      bandwidth=100, measure='S21'):
    """
    list_freq_ranges is a list that contains the arays defining all the ranges of interest.
    """
    for idx, range_array in enumerate(list_freq_ranges):
        this_file = file_name + '_%d'%idx
        acquire_single_linear_frequency_span(file_name = this_file,
                                             start_freq=range_array[0],
                                             stop_freq=range_array[-1],
                                             nbr_points=len(range_array),
                                             power=power,
                                             bandwidth=bandwidth,
                                             measure=measure)
    packing_mmts(file_name, labels=file_name+'__', N=len(list_freq_ranges))

def packing_mmts(file_name, labels, N):
    # imports data
    # stacks it toghether
    for idx in range(N):
        this_file = file_name + '_%d'%idx
        ma_obj = ma.MeasurementAnalysis(label=this_file, auto=False)
        ma_obj.get_naming_and_values()
        if idx==0:
            data_points = ma_obj.sweep_points
            data_vector = ma_obj.measured_values
        else:
            data_points = np.hstack((data_points,ma_obj.sweep_points))
            data_vector = np.hstack((data_vector,ma_obj.measured_values))
        del ma_obj
    data_vector = np.array(data_vector)
    data_points = np.array(data_points)
    sort_mask = np.argsort(data_points)
    data_points = data_points[sort_mask]
    data_vector = data_vector[:,sort_mask]

    # keeps track of sweeppoints call
    class call_counter:
        def __init__(self):
            self.counter=0
    swp_points_counter = call_counter()

    def wrapped_data_importing(counter):
        counter.counter += 1
        return data_vector[:,counter.counter-1]

    detector_inst = det.Function_Detector_list(
        sweep_function=wrapped_data_importing,
        result_keys=['Amp', 'phase', 'I', 'Q'],
        msmt_kw={'counter': swp_points_counter})
    MC_instr.set_sweep_function(swf.None_Sweep(name='VNA sweep'))
    MC_instr.set_sweep_points(data_points.flatten())
    MC_instr.set_detector_function(detector_inst)
    MC_instr.run(name=file_name)
