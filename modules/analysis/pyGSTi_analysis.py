import numpy as np


def perform_extended_GST_on_data(filename_data_input, filename_target_gateset,
                                 filename_fiducials, filename_germs,
                                 maxlength_germseq):
    """Performs extended/long GST on data
    --------------------------------------------------------------------------
    Parameters:

    filename_data_input: string
    Filename of the .txt containing the listofexperiments with the counts as formatted
    accordign to the standard pyGSTi way,thus at every line a gatesequence, followed
    by the plus and minus count (|1> and |0>)

    filename_target_gateset: string
    Filename of the .txt file which contains the target gateset

    filename_fiducials: string
    Filename of the .txt file containing the fiducials

    filename_germs: string
    Filename of .txt file containing the germs

    maxlength_germpowerseq: integer
    Integer specifying the maximum length of the of the germ-power sequence
    used in the gates sequence. The germs are repeated an integer number of
    times, such that the total length of this repetition is smaller than a
    specified maxlength. The list of experiments is composed of all the
    gatesequences for a specified maxlength which is increasing. The maximum
    of this maxlength is the parameter ' maxlength_germpowerseq'. #This is
    still quite unclear/needs better clarification


    --------------------------------------------------------------------------
    Returns:

    pyGSTi .Results object, on which you can apply several functions to create
    reports and get the estimates such as:
    results.gatesets['final estimate'] --> gives final estimate of the gateset
    results.create_full_report_pdf,   --> creates full report
    results.create_brief_report_pdf   ---> creates brief report
    """
    import pygsti
    maxLengths = [0]
    for i in range(int(np.log(maxlength_germseq)/np.log(2))+1):
        maxLengths.append((2)**i)
    results = pygsti.do_long_sequence_gst(
        filename_data_input, filename_target_gateset,
        filename_fiducials, filename_fiducials,
        filename_germs, maxLengths,
        gaugeOptRatio=1e-3)
    return results


def extract_RB_fidelity_from_GST_result(
        result, clifford_gate_decomposition,
        gateset_5_primitives_as_9_gateset=True,
        dimension=2, weigthed_average=True):
    """Function which extracts the RB fidelity per gate and RB fidelity per
    clifford out of GST result. The weighted average is taken of the
    processinfidelities (where the weights are determined by the relative
    frequencies of gates within the clifford decomposition), after which the
    average processinfidelity is related to the RB fidelity according to
    equation 3 from the paper by Nielsen: 'A simple formula for the average
    gate fidelity of a quantum dynamical operation'.Then the RB fidelity per
    clifford and the RB fidelity per gate are returned.


    Parameters
    ----------------
    result: pygsti.report.results.Results object
    Result from standard GST analysis.

    clifford_gate_decomposition: nested list
    List containing the decomposition of the clifford in elementary gates.
    For comparing with RB results, use the same decomposition as was used in RB.
    Which is at the moment: Gate decomposition decomposition of the clifford
    group as per Eptstein et al. Phys. Rev. A 89, 062321 (2014).
    (called: gate_decomposition, in:
    pycQED-3.modules.measurement.randomized_benchmarking.clifford_decompositions)

    gateset_5_primitives_as_9_gateset: optional, default = True,
    In pycQED-3 RB, the clifford gates are decomposed in 9 elementary gates,
    I, and rotation about X, Y of Pi or Pi/2. The targetset of GST can be less
    than those 9 gates, where then is assumed that X90=mX90, Y90=mY90, etc.
    In that case, leave gateset_5_primitives_as_9_gateset at True.
    Otherwise if GST's target set was the 9 elementary gates, put it at False.

    dimension: optional, default=2
    Dimension of the Hilbert-space. In case of single qubit offcourse =2.

    weighted_average: optional, default =True
    Specify whether when averaging over the process infidelities, want to take
    into account the frequencies with which the elementary gates occur in the
    clifford gate decomposition.

    Returns
    -------------
    Randomized benchmarking fidelity per gate AND Randomzied benchmarking
    fidelity per clifford
    """
    GST_gateset = result.gatesets['target']
    gatesinfidelities = extract_processinfidelities_from_GST_result(result)
    if gateset_5_primitives_as_9_gateset == False:
        pass
    elif gateset_5_primitives_as_9_gateset == True:
        gatesinfidelities_ext9 = list(gatesinfidelities)
        gatesinfidelities_ext9 = gatesinfidelities_ext9+list(gatesinfidelities[1::])
        gatesinfidelities = gatesinfidelities_ext9
    weights = make_weights_of_gates_Clifford_decomp(clifford_gate_decomposition)
    average_process_infidelity_for_RB_calc = np.average(
        gatesinfidelities, weights=weights)
    RB_fidelity_per_gate = (
        dimension*(1-average_process_infidelity_for_RB_calc)+1)/(dimension+1)
    RB_fidelity_per_clifford = RB_fidelity_per_gate**1.875
    return RB_fidelity_per_gate, RB_fidelity_per_clifford


def extract_processinfidelities_from_GST_result(result):
    """Function which extracts the process infidelities from the GST results
    objects.

    Parameters
    -------------------------------------------------------------------------
    results: pygsti.report.results.Results object

    Returns
    --------
    numpy array containing process infidelities
    """

    GST_gateset = result.gatesets['target']
    infidelitytable = result.tables['bestGatesetVsTargetTable']
    gatelabels = GST_gateset.gates.keys()
    gatesinfidelities = np.ones((len(gatelabels),1))
    for i in range(len(gatelabels)):
        gatesinfidelities[i] = infidelitytable[gatelabels[i]]['Process Infidelity']['value']
    return gatesinfidelities


def make_weights_of_gates_Clifford_decomp(
        clifford_gate_decomposition,
        gates=['I', 'X90', 'X180', 'Y90', 'Y180',
               'mX90', 'mX180', 'mY90', 'mY180']):
    """Calculates frequencies of gates in ['I','X90','X180','Y90','Y180',
    'mX90','mX180','mY90','mY180'],
    in the specified Clifford decomposition

    Parameters:
    ----------------------------
    clifford_gate_decomposition: nested list
    List containing lists with the clifford decomposition.

    gates: list containing gatelabels

    Returns
    ---------------------------
    list of weights of frequencies of elementary gates in clifford gates
    """

    total_count = 0
    individual_gate_counts = np.zeros((len(gates),1))
    for i, gate in enumerate(gates):
        for j in range(len(clifford_gate_decomposition)):
            for k in range(len(clifford_gate_decomposition[j])):
                if clifford_gate_decomposition[j][k] == gate:
                    total_count += 1
                    individual_gate_counts[i] += 1
    weights = individual_gate_counts/total_count
    return weights


def extract_gates_T1_from_result_GST(result, gate_operation_time):
    """Calculates the T1 from the extracted diagonal decay out of the result of GST analysis.
    Which is given as a number e.g. 0.001, from which T1 can be calculated.
    Which is done by: (1-decayrate)**N=e**(N*ln(1-decayrate)), where N is the
    number of gates applied. And: e**(N*ln(1-decayrate))=e**(-t/T1).
    This gives: T1=-time_per_gate/ln(1-decayrate)

    Parameters
    --------------------------------
    result: pygsti.report.results.Results object
    Result from standard GST analysis.

    gate_operation_time: float
    The time each gate operation (pulse) takes in seconds.

    Returns
    ---------------------------------
    T1 in seconds"""
    decomptable = result.tables['bestGatesetDecompTable']
    Diagonal_decay_rates = []
    gatelabels = result.gatesets['target'].gates.keys()
    for i in range(len(gatelabels)):
        Diagonal_decay_rates.append(decomptable[gatelabels[i]]['Diag. decay']['value'])
    T1_gates = [-gate_operation_time/np.log((1-i)) for i in Diagonal_decay_rates]
    return T1_gates


def extract_gates_T2_from_result_GST(result, gate_operation_time):
    """Calculates the T2 from the extracted diagonal decay out of the result of GST analysis.
    Which is given as a number e.g. 0.001, from which T2 can be calculated.
    Which is done by: (1-decayrate)**N=e**(N*ln(1-decayrate)), where N is the
    number of gates applied. And: e**(N*ln(1-decayrate))=e**(-t/T2).
    This gives: T2=-time_per_gate/ln(1-decayrate)

    Parameters
    --------------------------------
    result: pygsti.report.results.Results object
    Result from standard GST analysis.

    gate_operation_time: float
    The time each gate operation (pulse) takes in seconds.

    Returns
    ---------------------------------
    T2 in seconds"""
    decomptable = result.tables['bestGatesetDecompTable']
    Off_diagonal_decay_rates = []
    gatelabels = result.gatesets['target'].gates.keys()
    for i in range(len(gatelabels)):
        Off_diagonal_decay_rates.append(decomptable[gatelabels[i]]['Off-diag. decay']['value'])
    T2_gates = [-gate_operation_time/np.log((1-i))for i in Off_diagonal_decay_rates]
    return T2_gates


def extract_average_T1_from_result_GST(result, gate_operation_time):
    """Calculates the T1 from the extracted diagonal decay out of the result of GST analysis.
    Which is given as a number e.g. 0.001, from which T1 can be calculated.
    Which is done by: (1-decayrate)**N=e**(N*ln(1-decayrate)), where N is the
    number of gates applied. And: e**(N*ln(1-decayrate))=e**(-t/T1).
    This gives: T1=-time_per_gate/ln(1-decayrate).From there the average is
    calculated.

    Parameters
    --------------------------------
    result: pygsti.report.results.Results object
    Result from standard GST analysis.

    gate_operation_time: float
    The time each gate operation (pulse) takes in seconds.

    Returns
    ---------------------------------
    Average T1 of the gateset in seconds"""
    T1_gates = extract_gates_T1_from_result_GST(result, gate_operation_time)
    T1_average = np.average(T1_gates)
    return T1_average


def extract_average_T2_from_result_GST(result, gate_operation_time):
    """Calculates the T2 from the extracted diagonal decay out of the result of GST analysis.
    Which is given as a number e.g. 0.001, from which T2 can be calculated.
    Which is done by: (1-decayrate)**N=e**(N*ln(1-decayrate)), where N is the
    number of gates applied. And: e**(N*ln(1-decayrate))=e**(-t/T2).
    This gives: T2=-time_per_gate/ln(1-decayrate). From there the average is
     calculated.

    Parameters
    --------------------------------
    result: pygsti.report.results.Results object
    Result from standard GST analysis.

    gate_operation_time: float
    The time each gate operation (pulse) takes in seconds.

    Returns
    ---------------------------------
    Average T2 of the gateset in seconds"""
    T2_gates = extract_gates_T2_from_result_GST(result, gate_operation_time)
    T2_average = np.average(T2_gates)
    return T2_average


def perform_GST_analysis_on_multiple_experiment_runs(
        number_of_runs_must_be_even, filename_target_gateset,
        filename_fiducials, filename_germs, maxlength_germseq):
    """For now only applicable for the specific case of an experiment consisting of multiple runs,
    alternating between conventional
    and restless tuned pulses, starting of with conventional tuning.
    At the moment even only works for text files named in specific format of:
    'GST_data_paper_L%s_conventional_run%s'%(maxlength, i) +'.txt'
    --------------------------------------------------------------------------------------------------------
    Parameters:

    number_of_runs_must_be_even: even integer,
    Even integer specificying how much runs are performed of each the conventional and restless runs were performed.


    """
    results_list_conventional = []
    results_list_restless = []
    for i in range(number_of_runs_must_be_even):
        filename_data_input = 'GST_data_paper_L%s_conventional_run%s'%(
            maxlength_germseq, i) +'.txt'
        results = perform_extended_GST_on_data(
            filename_data_input, filename_target_gateset,
            filename_fiducials, filename_germs, maxlength_germseq)
        results_list_conventional.append(results)

        filename_data_input = ('GST_data_paper_L%s_restless_run%s'%(
            maxlength_germseq, i) + '.txt')
        results = perform_extended_GST_on_data(
            filename_data_input, filename_target_gateset,
            filename_fiducials, filename_germs,
            maxlength_germseq)
        results_list_restless.append(results)
    return results_list_conventional, results_list_restless
