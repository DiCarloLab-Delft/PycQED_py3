import logging
import numpy as np
from copy import deepcopy
from ..waveform_control import element
from ..waveform_control import pulse
from ..waveform_control import sequence
from modules.measurement.randomized_benchmarking import randomized_benchmarking as rb
from modules.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from modules.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars
from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)
station = None
reload(element)


def GST_from_textfile(pulse_pars, RO_pars, filename,
                      upload=True, seq_name=None,
                      verbose=False):
    '''
    Input pars:
        pulse_pars:     dict containing the pulse parameters
        RO_pars:        dict containing the RO parameters
        filename:       name of a pygsti generated text file
        upload:         upload to AWG or not, if returns seq, el_list
    '''
    if seq_name is None:
        seq_name = 'GST_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # Create a dict with the parameters for all the pulses
    pulse_dict = get_pulse_dict_from_pars(pulse_pars)
    pulse_dict['RO'] = RO_pars
    pulse_combinations = create_experiment_list_pyGSTi_general(filename)
    for i, pulse_comb in enumerate(pulse_combinations):
        pulse_list = []
        for pulse_key in pulse_comb:
            pulse_list += [pulse_dict[pulse_key]]
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq, el_list


def create_experiment_list_pyGSTi(filename):
    """
    Extracting list of experiments from .txt file

    Parameters:

    filename: string
        Name of the .txt file. File must be formatted in the way as done by
        pyGSTi.
        One gatesequence per line, formatted as e.g.:Gx(Gy)^2Gx.

    Returns:

    Nested list containing all gate sequences for experiment. Every gate
    sequence is also a nested list, []

    """
    import re
    experiments = open(filename)
    sequences = experiments.read().split("\n")
    experimentlist = []
    for i in range(len(sequences)):
        clean_seq = sequences[i].strip()
        gateseq = []
        if "{}" in clean_seq:   # special case (no fiducials &no germs)
            gateseq.insert(0, "RO")
            experimentlist.append(gateseq)
        if "(" in clean_seq:
            fiducial = []
            germs = []
            measfiducial = []
            if "^" in clean_seq:
                power = int(re.findall("\d+", clean_seq)[0])
                result = re.split("[(]|\)\^\d", clean_seq)

                regsplit1 = re.findall("G[xyi]", result[0])
                for i in range(len(regsplit1)):
                    if regsplit1[i] == "Gi":
                        fiducial.append("I")
                    elif regsplit1[i] == "Gx":
                        fiducial.append("X90")
                    elif regsplit1[i] == "Gy":
                        fiducial.append("Y90")

                regsplit2 = re.findall("G[xyi]", result[1])
                for i in range(len(regsplit2)):
                    if regsplit2[i] == "Gi":
                        germs.append("I")
                    elif regsplit2[i] == "Gx":
                        germs.append("X90")
                    elif regsplit2[i] == "Gy":
                        germs.append("Y90")

                regsplit3 = re.findall("G[xyi]", result[2])
                for i in range(len(regsplit3)):
                    if regsplit3[i] == "Gi":
                        measfiducial.append("I")
                    elif regsplit3[i] == "Gx":
                        measfiducial.append("X90")
                    elif regsplit3[i] == "Gy":
                        measfiducial.append("Y90")
            else:
                power = 1
                result = re.split("[()]", clean_seq)

                regsplit1 = re.findall("G[xyi]", result[0])
                for i in range(len(regsplit1)):
                    if regsplit1[i] == "Gi":
                        fiducial.append("I")
                    elif regsplit1[i] == "Gx":
                        fiducial.append("X90")
                    elif regsplit1[i] == "Gy":
                        fiducial.append("Y90")

                regsplit2 = re.findall("G[xyi]", result[1])
                for i in range(len(regsplit2)):
                    if regsplit2[i] == "Gi":
                        germs.append("I")
                    elif regsplit2[i] == "Gx":
                        germs.append("X90")
                    elif regsplit2[i] == "Gy":
                        germs.append("Y90")

                regsplit3 = re.findall("G[xyi]", result[2])
                for i in range(len(regsplit3)):
                    if regsplit3[i] == "Gi":
                        measfiducial.append("I")
                    elif regsplit3[i] == "Gx":
                        measfiducial.append("X90")
                    elif regsplit3[i] == "Gy":
                        measfiducial.append("Y90")
            if len(fiducial) != 0:
                gateseq.append(fiducial)
            if len(germs) != 0:
                gateseq.append(germs*power)
            if len(measfiducial) != 0:
                gateseq.append(measfiducial)
            gateseq.append(["RO"])
            gateseq = list(flatten_list(gateseq))
            experimentlist.append(gateseq)
        elif ("Gi" in clean_seq) or ("Gx" in clean_seq) or ("Gy" in clean_seq):
            regsplit = re.findall("G[xyi]", clean_seq)
            loopseq = []
            for i in range(len(regsplit)):
                if regsplit[i] == "Gi":
                    loopseq.append("I")
                elif regsplit[i] == "Gx":
                    loopseq.append("X90")
                elif regsplit[i] == "Gy":
                    loopseq.append("Y90")
            gateseq.append(loopseq)
            gateseq.append(["RO"])
            gateseq = list(flatten_list(gateseq))
            experimentlist.append(gateseq)
    if len(experimentlist) < (len(sequences)-2):
        print("Lenght list of experiments too short, probably something wrong")
    experiments.close()
    return experimentlist


def flatten_list(lis):
    """
    Help function for using:create_experiment_list_pyGSTi

    """
    from collections import Iterable
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten_list(item):
                yield x
        else:
            yield item

def insert_counts_into_dataset(filename_gateseq, filename_dataoutput, counts):
    """
    Function for inserting counts retreived from experiment
    into pyGSTi. Counts are added to .text file which was used
    to create the gate sequences for the experiment.
    Formatted as pyGSTi .txt style
    Function only works in python >3

    Parameters:
    ----------------------------------------------------
    filename_gatseq: string referring to a .txt file
    must be a .txt file

    filename_dataoutput: string
    This is the name of the output file

    counts: #ofgateseq by 2 np.array, with first column plus counts, second minus counts    # or list??
    Counts

    """
    experiment_text_file_gateseq = open(filename_gateseq)
    sequences = experiment_text_file_gateseq.read().split("\n") #this thus
    #also includes the first line(title), and last whiteline
    experiment_text_file_gateseq.close()
    f = open(filename_dataoutput, "w")
    print("## Columns = plus count, minus count",file=f)
    for i in range(len(sequences)-2):
            if i == 0:
                print("{}"+"  %s" % counts[0, 0]+"  %s" % counts[0, 1], file=f)
                # gatesquence + 2 spaces + # count_plus+2 spaces+count_min
            else:
                cleanseq = sequences[i+1].strip()
                print(cleanseq+'  %s' % counts[i, 0]+'  %s' % counts[i, 1], file=f)#
                #gatesquence + 2 spaces + count_plus+2 spaces+count_min
    f.close()

def write_textfile_data_for_GST_input(filename_input, filename_output,
                                      filename_gateseq,  maxLength_gateseq,
                                      number_of_gateseq):
    """Writes data from h5py file to text file for input into pyGSTi
    --------------------------------------------------------------------------
    Parameters:
    filename_input:string
    .hdf5 file containing experimantal data. 'Data' folder must contain data
    formattted in 3 columns,first being the shot-index, second column only
    zeros, third column the counts (0 or 1).

    filename_output: string
    text file for output of data into pyGSTi

    filename_gateseq:string
    text file containg the gatesequences used in the GST experiment

    maxLength_gateseq: integer
    max length of the germ sequence (germs are raised to such a power
    so that they don't exceed the maximum length.
    That maximum length is this parameter )

    number_of_gateseq: integer
    Number of gatesequences used in experiment

    """
    import h5py
    import numpy as np
    datafileGST = h5py.File(filename_input,'r')
    expdata = datafileGST['Experimental Data/Data']
    pluscountbins = np.zeros((1,number_of_gateseq))
    for gateseqindex in range(number_of_gateseq):
        pluscountbins[0, gateseqindex] = np.sum(expdata[gateseqindex::(number_of_gateseq), 2])
    plus_counts = np.transpose(pluscountbins)
    number_of_datapoints = np.shape(expdata)[0]
    if (number_of_datapoints % number_of_gateseq) == 0:
        datapoints_per_seq = number_of_datapoints/number_of_gateseq
        total_counts = datapoints_per_seq*np.ones((number_of_gateseq))
        minus_counts = total_counts-plus_counts
        counts = (np.concatenate((plus_counts, minus_counts), axis=1)).astype(int)
        insert_counts_into_dataset(filename_gateseq, filename_output, counts)
    else:
        raise ValueError("""Number is not integer times the number of gatesequences
                         (datapoints are unevenly distributed among gatesequences""")



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
    for i in range(len(maxlength_germseq)):
        maxLengths.append((2)**i)
    results = pygsti.do_long_sequence_gst(filename_data_input, maxlength_germseq,
                                        filename_fiducials, filename_fiducials,
                                        filename_germs, maxLengths,
                                        gaugeOptRatio=1e-3)
    return results


def create_experiment_list_pyGSTi_general(filename):
    """
    Extracting list of experiments from .txt file

    Parameters:

    filename: string
        Name of the .txt file. File must be formatted in the way as done by
        pyGSTi.
        One gatesequence per line, formatted as e.g.:Gx(Gy)^2Gx.
        This function only works for the  5 primitives gateset.
        Gi, Gx90, Gy90, Gx180, Gy180. And still needs to be generalized to
        be able to handle any dataset.

    Returns:

    Nested list containing all gate sequences for experiment. Every gate
    sequence is also a nested list, []

    """
    import re
    experiments = open(filename)
    sequences = experiments.read().split("\n")
    experimentlist = []
    for i in range(len(sequences)):
        clean_seq = sequences[i].strip()
        gateseq = []
        if "{}" in clean_seq:   # special case (no fiducials &no germs)
            gateseq.insert(0, "RO")
            experimentlist.append(gateseq)
        if "(" in clean_seq:
            fiducial = []
            germs = []
            measfiducial = []
            if "^" in clean_seq:
                powerstring_with_exponent=re.findall('\^\d+',clean_seq)
                powerstring=re.findall('\d+',powerstring_with_exponent[0])
                power=int(powerstring[0])
                result = re.split("[(]|\)\^\d", clean_seq)

                regsplit1 = re.findall('G[xy]180|G[xy]90|Gi', result[0])
                for i in range(len(regsplit1)):
                    if regsplit1[i]=="Gi":
                        fiducial.append("I")
                    elif regsplit1[i]=="Gx90":
                        fiducial.append("X90")
                    elif regsplit1[i]=="Gy90":
                        fiducial.append("Y90")
                    elif regsplit1[i]=='Gx180':
                        fiducial.append('X180')
                    elif regsplit1[i]==('Gy180'):
                        fiducial.append('Y180')

                regsplit2 = re.findall('G[xy]180|G[xy]90|Gi',result[1])
                for i in range(len(regsplit2)):
                    if regsplit2[i]=="Gi":
                        germs.append("I")
                    elif regsplit2[i]=="Gx90":
                        germs.append("X90")
                    elif regsplit2[i]=="Gy90":
                        germs.append("Y90")
                    elif regsplit2[i]=='Gx180':
                        germs.append('X180')
                    elif regsplit2[i]==('Gy180'):
                        germs.append('Y180')

                regsplit3=re.findall('G[xy]180|G[xy]90|Gi', result[2])
                for i in range(len(regsplit3)):
                    if regsplit3[i]=="Gi":
                        measfiducial.append("I")
                    elif regsplit3[i]=="Gx90":
                        measfiducial.append("X90")
                    elif regsplit3[i]=="Gy90":
                        measfiducial.append("Y90")
                    elif regsplit3[i]=='Gx180':
                        measfiducial.append('X180')
                    elif regsplit3[i]==('Gy180'):
                        measfiducial.append('Y180')
            else:
                power = 1
                result = re.split("[()]",clean_seq)

                regsplit1 = re.findall('G[xy]180|G[xy]90|Gi',result[0])
                for i in range(len(regsplit1)):
                    if regsplit1[i]=="Gi":
                        fiducial.append("I")
                    elif regsplit1[i]=="Gx90":
                        fiducial.append("X90")
                    elif regsplit1[i]=="Gy90":
                        fiducial.append("Y90")
                    elif regsplit1[i]=='Gx180':
                        fiducial.append('X180')
                    elif regsplit1[i]==('Gy180'):
                        fiducial.append('Y180')


                regsplit2 = re.findall('G[xy]180|G[xy]90|Gi',result[1])
                for i in range(len(regsplit2)):
                    if regsplit2[i]=="Gi":
                        germs.append("I")
                    elif regsplit2[i]=="Gx90":
                        germs.append("X90")
                    elif regsplit2[i]=="Gy90":
                        germs.append("Y90")
                    elif regsplit2[i]=='Gx180':
                        germs.append('X180')
                    elif regsplit2[i]==('Gy180'):
                        germs.append('Y180')

                regsplit3 = re.findall('G[xy]180|G[xy]90|Gi',result[2])
                for i in range(len(regsplit3)):
                    if regsplit3[i]=="Gi":
                        measfiducial.append("I")
                    elif regsplit3[i]=="Gx90":
                        measfiducial.append("X90")
                    elif regsplit3[i]=="Gy90":
                        measfiducial.append("Y90")
                    elif regsplit3[i]=='Gx180':
                        measfiducial.append('X180')
                    elif regsplit3[i]==('Gy180'):
                        measfiducial.append('Y180')
            if len(fiducial)!=0:
                gateseq.append(fiducial)
            if len(germs)!=0:
                gateseq.append(germs*power)
            if len(measfiducial)!=0:
                gateseq.append(measfiducial)
            gateseq.append(["RO"])
            gateseq=list(flatten_list(gateseq))
            experimentlist.append(gateseq)
        elif ("Gi" in clean_seq) or ("Gx" in clean_seq) or ("Gy" in clean_seq):
            regsplit =re.findall('G[xy]180|G[xy]90|Gi',clean_seq)
            loopseq = []
            for i in range(len(regsplit)):
                if regsplit[i]=="Gi":
                    loopseq.append("I")
                elif regsplit[i]=="Gx90":
                    loopseq.append("X90")
                elif regsplit[i]=="Gy90":
                    loopseq.append("Y90")
                elif regsplit[i]=='Gx180':
                    loopseq.append('X180')
                elif regsplit[i]==('Gy180'):
                    loopseq.append('Y180')
            gateseq.append(loopseq)
            gateseq.append(["RO"])
            gateseq=list(flatten_list(gateseq))
            experimentlist.append(gateseq)
    if len(experimentlist)<(len(sequences)-2):
        print("Lenght list of experiments too short, probably something wrong")
    experiments.close()
    return experimentlist
def flatten_list(lis):
    """
    Help function for using:create_experiment_list_pyGSTi

    """
    from collections import Iterable
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten_list(item):
                yield x
        else:
            yield item


