from __future__ import print_function

import h5py
import numpy as np
from ..waveform_control import element
from ..waveform_control import pulse
from ..waveform_control import sequence
# from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars
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
    station.pulsar.update_channel_settings()
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


def write_textfile_data_for_GST_input(filename_input, filename_output,
                                      filename_gateseq, number_of_gateseq,
                                      zero_one_inverted=False):
    """
    Writes data from h5py file to text file for input into pyGSTi

    Parameters:
    ----------------------------------------------------------------------
    filename_input: string
    .hdf5 file containing experimantal data. 'Data' folder must contain
    data formattted in 3 columns,first being the shot-index, second
    column only zeros, third column the counts (0 or 1).

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

    zero_one_inverted: Boolean (optional), or string:'automatic'
    Due to way which experimental data is converted to counts: by
    determining a threshold,however it is not know which side of the
    threshold is 1 and which side is 0, therefore it could be that
    0 and 1 are inverted. This needs to be checked, for example by
    looking at what identity gate does. If inverted, specifiy
    zero_one_inverted=True, but often the better thing is to put it on 'automatic'
    otherwise leave at default (False).

    Returns
    ------------------------------------------------------------
    Next to writing a .txt file containing the GST data in the standard
    pyGSTi format (each line contains a gatesequences, followed by plus
    and minus counts respectively), it returns a list containing the counts,
    with the first column containing plus counts, and second minus, for easy
    checking.

    """
    datafileGST = h5py.File(filename_input,'r')
    expdata = datafileGST['Experimental Data/Data']
    pluscountbins = np.zeros((1,number_of_gateseq))
    number_of_datapoints = np.shape(expdata)[0]
    datapoints_per_seq = number_of_datapoints/number_of_gateseq
    shapedarr = reshape_array_h5pyfile(expdata[:,2],number_of_gateseq,datapoints_per_seq)
    for gateseqindex in range(number_of_gateseq):
        pluscountbins[0,gateseqindex] = np.count_nonzero(shapedarr[gateseqindex,:])
    plus_counts = np.transpose(pluscountbins)
    plus_counts = np.reshape(plus_counts,(number_of_gateseq,1))
    if zero_one_inverted == 'automatic':
        if plus_counts[0] >= (datapoints_per_seq/2):
            zero_one_inverted = True
        elif plus_counts[0] < (datapoints_per_seq/2):
            zero_one_inverted = False
    print(np.shape(plus_counts))

    if (number_of_datapoints%number_of_gateseq) == 0:
        total_counts = datapoints_per_seq*np.ones((number_of_gateseq,1))
        minus_counts = total_counts-plus_counts
        minus_counts = np.reshape(minus_counts,(number_of_gateseq,1))
        print(np.shape(minus_counts))
        counts =(np.concatenate((plus_counts,minus_counts),axis =1)).astype(int)
        if zero_one_inverted == False:
            insert_counts_into_dataset(filename_gateseq,filename_output,counts)
            if plus_counts[0] >= (datapoints_per_seq/2):
                import warnings
                warnings.warn("""Plus counts for the first gatesequence
                (which is normally, initialize and readout, and #pluscounts
                should be approx zero) is more than half the total of counts
                of the first gatesequence, probably the 0 and 1 threshold are
                reversed,check whether this is the case and possibly switch
                to the setting: zero_one_inverted=True or zero_one_inverted='automatic' """)
        elif zero_one_inverted == True :
            counts = np.fliplr(counts)
            insert_counts_into_dataset(filename_gateseq,filename_output,counts)
    else:
        raise ValueError("""Number is not integer times the number of gatesequences
                         (datapoints are unevenly distributed among gatesequences""")
    return counts


def reshape_array_h5pyfile(array,number_of_gatesequences,datapoints_per_seq):
    """reshaping function"""
    new_array = np.reshape(array,(number_of_gatesequences,datapoints_per_seq),order='F') #order is important, for data as column,
    #use order F, this will give an number_of_gatesequences x datapoints_per_seq matrix
    return new_array


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

def write_experiment_runs_to_text_files(starttime,endtime,maxlength,filename_gateseq,number_of_gateseq):

    """Function for writing experiment runs to text files. At the moment only
    works for an experiment which is repeated for same parameters several
    times in time. And filenames much be formatted as usal within PycQED-3
    --------------------------------------------------------------------------------------------
    Parameters:

    starttime:string
    Formatted in the standard form as used in PycQED_py3, in the form of e.g.
    20160614_000000.

    endtime:string
    Formatted in the standard form as used in PycQED_py3, in the form of e.g.
    20160615_235959

    maxlength: integer
    Maximum length of germ power sequence

    filename_gateseq: string
    Text file containing the list of experiments/gatesequences

    number_of_gateseq:
    the number of gatesequences in each experiment

    """
    from pycqed.analysis import measurement_analysis as ma
    from pycqed.analysis import analysis_toolbox as a_tools
    import measurement.pulse_sequences.gate_set_tomography as _gst
    from importlib import reload
    import h5py
    reload(_gst)
    experiment_timestamp_strings = a_tools.get_timestamps_in_range(starttime, endtime, label='GST')
    file_path_names = []

    for i in range(len(experiment_timestamp_strings)):
        file_path_names.append(a_tools.data_from_time(experiment_timestamp_strings[i]))

    for i in range(len(file_path_names)):
        fn = file_path_names[i]
        suffix = fn[len(a_tools.datadir)+10:]
        filename_input = file_path_names[i]+'\\'+suffix+'.hdf5'
        filename_output = 'GST_data%s_run%s' % (maxlength, suffix, i) +'.txt'
        _gst.write_textfile_data_for_GST_input_adapt_develop(filename_input,
                                      filename_output, filename_gateseq,
                                      number_of_gateseq, zero_one_inverted='automatic')


def write_experiment_runs_to_text_files_conv_vs_restless(starttime,endtime,maxlength,filename_gateseq,number_of_gateseq):

    """
    Function for writing experiment runs to text files. At the moment only
    works for an experiment which is repeated for same parameters several times in time. It also assumes that one starts with conventional tuning, then restless etc. It's specialised for only experiment and is more of a One-fit.
    ----------------------------------------------------------------------------------------
    Parameters:

    starttime:string
    Formatted in the standard form as used in PycQED_py3, in the form of e.g. 20160614_000000.

    endtime:string
    Formatted in the standard form as used in PycQED_py3, in the form of e.g.20160615_235959

    maxlength: integer
    Maximum length of germ power sequence

    filename_gateseq: string
    Text file containing the list of experiments/gatesequences

    number_of_gateseq:
    the number of gatesequences in each experiment

    """
    from pycqed.analysis import measurement_analysis as ma
    from pycqed.analysis import analysis_toolbox as a_tools
    import measurement.pulse_sequences.gate_set_tomography as _gst
    from importlib import reload
    import h5py
    reload(_gst)
    experiment_timestamp_strings = a_tools.get_timestamps_in_range(starttime, endtime, label = 'GST')
    file_path_names = []

    for i in range(len(experiment_timestamp_strings)):
        file_path_names.append(a_tools.data_from_time(experiment_timestamp_strings[i]))
    file_path_names_conventional = file_path_names[0::2]
    file_path_names_restless = file_path_names[1::2]

    for i in range(len(file_path_names_conventional)):
        fn = file_path_names_conventional[i]
        suffix = fn[len(a_tools.datadir)+10:]
        filename_input = file_path_names_conventional[i]+'\\'+suffix+'.hdf5'
        filename_output = 'GST_data_paper_L%s_conv_%s_run%s_17june' %(maxlength,suffix, i) +'.txt'
        _gst.write_textfile_data_for_GST_input_adapt_develop(filename_input,
                                       filename_output,filename_gateseq,
                                      number_of_gateseq,zero_one_inverted = 'automatic')

    for i in range(len(file_path_names_restless)):
        fn = file_path_names_restless[i]
        suffix = fn[len(a_tools.datadir)+10:]
        filename_input = file_path_names_restless[i]+'\\'+suffix+'.hdf5'
        filename_output = 'GST_data_paper_L%s_rest_%s_run%s_17june' %(maxlength,suffix, i) +'.txt'
        _gst.write_textfile_data_for_GST_input_adapt_develop(filename_input,
                                       filename_output,filename_gateseq,
                                      number_of_gateseq,zero_one_inverted = 'automatic')
