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
                      upload=True,
                      verbose=False):
    '''
    Input pars:
        pulse_pars:     dict containing the pulse parameters
        RO_pars:        dict containing the RO parameters
        filename:       name of a pygsti generated text file
        upload:         upload to AWG or not, if returns seq, el_list
    '''
    seq_name = 'GST_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # Create a dict with the parameters for all the pulses
    pulse_dict = get_pulse_dict_from_pars(pulse_pars)
    pulse_dict['RO'] = RO_pars
    pulse_combinations = create_experiment_list_pyGSTi(filename)
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
