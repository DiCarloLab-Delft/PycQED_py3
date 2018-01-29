"""
This file reads in a pygsti dataset file and converts it to a valid
OpenQL sequence.
"""


def pygsti_gateset_from_dataset(filename:str):
    with open(filename, 'r') as f:
        rawLines = f.readlines()
    # lines[1:]

    lines = []
    for line in rawLines:
        lines.append(line.split(' ')[0])
    gateStrList = [
        pygsti.objects.GateString(None, stringRepresentation=line)
        for line in lines[1:]]

    return gateStrList


def