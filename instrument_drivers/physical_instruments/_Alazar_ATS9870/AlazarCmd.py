import numpy
from ctypes import Structure
from ctypes import *


class ATS9870_BoardDef(Structure):
    _fields_ = [
            ('RecordCount', c_ulong),
            ('RecLength', c_ulong),
            ('PreDepth', c_ulong),
            ('ClockSource', c_ulong),
            ('ClockEdge', c_ulong),
            ('SampleRate', c_ulong),
            ('CouplingChanA', c_ulong),
            ('InputRangeChanA', c_ulong),
            ('InputImpedChanA', c_ulong),
            ('CouplingChanB', c_ulong),
            ('InputRangeChanB', c_ulong),
            ('InputImpedChanB', c_ulong),
            ('TriEngOperation', c_ulong),
            ('TriggerEngine1', c_ulong),
            ('TrigEngSource1', c_ulong),
            ('TrigEngSlope1', c_ulong),
            ('TrigEngLevel1', c_ulong),
            ('TriggerEngine2', c_ulong),
            ('TrigEngSource2', c_ulong),
            ('TrigEngSlope2', c_ulong),
            ('TrigEngLevel2', c_ulong),
            ]

clock_sources = {
        'internal': 1,
        'slow_external': 4,  # use this for 10 MHz
        'external_AC': 5,  # 'external_10MHz': 7}
        'external_10MHz': 7}

sample_rates = {
        1: 0x1,
        2: 0x2,
        5: 0x4,
        10: 0x8,
        20: 0xA,
        50: 0xC,
        100: 0xE,
        200: 0x10,
        500: 0x12,
        1e3: 0x14,
        2e3: 0x18,
        5e3: 0x1A,
        10e3: 0x1C,
        20e3: 0x1E,
        50e3: 0x22,
        100e3: 0x24,
        250e3: 0x2B,
        500e3: 0x30,
        1e6: 0x35,
        'user_defined': 1e9
        }

channels = {
        'A' : 1,
        'B' : 2,
        'AB' : 0}

couplings = {
    'AC': 1,
    'DC': 2}

impedances = {
    1e3: 1,
    50: 2,
    75: 4,
    300: 8}
k = 1
ranges = {}
ranges_list = [
    0.02,
    0.04,
    0.05,
    0.08,
    0.1,
    0.2,
    0.4,
    0.5,
    0.8,
    1,
    2,
    4,
    5,
    8,
    10,
    20,
    40,
    16]

for range in ranges_list:
    ranges[range] = k
    k += 1
ranges[numpy.inf] = k+1

trigger_oper = {
    'eng_1_only': 0,
    'eng_2_only': 1,
    'eng_1_OR_2': 2,
    'eng_1_AND_2': 3,
    'eng_1_XOR_2': 4,
    'eng_1_AND_NOT_2': 5}

trigger_sources = {
    'chA': 0,
    'chB': 1,
    'external': 2,
    'disable': 3}

trig_slope = {'pos': 1, 'neg': 2}

typedef = {
    'ATS850': 1,
    'ATS310': 2,
    'ATS330': 3,
    'ATS460': 7,
    'ATS860': 8,
    'ATS660': 9,
    'ATS9462': 11,
    'ATS9870': 13,
    'ATS9350': 14,
    'ATS9325': 15,
    'ATS9440': 16,
    'ATS9410': 17,
    'ATS9351': 18,
    'ATS9850': 21}


capabilities = {
    'SERIAL_NUMBER': 0x10000024,
    'LATEST_CAL_DATE': 0x10000026,
    'LATEST_CAL_DATE_MONTH': 0x1000002D,
    'LATEST_CAL_DATE_DAY': 0x1000002E,
    'LATEST_CAL_DATE_YEAR': 0x1000002F,
    'MEMORY_SIZE': 0x1000002A,
    'BOARD_TYPE': 0x1000002B,
    'ASOPC_TYPE': 0x1000002C,
    'PCIE_LINK_SPEED': 0x10000030,
    'PCIE_LINK_WIDTH': 0x10000031}
