'''
Functions for Quantum Simulations on DiCarlo lab
Packages: quantumsim, quantumsim_overlay
General functions to be used in any quantum simulation
'''

import qsoverlay.circuit_builder as circuit_builder
from qsoverlay import DiCarlo_setup as setup
#from qsoverlay import gate_redesign as gates
import numpy as np
from numpy import pi
import quantumsim.circuit
import quantumsim.sparsedm


def qubitdef(qubit_parameters, qubit_gate={}, **kwargs):
    # The commented lines are important to know how to define the variables
    '''
    qubit_parameters=(['q1', 'q2', 'q3'...],
    [{noise_flag=True, t1=30000, t2=30000},
    {noise_flag=True, t1=30000, t2=30000},
    ()])
    '''

    '''
    qubit_gate = {msmt_time=600, gate_time_1q=20, gate_time_2q=40}
    '''

    gate_set = setup.get_gate_dic(**qubit_gate, **kwargs)
    composite_gate_set = setup.get_composites()
    qubit_dic = {}

    for qubit, param in zip(*qubit_parameters):
        qubit_dic[qubit] = setup.get_qubit(**param, **kwargs)

    return(qubit_dic, gate_set, composite_gate_set)


def create_circuitbuilder(qubit_dic, gate_set, composite_gate_set):

    cir = circuit_builder.Builder(qubit_dic=qubit_dic, gate_dic=gate_set,
                                  composite_gate_set=composite_gate_set)

    return cir


def Sz_measure(qu_list, sdm, **kwargs):

    # qu_list is an array with the names of the qubits with the same name as in the circuit
    # sdm is the Sparse Density Matrix after the circuit
    # This function performs a Sz measurement on the listed qubits
    # It keeps track on the position of the qubit such that the measurement is performed correctly

    qid = [sdm.idx_in_full_dm[qubits] for qubits in qu_list]

    diag = sdm.full_dm.get_diag()

    if 'readout_error' in kwargs.keys():
        eps_readout = kwargs['readout_error']
    else:
        eps_readout = 0

    # if 'qu_cond_list' in kwargs.keys():
    #    diag_short = [diag[1], diag[2]]
    #    measure_Sz = sum([x*(-1)**(sum([((n//(2**q)) % 2) for q in qid])) for n, x in enumerate(diag_short, 1)])/(sum(diag_short))
    # else:
    measure_Sz = (sum([x*(-1)**(sum([((n//(2**q)) % 2) for q in qid]))
                  for n, x in enumerate(diag)]))*(1-2*eps_readout)

    return measure_Sz


def prepare_state(state_prep_ops, cir, **kwargs):
    '''
    state_prep_ops= (['RX', 'RY', 'ISwap'],['q1', 'q2', ('q1','q2')],[pi/2, pi/2, angle1])
    '''
    prep_ops_dic = {'RX': lambda qubit, ang: cir.add_gate('RX', qubit_list=[qubit], angle=ang),
                    'RY': lambda qubit, ang: cir.add_gate('RY', qubit_list=[qubit], angle=ang),
                    'RZ': lambda qubit, ang: cir.add_gate('RZ', qubit_list=[qubit], angle=ang),
                    'CZ': lambda qubit_list, ang: cir.add_gate('CZ', qubit_list=qubit_list),
                    'CNOT': lambda qubit_list: cir.add_gate('CNOT', qubit_list=qubit_list),
                    'ISwapRot': lambda qubit_list, ang: cir.add_gate('ISwapRotation',
                                                                     qubit_list=qubit_list,
                                                                     angle=ang),
                    'Hadamard': lambda qubit: cir.add_gate('Had', qubit=[qubit])}

    if state_prep_ops is not None:
        for operation, qubit, ang in zip(*state_prep_ops, **kwargs):
            prep_ops_dic[operation](qubit, ang)
    else:
        raise ValueError('State needs to be prepared')


def construct_measurement(measurement_list, cir,
                          finalize=True, **kwargs):
    """
    Includes what measurement needs to be done and in which qubit
    measurement_list = (["X", "Y", "X", "CNOT"],
                           ["q1", "q2", "q2", ("q1","q2")],[])
                           """

    msmt_time = cir.qubit_dic.get('QL').get('msmt_time', 1400)

    op_dic = {'X': lambda qubit, **kwargs: cir.add_gate('RY', qubit_list=[qubit], angle=pi/2),
              'Xinv': lambda qubit, **kwargs: cir.add_gate('RY', qubit_list=[qubit], angle=-pi/2),
              'Y': lambda qubit, **kwargs: cir.add_gate('RX', qubit_list=[qubit], angle=pi/2),
              'Yinv': lambda qubit, **kwargs: cir.add_gate('RX', qubit_list=[qubit], angle=-pi/2),
              'CNOT': lambda qubit_list, **kwargs: cir.add_gate('CNOT', qubit_list=qubit_list),
              'CZ': lambda qubit_list, **kwargs: cir.add_gate('CZ', qubit_list=qubit_list),
              'Measure': lambda qubit, **kwargs: cir.add_gate('Measure', qubit_list=[qubit], sampler=kwargs['sampler'])}

    if measurement_list is not None:
        for operation, qubit in zip(*measurement_list):
            op_dic[operation](qubit, **kwargs)

    if finalize is True:
        cir.finalize(t_add=msmt_time)


def construct_sdm(cir):
    sdm = quantumsim.sparsedm.SparseDM(cir.circuit.get_qubit_names())
    cir.circuit.order()
    cir.circuit.apply_to(sdm)
    return sdm


def state_prep_measure(qubit_param, state_prep_ops, measurement_list,
                       qubit_gate={}, circuit=False, **kwargs):
    '''
    General function, prepares initial state, creates measurement circuit.
    Returns Sparse Density Matrix (sdm)
    Optional: Return circuit_builder(cir) if circuit==True
    '''
    '''
    qubit_param=(['q1', 'q2', 'q3'...],
    [{'noise_flag':True, 't1':30000, 't2':30000},
    {'noise_flag':True, 't1':30000, 't2':30000},
    {'noise_flag':True, 't1':30000, 't2':30000}])
    The parameters can be different for each qubit
    '''
    '''
    state_prep_ops= (['RX', 'RY', 'ISwap'],['q1', 'q2', ('q1','q2')],[pi/2, pi/2, angle1])
    '''
    '''
    measurement_list=(['Measure', 'Measure', 'X', 'Y'], ['q1','q2','q2','q1])
    '''
    '''
    qubit_gate = {'msmt_time':600, 'gate_time_1q':20, 'gate_time_2q':40}
    if qubit_gate is not introduced it takes the default values from qsoverlay 600ns
    '''
    # readout_error is contained it **kwargs and passed to Sz_measure function

    # Variable assignation used for all parts of the function

    qubit_gate = qubit_gate

    qubit_dic, gate_set, composite_gate_set = qubitdef(qubit_list=qubit_list,
                                                       qubit_gate=qubit_gate)

    cir = create_circuitbuilder(qubit_dic=qubit_dic, gate_set=gate_set,
                                composite_gate_set=composite_gate_set)

    prepare_state(state_prep_ops=state_prep_ops,
                  cir=cir)

    construct_measurement(measurement_list=measurement_list,
                          cir=cir,
                          **kwargs)
    sdm = construct_sdm(cir)

    if circuit is False:
        return sdm
    else:
        return(sdm, cir)
