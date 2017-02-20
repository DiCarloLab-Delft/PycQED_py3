from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import(gate_decomposition)
'''
two-qubit clifford decomposition as per
http://journals.aps.org/pra/pdf/10.1103/PhysRevA.87.030301
'''

#########################################################################l##
#2-qubit clifford decompositions
############################################################################
#We divide the 11520 two-qubit clifford group into several classes:
###########################################################################
# class_0: single-qubit class
#q0      -C1-
#q1      -C1-
#With C1, the single-qubit clifford group. As C1 consists of 24 cliffords,
#this class has 24^2=576 elements.

############################################################################
C1_q0 = [[s + ' q0' for s in ss] for ss in gate_decomposition] #adding qubit suffix to the single qubit cliffords
C1_q1 = [[s + ' q1' for s in ss] for ss in gate_decomposition] #can easiliy be changed to the 5 primitives method
# C1_q1 = [s + "q1" for s in gate_decomposition]

twoQ_class_0 = [[]]*(24*24)

for i in range(24):
    for j in range(24):
        twoQ_class_0[i*24+j] = C1_q0[i]+C1_q1[j]

# class_1: c-not like class
#q0      -C1  -c-  S1
#q1      -C1 -not- S1
#with c the control and not the target of the cnot gate and S1 a group of
#three single-qubit cliffords

gate_decomposition_2Q = twoQ_class_0



