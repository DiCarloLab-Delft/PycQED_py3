from pycqed.measurement.randomized_benchmarking.clifford_decompositions \
    import(gate_decomposition)
"""
This file contains Clifford decompositions for the two qubit Clifford group.

The Clifford decomposition follows closely two papers:
Corcoles et al. Process verification .... Phys. Rev. A. 2013
    http://journals.aps.org/pra/pdf/10.1103/PhysRevA.87.030301
for the different classes of two-qubit Cliffords.

and
Barends et al. Superconducting quantum circuits at the ... Nature 2014
    https://www.nature.com/articles/nature13171?lang=en
for writing the cliffords in terms of CZ gates.


###########################################################################
2-qubit clifford decompositions

The two qubit clifford group (C2) consists of 11520 two-qubit cliffords
These gates can be subdivided into four classes.
    1. The Single-qubit like class  | 576 elements  (24^2)
    2. The CNOT-like class          | 5184 elements (24^2 * 3^2)
    3. The iSWAP-like class         | 5184 elements (24^2 * 3^2)
    4. The SWAP-like class          | 576  elements (24^2)
    --------------------------------|------------- +
    Two-qubit Clifford group C2     | 11520 elements


1. The Single-qubit like class
    -- C1 --
    -- C1 --

2. The CNOT-like class
    --C1--•--S1--      --C1--•--S1------
          |        ->        |
    --C1--⊕--S1--      --C1--•--S1^Y90--

3. The iSWAP-like class
    --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
          |       ->        |        |
    --C1--*--S1--     --C1--•--mY90--•--S1^X90--

4. The SWAP-like class
    --C1--x--     --C1--•-mY90--•--Y90--•-------
          |   ->        |       |       |
    --C1--x--     --C1--•--Y90--•-mY90--•--Y90--


C1: element of the single qubit Clifford group
S1: element of the S1 group, a subgroup of the single qubit Clifford group

The S1 subgroups of C1 used above.
Sets written as physical gates in time. Left gate is applied first.
Index in the single qubit Clifford group
S1:
 C1[0]:   I
 C1[1]:   Y90, X90
 C1[2]:   mX90, mY90

S1^X90:
 C1[16]:   X90
 C1[17]:   X90, Y90, X90
 C1[15]:   mY90

S1^Y90:
 C1[21]:   Y90
 C1[xx]:   Y180, X90
 C1[xx]:   mX90, mY90, X90



"""



