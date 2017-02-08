import numpy as np
import matplotlib.pyplot as plt
import qutip as qtp
from pycqed.analysis import tomography as tomo
from qcodes.plots.pyqtgraph import QtPlot


# phase_q0 = np.pi/50
phase_q0 = 0
# phase_q1 = np.pi/20
phase_q1 = 0
phase_2q = np.pi/50
# phase_2q = 0

up = qtp.basis(2, 0)
dn = qtp.basis(2, 1)
bell_state = 0

if bell_state == 0:  # |Phi_m>=|00>-|11>
    gate1 = 'Y90 ' + qS
    gate2 = 'Y90 ' + qCZ
    after_pulse = 'mY90 ' + qCZ
elif bell_state == 1:  # |Phi_p>=|00>+|11>
    gate1 = 'mY90 ' + qS
    gate2 = 'Y90 ' + qCZ
    after_pulse = 'mY90 ' + qCZ
elif bell_state == 2:  # |Psi_m>=|01> - |10>
    gate1 = 'Y90 ' + qS
    gate2 = 'mY90 ' + qCZ
    after_pulse = 'mY90 ' + qCZ
elif bell_state == 3:  # |Psi_p>=|01> + |10>
    gate1 = 'mY90 ' + qS
    gate2 = 'mY90 ' + qCZ
    after_pulse = 'mY90 ' + qCZ


theta_0 = np.exp(-1j*phase_q0)
theta_1 = np.exp(-1j*phase_q1)
theta_2q = np.exp(-1j*phase_2q)

psi_b0 = (theta_2q*qtp.tensor([theta_0*up,
                              theta_1*up])
          + qtp.tensor([dn, dn])).unit()


bell0 = qtp.ket2dm(psi_b0)

print(bell0)
bell0_p = tomo.pauli_ops_from_density_matrix(bell0)

f, ax = plt.subplots()
tomo.plot_operators(bell0_p, ax)

# tomo.plot_operators(bell0, ax)
plt.show()

