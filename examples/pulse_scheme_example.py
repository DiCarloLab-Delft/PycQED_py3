import matplotlib.pyplot as plt
import pycqed.utilities.pulse_scheme as pls
import numpy as np

# Simple example of how to use the pulse_scheme module

cm = 1 / 2.54  # inch to cm conversion

fig, ax = pls.new_pulse_fig((7*cm, 3*cm))

# Plot pulses
p1 = pls.mwPulse(ax, 0, width=1.5, label='$X_{\\pi/2}$')
p2 = pls.ramZPulse(ax, p1, width=2.5, sep=1.5)
p3 = pls.separator(ax, p2)
p4 = pls.mwPulse(ax, p3, width=1.5, phase=np.pi/2, label='$Y_{\\pi/2}$')

# Add some arrows and labeling
pls.interval(ax, p1, p1 + 1.5, height=1.7, label='$T_\\mathsf{p}$')
pls.interval(ax, p1, p3, height=-.6, labelHeight=-0.5, label='$\\tau$',
             vlines=False)

# Adjust plot range to fit the whole figure
ax.set_ylim(-1.2, 2.5)

plt.show()
