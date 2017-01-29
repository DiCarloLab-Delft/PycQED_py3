"""
This example script contains all the elements required for the dac arcs
characterization.

Please use it as an example to base your own scripts on.
It is based on a notebook originally by Ramiro on the 5Q chip.
"""

import time
import datetime
print(time.ctime())

%matplotlib inline

from pycqed.init.LaMaserati_JrJr import *

# Cell 3

# @Ramiro, what is this Omega used for?
omega = lambda flux, f_max, EC, asym: (
    f_max + EC) * (asym**2 + (1-asym**2)*np.cos(np.pi*flux)**2)**0.25 - EC

f_flux = [None] * 5

f_flux[0] = lambda flux: omega(flux=flux,
                               f_max=5.9433029198374685,
                               EC=0.28,
                               asym=0.)


f_flux[1] = lambda flux: omega(flux=flux,
                               f_max=6.3774980297713189,
                               EC=0.28,
                               asym=0.)


f_flux[2] = lambda flux: omega(flux=flux,
                               f_max=5.6883959089701035,
                               EC=0.28,
                               asym=0.)


f_flux[3] = lambda flux: omega(flux=flux,
                               f_max=6.1113712558694182,
                               EC=0.28,
                               asym=0.)


f_flux[4] = lambda flux: omega(flux=flux,
                               f_max=6.7138650690678894,
                               EC=0.28,
                               asym=0.)


def sweep_flux(fl, idx):
    this_flux = Flux_Control.flux_vector()
    this_flux[idx] = fl
    Flux_Control.flux_vector(this_flux)


# Cell 4

params_1 = {'E_c': 0.28,
            'asymmetry': 0,
            'dac_flux_coefficient': 0.0014818919359166486,
            'dac_sweet_spot': -64.682660992718183,
            'f_max': 5.9433029198374685}
params_2 = {'E_c': 0.28,
            'asymmetry': 0,
            'dac_flux_coefficient': 0.0020123973260009367,
            'dac_sweet_spot': 45.588289267501978,
            'f_max': 6.3774980297713189}
params_3 = {'E_c': 0.28,
            'asymmetry': 0,
            'dac_flux_coefficient': 0.0016806934639465709,
            'dac_sweet_spot': -53.472554718672427,
            'f_max': 5.6883959089701035}
params_4 = {'E_c': 0.28,
            'asymmetry': 0,
            'dac_flux_coefficient': 0.0012685027014113798,
            'dac_sweet_spot': 2.4196012752483966,
            'f_max': 6.1113712558694182}
params_5 = {'E_c': 0.28,
            'asymmetry': 0,
            'dac_flux_coefficient': 0.00094498809508039799,
            'dac_sweet_spot': 31.549597601272581,
            'f_max': 6.7138650690678894}

# Arches parameters for all the qubits
from pycqed.analysis import fitting_models as fit_mods
f_dac1 = lambda d: fit_mods.QubitFreqDac(dac_voltage=d, **params_1)*1e9
f_dac2 = lambda d: fit_mods.QubitFreqDac(dac_voltage=d, **params_2)*1e9
f_dac3 = lambda d: fit_mods.QubitFreqDac(dac_voltage=d, **params_3)*1e9
f_dac4 = lambda d: fit_mods.QubitFreqDac(dac_voltage=d, **params_4)*1e9
f_dac5 = lambda d: fit_mods.QubitFreqDac(dac_voltage=d, **params_5)*1e9


#######################################################
# Start of notebook
#######################################################

# Cell 5
q1, q0, q3, q2, q4 = AncT, DataT, AncB, DataM, DataB

list_qubits = [AncT, DataT, AncB, DataM, DataB]
for qubit in list_qubits:
    qubit.f_RO_mod(-20e6)
    qubit.f_pulse_mod(-100e6)
    qubit.RO_fixed_point_correction(True)
    qubit.RO_acq_averages(2**13)
    # we use the CBox as it has fast data acquisiotn compared to the UHFQC
    switch_to_pulsed_RO_CBox(qubit)
CBox.integration_length(500)

# Cell 9
IVVI.set_dacs_zero()
print_instr_params(IVVI)

#######################################################
# Find resonator + qubit
#######################################################

# This cell is used for finding the resonator and the qubit.
# Replace AncT with arbitrary qubit names
# Find resonator
IVVI.set_dacs_zero()
AncT.RO_acq_averages(2**13)
AncT.find_resonator_frequency(freqs=np.arange(7.09e9, 7.096e9, 0.05e6),
                              use_min=True)
# Find qubit
AncT.RO_acq_averages(2**14)
AncT.find_frequency(f_span=40e6, f_step=0.2e6, pulsed=True)
#########################


#######################################################
# Resonator power scan
#######################################################
freqs = np.arange(7.083e9, 7.09e9, 0.05e6)
powers = np.arange(7.78e9, 7.789e9, 0.05e6)
powers = np.arange(-55, 1, 5)

DataM.prepare_for_continuous_wave()
MC.set_sweep_function(HS.frequency)
MC.set_sweep_points(freqs)
MC.set_sweep_function_2D(RF.power)
MC.set_sweep_points_2D(powers)
MC.set_detector_function(det.Heterodyne_probe(HS, trigger_separation=2.8e-6))
MC.run(name='Resonator_power_scan_'+DataM.name, mode='2D')

#######################################################
# Find all qubit freqs
#######################################################
f_sweet_spot = [5.9424611697922414e9,
                6.3774736707949238e9,
                5.6889304935714922e9,
                6.108392498189497e9,
                6.7142613315373918e9]
f_span = 80e6


AncT.find_frequency(freqs=np.arange(f_sweet_spot[0]-f_span,
                                    f_sweet_spot[0], 0.2e6), pulsed=True)
AncB.find_frequency(freqs=np.arange(f_sweet_spot[1]-f_span,
                                    f_sweet_spot[1], 0.2e6), pulsed=True)
DataT.find_frequency(freqs=np.arange(f_sweet_spot[2]-f_span,
                                     f_sweet_spot[2], 0.2e6), pulsed=True)
DataM.find_frequency(freqs=np.arange(f_sweet_spot[3]-f_span,
                                     f_sweet_spot[3], 0.2e6), pulsed=True)
DataB.find_frequency(freqs=np.arange(f_sweet_spot[4]-f_span,
                                     f_sweet_spot[4], 0.2e6), pulsed=True)


#######################################################
# Resonator dac scan
#######################################################

IVVI.set_dacs_zero()
AncT.measure_resonator_dac(freqs=np.arange(7.09e9, 7.096e9, 0.05e6),
                           dac_voltages=np.arange(-1000, 1001, 200))
IVVI.set_dacs_zero()
DataM.measure_resonator_dac(freqs=np.arange(7.784e9, 7.789e9, 0.05e6),
                            dac_voltages=np.arange(-1000, 1001, 200))
IVVI.set_dacs_zero()
AncB.measure_resonator_dac(freqs=np.arange(7.083e9, 7.09e9, 0.05e6),
                           dac_voltages=np.arange(-1000, 1001, 200))
IVVI.set_dacs_zero()
DataB.measure_resonator_dac(freqs=np.arange(7.992e9, 8e9, 0.05e6),
                            dac_voltages=np.arange(-1000, 1001, 200))
IVVI.set_dacs_zero()
DataT.measure_resonator_dac(freqs=np.arange(7.59e9, 7.602e9, 0.05e6),
                            dac_voltages=np.arange(-1000, 1001, 200))
IVVI.set_dacs_zero()


#######################################################
#######################################################
# Arches spectroscoy
#######################################################
#######################################################

# Set frequency calculation to flux
# setting qubit calc to flux seems to contain a bug. setting it to dac also works
AncT.f_qubit_calc('flux')
DataT.f_qubit_calc('flux')
AncB.f_qubit_calc('flux')
DataB.f_qubit_calc('flux')
DataT.f_qubit_calc('flux')


# First test that find frequency and find qubit works
for q in [AncT, AncB, DataT, DataM, DataB]:
    q.RO_fixed_point_correction(True)
    q.f_qubit_calc('dac')
    q.RO_acq_averages(2**13)
    q.find_resonator_frequency(freqs=None,
                              use_min=True)

    q.RO_acq_averages(2**14)
    q.find_frequency(f_span=30e6, f_step=0.2e6, pulsed=True)


f_span = 40e6
dac_vector = np.arange(-200, 201, 5)

# this can be done for all qubits.
# TODO:  This should be combined into a composite detector
IVVI.set_dacs_zero()
for d in dac_vector:
    IVVI.dac1(d)
#     freq_center = f_dac1(d)
    q.RO_acq_averages(2**13)
    try:
        q.find_resonator_frequency(freqs=np.arange(7.09e9, 7.096e9, 0.05e6),
                                      use_min=True)
    except Exception as e:
        logging.warning(e)
    q.RO_acq_averages(2**14)
    try:
        q.find_frequency(f_span=100e6, f_step=0.2e6, pulsed=True)
    except Exception as e:
        logging.warning(e)


# Tracking the qubit vs flux
flux_vec = np.zeros(5)
fluxes=np.linspace(-.03, .03, 21)
q = DataT
f_span = 80e6
flux_id = int(q.dac_channel()-1)
for f in fluxes:
    flux_vec[flux_id] = f
    Flux_Control.flux_vector(flux_vec)
    q.RO_acq_averages(2**13)
    try:
        q.find_resonator_frequency(use_min=True)
    except Exception as e:
        logging.warning(e)
    q.RO_acq_averages(2**14)
    try:
        q.find_frequency(freqs=np.arange(q.f_max()-f_span,
                                         q.f_max()+f_span/2, 0.2e6),
                         pulsed=True)
    except Exception as e:
        logging.warning(e)




# After this the data can be extracted and pasted together.
# TODO: add relevant analysis snippet here.


#######################################################
#######################################################
# After tuning arches, update qubit parameters
#######################################################
#######################################################

AncT.E_c(0.28e9)
AncT.asymmetry(0)
AncT.dac_flux_coefficient(0.0014832606276941286)
AncT.dac_sweet_spot(-80.843401134877467)
AncT.f_max(5.942865842632016e9)
AncT.f_qubit_calc('flux')

AncB.E_c(0.28e9)
AncB.asymmetry(0)
AncB.dac_flux_coefficient(0.0020108167368328178)
AncB.dac_sweet_spot(46.64580507835808)
AncB.f_max(6.3772306731019359e9)
AncB.f_qubit_calc('flux')

DataT.E_c(0.28e9)
DataT.asymmetry(0)
DataT.dac_flux_coefficient(0.0016802077647335939)
DataT.dac_sweet_spot(-59.871260477923215)
DataT.f_max(5.6884932787721443e9)
DataT.f_qubit_calc('flux')


DataM.E_c(0.28e9)
DataM.asymmetry(0)
DataM.dac_flux_coefficient(0.0013648395455073477)
DataM.dac_sweet_spot(23.632250360310309)
DataM.f_max(6.1091409419040268e9)
DataM.f_qubit_calc('flux')

DataB.E_c(0.28e9)
DataB.asymmetry(0)
DataB.dac_flux_coefficient(0.00076044591994623627)
DataB.dac_sweet_spot(89.794843711783415)
DataB.f_max(6.7145280717783091e9)
DataB.f_qubit_calc('flux')

sweet_spots_mv = [-AncT.dac_sweet_spot(), -AncB.dac_sweet_spot(),
                  -DataT.dac_sweet_spot(), -DataM.dac_sweet_spot(),
                  -DataB.dac_sweet_spot()]

offsets = np.dot(Flux_Control.transfer_matrix(),sweet_spots_mv)

Flux_Control.flux_offsets(offsets)
Flux_Control.flux_vector(np.zeros(5))

AncT.dac_sweet_spot()

IVVI.dac1(0.)
IVVI.dac2(0.)
IVVI.dac3(0.)
IVVI.dac4(0.)
IVVI.dac5(0.)


