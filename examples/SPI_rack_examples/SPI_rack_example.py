"""
This is an example on how to control the SPI rack current sources (S4g) using
the FluxCurrent driver of PycQED.

The FluxCurrent driver combines several SPI modules into a single driver
and contains a crosstalk correction matrix.

This example will go over how to install the requirements and show how
to have basic control over the instrument.


1. Ensure the spirack dependencies are installed correctly.
    If on windows, download https://github.com/Rubenknex/SPI-rack and install
        the drivers.zip
    use `pip install spirack` to install the python drivers

2. Run the examples below to ensure everything is working correctly, the
    comments should make this self explanatory.

"""

from pycqed.instrument_drivers.physical_instruments.QuTech_SPI_S4g_FluxCurrent \
    import QuTech_SPI_S4g_FluxCurrent


# The first step is to find what com port is being used.
# to find this go to "Device Manager" (on windows) and look under Ports.
# The "SPI Rack Controller" should be listed here.
address = 'COM10'


# we define a mapping between parameter names and the module/channel being
# used. By convention we use the names of the qubits that are being biased.
# The format is {"Name": (module_number, channel)}.
# N.B. The module number is set by a dip switch if you open up the module and
# should be unique to each unit. Channels start counting from zero (not 1).
channel_map = {"QL": (2, 0),
               "QR": (2, 2),
               "test_current": (2, 1)}


# We instantiate the fluxcurrent instrument.
# note that the channel map maps
fluxcurrent = QuTech_SPI_S4g_FluxCurrent(
    "fluxcurrent", address=address,
    channel_map=channel_map)


# Currents can now simply be set using the parameters
fluxcurrent.test_current(5e-3)  # 50mA

# A convenience function reads out all the currents and prints them

fluxcurrent.print_overview()

# It should print something like this, showing both the parameter name,
# the module and channel used and the read out value.
"""
Name            Module  Channel    I
QL                 4       0    9.918 μA
QR                 4       2    0.0  A
test_current       4       1    5.0 mA
"""

