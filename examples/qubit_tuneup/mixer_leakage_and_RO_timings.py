import qcodes.instrument_drivers.signal_hound.USB_SA124B as sh

# Mixer offset calibration snippet
SH = sh.SignalHound_USB_SA124B('SH')
AncT.calibrate_mixer_offsets(SH)


# Calibrating pulse to readout timing

