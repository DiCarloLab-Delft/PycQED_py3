from pycqed.measurement.waveform_control_CC import waveform as wf

try:
    from qcodes import Instrument
except ImportError:
    print('could not import qcodes Instrument')


def mock_control_pulse_prepare(command, **kwargs):
    '''
    Mock function for testing purposes returns the kwargs
    '''
    # printing to be caught in test suite
    print('mock called with {}'.format(kwargs))
    return


def QWG_pulse_prepare(operation_name, **kwargs):
    QWG_name = kwargs.pop('QWG_name')
    codeword = kwargs.pop('codeword')
    channels = kwargs.pop('channels')

    QWG = Instrument.find_instrument(QWG_name)
    pulse_func = getattr(wf, kwargs['pulse_type'])
    waveform = pulse_func(kwargs)

    for i, ch in enumerate(channels):
        wf_name = operation_name+str(ch)
        QWG.createWaveformReal(wf_name, waveform[i], [], [])
        QWG.set(codeword, ch, wf_name)






