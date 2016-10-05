
def mock_control_pulse_prepare(**kwargs):
    '''
    Mock function for testing purposes returns the kwargs
    '''
    print('generating and uploading pulse with {}'.format(kwargs))
    return kwargs
