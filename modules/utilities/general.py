import os
# import qt
import h5py
from modules.analysis import analysis_toolbox as a_tools

import sys
import glob
import serial

def get_git_revision_hash():
    import logging
    import subprocess
    try:
        # Refers to the global qc_config
        PycQEDdir = qc_config['PycQEDdir']
        hash = subprocess.check_output(['git', 'rev-parse',
                                        '--short=7', 'HEAD'], cwd=PycQEDdir)
    except:
        logging.warning('Failed to get Git revision hash, using 00000 instead')
        hash = '00000'

    return hash


def dict_to_ordered_tuples(dic):
    '''Convert a dictionary to a list of tuples, sorted by key.'''
    if dic is None:
        return []
    keys = dic.keys()
    keys.sort()
    ret = [(key, dic[key]) for key in keys]
    return ret


def load_settings_onto_instrument(instrument, folder=None,
                                  label='Settings',
                                  timestamp=None, **kw):
        '''
        Loads settings from an hdf5 file onto the instrument handed to the
        function.
        By default uses the last hdf5 file in the datadirectory with
        'Settings' in the name. By giving a label or timestamp another file
        can be chosen as the settings file.
        '''
        older_than = None
        instrument_name = instrument.get_name()
        success = False
        count = 0
        while success is False and count < 10:
            try:
                if folder is None:
                    folder = a_tools.get_folder(timestamp, older_than, **kw)
                else:
                    folder = folder
                filepath = a_tools.measurement_filename(folder)
                f = h5py.File(filepath, 'r')
                sets_group = f['Instrument settings']
                ins_group = sets_group[instrument_name]
                print('Loaded Settings Successfully')
                success = True
            except:
                older_than = os.path.split(folder)[0][-8:] \
                             +'_'+ os.path.split(folder)[1][:6]
                folder = None
                success = False
            count+=1

        if not success:
            print('Could not open settings for instrument "%s"' % (
                instrument_name))
            return False

        for parameter, value in ins_group.attrs.items():
            try:
                if value != 'None':  # None is saved as string in hdf5
                    if type(value) == str:
                        if value == 'False':
                            exec("instrument.set_%s(False)" % (parameter))
                        else:
                            exec("instrument.set_%s('%s')" % (parameter, value))
                    else:
                        exec('instrument.set_%s(%s)' % (parameter, value))
            except:
                print('Could not set parameter: "%s" for instrument "%s"' % (
                    parameter, instrument_name))
        f.close()
        return True


def send_email(subject='PycQED needs your attention!',
               body = '', email=None):
    # Import smtplib for the actual sending function
    import smtplib
    # Here are the email package modules we'll need
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    if email is None:
        email = qt.config['e-mail']

    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg['Subject'] = subject
    family = 'serwan.asaad@gmail.com'
    msg['From'] = 'Lamaserati@tudelft.nl'
    msg['To'] = email
    msg.attach(MIMEText(body, 'plain'))


    # Send the email via our own SMTP server.
    s = smtplib.SMTP_SSL('smtp.gmail.com')
    s.login('DCLabemail@gmail.com', 'DiCarloLab')
    s.sendmail(email, family, msg.as_string())
    s.quit()



def list_available_serial_ports():
    """Lists serial ports

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of available serial ports

    Frunction from :
    http://stackoverflow.com/questions/12090503/
        listing-available-com-ports-with-python
    """
    if sys.platform.startswith('win'):
        ports = ['COM' + str(i + 1) for i in range(256)]

    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this is to exclude your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')

    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')

    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result
