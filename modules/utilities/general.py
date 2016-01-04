import subprocess
import os
import qt
import h5py
from modules.analysis import analysis_toolbox as a_tools

import sys
import glob
import serial

def get_git_revision_hash():
    import logging
    import subprocess
    try:
        PycQEDdir = qt.config['PycQEDdir']
        hash = subprocess.check_output(['git', 'rev-parse',
                                        '--short=7', 'HEAD'], cwd=PycQEDdir)
    except:
        logging.warning('Failed to get Git revision hash, using 00000 instead')
        hash = '00000'

    return hash


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
    s.login('DCLabemail@gmail.com','DiCarloLab')
    s.sendmail(email, family, msg.as_string())
    s.quit()


#This is code from Kwant that Anton showed me (Adriaan), it is located
# at http://git.kwant-project.org/kwant/tree/setup.py.
# it should in the future replace the current git get revision hash function.

# This is an exact copy of the function from kwant/version.py.  We can't import
# it here (because Kwant is not yet built when this scipt is run), so we just
# include a copy.
def get_version_from_git():
    PycQEDdir = qt.config['PycQEDdir']
    try:
        p = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                             cwd=PycQEDdir,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return
    if p.wait() != 0:
        return
    # TODO: use os.path.samefile once we depend on Python >= 3.3.
    if os.path.normpath(p.communicate()[0].rstrip('\n')) != PycQEDdir:
        # The top-level directory of the current Git repository is not the same
        # as the root directory of the Kwant distribution: do not extract the
        # version from Git.
        return



    # git describe --first-parent does not take into account tags from branches
    # that were merged-in.
    for opts in [['--first-parent'], []]:
        try:
            p = subprocess.Popen(['git', 'describe'] + opts, cwd=PycQEDdir,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            return
        if p.wait() == 0:
            break
    else:
        pass
        # return p
    version = p.communicate()[0].rstrip('\n')

    # if version[0] == 'v':
    #     version = version[1:]

    try:
        p = subprocess.Popen(['git', 'diff', '--quiet'], cwd=PycQEDdir)
        print('Try statement')
        return p
    except OSError:
        version += '-confused'  # This should never happen.
    else:
        if p.wait() == 1:
            version += '-dirty'
    return version


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
