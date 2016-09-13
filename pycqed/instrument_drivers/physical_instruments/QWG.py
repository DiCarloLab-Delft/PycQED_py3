'''
	File:				QWG.py
	Author:				Wouter Vlothuizen, TNO/QuTech
	Purpose:			Instrument driver for Qutech QWG
	Usage:				
	Notes:				does not depend on other software or drivers (e.g. Visa, etc). FIXME: not true anymore
	Bugs:

'''

from SCPI import SCPI

import numpy as np
from qcodes import validators as vals


class Transport:
	def __init__(self, address=None, timeout=5, terminator=''):
		self.address = address
	# FIXME: define empty virtual functions

class IPTransport(Transport):
	def __init__(self, address=None, port=None, timeout=5,
				 terminator='\n', persistent=True, write_confirmation=True,
				 **kwargs):
		self.address = address
		self.port = port
	# FIXME: define functions based on IPInstrument


class SocketTransport(Transport):  
	def __init__(self, logging=True, simMode=False, paranoid=False):
		# properties
		self.logging = logging		# enable logging
		self.simMode = simMode		# simulation: don't access hardware
		self.paranoid = paranoid	# be paranoid about cross checking, at the cost of performance
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


	def open(self, host, port=5025):
		''' open connection, e.g. open('192.168.0.16', 4000)
		'''
		if not self.simMode:
			self.socket.settimeout(1)												# first set timeout (before connect)
			self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 512*1024)	# beef up buffer, to prevent socket.send() not sending all our data in one go
			self.socket.connect( (host, port) )
	
	
	def close(self):
		''' close connection
		'''
		if not self.simMode:
			self.socket.close()


	def write(self, str):
		''' send a command string
			NB: send can be used by the end user directly, but this is not encouraged because it defeats our abstraction layer
		'''
		if not self.simMode:
			outStr = str+'\n'
			self.socket.send(outStr.encode('ascii'))		# FIXME: check return value, maybe encode() can be improved on by not using unicode strings?
		
		# FIXME: logging
	

	def writeBinary(self, data):
		''' send binary data
			Input:
				data	bytearray
		'''
		if not self.simMode:
			expLen = len(data)
			actLen = self.socket.send(data)
			if(actLen!=expLen):
				raise UserWarning('not all data sent: expected %d, actual %d' % (expLen, actLen))		# FIXME: handle this case by calling send again. Or enlarge socket.SO_SNDBUF even further
		
		# FIXME: logging
	
	
	def readBinary(self, byteCnt):
		''' read binary data
		'''
		if not self.simMode:
			data = self.socket.recv(byteCnt)
		else:
			data = zeros(byteCnt, 1)
		
		# FIXME: logging
		return data
	
	
	def ask(self, str):
		''' send a command, and receive a response
		'''
		self.send(str)
		
		if not self.simMode:
#			resp = self.socket.recv(4096)	# FIXME: do a readline
			resp = self.socket.makefile().readline()		# is this allowed when timeout is active (i.e. non blocking socket)?
		else:
			resp = '';

		return resp.rstrip()								# remove trailing white space, CR, LF
	
	
	def askDouble(self, str):
		resp = self.ask(str)
		return str2double(resp) 




class QWG(SCPI):
	def __init__(self, name, transport=None, **kwargs):
		super().__init__(name, transport, **kwargs)

		# AWG properties
		self.device_descriptor = type('', (), {})()
		self.device_descriptor.model = 'QWG'
		self.device_descriptor.numChannels = 4
		self.device_descriptor.numDacBits = 12
		self.device_descriptor.numMarkersPerChannel = 2
		self.device_descriptor.numMarkers = 8
		self.device_descriptor.numTriggers = 8

		# valid values
		self.device_descriptor.mvals_trigger_impedance = vals.Enum(50),
		self.device_descriptor.mvals_trigger_level = vals.Numbers(0, 2.5)
		self.device_descriptor.mvals_channel_amplitude = vals.Numbers(0,1)	# FIXME: not in [V]
		self.device_descriptor.mvals_channel_offset = vals.Numbers(-0.05,0.05)	# FIXME: not in [V]

		self.add_parameters()


	def add_parameters(self):
		##########################################################################
		## QWG specific
		##########################################################################

		for i in range(1, self.device_descriptor.numChannels+1):		# FIXME: 1 & 3 only
			sfreq_cmd = 'qutech:output{}:frequency'.format(i)
			sph_cmd = 'qutech:output{}:phase'.format(i)
			# NB: sideband frequency has a resolution of ~0.23 Hz:
			self.add_parameter('ch_pair{}_sideband_frequency'.format(i),
					   units='Hz',
					   label='Sideband frequency channel pair {} (Hz)'.format(i),
					   get_cmd=sfreq_cmd + '?',
					   set_cmd=sfreq_cmd + ' {}',
					   vals=vals.Numbers(-300e6, 300e6),
					   get_parser=float)
			self.add_parameter('ch_pair{}_sideband_phase'.format(i),
					   units='deg',
					   label='Sideband phase channel pair {} (deg)'.format(i),
					   get_cmd=sph_cmd + '?',
					   set_cmd=sph_cmd + ' {}',
					   vals=vals.Numbers(-180, 360),
					   get_parser=float)

		for i in range(1, self.device_descriptor.numTriggers+1):
			triglev_cmd = 'qutech:trigger{}:level'.format(i)
			# individual trigger level per trigger input:
			self.add_parameter('ch{}_trigger_level'.format(i),
					   units='V',
					   label='Trigger level channel {} (V)'.format(i),
					   get_cmd=triglev_cmd + '?',
					   set_cmd=triglev_cmd + ' {}',
					   vals=self.device_descriptor.mvals_trigger_level,
					   get_parser=float)

		##########################################################################
		## Tek 5014 compatible
		## NB: code below mostly copied from QCoDeS
		##########################################################################

	   # Compatibility: 5015, QWG: FIXME: QWG adds 'TBD' and does not support 'TBD'
		self.add_parameter('run_mode',
						   get_cmd='AWGC:RMOD?',
						   set_cmd='AWGC:RMOD ' + '{}',
						   vals=vals.Enum('CONT', 'TRIG', 'SEQ', 'GAT'))		

		# Trigger parameters #
		self.add_parameter('trigger_impedance',		
						   label='Trigger impedance (Ohm)',
						   units='Ohm',
						   get_cmd='TRIG:IMP?',
						   set_cmd='TRIG:IMP ' + '{}',
						   vals=vals.Enum(50, 1000),
						   get_parser=float)

		# Compatibility: 5014, QWG FIXME: different range
		self.add_parameter('trigger_level',
						   units='V',
						   label='Trigger level (V)',
						   get_cmd='TRIG:LEV?',
						   set_cmd='TRIG:LEV ' + '{:.3f}',
						   vals=vals.Numbers(-5, 5),
						   get_parser=float)
		
		self.add_parameter('trigger_slope',
						   get_cmd='TRIG:SLOP?',
						   set_cmd='TRIG:SLOP ' + '{}',
						   vals=vals.Enum('POS', 'NEG'))  # ,
						   # get_parser=self.parse_int_pos_neg)
		
		self.add_parameter('trigger_source',
						   get_cmd='TRIG:source?',
						   set_cmd='TRIG:source ' + '{}',
						   vals=vals.Enum('INT', 'EXT'))

		# Channel parameters #
		for i in range(1, self.device_descriptor.numChannels+1):
			amp_cmd = 'SOUR{}:VOLT:LEV:IMM:AMPL'.format(i)
			offset_cmd = 'SOUR{}:VOLT:LEV:IMM:OFFS'.format(i)
			state_cmd = 'OUTPUT{}:STATE'.format(i)
			waveform_cmd = 'SOUR{}:WAV'.format(i)
			# Set channel first to ensure sensible sorting of pars

			# Compatibility: 5014, QWG 
			self.add_parameter('ch{}_state'.format(i),
							   label='Status channel {}'.format(i),
							   get_cmd=state_cmd + '?',
							   set_cmd=state_cmd + ' {}',
							   vals=vals.Ints(0, 1))

			# Compatibility: 5014, QWG (FIXME: different range, not in V)
			self.add_parameter('ch{}_amp'.format(i),
							   label='Amplitude channel {} (Vpp)'.format(i),
							   units='Vpp',
							   get_cmd=amp_cmd + '?',
							   set_cmd=amp_cmd + ' {:.6f}',
							   vals=vals.Numbers(0.02, 4.5),
							   get_parser=float)

			# Compatibility: 5014, QWG (FIXME: different range, not in V)
			self.add_parameter('ch{}_offset'.format(i),
							   label='Offset channel {} (V)'.format(i),
							   units='V',
							   get_cmd=offset_cmd + '?',
							   set_cmd=offset_cmd + ' {:.3f}',
							   vals=vals.Numbers(-.1, .1),
							   get_parser=float)

			# FIXME: handle waveform differently?
#			self.add_parameter('ch{}_waveform'.format(i),
#							   label='Waveform channel {}'.format(i),
#							   get_cmd=waveform_cmd + '?',
#							   set_cmd=waveform_cmd + ' "{}"',
#							   vals=vals.Strings(),
#FIXME							   get_parser=parsestr)

			# Marker channel parameters #
			for j in range(1, self.device_descriptor.numMarkers+1):
				m_del_cmd = 'SOUR{}:MARK{}:DEL'.format(i, j)
				m_high_cmd = 'SOUR{}:MARK{}:VOLT:LEV:IMM:HIGH'.format(i, j)
				m_low_cmd = 'SOUR{}:MARK{}:VOLT:LEV:IMM:LOW'.format(i, j)

				self.add_parameter(
					'ch{}_m{}_del'.format(i, j),
					label='Channel {} Marker {} delay (ns)'.format(i, j),
					get_cmd=m_del_cmd + '?',
					set_cmd=m_del_cmd + ' {:.3f}e-9',
					vals=vals.Numbers(0, 1),
					get_parser=float)

				self.add_parameter(
					'ch{}_m{}_high'.format(i, j),
					label='Channel {} Marker {} high level (V)'.format(i, j),
					get_cmd=m_high_cmd + '?',
					set_cmd=m_high_cmd + ' {:.3f}',
					vals=vals.Numbers(-2.7, 2.7),
					get_parser=float)

				self.add_parameter(
					'ch{}_m{}_low'.format(i, j),
					label='Channel {} Marker {} low level (V)'.format(i, j),
					get_cmd=m_low_cmd + '?',
					set_cmd=m_low_cmd + ' {:.3f}',
					vals=vals.Numbers(-2.7, 2.7),
					get_parser=float)



	
	##########################################################################
	## QWG functions not very suitable to be implemented as Parameter
	##########################################################################

	def syncSidebandGenerators(self):
		''' 
		'''
		self.write('QUTEch:OUTPut:SYNCsideband')

	##########################################################################
	## QWG functions that are/could be implemented as Parameter
	## will be deprecated in the future
	##########################################################################

	def setRunModeCodeword(self):
		self.write('awgcontrol:rmode codeword')
	
	def setSidebandFrequency(self, chPair, frequency):
		"""
		Set the sideband frequency for a channel pair.

		Args:
			chPair (int): the channel pair to use, 1 or 3

			frequency (float): the sideband frequency in [Hz], range
			-MAXF..MAXF in 0.23 Hz steps. MAXF is currently 300 MHz
		"""
		self.write('qutech:output%d:frequency %f' % (chPair,frequency))

	def setSidebandPhase(self, chPair, phaseDeg):
		''' phaseDeg:			-180..180, or 0..360 in 65536 steps
		'''
		self.write('qutech:output%d:phase %f' % (chPair,phaseDeg))

	def setMatrix(self, chPair, mat):
		''' matrix:				2x2 matrix for mixer calibration
		'''
		self.write('qutech:output%d:matrix %f,%f,%f,%f' % (chPair,mat[0,0],mat[1,0],mat[0,1],mat[1,1]))		# FIXME

	def setChannelTriggerLevel(self, trigChannel, level):
		''' level:				0.0 V to 2.5 V, in very small steps
		'''
		self.write('qutech:trigger%d:level %f' % (trigChannel,level))

	#####################################################################################
	## AWG5014 functions: SOURCE
	#####################################################################################
	def setWaveform(self, ch, name):
		'''	ch:				1..4
			name:			waveform name excluding double quotes, e.g. '*Sine100'
			Compatibility:	5014, QWG
		'''
		self.write('source%d:waveform "%s"' % (ch, name))
	
	
	def setPhaseDeg(self, ch, phase):
		'''	NB: applies to waveforms only, in non-sequence mode
			ch:				1,2
			phase:			-180 to 180 [deg], steps not defined
			Compatibility:	5014
		'''
		self.write('source%d:phase %f' % (ch, phase))
	

	#####################################################################################
	## AWG5014 functions: SEQUENCE
	#####################################################################################
	def setSeqLength(self, length):
		''' length:		0..max. Allocates new, or trims existing sequence
		'''
		self.write('sequence:length %d' % length)
	
	
	def setSeqElemLoopInfiniteOn(self, element):
		''' element:		1..length
		'''
		self.write('sequence:element%d:loop:infinite on' % element)
	
	def setSeqElemWaveform(self, element, ch, name):
		''' element:		1..length
			Compatibility:	5014, QWG
		'''
		self.write('sequence:element%d:waveform%d "%s"' % (element, ch, name))
	
	#####################################################################################
	## AWG5014 functions: WLIST (Waveform list)
	#####################################################################################
	def getWlistSize(self):
		return self.askDouble('wlist:size?')
		
	def getWlistName(self, idx):
		'''	idx:			0..size-1
		'''
		return self.ask('wlist:name? %d' % idx)
	
	def getWlist(self):
		''' NB: takes a few seconds on 5014: our fault or Tek's?
		'''
		size = self.getWlistSize()
		wlist = []									# empty list
		for k in range(size):						# build list of names
			wlist.append(self.getWlistName(k))	
		return wlist
	
	def deleteWaveform(self, name):
		'''	name:		waveform name excluding double quotes, e.g. 'test'
			Compatibility:	5014, QWG
		'''
		self.write('wlist:waveform:delete "%s"' % name)
		
		'''	Compatibility:	5014, QWG
		'''
	def deleteWaveformAll(self):
		self.write('wlist:waveform:delete all')
	
	def getWaveformType(self, name):
		'''	name:		waveform name excluding double quotes, e.g. '*Sine100'
			Returns:	'INT' or 'REAL'
		'''
		return self.ask('wlist:waveform:type? "%s"' % name)
	
	def getWaveformLength(self, name):
		'''	name:		waveform name excluding double quotes, e.g. '*Sine100'
		'''
		return self.askDouble('wlist:waveform:length? "%s"' % name)
	
	def newWaveformReal(self, name, len):
		'''	name:		waveform name excluding double quotes, e.g. 'test'
			NB: seems to do nothing if waveform already exists
		'''
		self.write('wlist:waveform:new "%s",%d,real' % (name, len))
	
	def getWaveformData(self, name):
		'''	
			Input:
				name:		string				waveform name excluding double quotes, e.g. '*Sine100'
			Output:	
				tuple containing lists: (waveform, marker1, marker2)

			Compatibility:	5014, QWG

			Funny old Matlab timing results:
				tic;[waveform,marker1,marker2] = awg.getWaveformData('*Sine100');toc
				Elapsed time is 0.265559 seconds.
				tic;[waveform,marker1,marker2] = awg.getWaveformData('*Sine1000');toc
				Elapsed time is 0.101930 seconds.
				tic;[waveform,marker1,marker2] = awg.getWaveformData('*Sine3600');toc
				Elapsed time is 0.056023 seconds.
		'''
		self.write('wlist:waveform:data? "%s"' % name)							# response starts with header, e.g. '#3500'		
		binBlock = self.binBlockRead()
		# extract waveform and markers
		waveformLen = len(binBlock)/5 											# 5 bytes per record
		waveform = []
		marker1 = []
		marker2 = []
		for k in range(waveformLen):
			(waveform[i], markers) = struct.unpack(binBlock, '<fB')
			marker1[i] = markers & 0x01
			marker2[i] = markers>>1 & 0x01

		return (waveform, marker1, marker2)
	
	
	def sendWaveformDataReal(self, name, waveform, marker1, marker2):
		'''	send waveform and markers directly to AWG memory, i.e. not to a file on the AWG disk.
			NB: uses real data normalized to the range from -1 to 1 (independent of number of DAC bits of AWG)

			Input:
				name 		string				waveform name excluding double quotes, e.g. 'test'. Must already exits in AWG
				waveform 	float[numpoints]	vector defining the waveform, normalized between -1.0 and 1.0
				marker1 	int[numpoints]		vector of 0 and 1 defining the first marker
				marker2 	int[numpoints]		vector of 0 and 1 defining the second marker
		
			Compatibility:	5014, QWG
			Based on:
				Tektronix_AWG5014.py::send_waveform, which sends data to an AWG _file_, not a memory waveform
				'awg_transferRealDataWithMarkers', Author = Stefano Poletto, Compatibility = Tektronix AWG5014, AWG7102

		'''

		# parameter handling
		if len(marker1)==0 and len(marker2)==0:									# no marker data
			m = np.zeros(len(waveform))
		else:
			if (not((len(waveform) == len(marker1)) and ((len(marker1) == len(marker2))))):
				raise UserWarning('length mismatch between markers/waveform')
			# prepare markers
			m = marker1 + numpy.multiply(marker2, 2)
			m = int(numpy.round(m[i], 0))

		# FIXME: check waveform amplitude and marker values (if paranoid)
	

		# generate the binblock
		binBlock = b''
		for i in range(len(waveform)):
			binBlock = binBlock + struct.pack('<fB', waveform[i], int(m[i]))
		
		# write binblock
		hdr = 'wlist:waveform:data "{}",'.format(name)
		self.binBlockWrite(binBlock, hdr)

	
	def createWaveformReal(self, name, waveform, marker1, marker2):
		''' convenience function to create a waveform in the AWG and then send data to it
		'''
		waveLen = len(waveform);
#		if self.paranoid:
			# check waveform is there, problems might arise if it already existed
		self.newWaveformReal(name, waveLen)		
		self.sendWaveformDataReal(name, waveform, marker1, marker2)


	#####################################################################################
	## AWG5014 functions: MMEM (Mass Memory)
	#####################################################################################

#	None at the moment

	#####################################################################################
	## Generic (i.e. at least AWG520 and AWG5014) Tektronix AWG functions
	#####################################################################################

	## Tek_AWG functions: menu Setup|Waveform/Sequence 
	def loadWaveformOrSequence(self, awgFileName):
		''' awgFileName:		name referring to AWG file system
		'''
		self.write('source:def:user "%s"' % awgFileName)		# NB: we only support default Mass Storage Unit Specifier "Main", which is the internal harddisk


	## Tek_AWG functions: Button interface
	def run(self):
		self.write('awgcontrol:run:immediate')
	
	def stop(self):
		self.write('awgcontrol:stop:immediate')


	## functions: menu Setup|Horizontal 
	def setClockRefInternal(self):
		self.write(':roscillator:source internal')
	
	def setClockRefExternal(self):
		self.write(':roscillator:source external');
	
	## functions: menu Setup|Horizontal 
	def setClockSourceInternal(self):
		self.write('awgcontrol:clock:source internal')
	
	def setClockSourceExternal(self):
		self.write('awgcontrol:clock:source external')
	   
	def setClockFrequency(self, frequency):
		''' frequency:      AWG520:     TBD
							AWG5014:    10 MHz..10GHz (FIXME: TBC)
		'''
		self.write('source1:frequency %f' % frequency)
	
	def setTriggerInterval(self, interval):
		''' interval:           AWG520: 1.0 us to 10.0 s.
		'''
		self.write('trigger:timer %f' % interval)
	


	##########################################################################
	## Generic AWG functions also implemented as Parameter
	## to be deprecated in the future
	##########################################################################


	# NB: functions are organised by their appearance in the AWG520 user interface
	## functions: menu Setup|Vertical 
	def setOffset(self, ch, offset):
		''' ch:             AWG520: 1,2 (and 7,8 see documentation)
			offset:         AWG520: -1.000V to +1.000V in 1 mV steps
		'''
		self.write('source%d:voltage:level:immediate:offset %f' % (ch, offset))
			
	def getOffset(self, ch):
		''' ch:             AWG520: 1,2 (and 7,8 see documentation)
		'''
		return self.askDouble('source%d:voltage:level:immediate:offset?' % ch)
			
	## functions: menu Setup|Vertical 
	def setAmplitude(self, ch, amplitude):
		''' ch:             AWG520: 1,2 (and 7,8 see documentation)
			amplitude:      AWG520: 0.020Vpp to 2.000Vpp in 1 mV steps
		'''
		self.write('source%d:voltage:level:immediate:amplitude %f' % (ch, amplitude))
	
	def getAmplitude(self, ch):
		''' ch:             AWG520: 1,2 (and 7,8 see documentation)
		'''
		return self.askDouble('source%d:voltage:level:immediate:amplitude?' % ch)
	
	def setMarkerVoltageLow(self, ch, marker, voltage):
		''' ch:             AWG520: 1,2
			marker:         AWG520: 1,2
			amplitude:      AWG520: -2V to 2V in 0.05V steps
		'''
		self.write('source%d:marker%d:voltage:high %f' % (ch, marker, voltage))
	
	def setMarkerVoltageHigh(self, ch, marker, voltage):
		''' ch:             AWG520: 1,2
			marker:         AWG520: 1,2
			amplitude:      AWG520: -2V to 2V in 0.05V steps
		'''
		self.write('source%d:marker%d:voltage:high %f' % (ch, marker, voltage))
	
	
	## functions: Button interface
	def setOutputStateOn(self, ch):
		''' ch:             AWG520: 1,2 (and 7 see documentation)
			NB: only works if waveform is defined or def Generator is on
		'''
		self.write('output%d:state on' % ch)
	
	def setOutputStateOff(self, ch):
		''' ch:             AWG520: 1,2 (and 7 see documentation)
		'''
		self.write('output%d:state off' % ch);

	## functions: menu Setup|Trigger
	def setTriggerSourceInternal(self):
		self.write('trigger:source internal')

	def setTriggerSourceExternal(self):
		self.write('trigger:source external')

	def setTriggerSlopePositive(self):
		self.write('trigger:slope positive')
	
	def setTriggerSlopeNegative(self):
		self.write('trigger:slope negative')
	
	def setTriggerLevel(self, level):
		''' level:              AWG520: -5.0 V to +5.0 V, in 0.1 V steps
		'''
		self.write('trigger:level %f' % level)

	def setTriggerImpedance50ohm(self):
		self.write('trigger:impedance 50')
	

	## functions: menu Setup|Run Mode
	def setRunModeEnhanced(self):
		''' NB: note that for AWG5014, 'enhanced' is identical to 'sequence'
		'''
		self.write('awgcontrol:rmode enhanced') 
	
	def setRunModeSequence(self):
		self.write('awgcontrol:rmode seq')
	
	def setRunModeContinuous(self):
		self.write('awgcontrol:rmode cont')
	

# FIXME: old
#	def ReadPatFile(self)
#		fid = fopen('monoplexer-testsetup-datasheets\AWG\Nulti_520Ch10001.pat');
#		fileType = fscanf(fid, 'MAGIC %d\n');	 # AWG500/600 series: 2000
#		digitCnt = fscanf(fid, '#%1d');
#		formatString = sprintf('%%%dd', digitCnt);
#		byteCnt = fscanf(fid, formatString);	# FIXME: use getByteCntFromHeader()
#		data = fread(fid, byteCnt/2, 'uint16');
#		% FIXME: more follows: CLOCK
#		clockFreq = fscanf(fid, 'CLOCK %d\n')
#		fclose(fid);
#
#		plot(bitand(data,1023)-512, 'r')		# AWG520 has 10 bit DACs
#		% bit 13 = marker 1, bit 14 is marker 2
	

