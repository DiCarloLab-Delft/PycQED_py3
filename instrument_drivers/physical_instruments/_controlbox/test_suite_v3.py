import unittest
import numpy as np
from . import defHeaders_CBox_v3 as defHeaders
from . import test_suite


class CBox_tests_v3(test_suite.CBox_tests):
    def __init__(self, *args, **kwargs):
        super(test_suite.CBox_tests, self).__init__(*args, **kwargs)
        self.loadSaveDataFile = False

    def LoadSavedData(self):
        if(not self.loadSaveDataFile):
            try:
                DataFile = np.load("SaveData.npz")
                self.loadSaveDataFile = True

                self.SavedInputAvgRes0 = DataFile['SavedInputAvgRes0']

                self.SavedIntLogResult0_8 = DataFile['SavedIntLogResult0_8']
                self.SavedIntLogResult1_8 = DataFile['SavedIntLogResult1_8']

                self.SavedIntLogResult0_200 = \
                    DataFile['SavedIntLogResult0_200']
                self.SavedIntLogResult1_200 = \
                    DataFile['SavedIntLogResult1_200']

                self.SavedIntAvgResult0 = DataFile['SavedIntAvgResult0']
                self.SavedIntAvgResult1 = DataFile['SavedIntAvgResult1']

                self.SavedCh0Counters = DataFile['SavedCh0Counters']
                self.SavedCh0Result = DataFile['SavedCh0Result']
                self.SavedCh1Result = DataFile['SavedCh1Result']
                self.SavedCh1Counters = DataFile['SavedCh1Counters']
                self.SavedTimingtapeResult0 = \
                    DataFile['SavedTimingtapeResult0']
                self.SavedTimingtapeResult7 = \
                    DataFile['SavedTimingtapeResult7']
            except:
                print("Cannot open the saved data file.")
                self.loadSaveDataFile = False
                self.assertTrue(False)

    def test_firmware_version(self):
        v = self.CBox.get('firmware_version')
        print(v)
        self.assertTrue(int(v[1]) == 3)  # major version
        self.assertTrue(int(v[3]) == 2)  # minor version

    def test_setting_mode(self):
        # test acquisition_mode
        for i in range(len(defHeaders.acquisition_modes)):
            self.CBox.set('acquisition_mode', i)
            self.assertEqual(self.CBox.get('acquisition_mode'),
                             defHeaders.acquisition_modes[i])
        # test setting core state to 'active'
        self.CBox.set('core_state', 'active')
        self.assertEqual(self.CBox.get('core_state'),
                         defHeaders.core_states[1])
        # test run mode
        for i in range(len(defHeaders.run_modes)):
            self.CBox.set('run_mode', i)
            self.assertEqual(self.CBox.get('run_mode'),
                             defHeaders.run_modes[i])
        # test setting core state to 'idle'
        self.CBox.set('core_state', 'idle')
        self.assertEqual(self.CBox.get('core_state'),
                         defHeaders.core_states[0])

        # test trigger source
        for i in range(3):
            self.CBox.set('trigger_source', i)
            self.assertEqual(self.CBox.get('trigger_source'),
                             defHeaders.trigger_sources[i])
        # test awg mode
        for j in range(3):
            for i in range(len(defHeaders.awg_modes)):
                self.CBox.set('AWG{}_mode'.format(j), i)
                self.assertEqual(self.CBox.get('AWG{}_mode'.format(j)),
                                 defHeaders.awg_modes[i])

        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.set('acquisition_mode', 'idle')
        for j in range(3):
            self.CBox.set('AWG{}_mode'.format(j), 0)

    def test_input_avg_mode(self):
        self.LoadSavedData()
        # Perform input average and plot the result
        sine_numbers = 8
        waveLength = 120
        sine_waves = [[0]*(waveLength+1) for i in range(sine_numbers)]
        cosine_waves = [[0]*(waveLength+1) for i in range(sine_numbers)]

        for sin_nr in range(sine_numbers):
            for sample_nr in range(waveLength):
                sine_waves[sin_nr][sample_nr] = \
                    min(2**13-1, np.floor(6*1024*np.sin(
                        sample_nr/float(waveLength)*2*np.pi*(sin_nr+1))))
                cosine_waves[sin_nr][sample_nr] = \
                    min(2**13-1, np.floor(6*1024*np.cos(
                        sample_nr/float(waveLength)*2*np.pi*(sin_nr+1))))

            for awg_nr in range(3):
                for dac_nr in range(2):
                    for pulse_nr in range(8):
                            self.CBox.set_awg_lookuptable(
                                 awg_nr, pulse_nr, dac_nr,
                                 sine_waves[pulse_nr],
                                 units='dac', length=waveLength)

        self.CBox.set('acquisition_mode', 'idle')    # set to idle state
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.load_instructions('programs\\input_avg.asm')
        self.CBox.set_master_controller_working_state(1, 0, 0)

        self.CBox.set('run_mode', 1)
        NoSamples = 400
        self.CBox.set("nr_samples", NoSamples)
        self.CBox.set('nr_averages', 2**4)
        self.CBox.set('signal_delay', 0)
        self.CBox.set('acquisition_mode', 'input averaging')
        [InputAvgRes0, InputAvgRes1] = self.CBox.get_input_avg_results()
        self.assertTrue(self.Appx_Cmp_Wave(InputAvgRes0,
                                           self.SavedInputAvgRes0))

    def test_integration_logging_mode(self):
        self.LoadSavedData()
        # initalizing waveform LUT in awgs
        triggerlength = 2
        plane = [-6*1024]*(triggerlength)
        for awg_nr in range(3):
            for dac_nr in range(2):
                for pulse_nr in range(8):
                    self.CBox.set_awg_lookuptable(awg_nr, pulse_nr, dac_nr,
                                                  plane, units='dac',
                                                  length=triggerlength-1)

        # Setting parameters.
        integration_length = 200
        nr_samples = 9
        self.CBox.set('signal_delay', 0)
        self.CBox.set('integration_length', integration_length)
        self.CBox.set('nr_averages', 2**4)
        self.CBox.set('nr_samples', nr_samples)
        self.CBox.set('lin_trans_coeffs', [1, 0, 0, 1])
        self.CBox.set('adc_offset', -1)
        weights0 = 1*np.ones(512)
        weights1 = 2*np.ones(512)
        self.CBox.set('sig0_integration_weights', weights0)
        self.CBox.set('sig1_integration_weights', weights1)

        self.CBox.set('log_length', 8)
        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.load_instructions('programs\\int_log.txt')
        self.CBox.set_master_controller_working_state(1, 0, 0)
        self.CBox.set('acquisition_mode', 'integration logging')
        self.CBox.set('run_mode', 1)
        [IntLogResult0_8, IntLogResult1_8] = \
            self.CBox.get_integration_log_results()

        weights1 = 0*np.ones(512)
        self.CBox.set('sig1_integration_weights', weights1)
        self.CBox.set('log_length', 200)

        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.load_instructions('programs\\int_log.txt')
        self.CBox.set_master_controller_working_state(1, 0, 0)
        self.CBox.set('acquisition_mode', 'integration logging')
        self.CBox.set('run_mode', 1)
        [IntLogResult0_200, IntLogResult1_200] = \
            self.CBox.get_integration_log_results()
        self.CBox.set('acquisition_mode', 'idle')

        self.assertTrue(self.Appx_Cmp_Wave(IntLogResult0_8,
                                           self.SavedIntLogResult0_8))
        self.assertTrue(self.Appx_Cmp_Wave(IntLogResult1_8,
                                           self.SavedIntLogResult1_8))
        self.assertTrue(self.Appx_Cmp_Wave(IntLogResult0_200,
                                           self.SavedIntLogResult0_200))
        self.assertTrue(self.Acc_Cmp_Wave(IntLogResult1_200,
                                          self.SavedIntLogResult1_200))

    def test_integration_average_mode(self):
        self.LoadSavedData()

        triggerlength = 20
        plane = [-6*1024]*(triggerlength)
        for awg_nr in range(3):
            for dac_nr in range(2):
                for pulse_nr in range(8):
                    self.CBox.set_awg_lookuptable(awg_nr, pulse_nr, dac_nr,
                                                  plane, units='dac',
                                                  length=triggerlength-1)

        # load instructions
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.load_instructions('programs\\int_avg.txt')

        # Set the parameters
        integration_length = 500
        nr_samples = 9
        self.CBox.set('signal_delay', 0)
        self.CBox.set('integration_length', integration_length)
        self.CBox.set('nr_averages', 2**4)
        self.CBox.set('nr_samples', nr_samples)
        self.CBox.set('lin_trans_coeffs', [1, 0, 0, 1])
        self.CBox.set('adc_offset', 0)

        # set the integration weights
        weights0 = 1*np.ones(512)
        # weights1 = np.zeros(512)
        weights1 = -1*np.ones(512)
        self.CBox.set('sig0_integration_weights', weights0)
        self.CBox.set('sig1_integration_weights', weights1)

        # Perform integration average and plot the result
        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.set_master_controller_working_state(1, 0, 0)
        self.CBox.set('acquisition_mode', 'integration averaging')
        self.CBox.set('run_mode', 1)
        [IntAvgRst0, IntAvgRst1] = self.CBox.get_integrated_avg_results()
        self.CBox.set('acquisition_mode', 'idle')
        self.assertTrue(self.Appx_Cmp_Wave(IntAvgRst0,
                                           self.SavedIntAvgRst0))
        self.assertTrue(self.Appx_Cmp_Wave(IntAvgRst1,
                                           self.SavedIntAvgRst1))

    def test_qubit_state_log_mode(self):
        self.LoadSavedData()
        # initalizing waveform LUT in awgs
        triggerlength = 20
        plane = [-6*1024]*(triggerlength)
        for awg_nr in range(3):
            for dac_nr in range(2):
                for pulse_nr in range(8):
                    self.CBox.set_awg_lookuptable(awg_nr, pulse_nr, dac_nr,
                                                  plane, units='dac',
                                                  length=triggerlength-1)
        # load instructions
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.load_instructions('programs\\QubitStateLog.asm')

        # Set the parameters
        integration_length = 500
        self.CBox.set('log_length', 100)
        self.CBox.set('signal_delay', 0)
        self.CBox.set('integration_length', integration_length)
        self.CBox.set('lin_trans_coeffs', [1, 0, 0, 1])
        self.CBox.set('adc_offset', 0)
        self.CBox.set('sig0_threshold_line', 10000)
        self.CBox.set('sig1_threshold_line', 1000000)

        # set the integration weights
        weights0 = 1*np.ones(512)
        weights1 = -1*np.ones(512)
        self.CBox.set('sig0_integration_weights', weights0)
        self.CBox.set('sig1_integration_weights', weights1)

        # Perform integration average and plot the result
        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.set_master_controller_working_state(1, 0, 0)
        self.CBox.set('acquisition_mode', 'integration logging')
        self.CBox.set('run_mode', 1)
        [ch0_counters, ch1_counters] = self.CBox.get_qubit_state_log_counters()
        [ch0_result, ch1_result] = self.CBox.get_qubit_state_log_results()
        self.CBox.set('acquisition_mode', 'idle')
        self.assertEqual(ch0_counters, self.SavedCh0Counters)
        self.assertEqual(ch1_counters, self.SavedCh1Counters)
        self.assertEqual(ch0_result, self.SavedCh0Result)
        self.assertEqual(ch1_result, self.SavedCh1Result)

    def test_tape_mode(self):
        self.LoadSavedData()
        self.loadSineWaves(100)

        for i in range(7):
            tape = []
            tape.extend(self.CBox.create_timing_tape_entry(0, 7, False))
            for j in range(i+1):
                tape.extend(self.CBox.create_timing_tape_entry(i*10, i, False))
            tape.extend(self.CBox.create_timing_tape_entry(0, 7, True))
            print(tape)
            for awg_nr in range(3):
                self.CBox.set_conditional_tape(awg_nr, i, tape)

        tape = []
        for i in range(8):
            tape.extend(self.CBox.create_timing_tape_entry(0, 0, False))
            tape.extend(self.CBox.create_timing_tape_entry(0, i, True))

        for awg_nr in range(3):
            self.CBox.set_segmented_tape(awg_nr, tape)

        self.CBox.AWG0_mode.set('tape')
        self.CBox.AWG1_mode.set('tape')
        self.CBox.AWG2_mode.set('tape')

        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.load_instructions('programs\\TimingTapeTest0.asm')
        self.CBox.set_master_controller_working_state(1, 0, 0)

        NoSamples = 400
        self.CBox.set("nr_samples", NoSamples)
        self.CBox.set('nr_averages', 2**4)
        self.CBox.set('signal_delay', 0)
        self.CBox.set('acquisition_mode', 'input averaging')
        self.CBox.set('run_mode', 1)

        [TimingtapeResult0, _] = self.CBox.get_input_avg_results()

        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set_master_controller_working_state(0, 0, 0)
        self.CBox.load_instructions('programs\\TimingTapeTest7.asm')
        self.CBox.set_master_controller_working_state(1, 0, 0)

        NoSamples = 400
        self.CBox.set("nr_samples", NoSamples)
        self.CBox.set('nr_averages', 2**10)
        self.CBox.set('signal_delay', 0)
        self.CBox.set('acquisition_mode', 'input averaging')
        self.CBox.set('run_mode', 1)

        [TimingtapeResult7, _] = self.CBox.get_input_avg_results()

        assert(self.Appx_Cmp_Wave(self.SavedTimingtapeResult0,
                                  TimingtapeResult0))
        assert(self.Appx_Cmp_Wave(self.SavedTimingtapeResult7,
                                  TimingtapeResult7))

    def loadSineWaves(self, waveLength=120):
        sine_numbers = 8
        sine_waves = [[0]*(waveLength+1) for i in range(sine_numbers)]
        cosine_waves = [[0]*(waveLength+1) for i in range(sine_numbers)]

        for sin_nr in range(sine_numbers):
            for sample_nr in range(waveLength):
                sine_waves[sin_nr][sample_nr] = np.floor(
                    -6*1024*np.sin(
                        sample_nr/float(waveLength)*2*np.pi*(sin_nr+1)))
                cosine_waves[sin_nr][sample_nr] = np.floor(
                    -6*1024*np.cos(
                        sample_nr/float(waveLength)*2*np.pi*(sin_nr+1)))

        for awg_nr in range(3):
            for pulse_nr in range(8):
                self.CBox.set_awg_lookuptable(awg_nr, pulse_nr, 0,
                                              sine_waves[pulse_nr],
                                              units='dac', length=waveLength)
                self.CBox.set_awg_lookuptable(awg_nr, pulse_nr, 1,
                                              cosine_waves[pulse_nr],
                                              units='dac', length=waveLength)

    def Appx_Cmp_Wave(self, wave1, wave2, allowedDev=0.02):
        (dev, pos) = self.MaxDev(wave1, wave2)
        if (dev < allowedDev):
            return True
        else:
            for i in range(len(wave1)):
                if ((wave1[0] == 0 and np.abs(wave1[1]) > 100) or
                        (wave1[1] == 0 and np.abs(wave1[0]) > 100)):
                    return False

        return True

    def Acc_Cmp_Wave(self, wave1, wave2):
        self.assertEqual(len(wave1), len(wave2))
        return np.array_equal(wave1, wave2)

    def MaxDev(self, wave1, wave2):
        self.assertEqual(len(wave1), len(wave2))
        wave_len = len(wave1)

        deviation = 0
        pos = None

        max_wave = np.maximum(wave1, wave2)

        for i in range(wave_len):
            if (max_wave[i] != 0):
                dev = np.abs((wave1[i]-wave2[i])/max_wave[i])
                if (dev > deviation):
                    deviation = dev
                    pos = i

        return deviation, pos
