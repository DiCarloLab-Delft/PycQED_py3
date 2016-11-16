import logging
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.detector_functions as det


class ASM_Sweep(swf.Hard_Sweep):

    def __init__(self, filename, CBox, upload=True):
        super().__init__()
        self.name = 'ASM'
        # self.parameter_name = 'tau'
        # self.unit = 's'
        self.filename = filename
        self.upload = upload
        self.CBox = CBox

    def prepare(self, **kw):
        if self.upload:
            self.CBox.AWG0_mode('Codeword-trigger mode')
            self.CBox.AWG1_mode('Codeword-trigger mode')
            self.CBox.AWG2_mode('Codeword-trigger mode')
            # self.CBox.set_master_controller_working_state(0, 0, 0)
            self.CBox.core_state('idle')
            self.CBox.acquisition_mode('idle')
            self.CBox.trigger_source('internal')
            self.CBox.demodulation_mode('double')
            self.CBox.load_instructions(self.filename)

            # self.CBox.set_master_controller_working_state(1, 0, 0)


class CBox_integrated_average_detector_CC(det.Hard_Detector):

    def __init__(self, CBox, seg_per_point=1, normalize=False, rotate=False,
                 nr_averages=1024, integration_length=1e-6, **kw):
        '''
        Integration average detector.
        Defaults to averaging data in a number of segments equal to the
        nr of sweep points specificed.

        seg_per_point allows you to use more than 1 segment per sweeppoint.
        this is for example useful when doing a MotzoiXY measurement in which
        there are 2 datapoints per sweep point.
        Normalize/Rotate adds a third measurement with the rotated/normalized data.
        '''
        super().__init__(**kw)
        self.CBox = CBox
        self.name = 'CBox_integrated_average_detector_CC'
        self.value_names = ['I', 'Q']
        self.value_units = ['a.u.', 'a.u.']
        self.seg_per_point = seg_per_point
        self.rotate = rotate
        self.normalize = normalize
        self.cal_points = kw.get('cal_points', None)
        self.nr_averages = nr_averages
        self.integration_length = integration_length

    def get_values(self):
        succes = False
        i = 0
        while not succes:
            try:
                self.CBox.set('run_mode', 0)
                self.CBox.set('acquisition_mode', 'idle')
                self.CBox.set('acquisition_mode', 'integration averaging')
                self.CBox.core_state('active')
                self.CBox.set('run_mode', 1)
                # does not restart AWG tape in CBox as we don't use it anymore
                data = self.CBox.get_integrated_avg_results()
                succes = True
            except Exception as e:
                logging.warning('Exception caught retrying')
                logging.warning(e)
                self.CBox.set('acquisition_mode', 'idle')
                self.AWG.stop()

                self.CBox.set('acquisition_mode', 'integration averaging')

            i += 1
            if i > 20:
                break
        if self.rotate or self.normalize:
            return self.rotate_and_normalize(data)
        else:
            return data

    def prepare(self, sweep_points):
        self.CBox.set('nr_samples', self.seg_per_point*len(sweep_points))
        self.CBox.nr_averages(int(self.nr_averages))
        self.CBox.integration_length(int(self.integration_length/(5e-9)))
        self.CBox.set('acquisition_mode', 'idle')
        self.CBox.set('acquisition_mode', 'integration averaging')

    def finish(self):
        self.CBox.set('acquisition_mode', 'idle')