"""
Base detector functions, i.e. Detector_Function and its first order descendents
extracted from pycqed/measurement/detector_functions.py commit 0da380ad2adf2dc998f5effef362cdf264b87948
"""

import numpy as np


class Detector_Function(object):

    '''
    Detector_Function class for MeasurementControl
    '''

    def __init__(self, **kw):
        self.name = self.__class__.__name__
        self.detector_control = ''
        self.set_kw()
        self.value_names = ['val A', 'val B']
        self.value_units = ['arb. units', 'arb. units']

        self.prepare_function = None
        self.prepare_function_kwargs = None

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s' % (key, kw[key]))

    def arm(self):
        """
        Ensures acquisition instrument is ready to measure on first trigger.
        """
        pass

    def get_values(self):
        pass

    def prepare(self, **kw):
        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

    def set_prepare_function(self,
                             prepare_function,
                             prepare_function_kwargs: dict = dict()):
        """
        Set an optional custom prepare function.

        prepare_function: function to call during prepare
        prepare_function_kwargs: keyword arguments to be passed to the
            prepare_function.

        N.B. Note that not all detectors support a prepare function and
        the corresponding keywords.
        Detectors that do not support this typicaly ignore these attributes.
        """
        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs

    def finish(self, **kw):
        pass


class Mock_Detector(Detector_Function):
    def __init__(
            self,
            value_names=['val'],
            value_units=['arb. units'],
            detector_control='soft',
            mock_values=np.zeros([20, 1]),
            **kw
    ):
        self.name = self.__class__.__name__
        self.set_kw()
        self.value_names = value_names
        self.value_units = value_units
        self.detector_control = detector_control
        self.mock_values = mock_values
        self._iteration = 0

    def acquire_data_point(self, **kw):
        '''
        Returns something random for testing
        '''
        idx = self._iteration % (np.shape(self.mock_values)[0])
        self._iteration += 1
        return self.mock_values[idx]

    def get_values(self):
        return self.mock_values

    def prepare(self, **kw):
        pass

    def finish(self, **kw):
        pass


class Multi_Detector(Detector_Function):
    """
    Combines several detectors of the same type (hard/soft) into a single
    detector.
    """

    def __init__(
            self,
            detectors: list,
            detector_labels: list = None,
            det_idx_prefix: bool = True,
            **kw
    ):
        """
        detectors     (list):
            a list of detectors to combine.
        det_idx_prefix(bool):
            if True prefixes the value names with
        detector_labels (list):
            if not None, will be used instead instead of
            "det{idx}_" as a prefix for the different channels
        """
        self.detectors = detectors
        self.name = 'Multi_detector'
        self.value_names = []
        self.value_units = []
        for i, detector in enumerate(detectors):
            for detector_value_name in detector.value_names:
                if det_idx_prefix:
                    if detector_labels is None:
                        val_name = 'det{} '.format(i) + detector_value_name
                    else:
                        val_name = detector_labels[i] + \
                            ' ' + detector_value_name
                else:
                    val_name = detector_value_name
                self.value_names.append(val_name)
            for detector_value_unit in detector.value_units:
                self.value_units.append(detector_value_unit)

        self.detector_control = self.detectors[0].detector_control
        for d in self.detectors:
            if d.detector_control != self.detector_control:
                raise ValueError('All detectors should be of the same type')

    def prepare(self, **kw):
        for detector in self.detectors:
            detector.prepare(**kw)

    def set_prepare_function(self,
                             prepare_function,
                             prepare_function_kw: dict = dict(),
                             detectors: str = 'all'):
        """
        Set an optional custom prepare function.

        prepare_function: function to call during prepare
        prepare_function_kw: keyword arguments to be passed to the
            prepare_function.
        detectors :  |"all"|"first"|"last"|
            sets the prepare function to "all" child detectors, or only
            on the "first" or "last"

        The multi detector passes the arguments to the set_prepare_function
        method of all detectors it contains.
        """
        if detectors == "all":
            for detector in self.detectors:
                detector.set_prepare_function(
                    prepare_function, prepare_function_kw)
        elif detectors == 'first':
            self.detectors[0].set_prepare_function(
                prepare_function, prepare_function_kw)
        elif detectors == 'last':
            self.detectors[-1].set_prepare_function(
                prepare_function, prepare_function_kw)

    def set_child_attr(self, attr, value, detectors: str = 'all'):
        """
        Set an attribute of child detectors.

        attr (str): the attribute to set
        value   : the value to set the attribute to

        detectors :  |"all"|"first"|"last"|
            sets the attribute on "all" child detectors, or only
            on the "first" or "last"
        """
        if detectors == "all":
            for detector in self.detectors:
                setattr(detector, attr, value)
        elif detectors == 'first':
            setattr(self.detectors[0], attr, value)
        elif detectors == 'last':
            setattr(self.detectors[-1], attr, value)

    def get_values(self):
        values_list = []
        for detector in self.detectors:
            detector.arm()
        for detector in self.detectors:
            new_values = detector.get_values()
            values_list.append(new_values)
        values = np.concatenate(values_list)
        return values

    def acquire_data_point(self):
        # N.B. get_values and acquire_data point are virtually identical.
        # the only reason for their existence is a historical distinction
        # between hard and soft detectors that leads to some confusing data
        # shape related problems, hence the append vs concatenate
        values = []
        for detector in self.detectors:
            new_values = detector.acquire_data_point()
            values = np.append(values, new_values)
        return values

    def finish(self):
        for detector in self.detectors:
            detector.finish()


class None_Detector(Detector_Function):

    def __init__(self, **kw):
        super(None_Detector, self).__init__()
        self.detector_control = 'soft'
        self.set_kw()
        self.name = 'None_Detector'
        self.value_names = ['None']
        self.value_units = ['None']

    def acquire_data_point(self, **kw):
        '''
        Returns something random for testing
        '''
        return np.random.random()


class Soft_Detector(Detector_Function):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.detector_control = 'soft'

    def acquire_data_point(self, **kw):
        return np.random.random()

    def prepare(self, sweep_points=None):
        pass


class Hard_Detector(Detector_Function):

    def __init__(self, **kw):
        super().__init__()
        self.detector_control = 'hard'

    def prepare(self, sweep_points=None):
        pass

    def finish(self):
        pass
