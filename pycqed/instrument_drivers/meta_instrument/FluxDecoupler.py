import numpy as np
from qcodes import Instrument
from qcodes.utils import validators as vals
from pycqed.instrument_drivers.pq_parameters import InstrumentParameter


class FluxDecoupler(Instrument):
    """
    Decouples a system of flux sources $\vec{V}$ (qubits' individual fluxlines 
    or global coils) at the specified target points (qubits).

    Each flux source and target can be inactivated using the `active_<name>` 
    parameter. Inactive flux sources are not controlled and the fluxes at the 
    inactive flux targets are ignored. The number of active flux sources should
    always equal the number of active flux targets.
    """

    def __init__(self, name, dc_source=None, **kw):
        super().__init__(name, **kw)

        self.add_parameter('dc_source', parameter_class=InstrumentParameter)
        if dc_source is not None:
            self.dc_source(dc_source.name)

        self._couplings = np.zeros((0, 0), dtype=np.float)
        self._offsets = np.zeros((0,), dtype=np.float)

        self._target_names = []
        self._target_active = []
        self._source_names = []
        self._source_active = []

    def add_target(self, name):
        if name in self._target_names:
            raise KeyError(
                "A flux target with name '{}' already exists".format(name))
        elif name in self._source_names:
            raise KeyError(
                "A flux source with name '{}' already exists".format(name))
        i = len(self._target_names)
        self._target_names.append(name)
        self._target_active.append(True)
        self._offsets = np.append(self._offsets, [0])
        self._couplings = np.vstack((self._couplings,
                                     np.zeros((1, self._couplings.shape[1]))))
        self.add_parameter('offset_{}'.format(name),
                           vals=vals.Numbers(),
                           get_cmd=(lambda self=self, i=i: self._offsets[i]),
                           set_cmd=(lambda x, self=self, i=i:
                               self._offsets.__setitem__(i, x)))
        for j, src_name in enumerate(self._source_names):
            self.add_parameter('c_{}_{}'.format(name, src_name), unit='1/V',
                               vals=vals.Numbers(),
                               get_cmd=(lambda self=self, i=i, j=j:
                                   self._couplings[i, j]),
                               set_cmd=(lambda x, self=self, i=i, j=j:
                                   self._couplings.__setitem__((i, j), x)))
        self.add_parameter('active_{}'.format(name), vals=vals.Bool(),
                           get_cmd=(lambda self=self, i=i:
                               self._target_active[i]),
                           set_cmd=(lambda x, self=self, i=i:
                               self._target_active.__setitem__(i, x)))
        self.add_parameter('flux_{}'.format(name), vals=vals.Numbers(),
                           get_cmd=(lambda self=self, name=name:
                               self.get_fluxes()[name]),
                           set_cmd=(lambda x, self=self, name=name:
                               self.set_fluxes({name: x})))

    def add_source(self, name):
        """
        The name of the source needs to match the name of the module on the
        DC source
        """
        if name in self._target_names:
            raise KeyError(
                "A flux target with name '{}' already exists".format(name))
        elif name in self._source_names:
            raise KeyError(
                "A flux source with name '{}' already exists".format(name))
        j = len(self._source_names)
        self._source_names.append(name)
        self._source_active.append(True)
        self._couplings = np.hstack((self._couplings,
                                     np.zeros((self._couplings.shape[0], 1))))
        for i, trg_name in enumerate(self._target_names):
            self.add_parameter('c_{}_{}'.format(trg_name, name), unit='1/V',
                               vals=vals.Numbers(),
                               get_cmd=(lambda self=self, i=i, j=j:
                                   self._couplings[i, j]),
                               set_cmd=(lambda x, self=self, i=i, j=j:
                                   self._couplings.__setitem__((i, j), x)))
        self.add_parameter('active_{}'.format(name), vals=vals.Bool(),
                           get_cmd=(lambda self=self, j=j:
                               self._source_active[j]),
                           set_cmd=(lambda x, self=self, j=j:
                               self._source_active.__setitem__(j, x)))

    def set_active_sources(self, sources):
        self._source_active = [name in sources for name in self._source_names]

    def get_active_sources(self):
        return [name for name, active in
                zip(self._source_names, self._source_active) if active]

    def set_active_targets(self, targets):
        self._target_active = [name in targets for name in self._target_names]

    def get_active_targets(self):
        return [name for name, active in
                zip(self._target_names, self._target_active) if active]

    def get_fluxes(self):
        dc_source = self.dc_source.get_instr()
        source_fluxes = [dc_source.get('volt_{}'.format(name))
                         for name in self._source_names]
        target_fluxes = self._couplings.dot(source_fluxes) + self._offsets
        return dict(zip(self._target_names, target_fluxes))

    def set_fluxes(self, fluxes):
        dc_source = self.dc_source.get_instr()
        all_trg_fluxes = self.get_fluxes()
        all_trg_fluxes.update(fluxes)
        active_couplings = self.get_active_couplings()
        active_trg_fluxes = np.array([all_trg_fluxes[name]
            for name, active in zip(self._target_names, self._target_active)
            if active])
        active_trg_offsets = self._offsets[self._target_active]
        active_src_fluxes = np.linalg.inv(active_couplings).dot(
            active_trg_fluxes - active_trg_offsets)
        active_src_names = [name
            for name, active in zip(self._source_names, self._source_active)
            if active]
        dc_source.set_smooth(dict(zip(active_src_names, active_src_fluxes)))

    def get_active_couplings(self):
        return self._couplings[self._target_active].T[self._source_active].T
