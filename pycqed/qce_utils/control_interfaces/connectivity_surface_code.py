# -------------------------------------------
# Module containing implementation of surface-code connectivity structure.
# -------------------------------------------
from dataclasses import dataclass, field
import warnings
from typing import List, Union, Dict, Tuple
from enum import Enum, unique, auto
from pycqed.qce_utils.definitions import SingletonABCMeta
from pycqed.qce_utils.custom_exceptions import ElementNotIncludedException
from pycqed.qce_utils.control_interfaces.intrf_channel_identifier import (
    IChannelIdentifier,
    IFeedlineID,
    IQubitID,
    IEdgeID,
    FeedlineIDObj,
    QubitIDObj,
    EdgeIDObj,
)
from pycqed.qce_utils.control_interfaces.intrf_connectivity_surface_code import (
    ISurfaceCodeLayer,
    IParityGroup,
    ParityType,
)
from pycqed.qce_utils.control_interfaces.intrf_connectivity import (
    IDeviceLayer
)


@unique
class FrequencyGroup(Enum):
    LOW = auto()
    MID = auto()
    HIGH = auto()


@dataclass(frozen=True)
class FrequencyGroupIdentifier:
    """
    Data class, representing (qubit) frequency group identifier.
    """
    _id: FrequencyGroup

    # region Class Properties
    @property
    def id(self) -> FrequencyGroup:
        """:return: Self identifier."""
        return self._id
    # endregion

    # region Class Methods
    def is_equal_to(self, other: 'FrequencyGroupIdentifier') -> bool:
        """:return: Boolean, whether other frequency group identifier is equal self."""
        return self.id == other.id

    def is_higher_than(self, other: 'FrequencyGroupIdentifier') -> bool:
        """:return: Boolean, whether other frequency group identifier is 'lower' than self."""
        # Guard clause, if frequency groups are equal, return False
        if self.is_equal_to(other):
            return False
        if self.id == FrequencyGroup.MID and other.id == FrequencyGroup.LOW:
            return True
        if self.id == FrequencyGroup.HIGH:
            return True
        return False

    def is_lower_than(self, other: 'FrequencyGroupIdentifier') -> bool:
        """:return: Boolean, whether other frequency group identifier is 'higher' than self."""
        # Guard clause, if frequency groups are equal, return False
        if self.is_equal_to(other):
            return False
        if self.is_higher_than(other):
            return False
        return True
    # endregion


@dataclass(frozen=True)
class DirectionalEdgeIDObj(EdgeIDObj, IEdgeID):
    """
    Data class, implementing IEdgeID interface.
    Overwrites __hash__ and __eq__ to make qubit-to-qubit direction relevant.
    """

    # region Class Methods
    def __hash__(self):
        """
        Sorts individual qubit hashes such that the order is NOT maintained.
        Making hash comparison independent of order.
        """
        return hash((self.qubit_id0.__hash__(), self.qubit_id1.__hash__()))

    def __eq__(self, other):
        if isinstance(other, DirectionalEdgeIDObj):
            # Edge is equal if they share the same qubit identifiers, order does not matter
            return other.__hash__() == self.__hash__()
        if isinstance(other, EdgeIDObj):
            warnings.warn(message=f"Comparing directional edge to non-directional edge returns False by default.")
            return False
        return False
    # endregion


@dataclass(frozen=True)
class ParityGroup(IParityGroup):
    """
    Data class, implementing IParityGroup interface.
    """
    _parity_type: ParityType = field(init=True)
    """X or Z type stabilizer."""
    _ancilla_qubit: IQubitID = field(init=True)
    """Ancilla qubit."""
    _data_qubits: List[IQubitID] = field(init=True)
    """Data qubits."""
    _edges: List[IEdgeID] = field(init=False)
    """Edges between ancilla and data qubits."""

    # region Interface Properties
    @property
    def parity_type(self) -> ParityType:
        """:return: Parity type (X or Z type stabilizer)."""
        return self._parity_type

    @property
    def ancilla_id(self) -> IQubitID:
        """:return: (Main) ancilla-qubit-ID from parity."""
        return self._ancilla_qubit

    @property
    def data_ids(self) -> List[IQubitID]:
        """:return: (All) data-qubit-ID's from parity."""
        return self._data_qubits

    @property
    def edge_ids(self) -> List[IEdgeID]:
        """:return: (All) edge-ID's between ancilla and data qubit-ID's."""
        return self._edges
    # endregion

    # region Interface Methods
    def contains(self, element: Union[IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of parity group or not."""
        if element in self.data_ids:
            return True
        if element in self.edge_ids:
            return True
        if element == self.ancilla_id:
            return True
        return False
    # endregion

    # region Class Methods
    def __post_init__(self):
        edges: List[IEdgeID] = [
            EdgeIDObj(
                qubit_id0=self.ancilla_id,
                qubit_id1=data_qubit_id,
            )
            for data_qubit_id in self.data_ids
        ]
        object.__setattr__(self, '_edges', edges)
    # endregion


@dataclass(frozen=True)
class FluxDanceLayer:
    """
    Data class, containing directional gates played during 'flux-dance' layer.
    """
    _edge_ids: List[IEdgeID]
    """Non-directional edges, part of flux-dance layer."""

    # region Class Properties
    @property
    def qubit_ids(self) -> List[IQubitID]:
        """:return: All qubit-ID's."""
        return list(set([qubit_id for edge in self.edge_ids for qubit_id in edge.qubit_ids]))

    @property
    def edge_ids(self) -> List[IEdgeID]:
        """:return: Array-like of directional edge identifiers, specific for this flux dance."""
        return self._edge_ids
    # endregion

    # region Class Methods
    def contains(self, element: Union[IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of flux-dance layer or not."""
        if element in self.qubit_ids:
            return True
        if element in self.edge_ids:
            return True
        return False

    def get_involved_edge(self, qubit_id: IQubitID) -> IEdgeID:
        """:return: Edge in which qubit-ID is involved. If qubit-ID not part of self, raise error."""
        for edge in self.edge_ids:
            if edge.contains(element=qubit_id):
                return edge
        raise ElementNotIncludedException(f'Element {qubit_id} is not part of self ({self}) and cannot be part of an edge.')

    def get_spectating_qubit_ids(self, device_layer: IDeviceLayer) -> List[IQubitID]:
        """:return: Direct spectator (nearest neighbor) to qubit-ID's participating in flux-dance."""
        participating_qubit_ids: List[IQubitID] = self.qubit_ids
        nearest_neighbor_ids: List[IQubitID] = [neighbor_id for qubit_id in participating_qubit_ids for neighbor_id in device_layer.get_neighbors(qubit_id, order=1)]
        filtered_nearest_neighbor_ids: List[IQubitID] = list(set([qubit_id for qubit_id in nearest_neighbor_ids if qubit_id not in participating_qubit_ids]))
        return filtered_nearest_neighbor_ids

    def requires_parking(self, qubit_id: IQubitID, device_layer: ISurfaceCodeLayer) -> bool:
        """
        Determines whether qubit-ID is required to park based on participation in flux dance and frequency group.
        :return: Boolean, whether qubit-ID requires some form of parking.
        """
        spectating_qubit_ids: List[IQubitID] = self.get_spectating_qubit_ids(device_layer=device_layer)
        # Guard clause, if qubit-ID does not spectate the flux-dance, no need for parking
        if qubit_id not in spectating_qubit_ids:
            return False
        # Check if qubit-ID requires parking based on its frequency group ID and active two-qubit gates.
        frequency_group: FrequencyGroupIdentifier = device_layer.get_frequency_group_identifier(element=qubit_id)
        # Parking is required if any neighboring qubit from a higher frequency group is part of an edge.
        neighboring_qubit_ids: List[IQubitID] = device_layer.get_neighbors(qubit=qubit_id, order=1)
        involved_neighbors: List[IQubitID] = [qubit_id for qubit_id in neighboring_qubit_ids if self.contains(qubit_id)]
        involved_frequency_groups: List[FrequencyGroupIdentifier] = [device_layer.get_frequency_group_identifier(element=qubit_id) for qubit_id in involved_neighbors]
        return any([neighbor_frequency_group.is_higher_than(frequency_group) for neighbor_frequency_group in involved_frequency_groups])
    # endregion



@dataclass(frozen=True)
class VirtualPhaseIdentifier(IChannelIdentifier):
    """
    Data class, describing (code-word) identifier for virtual phase.
    """
    _id: str

    # region Interface Properties
    @property
    def id(self) -> str:
        """:returns: Reference Identifier."""
        return self._id
    # endregion

    # region Interface Methods
    def __hash__(self):
        """:returns: Identifiable hash."""
        return self._id.__hash__()

    def __eq__(self, other):
        """:returns: Boolean if other shares equal identifier, else InterfaceMethodException."""
        if isinstance(other, VirtualPhaseIdentifier):
            return self.id.__eq__(other.id)
        return False
    # endregion


@dataclass(frozen=True)
class FluxOperationIdentifier(IChannelIdentifier):
    """
    Data class, describing (code-word) identifier for flux operation.
    """
    _id: str

    # region Interface Properties
    @property
    def id(self) -> str:
        """:returns: Reference Identifier."""
        return self._id
    # endregion

    # region Interface Methods
    def __hash__(self):
        """:returns: Identifiable hash."""
        return self._id.__hash__()

    def __eq__(self, other):
        """:returns: Boolean if other shares equal identifier, else InterfaceMethodException."""
        if isinstance(other, FluxOperationIdentifier):
            return self.id.__eq__(other.id)
        return False
    # endregion


class Surface17Layer(ISurfaceCodeLayer, metaclass=SingletonABCMeta):
    """
    Singleton class, implementing ISurfaceCodeLayer interface to describe a surface-17 layout.
    """
    _feedline_qubit_lookup: Dict[IFeedlineID, List[IQubitID]] = {
        FeedlineIDObj('FL1'): [QubitIDObj('D9'), QubitIDObj('D8'), QubitIDObj('X4'), QubitIDObj('Z4'), QubitIDObj('Z2'), QubitIDObj('D6')],
        FeedlineIDObj('FL2'): [QubitIDObj('D3'), QubitIDObj('D7'), QubitIDObj('D2'), QubitIDObj('X3'), QubitIDObj('Z1'), QubitIDObj('X2'), QubitIDObj('Z3'), QubitIDObj('D5'), QubitIDObj('D4')],
        FeedlineIDObj('FL3'): [QubitIDObj('D1'), QubitIDObj('X1')],
    }
    _qubit_edges: List[IEdgeID] = [
        EdgeIDObj(QubitIDObj('D1'), QubitIDObj('Z1')),
        EdgeIDObj(QubitIDObj('D1'), QubitIDObj('X1')),
        EdgeIDObj(QubitIDObj('D2'), QubitIDObj('X1')),
        EdgeIDObj(QubitIDObj('D2'), QubitIDObj('Z1')),
        EdgeIDObj(QubitIDObj('D2'), QubitIDObj('X2')),
        EdgeIDObj(QubitIDObj('D3'), QubitIDObj('X2')),
        EdgeIDObj(QubitIDObj('D3'), QubitIDObj('Z2')),
        EdgeIDObj(QubitIDObj('D4'), QubitIDObj('Z3')),
        EdgeIDObj(QubitIDObj('D4'), QubitIDObj('X3')),
        EdgeIDObj(QubitIDObj('D4'), QubitIDObj('Z1')),
        EdgeIDObj(QubitIDObj('D5'), QubitIDObj('Z1')),
        EdgeIDObj(QubitIDObj('D5'), QubitIDObj('X3')),
        EdgeIDObj(QubitIDObj('D5'), QubitIDObj('Z4')),
        EdgeIDObj(QubitIDObj('D5'), QubitIDObj('X2')),
        EdgeIDObj(QubitIDObj('D6'), QubitIDObj('X2')),
        EdgeIDObj(QubitIDObj('D6'), QubitIDObj('Z4')),
        EdgeIDObj(QubitIDObj('D6'), QubitIDObj('Z2')),
        EdgeIDObj(QubitIDObj('D7'), QubitIDObj('Z3')),
        EdgeIDObj(QubitIDObj('D7'), QubitIDObj('X3')),
        EdgeIDObj(QubitIDObj('D8'), QubitIDObj('X3')),
        EdgeIDObj(QubitIDObj('D8'), QubitIDObj('X4')),
        EdgeIDObj(QubitIDObj('D8'), QubitIDObj('Z4')),
        EdgeIDObj(QubitIDObj('D9'), QubitIDObj('Z4')),
        EdgeIDObj(QubitIDObj('D9'), QubitIDObj('X4')),
    ]
    _parity_group_x: List[IParityGroup] = [
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X1'),
            _data_qubits=[QubitIDObj('D1'), QubitIDObj('D2')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X2'),
            _data_qubits=[QubitIDObj('D2'), QubitIDObj('D3'), QubitIDObj('D5'), QubitIDObj('D6')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X3'),
            _data_qubits=[QubitIDObj('D4'), QubitIDObj('D5'), QubitIDObj('D7'), QubitIDObj('D8')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X4'),
            _data_qubits=[QubitIDObj('D8'), QubitIDObj('D9')]
        ),
    ]
    _parity_group_z: List[IParityGroup] = [
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z1'),
            _data_qubits=[QubitIDObj('D1'), QubitIDObj('D2'), QubitIDObj('D4'), QubitIDObj('D5')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z2'),
            _data_qubits=[QubitIDObj('D3'), QubitIDObj('D6')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z3'),
            _data_qubits=[QubitIDObj('D4'), QubitIDObj('D7')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z4'),
            _data_qubits=[QubitIDObj('D5'), QubitIDObj('D6'), QubitIDObj('D8'), QubitIDObj('D9')]
        ),
    ]
    _frequency_group_lookup: Dict[IQubitID, FrequencyGroupIdentifier] = {
        QubitIDObj('D1'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D2'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D3'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D4'): FrequencyGroupIdentifier(_id=FrequencyGroup.HIGH),
        QubitIDObj('D5'): FrequencyGroupIdentifier(_id=FrequencyGroup.HIGH),
        QubitIDObj('D6'): FrequencyGroupIdentifier(_id=FrequencyGroup.HIGH),
        QubitIDObj('D7'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D8'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D9'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('Z1'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('Z2'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('Z3'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('Z4'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('X1'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('X2'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('X3'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('X4'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
    }

    # region ISurfaceCodeLayer Interface Properties
    @property
    def parity_group_x(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of X-stabilizers."""
        return self._parity_group_x

    @property
    def parity_group_z(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of Z-stabilizers."""
        return self._parity_group_z
    # endregion

    # region Class Properties
    @property
    def feedline_ids(self) -> List[IFeedlineID]:
        """:return: All feedline-ID's."""
        return list(self._feedline_qubit_lookup.keys())

    @property
    def qubit_ids(self) -> List[IQubitID]:
        """:return: All qubit-ID's."""
        return [qubit_id for qubit_ids in self._feedline_qubit_lookup.values() for qubit_id in qubit_ids]

    @property
    def edge_ids(self) -> List[IEdgeID]:
        """:return: All edge-ID's."""
        return self._qubit_edges
    # endregion

    # region ISurfaceCodeLayer Interface Methods
    def get_parity_group(self, element: Union[IQubitID, IEdgeID]) -> IParityGroup:
        """:return: Parity group of which element (edge- or qubit-ID) is part of."""
        # Assumes element is part of only a single parity group
        for parity_group in self.parity_group_x + self.parity_group_z:
            if parity_group.contains(element=element):
                return parity_group
        raise ElementNotIncludedException(f"Element: {element} is not included in any parity group.")
    # endregion

    # region IDeviceLayer Interface Methods
    def get_connected_qubits(self, feedline: IFeedlineID) -> List[IQubitID]:
        """:return: Qubit-ID's connected to feedline-ID."""
        # Guard clause, if feedline not in lookup, raise exception
        if feedline not in self._feedline_qubit_lookup:
            raise ElementNotIncludedException(f"Element: {feedline} is not included in any feedline group.")
        return self._feedline_qubit_lookup[feedline]

    def get_neighbors(self, qubit: IQubitID, order: int = 1) -> List[IQubitID]:
        """
        Requires :param order: to be higher or equal to 1.
        :return: qubit neighbors separated by order. (order=1, nearest neighbors).
        """
        if order > 1:
            raise NotImplementedError("Apologies, so far there has not been a use for. But feel free to implement.")
        edges: List[IEdgeID] = self.get_edges(qubit=qubit)
        result: List[IQubitID] = []
        for edge in edges:
            result.append(edge.get_connected_qubit_id(element=qubit))
        return result

    def get_edges(self, qubit: IQubitID) -> List[IEdgeID]:
        """:return: All qubit-to-qubit edges from qubit-ID."""
        result: List[IEdgeID] = []
        for edge in self.edge_ids:
            if edge.contains(element=qubit):
                result.append(edge)
        return result

    def contains(self, element: Union[IFeedlineID, IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of device layer or not."""
        if element in self.feedline_ids:
            return True
        if element in self.qubit_ids:
            return True
        if element in self.edge_ids:
            return True
        return False
    
    def get_frequency_group_identifier(self, element: IQubitID) -> FrequencyGroupIdentifier:
        """:return: Frequency group identifier based on qubit-ID."""
        return self._frequency_group_lookup[element]
    # endregion


class Repetition9Layer(ISurfaceCodeLayer, metaclass=SingletonABCMeta):
    """
    Singleton class, implementing ISurfaceCodeLayer interface to describe a repetition-9 layout.
    """
    _parity_group_x: List[IParityGroup] = [
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X1'),
            _data_qubits=[QubitIDObj('D2'), QubitIDObj('D1')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X2'),
            _data_qubits=[QubitIDObj('D2'), QubitIDObj('D3')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X3'),
            _data_qubits=[QubitIDObj('D8'), QubitIDObj('D7')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X4'),
            _data_qubits=[QubitIDObj('D9'), QubitIDObj('D8')]
        ),
    ]
    _parity_group_z: List[IParityGroup] = [
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z1'),
            _data_qubits=[QubitIDObj('D4'), QubitIDObj('D5')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z2'),
            _data_qubits=[QubitIDObj('D6'), QubitIDObj('D3')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z3'),
            _data_qubits=[QubitIDObj('D7'), QubitIDObj('D4')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z4'),
            _data_qubits=[QubitIDObj('D5'), QubitIDObj('D6')]
        ),
    ]
    _virtual_phase_lookup: Dict[DirectionalEdgeIDObj, VirtualPhaseIdentifier] = {
        DirectionalEdgeIDObj(QubitIDObj('D1'), QubitIDObj('Z1')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D1')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('D1'), QubitIDObj('X1')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('X1'), QubitIDObj('D1')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('D2'), QubitIDObj('X1')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('X1'), QubitIDObj('D2')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('D2'), QubitIDObj('Z1')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D2')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('D2'), QubitIDObj('X2')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('X2'), QubitIDObj('D2')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('D3'), QubitIDObj('X2')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('X2'), QubitIDObj('D3')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('D3'), QubitIDObj('Z2')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('Z2'), QubitIDObj('D3')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('D4'), QubitIDObj('Z3')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('Z3'), QubitIDObj('D4')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('D4'), QubitIDObj('X3')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('X3'), QubitIDObj('D4')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('D4'), QubitIDObj('Z1')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D4')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('D5'), QubitIDObj('Z1')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D5')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('D5'), QubitIDObj('X3')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('X3'), QubitIDObj('D5')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('D5'), QubitIDObj('Z4')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('Z4'), QubitIDObj('D5')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('D5'), QubitIDObj('X2')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('X2'), QubitIDObj('D5')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('D6'), QubitIDObj('X2')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('X2'), QubitIDObj('D6')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('D6'), QubitIDObj('Z4')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('Z4'), QubitIDObj('D6')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('D6'), QubitIDObj('Z2')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('Z2'), QubitIDObj('D6')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('D7'), QubitIDObj('Z3')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('Z3'), QubitIDObj('D7')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('D7'), QubitIDObj('X3')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('X3'), QubitIDObj('D7')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('D8'), QubitIDObj('X3')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('X3'), QubitIDObj('D8')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('D8'), QubitIDObj('X4')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('X4'), QubitIDObj('D8')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('D8'), QubitIDObj('Z4')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
        DirectionalEdgeIDObj(QubitIDObj('Z4'), QubitIDObj('D8')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('D9'), QubitIDObj('Z4')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SW'),
        DirectionalEdgeIDObj(QubitIDObj('Z4'), QubitIDObj('D9')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NE'),
        DirectionalEdgeIDObj(QubitIDObj('D9'), QubitIDObj('X4')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_NW'),
        DirectionalEdgeIDObj(QubitIDObj('X4'), QubitIDObj('D9')): VirtualPhaseIdentifier('vcz_virtual_q_ph_corr_SE'),
    }
    _flux_dances: List[Tuple[FluxDanceLayer, FluxOperationIdentifier]] = [
        (
            FluxDanceLayer(
                _edge_ids=[
                    EdgeIDObj(QubitIDObj('X1'), QubitIDObj('D1')),
                    EdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D4')),
                    EdgeIDObj(QubitIDObj('X3'), QubitIDObj('D7')),
                    EdgeIDObj(QubitIDObj('Z2'), QubitIDObj('D6')),
                ]
            ),
            FluxOperationIdentifier(_id='repetition_code_1')
        ),
        (
            FluxDanceLayer(
                _edge_ids=[
                    EdgeIDObj(QubitIDObj('X1'), QubitIDObj('D2')),
                    EdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D5')),
                    EdgeIDObj(QubitIDObj('X3'), QubitIDObj('D8')),
                    EdgeIDObj(QubitIDObj('Z2'), QubitIDObj('D3')),
                ]
            ),
            FluxOperationIdentifier(_id='repetition_code_2')
        ),
        (
            FluxDanceLayer(
                _edge_ids=[
                    EdgeIDObj(QubitIDObj('Z3'), QubitIDObj('D7')),
                    EdgeIDObj(QubitIDObj('X4'), QubitIDObj('D8')),
                    EdgeIDObj(QubitIDObj('Z4'), QubitIDObj('D5')),
                    EdgeIDObj(QubitIDObj('X2'), QubitIDObj('D2')),
                ]
            ),
            FluxOperationIdentifier(_id='repetition_code_3')
        ),
        (
            FluxDanceLayer(
                _edge_ids=[
                    EdgeIDObj(QubitIDObj('Z3'), QubitIDObj('D4')),
                    EdgeIDObj(QubitIDObj('X4'), QubitIDObj('D9')),
                    EdgeIDObj(QubitIDObj('Z4'), QubitIDObj('D6')),
                    EdgeIDObj(QubitIDObj('X2'), QubitIDObj('D3')),
                ]
            ),
            FluxOperationIdentifier(_id='repetition_code_4')
        ),
    ]

    # region ISurfaceCodeLayer Interface Properties
    @property
    def parity_group_x(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of X-stabilizers."""
        return self._parity_group_x

    @property
    def parity_group_z(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of Z-stabilizers."""
        return self._parity_group_z
    # endregion

    # region Class Properties
    @property
    def feedline_ids(self) -> List[IFeedlineID]:
        """:return: All feedline-ID's."""
        return Surface17Layer().feedline_ids

    @property
    def qubit_ids(self) -> List[IQubitID]:
        """:return: All qubit-ID's."""
        return Surface17Layer().qubit_ids

    @property
    def edge_ids(self) -> List[IEdgeID]:
        """:return: All edge-ID's."""
        return Surface17Layer().edge_ids
    # endregion

    # region ISurfaceCodeLayer Interface Methods
    def get_parity_group(self, element: Union[IQubitID, IEdgeID]) -> IParityGroup:
        """:return: Parity group of which element (edge- or qubit-ID) is part of."""
        # Assumes element is part of only a single parity group
        for parity_group in self.parity_group_x + self.parity_group_z:
            if parity_group.contains(element=element):
                return parity_group
        raise ElementNotIncludedException(f"Element: {element} is not included in any parity group.")
    # endregion

    # region IGateDanceLayer Interface Methods
    def get_flux_dance_at_round(self, index: int) -> FluxDanceLayer:
        """:return: Flux-dance object based on round index."""
        try:
            flux_dance_layer: FluxDanceLayer = self._flux_dances[index]
            return flux_dance_layer
        except:
            raise ElementNotIncludedException(f"Index: {index} is out of bounds for flux dance of length: {len(self._flux_dances)}.")
    # endregion

    # region IDeviceLayer Interface Methods
    def get_connected_qubits(self, feedline: IFeedlineID) -> List[IQubitID]:
        """:return: Qubit-ID's connected to feedline-ID."""
        return Surface17Layer().get_connected_qubits(feedline=feedline)

    def get_neighbors(self, qubit: IQubitID, order: int = 1) -> List[IQubitID]:
        """
        Requires :param order: to be higher or equal to 1.
        :return: qubit neighbors separated by order. (order=1, nearest neighbors).
        """
        return Surface17Layer().get_neighbors(qubit=qubit, order=order)

    def get_edges(self, qubit: IQubitID) -> List[IEdgeID]:
        """:return: All qubit-to-qubit edges from qubit-ID."""
        return Surface17Layer().get_edges(qubit=qubit)

    def contains(self, element: Union[IFeedlineID, IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of device layer or not."""
        return Surface17Layer().contains(element=element)
    # endregion

    # region Class Methods
    def _get_flux_dance_layer(self, element: IEdgeID) -> FluxDanceLayer:
        """:return: Flux-dance layer of which edge element is part of."""
        # Assumes element is part of only a single flux-dance layer
        for flux_dance_layer, _ in self._flux_dances:
            if flux_dance_layer.contains(element=element):
                return flux_dance_layer
        raise ElementNotIncludedException(f"Element: {element} is not included in any flux-dance layer.")

    def _get_flux_operation_identifier(self, element: IEdgeID) -> FluxOperationIdentifier:
        """:return: Identifier describing flux-dance layer."""
        for flux_dance_layer, flux_operation_identifier in self._flux_dances:
            if flux_dance_layer.contains(element=element):
                return flux_operation_identifier
        raise ElementNotIncludedException(f"Element: {element} is not included in any flux-dance layer.")


    def get_flux_operation_identifier(self, qubit_id0: str, qubit_id1: str) -> str:
        """:return: Identifier describing flux-dance layer."""
        edge: IEdgeID = EdgeIDObj(
            qubit_id0=QubitIDObj(_id=qubit_id0),
            qubit_id1=QubitIDObj(_id=qubit_id1),
        )
        return self._get_flux_operation_identifier(element=edge).id

    def get_edge_flux_operation_identifier(self, ancilla_qubit: str) -> List[str]:
        """:return: Identifier describing flux-dance layer."""
        qubit_id: IQubitID = QubitIDObj(_id=ancilla_qubit)
        parity_group: IParityGroup = self.get_parity_group(element=qubit_id)
        return [
            self._get_flux_operation_identifier(
                element=edge_id,
            ).id
            for edge_id in parity_group.edge_ids
        ]

    def _get_virtual_phase_identifier(self, directional_edge: DirectionalEdgeIDObj) -> VirtualPhaseIdentifier:
        """:return: Identifier for virtual phase correction. Based on element and parity group."""
        return self._virtual_phase_lookup[directional_edge]

    def get_virtual_phase_identifier(self, from_qubit: str, to_qubit: str) -> VirtualPhaseIdentifier:
        """:return: Identifier for virtual phase correction. Based on element and parity group."""
        directional_edge: DirectionalEdgeIDObj = DirectionalEdgeIDObj(
            qubit_id0=QubitIDObj(_id=from_qubit),
            qubit_id1=QubitIDObj(_id=to_qubit),
        )
        return self._get_virtual_phase_identifier(directional_edge=directional_edge)

    def get_ancilla_virtual_phase_identifier(self, ancilla_qubit: str) -> str:
        """:return: Arbitrary virtual phase from ancilla used in parity group."""
        qubit_id: IQubitID = QubitIDObj(_id=ancilla_qubit)
        parity_group: IParityGroup = self.get_parity_group(element=qubit_id)
        directional_edge: DirectionalEdgeIDObj = DirectionalEdgeIDObj(
            qubit_id0=parity_group.ancilla_id,
            qubit_id1=parity_group.data_ids[0],
        )
        return self._get_virtual_phase_identifier(directional_edge=directional_edge).id

    def get_data_virtual_phase_identifiers(self, ancilla_qubit: str) -> List[str]:
        """:return: Arbitrary virtual phase from ancilla used in parity group."""
        qubit_id: IQubitID = QubitIDObj(_id=ancilla_qubit)
        parity_group: IParityGroup = self.get_parity_group(element=qubit_id)
        return [
            self._get_virtual_phase_identifier(
                directional_edge=DirectionalEdgeIDObj(
                    qubit_id0=data_id,
                    qubit_id1=parity_group.ancilla_id,
                )
            ).id
            for data_id in parity_group.data_ids
        ]

    def get_parity_data_identifier(self, ancilla_qubit: str) -> List[str]:
        """
        Iterates over provided ancilla qubit ID's.
        Construct corresponding IQubitID's.
        Obtain corresponding IParityGroup's.
        Flatten list of (unique) data qubit ID's part of these parity groups.
        :return: Array-like of (unique) data qubit ID's part of ancilla qubit parity groups.
        """
        ancilla_qubit_id: IQubitID = QubitIDObj(ancilla_qubit)
        parity_group: IParityGroup = self.get_parity_group(element=ancilla_qubit_id)
        data_qubit_ids: List[IQubitID] = [qubit_id for qubit_id in parity_group.data_ids]
        return [qubit_id.id for qubit_id in data_qubit_ids]

    def get_parity_data_identifiers(self, ancilla_qubits: List[str]) -> List[str]:
        """
        Iterates over provided ancilla qubit ID's.
        Construct corresponding IQubitID's.
        Obtain corresponding IParityGroup's.
        Flatten list of (unique) data qubit ID's part of these parity groups.
        :return: Array-like of (unique) data qubit ID's part of ancilla qubit parity groups.
        """
        return [unique_qubit_id for ancilla_qubit in ancilla_qubits for unique_qubit_id in set(self.get_parity_data_identifier(ancilla_qubit=ancilla_qubit))]
    
    def get_frequency_group_identifier(self, element: IQubitID) -> FrequencyGroupIdentifier:
        """:return: Frequency group identifier based on qubit-ID."""
        return Surface17Layer().get_frequency_group_identifier(element=element)
    # endregion


if __name__ == '__main__':

    flux_dance_0 = Repetition9Layer().get_flux_dance_at_round(0)
    print(flux_dance_0.edge_ids)    
