# -------------------------------------------
# Module containing implementation of surface-code connectivity structure.
# -------------------------------------------
from dataclasses import dataclass, field
import warnings
from typing import List, Union, Dict, Tuple
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
    # endregion


if __name__ == '__main__':

    for parity_group in Repetition9Layer().parity_group_x + Repetition9Layer().parity_group_z:
        print(parity_group.ancilla_id.id)
        print(f'(Ancilla) phase cw: {Repetition9Layer().get_ancilla_virtual_phase_identifier(parity_group.ancilla_id.id)}')
        print(f'(Data) phase cw: ', [phase_id for phase_id in Repetition9Layer().get_data_virtual_phase_identifiers(parity_group.ancilla_id.id)])
