# -------------------------------------------
# Interface for unique channel references
# For example:
# Qubit identifier, Feedline identifier, Flux channel identifier, etc.
# -------------------------------------------
from abc import ABCMeta, abstractmethod, ABC
from dataclasses import dataclass, field
from typing import List, Dict
from pycqed.qce_utils.custom_exceptions import InterfaceMethodException, IsolatedGroupException

QID = str  # Might become int in future
QName = str


class IChannelIdentifier(ABC):
    """
    Interface class, describing unique identifier.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def id(self) -> str:
        """:returns: Reference Identifier."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def __hash__(self):
        """:returns: Identifiable hash."""
        raise InterfaceMethodException

    @abstractmethod
    def __eq__(self, other):
        """:returns: Boolean if other shares equal identifier, else InterfaceMethodException."""
        raise InterfaceMethodException
    # endregion


class IQubitID(IChannelIdentifier, metaclass=ABCMeta):
    """
    Interface for qubit reference.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def name(self) -> QName:
        """:returns: Reference name for qubit."""
        raise InterfaceMethodException
    # endregion


class IFeedlineID(IChannelIdentifier, metaclass=ABCMeta):
    """
    Interface for feedline reference.
    """
    pass


class IEdgeID(IChannelIdentifier, metaclass=ABCMeta):
    """
    Interface class, for qubit-to-qubit edge reference.
    """

    # region Interface Methods
    @abstractmethod
    def contains(self, element: IQubitID) -> bool:
        """:return: Boolean, whether element is part of edge or not."""
        raise InterfaceMethodException

    @abstractmethod
    def get_connected_qubit_id(self, element: IQubitID) -> IQubitID:
        """:return: Qubit-ID, connected to the other side of this edge."""
        raise InterfaceMethodException
    # endregion


class IQubitIDGroups(ABC):
    """
    Interface class, describing groups of IQubitID's.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def groups(self) -> List[List[IQubitID]]:
        """:return: Array-like of grouped (array) IQubitID's."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def get_group(self, group_member: IQubitID) -> List[IQubitID]:
        """
        Returns empty list if group_member not part of this lookup.
        :return: Array-like of group members. Including provided group_member.
        """
        raise InterfaceMethodException
    # endregion


@dataclass(frozen=True)
class QubitIDObj(IQubitID):
    """
    Contains qubit label ID.
    """
    _id: QName

    # region Interface Properties
    @property
    def id(self) -> QID:
        """:returns: Reference ID for qubit."""
        return self._id

    @property
    def name(self) -> QName:
        """:returns: Reference name for qubit."""
        return self.id
    # endregion

    # region Class Methods
    def __hash__(self):
        """:returns: Identifiable hash."""
        return self.id.__hash__()

    def __eq__(self, other):
        """:returns: Boolean if other shares equal identifier, else InterfaceMethodException."""
        if isinstance(other, IQubitID):
            return self.id.__eq__(other.id)
        # raise NotImplementedError('QubitIDObj equality check to anything other than IQubitID interface is not implemented.')
        return False

    def __repr__(self):
        return f'<Qubit-ID>{self.id}'
    # endregion


@dataclass(frozen=True)
class QubitIDGroups(IQubitIDGroups):
    """
    Data class, implementing IQubitIDGroups interface.
    """
    group_lookup: Dict[IQubitID, int] = field(default_factory=dict)
    """Lookup dictionary where each IQubitID is matched to a specific (integer) group identifier."""

    # region Interface Properties
    @property
    def groups(self) -> List[List[IQubitID]]:
        """:return: Array-like of grouped (array) IQubitID's."""
        return list(self.group_id_to_members.values())
    # endregion

    # region Class Properties
    @property
    def group_id_to_members(self) -> Dict[int, List[IQubitID]]:
        """:return: Intermediate lookup table from group-id to its members."""
        group_lookup: Dict[int, List[IQubitID]] = {}
        for qubit_id, group_id in self.group_lookup.items():
            if group_id not in group_lookup:
                group_lookup[group_id] = [qubit_id]
            else:
                group_lookup[group_id].append(qubit_id)
        return group_lookup
    # endregion

    # region Interface Methods
    def get_group(self, group_member: IQubitID) -> List[IQubitID]:
        """
        Returns empty list if group_member not part of this lookup.
        :return: Array-like of group members. Including provided group_member.
        """
        group_id_to_members: Dict[int, List[IQubitID]] = self.group_id_to_members
        # Guard clause, if provided group member not in this lookup, return empty list.
        if group_member not in self.group_lookup:
            return []
        group_id: int = self.group_lookup[group_member]
        return group_id_to_members[group_id]
    # endregion

    # region Class Methods
    def __post_init__(self):
        # Verify group member uniqueness.
        all_group_members: List[IQubitID] = [qubit_id for group in self.groups for qubit_id in group]
        isolated_groups: bool = len(set(all_group_members)) == len(all_group_members)
        if not isolated_groups:
            raise IsolatedGroupException(f'Expects all group members to be part of a single group.')

    @classmethod
    def from_groups(cls, groups: List[List[IQubitID]]) -> 'QubitIDGroups':
        """:return: Class method constructor based on list of groups of QUbitID's."""
        group_lookup: Dict[IQubitID, int] = {}
        for group_id, group in enumerate(groups):
            for qubit_id in group:
                if qubit_id in group_lookup:
                    raise IsolatedGroupException(f'{qubit_id} is already in another group. Requires each group member to be part of only one group.')
                group_lookup[qubit_id] = group_id
        return QubitIDGroups(
            group_lookup=group_lookup,
        )
    # endregion


@dataclass(frozen=True)
class FeedlineIDObj(IFeedlineID):
    """
    Data class, implementing IFeedlineID interface.
    """
    name: QID

    # region Interface Properties
    @property
    def id(self) -> QID:
        """:returns: Reference ID for feedline."""
        return self.name
    # endregion

    # region Class Methods
    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        if isinstance(other, IFeedlineID):
            return self.id.__eq__(other.id)
        # raise NotImplementedError('FeedlineIDObj equality check to anything other than IFeedlineID interface is not implemented.')
        return False

    def __repr__(self):
        return f'<Feedline-ID>{self.id}'
    # endregion


@dataclass(frozen=True)
class EdgeIDObj(IEdgeID):
    """
    Data class, implementing IEdgeID interface.
    """
    qubit_id0: IQubitID
    """Arbitrary edge qubit-ID."""
    qubit_id1: IQubitID
    """Arbitrary edge qubit-ID."""

    # region Interface Properties
    @property
    def id(self) -> QID:
        """:returns: Reference ID for edge."""
        return f"{self.qubit_id0.id}-{self.qubit_id1.id}"
    # endregion

    # region Interface Methods
    def contains(self, element: IQubitID) -> bool:
        """:return: Boolean, whether element is part of edge or not."""
        if element in [self.qubit_id0, self.qubit_id1]:
            return True
        return False

    def get_connected_qubit_id(self, element: IQubitID) -> IQubitID:
        """:return: Qubit-ID, connected to the other side of this edge."""
        if element == self.qubit_id0:
            return self.qubit_id1
        if element == self.qubit_id1:
            return self.qubit_id0
        # If element is not part of this edge
        raise ValueError(f"Element: {element} is not part of this edge: {self}")
    # endregion

    # region Class Methods
    def __hash__(self):
        """
        Sorts individual qubit hashes such that the order is NOT maintained.
        Making hash comparison independent of order.
        """
        return hash((min(self.qubit_id0.__hash__(), self.qubit_id1.__hash__()), max(self.qubit_id0.__hash__(), self.qubit_id1.__hash__())))

    def __eq__(self, other):
        if isinstance(other, IEdgeID):
            # Edge is equal if they share the same qubit identifiers, order does not matter
            return other.contains(self.qubit_id0) and other.contains(self.qubit_id1)
        # raise NotImplementedError('EdgeIDObj equality check to anything other than IEdgeID interface is not implemented.')
        return False

    def __repr__(self):
        return f'<Edge-ID>{self.id}'
    # endregion
