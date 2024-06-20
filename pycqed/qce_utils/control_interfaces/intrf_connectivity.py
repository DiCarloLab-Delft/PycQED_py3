# -------------------------------------------
# Module containing interface for device connectivity structure.
# -------------------------------------------
from abc import ABC, ABCMeta, abstractmethod
from multipledispatch import dispatch
from typing import List, Tuple, Union
from pycqed.qce_utils.custom_exceptions import InterfaceMethodException
from pycqed.qce_utils.control_interfaces.intrf_channel_identifier import (
    IFeedlineID,
    IQubitID,
    IEdgeID,
)


class IIdentifier(ABC):
    """
    Interface class, describing equality identifier method.
    """

    # region Interface Methods
    @abstractmethod
    def __eq__(self, other):
        """:return: Boolean, whether 'other' equals 'self'."""
        raise InterfaceMethodException
    # endregion


class INode(IIdentifier, metaclass=ABCMeta):
    """
    Interface class, describing the node in a connectivity layer.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def edges(self) -> List['IEdge']:
        """:return: (N) Edges connected to this node."""
        raise InterfaceMethodException
    # endregion


class IEdge(IIdentifier, metaclass=ABCMeta):
    """
    Interface class, describing a connection between two nodes.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def nodes(self) -> Tuple[INode, INode]:
        """:return: (2) Nodes connected by this edge."""
        raise InterfaceMethodException
    # endregion


class IConnectivityLayer(ABC):
    """
    Interface class, describing a connectivity (graph) layer containing nodes and edges.
    Note that a connectivity layer can include 'separated' graphs
    where not all nodes have a connection path to all other nodes.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def nodes(self) -> List[INode]:
        """:return: Array-like of nodes."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def edges(self) -> List[IEdge]:
        """:return: Array-like of edges."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @dispatch(node=INode)
    @abstractmethod
    def get_connected_nodes(self, node: INode, order: int) -> List[INode]:
        """
        :param node: (Root) node to base connectivity on.
            If node has no edges, return an empty list.
        :param order: Connectivity range.
            Order <=0: empty list, 1: first order connectivity, 2: second order connectivity, etc.
        :return: Array-like of nodes connected to 'node' within order of connection (excluding 'node' itself).
        """
        raise InterfaceMethodException

    @dispatch(edge=IEdge)
    @abstractmethod
    def get_connected_nodes(self, edge: IEdge, order: int) -> List[INode]:
        """
        :param edge: (Root) edge to base connectivity on.
        :param order: Connectivity range.
            Order <=0: empty list, 1: first order connectivity, 2: second order connectivity, etc.
        :return: Array-like of nodes connected to 'edge' within order of connection.
        """
        raise InterfaceMethodException
    # endregion


class IConnectivityStack(ABC):
    """
    Interface class, describing an array-like of connectivity layers.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def layers(self) -> List[IConnectivityLayer]:
        """:return: Array-like of connectivity layers."""
        raise InterfaceMethodException
    # endregion


class IDeviceLayer(ABC):
    """
    Interface class, describing relation based connectivity.
    """

    # region Interface Methods
    @abstractmethod
    def get_connected_qubits(self, feedline: IFeedlineID) -> List[IQubitID]:
        """:return: Qubit-ID's connected to feedline-ID."""
        raise InterfaceMethodException

    @abstractmethod
    def get_neighbors(self, qubit: IQubitID, order: int = 1) -> List[IQubitID]:
        """
        Requires :param order: to be higher or equal to 1.
        :return: qubit neighbors separated by order. (order=1, nearest neighbors).
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_edges(self, qubit: IQubitID) -> List[IEdgeID]:
        """:return: All qubit-to-qubit edges from qubit-ID."""
        raise InterfaceMethodException

    @abstractmethod
    def contains(self, element: Union[IFeedlineID, IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of device layer or not."""
        raise InterfaceMethodException
    # endregion
