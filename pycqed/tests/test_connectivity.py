import unittest
from typing import List
from pycqed.qce_utils.control_interfaces.intrf_channel_identifier import (
    IQubitID,
    IEdgeID,
    QubitIDObj,
    EdgeIDObj,
)
from pycqed.qce_utils.control_interfaces.intrf_connectivity_surface_code import (
    ISurfaceCodeLayer,
    IParityGroup,
)
from pycqed.qce_utils.control_interfaces.connectivity_surface_code import Surface17Layer


class Surface17ConnectivityTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        cls.layer: Surface17Layer = Surface17Layer()
        cls.expected_qubit_ids: List[IQubitID] = [
            QubitIDObj('D9'), QubitIDObj('D8'), QubitIDObj('X4'), QubitIDObj('Z4'), QubitIDObj('Z2'), QubitIDObj('D6'),
            QubitIDObj('D3'), QubitIDObj('D7'), QubitIDObj('D2'), QubitIDObj('X3'), QubitIDObj('Z1'), QubitIDObj('X2'),
            QubitIDObj('Z3'), QubitIDObj('D5'), QubitIDObj('D4'), QubitIDObj('D1'), QubitIDObj('X1'),
        ]

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_qubit_inclusion(self):
        """Tests if all 17 expected qubits are included in the connectivity layer."""
        for qubit_id in self.expected_qubit_ids:
            with self.subTest(msg=f'{qubit_id.id}'):
                self.assertTrue(self.layer.contains(element=qubit_id))

    def test_qubit_edge_count(self):
        """Tests 24 unique edges are present."""
        edges: List[IEdgeID] = self.layer.edge_ids
        self.assertEquals(
            len(set(edges)),
            24,
            msg=f"Expect 24 unique edges in a Surface-17 layout. Got: {len(set(edges))}."
        )

    def test_qubit_edge_getter(self):
        """Tests various cases of obtaining qubit edges."""
        edges: List[IEdgeID] = self.layer.get_edges(qubit=QubitIDObj('D5'))
        expected_edges: List[IEdgeID] = [
            EdgeIDObj(QubitIDObj('Z1'), QubitIDObj('D5')), EdgeIDObj(QubitIDObj('D5'), QubitIDObj('X3')),
            EdgeIDObj(QubitIDObj('D5'), QubitIDObj('Z4')), EdgeIDObj(QubitIDObj('D5'), QubitIDObj('X2')),
        ]
        self.assertSetEqual(
            set(edges),
            set(expected_edges),
            msg=f"Expects these edges: {set(expected_edges)}, instead got: {set(edges)}."
        )

    def test_get_neighbor_qubits(self):
        """Tests various cases of obtaining (nearest) neighboring qubits."""
        qubits: List[IQubitID] = self.layer.get_neighbors(qubit=QubitIDObj('D5'), order=1)
        expected_qubits: List[IQubitID] = [
            QubitIDObj('Z1'), QubitIDObj('X2'), QubitIDObj('X3'), QubitIDObj('Z4')
        ]
        self.assertSetEqual(
            set(qubits),
            set(expected_qubits),
            msg=f"Expects these neighboring qubits: {set(expected_qubits)}, instead got: {set(qubits)}."
        )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion
