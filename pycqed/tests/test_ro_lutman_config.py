import unittest
from pycqed.instrument_drivers.meta_instrument.LutMans.ro_lutman_config import (
    get_default_map_collection,
    FeedlineMapCollection,
    FeedlineMap,
    FeedlineBitMap,
    read_ro_lutman_bit_map,
)


class ReadConfigTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        cls.example_map: FeedlineMapCollection = get_default_map_collection()
        cls.existing_map_id: str = 'S17'
        cls.existing_feedline_nr: int = 0
        cls.not_existing_map_id: str = 'NULL'
        cls.not_existing_feedline_nr: int = -1

    def setUp(self) -> None:
        """Set up for every test case"""
        pass

    # endregion

    # region Test Cases
    def test_case_assumptions(self):
        """Tests assumptions made by testcase about default map collection."""
        self.assertTrue(
            self.existing_map_id in self.example_map.feedline_map_lookup,
            msg=f'Expects {self.existing_map_id} to be in example map keys ({list(self.example_map.feedline_map_lookup.keys())}).'
        )
        feedline_map: FeedlineMap = self.example_map.feedline_map_lookup[self.existing_map_id]
        self.assertTrue(
            self.existing_feedline_nr in feedline_map.bitmap_lookup,
            msg=f'Expects {self.existing_feedline_nr} to be in example map keys ({list(feedline_map.bitmap_lookup.keys())}).'
        )
        self.assertFalse(
            self.not_existing_map_id in self.example_map.feedline_map_lookup,
            msg=f'Expects {self.existing_map_id} NOT to be in example map keys ({list(self.example_map.feedline_map_lookup.keys())}).'
        )
        self.assertFalse(
            self.not_existing_feedline_nr in feedline_map.bitmap_lookup,
            msg=f'Expects {self.not_existing_feedline_nr} NOT to be in example map keys ({list(feedline_map.bitmap_lookup.keys())}).'
        )

    def test_core_functionality(self):
        """
        Tests default functionality based on example map collection.
        """
        # Test typing
        self.run_map_collection(
            _map_collection=self.example_map
        )

    def test_read_from_config(self):
        """Tests correct construction of data classes from config file."""
        map_collection = read_ro_lutman_bit_map()
        self.assertIsInstance(
            map_collection,
            FeedlineMapCollection,
        )
        self.run_map_collection(
            _map_collection=map_collection
        )

    def run_map_collection(self, _map_collection: FeedlineMapCollection):
        """
        WARNING: Change these assertions based on changes in get_default_map_collection().
        Tests default functionality based on any map collection.
        """
        # Test typing
        i_len: int = len(_map_collection.feedline_map_lookup)
        for i, (map_id, feedline_map) in enumerate(_map_collection.feedline_map_lookup.items()):
            with self.subTest(line=i):
                self.assertIsInstance(
                    map_id,
                    str,
                )
                self.assertIsInstance(
                    feedline_map,
                    FeedlineMap,
                )
            for j, (feedline_nr, bit_map) in enumerate(feedline_map.bitmap_lookup.items()):
                with self.subTest(line=i * i_len + j):
                    self.assertIsInstance(
                        feedline_nr,
                        int,
                    )
                    self.assertIsInstance(
                        bit_map,
                        FeedlineBitMap,
                    )
                    resonator_bit_map = _map_collection.get_bitmap(
                        map_id=map_id,
                        feedline_nr=feedline_nr,
                    )
                    self.assertIsNotNone(
                        resonator_bit_map,
                    )
                    self.assertTrue(
                        len(resonator_bit_map) > 0,
                        msg='Expects bit-map to have non-zero length',
                    )
                    self.assertTrue(
                        all(isinstance(x, int) for x in resonator_bit_map),
                        msg='Expects all elements to be integers',
                    )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion
