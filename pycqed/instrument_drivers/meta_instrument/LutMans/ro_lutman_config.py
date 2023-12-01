# -------------------------------------------
# Data classes for resonator bit-map config.
# -------------------------------------------
from dataclasses import dataclass, field
from typing import List, Dict
from pycqed.utilities.custom_exceptions import (
    IdentifierFeedlineException,
)
from pycqed.utilities.readwrite_yaml import (
    os,
    get_yaml_file_path,
    write_yaml,
    read_yaml,
)


CONFIG_FILENAME: str = 'ro_lutman_config.yaml'


@dataclass()
class FeedlineBitMap:
    """
    Contains Feedline-ID to (resonator) bit-map lookup data.
    Intended as data-class for configuration.
    """
    id: int = field()
    bit_map: List[int] = field(default_factory=list)


@dataclass()
class FeedlineMap:
    """
    Contains collection of FeedlineBitMap data-classes.
    Conveys complete mapping for a given setup.
    Example:
        label := 'S17'
        bitmap_array := [
            FeedlineBitMap(id=0, bit_map=[0, 1, 2]),
            ...
        ]
    """
    id_label: str = field()
    bitmap_array: List[FeedlineBitMap] = field(default_factory=list)

    # region Class Properties
    @property
    def bitmap_lookup(self) -> Dict[int, FeedlineBitMap]:
        """:return: Lookup dictionary that maps FeedlineBitMap.id to FeedlineBitMap."""
        return {bitmap.id: bitmap for bitmap in self.bitmap_array}
    # endregion

    # region Class Methods
    def get_bitmap(self, feedline_nr: int) -> List[int]:
        """
        :param feedline_nr: Identifier number for feedline to retrieve bit-map from.
        :return: Array-like of resonator codeword bits corresponding to feedline identifier.
        """
        _lookup: Dict[int, FeedlineBitMap] = self.bitmap_lookup
        if feedline_nr not in _lookup:
            raise IdentifierFeedlineException(f'Bit map id {feedline_nr} not present in {list(_lookup.keys())}.')
        return _lookup[feedline_nr].bit_map
    # endregion


@dataclass()
class FeedlineMapCollection:
    """
    Contains a collection of FeedlineMap data-classes.
    Exposes getter for retrieving resonator bit-map for a specific:
        - FeedlineMap.id_label ('S5', 'S7', 'S17', etc.)
        - FeedlineBitMap.id (0, 1, 2, ...)
    """
    feedline_map_array: List[FeedlineMap] = field(default_factory=list)

    # region Class Properties
    @property
    def feedline_map_lookup(self) -> Dict[str, FeedlineMap]:
        """:return: Lookup dictionary that maps FeedlineMap.id_label to FeedlineMap."""
        return {bitmap.id_label: bitmap for bitmap in self.feedline_map_array}
    # endregion

    # region Class Methods
    def get_bitmap(self, map_id: str, feedline_nr: int) -> List[int]:
        """
        :param map_id: Identifier string for feedline map.
        :param feedline_nr: Identifier number for feedline to retrieve bit-map from.
        :return: Array-like of resonator codeword bits corresponding to feedline identifier.
        """
        _lookup: Dict[str, FeedlineMap] = self.feedline_map_lookup
        if map_id not in _lookup:
            raise IdentifierFeedlineException(f'Feedline map {map_id} not present in {list(_lookup.keys())}.')
        if feedline_nr not in _lookup[map_id].bitmap_lookup:
            raise IdentifierFeedlineException(f'Bit map id {feedline_nr} not present in {list(_lookup.keys())}.')
        return _lookup[map_id].bitmap_lookup[feedline_nr].bit_map
    # endregion


def get_default_map_collection() -> FeedlineMapCollection:
    """
    Purpose: backwards compatibility with hardcoded Base_RO_LutMan class constructor.
    :return: Default feedline-bit-map data-class.
    """
    map_collection: FeedlineMapCollection = FeedlineMapCollection(
        feedline_map_array=[
            FeedlineMap(
                id_label='S5',
                bitmap_array=[
                    FeedlineBitMap(
                        id=0,
                        bit_map=[0, 2, 3, 4],
                    ),
                    FeedlineBitMap(
                        id=1,
                        bit_map=[1],
                    )
                ]
            ),
            FeedlineMap(
                id_label='S7',
                bitmap_array=[
                    FeedlineBitMap(
                        id=0,
                        bit_map=[0, 2, 3, 5, 6],
                    ),
                    FeedlineBitMap(
                        id=1,
                        bit_map=[1, 4],
                    )
                ]
            ),
            FeedlineMap(
                id_label='S17',
                bitmap_array=[
                    FeedlineBitMap(
                        id=0,
                        bit_map=[6, 11],
                    ),
                    FeedlineBitMap(
                        id=1,
                        bit_map=[0, 1, 2, 3, 7, 8, 12, 13, 15],
                    ),
                    FeedlineBitMap(
                        id=2,
                        bit_map=[4, 5, 9, 10, 14, 16],
                    )
                ]
            )
        ]
    )
    return map_collection


def read_ro_lutman_bit_map() -> FeedlineMapCollection:
    """
    Reads config yaml, extracts FeedlineMapCollection and returns it.
    If yaml does not exist, create one and populate with default map collection.
    """
    file_path = get_yaml_file_path(filename=CONFIG_FILENAME)
    if not os.path.isfile(file_path):
        write_yaml(
            filename=CONFIG_FILENAME,
            packable=get_default_map_collection(),
            make_file=True,
        )
    return read_yaml(filename=CONFIG_FILENAME)


