"""
Created: 2020-07-12
Initial author: Victor Negirneac
Tools for extracting data
"""

import logging
from pycqed.analysis import measurement_analysis as ma
import pycqed.measurement.hdf5_data as hd5
import os
from datetime import datetime
import time

log = logging.getLogger(__name__)


def get_MC_settings(timestamp):
    """
    Retrieves Measurement Control setting from HDF5 file.
    """

    filepath = ma.a_tools.get_datafilepath_from_timestamp(timestamp)

    exctraction_spec = {
        "settings": ("MC settings", "attr:all_attr"),
        "begintime": ("MC settings/begintime", "dset"),
        "endtime": ("MC settings/endtime", "dset"),
        "preparetime": ("MC settings/preparetime", "dset"),
    }

    extracted = hd5.extract_pars_from_datafile(filepath, param_spec=exctraction_spec)

    for t_name in ["begintime", "endtime", "preparetime"]:
        struct = time.struct_time(extracted[t_name])
        dt = datetime.fromtimestamp(time.mktime(struct))
        extracted[t_name] = dt

    return extracted


def extract_qois_and_msmt_times(
    t_start: str,
    t_stop: str,
    label="",
    exact_label_match: bool = False,
    folder: str = None,
):
    """
    Extracts the all the `quantities_of_interest` from a measurement file
    and also the MC preparation and end times, for all the timestamps

    Return (<list of dicts of successful extractions>, <failed timestamps>)
    """
    timestamps = ma.a_tools.get_timestamps_in_range(
        timestamp_start=t_start,
        timestamp_end=t_stop,
        label=label,
        exact_label_match=exact_label_match,
        folder=folder,
    )
    timestamps.sort()
    extr_list = []
    failed_ts = []
    for ts in timestamps:
        try:
            # Get experiment times
            ext = get_MC_settings(ts)
            filepath = ma.a_tools.get_datafilepath_from_timestamp(ts)
            # Extract entire qois group
            dict_ = hd5.extract_pars_from_datafile(
                filepath,
                param_spec={"qois": ("Analysis/quantities_of_interest", "group")},
            )
            dict_["preparetime"] = ext["preparetime"]
            dict_["endtime"] = ext["endtime"]
            dict_["timestamp"] = ts
            dict_["filename"] = os.path.basename(filepath)
            extr_list.append(dict_)
        except Exception as e:
            failed_ts.append(ts)
            print(e)

    if failed_ts:
        print("Failed to extract from: ", failed_ts)
    return extr_list, failed_ts
