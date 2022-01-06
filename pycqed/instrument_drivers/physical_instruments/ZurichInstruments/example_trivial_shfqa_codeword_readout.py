# Copyright 2021 Zurich Instruments AG

"""
Requirements:
    * LabOne Version >= 22.1
    * Instruments:
        1 x SHFQA Instrument
    * Loopback configuration between input and output of channel 0

Usage:
    example_trivial_shfqa_codeword_readout.py [options] <device_id>
    example_trivial_shfqa_codeword_readout.py -h | --help

Arguments:
    <device_id>  The ID of the device to run the example with. [device_type: SHFQA]

Options:
    -h --help                 Show this screen.
    -s --server_host IP       Hostname or IP address of the dataserver [default: localhost]
    -p --server_port PORT     Port number of the data server [default: 8004]

Raises:
    Exception     If the specified device does not match the requirements.
    RuntimeError  If the device is not "discoverable" from the API.

See the "LabOne Programming Manual" for further help, available:
    - On Windows via the Start-Menu:
      Programs -> Zurich Instruments -> Documentation
    - On Linux in the LabOne .tar.gz archive in the "Documentation"
      sub-folder.
"""
import numpy as np
import zhinst.utils
import zhinst.deviceutils.shfqa as shfqa_utils
import helper_commons
import time


def run_example(
    device_id: str,
    server_host: str = "localhost",
    server_port: int = 8004,
):
    apilevel_example = 6
    (daq, _, _) = zhinst.utils.create_api_session(
        device_id, apilevel_example, server_host=server_host, server_port=server_port
    )

    channel_index = 0
    readout_duration = 600e-9

    shfqa_utils.configure_channel(
        daq,
        device_id,
        channel_index,
        center_frequency=5e9,
        input_range=0,
        output_range=-5,
        mode="readout",
    )
    path = f"/{device_id}/qachannels/{channel_index}/"
    daq.setInt(path + "input/on", 1)
    daq.setInt(path + "output/on", 1)

    # DIO configuration
    daq.setInt("/dev12023/dios/0/drive", 3)
    daq.setInt("/dev12023/dios/0/output", 0)

    shfqa_utils.configure_scope(
        daq,
        device_id,
        input_select={0: f"channel{channel_index}_signal_input"},
        num_samples=int(readout_duration * shfqa_utils.SHFQA_SAMPLING_FREQUENCY),
        trigger_input=f"channel{channel_index}_sequencer_monitor0",
        num_segments=1,
        num_averages=1,
        trigger_delay=200e-9,
    )
    excitation_pulses = helper_commons.generate_flat_top_gaussian(
        frequencies=np.linspace(2e6, 32e6, 2),
        pulse_duration=readout_duration,
        rise_fall_time=10e-9,
        sampling_rate=shfqa_utils.SHFQA_SAMPLING_FREQUENCY,
    )
    shfqa_utils.write_to_waveform_memory(
        daq, device_id, channel_index, waveforms=excitation_pulses, clear_existing=False
    )
    shfqa_utils.configure_sequencer_triggering(
        daq, device_id, channel_index, aux_trigger="software_trigger0"
    )
    shfqa_utils.load_sequencer_program(
        daq,
        device_id,
        channel_index,
        sequencer_program="""
waitDIOTrigger();
const CW_MASK = 0xffff;
var cwd = getDIOTriggered() & CW_MASK;
switch(cwd){
case 1:
startQA(QA_GEN_0, 0x0, true,  0, 0x0);
case 3:
startQA(QA_GEN_1, 0x0, true,  0, 0x0);
}
                """,
    )

    shfqa_utils.enable_scope(daq, device_id, single=0)

    cwds = [1, 3]
    for cwd in cwds:
        shfqa_utils.enable_sequencer(daq, device_id, channel_index, single=1)
        daq.syncSetInt("/dev12023/dios/0/output", cwd)
        daq.syncSetInt("/dev12023/dios/0/output", 0)
        time.sleep(1)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    cli_util_path = Path(__file__).resolve().parent / "../../utils/python"
    sys.path.insert(0, str(cli_util_path))
    cli_utils = __import__("cli_utils")
    cli_utils.run_commandline(run_example, __doc__)
    sys.path.remove(str(cli_util_path))
