import re
import json
"""
This file fixes issue 380 of https://github.com/DiCarloLab-Delft/PycQED_py3/issues/380
These are the assumptions I am making:
    1) Each kernel starts with the first prepz
    2) prepz cannot be in between the kernel
    3) CZ gate is always mapped to fl_cw_01
    4) The time tuples only show the time and gate cz
    5) Each instruction line starts with 'bs' string
"""

def get_qisa_tqisa_timing_tuples(qisa_file_path, tqisa_file_path, CCL_json_config):
    # Load the hardware config json file (hardcoded for now)
    with open(CCL_json_config,'r') as file_json:
        config_map = json.load(file_json)

    # Set the counter for number of kernels encountered
    kernel_idx = 0

    # These are switches which depending on whether they have encountered their respective
    # instructions, they will return false or true
    # They are initially set True because the program has not seen anything yet!
    first_prepz = True
    first_CZ = True

    # Open the output file
    with open(qisa_file_path + ".mod", 'w') as mod_qisa_file:
        # Open the original qisa file
        with open(qisa_file_path, 'r') as qisa_file :

            # We start our search line by line. Do note that this is a single pass of the file
            for line in qisa_file:
                output_line = line

                if re.search(r"prepz",line):
                # We search for the first instance of the prepz
                    if first_prepz:
                        kernel_idx += 1
                        # We increment the kernel counter the first time we see a prepz
                        first_prepz = False
                        first_CZ = True
                    else:
                        first_prepz = True

                # We now search for the first instance of the CZ gate
                # Assumption here is that there can never be a CZ t0 | CZ t1 case...

                if re.search(r"fl_cw_01",line):
                    if first_CZ:
                        output_line = line.replace("fl_cw_01","cw_" + "{:02}".format(kernel_idx))
                        first_CZ = False
                    else:
                        output_line = line.replace("fl_cw_01","cw_" + "{:02}".format(0))


                # We write out the line to the output file
                mod_qisa_file.write(output_line)


    # We should be able to do all these things in a single pass
    time_tuples = []
    with open(tqisa_file_path, 'r') as tq_file:
        linenum = 0
        for line in tq_file:
            linenum += 1
            # Get instruction line
            if re.search(r"bs", line):
                # Get the timing number
                timing_num = re.search(r'\d+' ,line)
                # Get the codewords
                codewords = re.search(r'(?<=bs\s1)\w+', line)
                result = ( int(timing_num.group(0)), codewords.group(0) )
                print(result)
                time_tuples.append(result)

    return time_tuples