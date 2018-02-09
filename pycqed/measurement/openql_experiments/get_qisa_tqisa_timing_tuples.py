import re
import json
"""
This file fixes issue 380 of https://github.com/DiCarloLab-Delft/PycQED_py3/issues/380
These are the assumptions I am making:
    1) Each kernel starts with the first prepz
    2) Other prepz within kernel is always within the first prepz and first measz
    3) CZ gate is always mapped to fl_cw_01
    4) The time tuples only show the time and gate cz
    5) Each instruction line starts with 'bs' string
Using the lazy way by separating the two files. 
In principle, we could do just regex on just the tqisa file,
but I am just not in the mood anymore to check for more regex...

-KKL 23/11/2017
"""

def get_qisa_tqisa_timing_tuples( qisa_file_path, tqisa_file_path, 
                                  CCL_json_config, output_path:str = None ) :
    # Load the hardware config json file
    # The idea is then to search for the codewords and then map it back to the gates
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
    if (output_path == None):
        output_file = qisa_file_path + ".out"
    else:
        output_file = output_path

    with open(output_file, 'w') as mod_qisa_file:
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
                        # Usually when we encounter our first measure, we would not expect a prepz
                        # So, let's call it the end of the kernel, and we turn on the first_prepz bool
                        if re.search(r"measz",line):
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
                codewords = re.split(r'bs\s1', line)
                # We now parse whether there is a | character
                try:

                    codewords2 = re.split(r'\s\|\s', codewords[1])

                    result = ( int(timing_num.group(0)), 
                               [ codewords2[0].strip() , codewords2[1].strip() ] 
                             )
                except:
                    result = ( int(timing_num.group(0)), codewords[1].strip() )
                
                time_tuples.append(result)

    return time_tuples