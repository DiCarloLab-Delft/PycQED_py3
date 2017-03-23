
# working_file_name=r'D:\\githubrepos\\pycqed_py3\\pycqed\\waveform_control_CC\\micro_instruction_files\\pulsed_spec.qumis'

working_file_name=r'D:\GitHubRepos\PycQED_py3\pycqed\measurement\waveform_control_CC\micro_instruction_files\pulsed_spec.qumis'

CBox.load_instructions(working_file_name)
CBox.start()


broken_file_name=r'D:\GitHubRepos\PycQED_py3\pycqed\measurement\waveform_control_CC\micro_instruction_files\CW_RO_sequence.qumis'
CBox.load_instructions(broken_file_name)
CBox.start()
