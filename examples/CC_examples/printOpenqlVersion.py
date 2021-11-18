import openql.openql as ql
from os.path import join, dirname, isfile

print(ql.get_version())

if 1:
	output_dir = join(dirname(__file__), 'output')
	ql.set_option('output_dir', output_dir)

if 1:
    print(ql.get_option('output_dir'))

