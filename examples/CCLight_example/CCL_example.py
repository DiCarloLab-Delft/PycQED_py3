
# coding: utf-8

# # This is a demonstration of how to use the CC-Light driver

# ## Start by initialization of the driver and any other important modules

from pycqed.instrument_drivers.physical_instruments.QuTech_CCL import CCL

"""
This is the driver initialization. We name this model as CCL_demo,
and connect to the hardware IP address of 192.168.0.252 using port 5025
"""
# ccl = CCL('CCL_demo', address='192.168.0.252', port=5025)
ccl = CCL('CCL_demo', address='192.168.42.11', port=5025)


# Now that we're finished with the initialization, we can start using
# the hardware!

"""
Let's start by listing out all the available parameters we can use
"""
ccl.parameters

"""
As we can see, there are quite a few functions we can test out. Let's test
them all. We start with IDN
"""
ccl.IDN()


"""
Now that the easy part is out of the way, let's ask CCLight if the timing
queue is empty
"""
ccl.timing_queue_empty()


"""
Let's test a random vsm channel. We set and then get the value
"""
ccl.vsm_channel_delay12(2)
ccl.vsm_channel_delay12()


"""
Nothing interesting there. Let's move on to more fun stuff. Let's set and get
the number of append points
"""
ccl.num_append_pts(6)
ccl.num_append_pts()


"""
This time, let's mess things up with the setting of the num_append_pts
parameter. Let's give a stupidly high value
"""
ccl.num_append_pts(1337)


"""
At least we know now that num_append_pts parameter should be between 0 and 7
inclusive. These ranges are automatically obtained from the hardware.
No need to mess about with them.
"""
ccl.num_append_pts(0)


"""
Let's enable the QuMA processor now, and then check the state
"""
ccl.enable(1)
ccl.enable()


"""
Let's upload our microcode to the control store
"""
ccl.upload_microcode('./microcode_example.txt')


"""
Let's next upload our instructions
"""
ccl.upload_instructions('./qisa_test_assembly/test_s_mask.qisa')


"""
We next run
"""
ccl.run(1)


"""
Suppose now we check if the execution is done
"""
ccl.execution_done()


"""
Now that it is complete, we turn off the run
"""
ccl.run(0)


"""
We then upload our next set of instructions
"""
ccl.upload_instructions('./qisa_test_assembly/test_assembly.qisa')


"""
And we run again...
"""
ccl.run(1)


"""
Check that the processor is running
"""
ccl.run()


"""
And then check if execution is done
"""
ccl.execution_done()


"""
All done! So, we turn off run and we disable the processor
"""
ccl.run(0)
ccl.enable(0)


"""
Let's clean up by disconnecting and tearing down the instrument
"""
ccl.close()

print("example ran successfully")

# So, you are now done with the walk through of the various functions on how
# to use the CCL python driver
