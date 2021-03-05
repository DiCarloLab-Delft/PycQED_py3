from importlib import reload
import json

#from . import HAL as hal
#from . import ShapeLib as s
import pycqed # just to guarantee pycqed is imported before qcodes (Yuk)
import ShapeLib as s
import HAL as hal



x = s.ge('ge')
x.mw_amp180 = 180
x.parameters


# HAL configuration should also be usable for OpenQL (or other compilers)
# FIXME: allow instruments within instruments?
# CC: define locations per slot, how to link to connected instrument, link to name?
with open("HAL_config.json", "r") as fp:
    cfg = json.load(fp)
h = hal.HAL()
h.from_JSON(cfg)

if 1:
    mw = hal.MicrowaveOutput("mwo")
    mw.lo.frequency = 4e9
    print(mw.snapshot())
    ##print(mw.print_readable_snapshot())

    #print(mw.parameters)
    print(mw.submodules)

