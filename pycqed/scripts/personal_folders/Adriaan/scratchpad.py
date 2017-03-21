UHFQC_1.quex_rl_source(2)
# only one sample to average over
UHFQC_1.quex_rl_length(1)
UHFQC_1.quex_rl_avgcnt(
    int(np.log2(HS.nr_averages())))
UHFQC_1.quex_wint_length(
    int(HS.RO_length()*1.8e9))
# Configure the result logger to not do any averaging
# The AWG program uses userregs/0 to define the number o
# iterations in the loop
UHFQC_1.awgs_0_userregs_0(
    int(HS.nr_averages()))
UHFQC_1.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg
UHFQC_1.acquisition_initialize([0, 1], 'rl')







sweep_points=None
if d.AWG is not None:
    d.AWG.stop()
if sweep_points is None:
    d.nr_sweep_points = 1
else:
    d.nr_sweep_points = 1
# # this sets the result to integration and rotation outcome
if d.cross_talk_suppression:
    # 2/0/1 raw/crosstalk supressed /digitized
    d.UHFQC.quex_rl_source(0)
else:
    # 2/0/1 raw/crosstalk supressed /digitized
    d.UHFQC.quex_rl_source(2)
d.UHFQC.quex_rl_length(d.nr_sweep_points)
d.UHFQC.quex_rl_avgcnt(int(np.log2(d.nr_averages)))
d.UHFQC.quex_wint_length(int(d.integration_length*(1.8e9)))
# Configure the result logger to not do any averaging
# The AWG program uses userregs/0 to define the number o iterations in
# the loop
d.UHFQC.awgs_0_userregs_0(
    int(d.nr_averages*d.nr_sweep_points))
d.UHFQC.awgs_0_userregs_1(0)  # 0 for rl, 1 for iavg
# d.UHFQC.awgs_0_single(1)
UHFQC_1.acquisition_initialize([0.0, 1.0], 'rl')
# d.UHFQC.acquisition_initialize(channels=d.channels, mode='rl')