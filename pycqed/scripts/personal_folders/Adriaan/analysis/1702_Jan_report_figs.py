from pycqed.analysis import measurement_analysis as ma
bell_tomos = ['']*4
for target_bell in range(4):
    bell_tomos[target_bell] = ma.Tomo_Multiplexed(
        label='BellTomo_{}'.format(target_bell),
        MLE=True, close_fig=False, target_bell=target_bell)
