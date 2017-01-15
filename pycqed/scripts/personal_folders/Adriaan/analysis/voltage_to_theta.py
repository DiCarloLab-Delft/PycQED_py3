from pycqed.analysis import measurement_analysis as ma


class V_to_theta_ana(ma.TwoD_Analysis):
    def __init__(self, **kw):
        super().__init__(**kw)

