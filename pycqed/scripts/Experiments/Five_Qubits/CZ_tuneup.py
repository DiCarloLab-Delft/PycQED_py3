from pycqed.scripts.Experiments.Five_Qubits import CZ_cost_analysis as ca
import pycqed.measurement.detector_functions as det
import pycqed.measurement.sweep_functions as swf
import numpy as np
import pycqed.measurement.multi_qubit_module as mq_mod


class CPhase_cost_func_det(det.Soft_Detector):
    def __init__(self, device, qS, qCZ,
                 MC_nested,
                 amplitudes,
                 upload=True, **kw):

        self.name = 'CPhase_cost_func_det'
        self.detector_control = 'soft'
        self.upload = upload
        self.amplitudes = amplitudes
        self.device = device
        self.qS = qS
        self.qCZ = qCZ
        self.AWG = qCZ.AWG
        self.MC = MC_nested
        self.qCZ = qCZ
        self.qS = qS

        self.value_names = ['Maximum amp diff', 'missing swap pop']
        self.value_units = ['a.u.', 'a.u.']

        self.theta_f = self.qCZ.CZ_theta()
        l_coeffs = self.qCZ.CZ_lambda_coeffs()
        self.lambda1 = l_coeffs[0]
        self.lambda2 = l_coeffs[1]
        self.lambda3 = l_coeffs[2]
        self.pars_changed = True
        self.rSWAP_param = kw.pop('rSWAP_param', None)
        self.last_rSWAP_amp = self.rSWAP_param.get()

    def acquire_data_point(self, **kw):
        new_rSWAP_amp = self.rSWAP_param.get()
        if new_rSWAP_amp != self.last_rSWAP_amp:
            self.pars_changed = True
        if self.pars_changed:
            self.qCZ.CZ_theta(self.theta_f)
            l_coeffs = np.array([self.lambda1, self.lambda2, self.lambda3])
            self.qCZ.CZ_lambda_coeffs(l_coeffs)
            mq_mod.measure_SWAP_CZ_SWAP(self.device, self.qS.name,
                                        self.qCZ.name,
                                        self.amplitudes,
                                        MC=self.MC,
                                        sweep_qubit=self.qCZ.name,
                                        upload=True)

            self.pars_changed = False
        else:
            mq_mod.measure_SWAP_CZ_SWAP(self.device, self.qS.name,
                                        self.qCZ.name,
                                        self.amplitudes,
                                        MC=self.MC,
                                        sweep_qubit=self.qCZ.name,
                                        upload=False)
        a = ca.CPhase_2Q_amp_cost_analysis()
        return a.cost_func_val


# sweep functions
class lambda1_sweep(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'Lambda1'
        self.parameter_name = 'lambda 1'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_lambda1 = self.comp_det.lambda1
        if old_lambda1 != val:
            self.comp_det.lambda1 = val
            self.comp_det.pars_changed = True


class lambda2_sweep(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'Lambda2'
        self.parameter_name = 'lambda 2'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_lambda2 = self.comp_det.lambda2
        if old_lambda2 != val:
            self.comp_det.lambda2 = val
            self.comp_det.pars_changed = True


class lambda3_sweep(swf.Soft_Sweep):

    def __init__(self, comp_det, **kw):
        self.sweep_control = 'soft'
        self.name = 'Lambda3'
        self.parameter_name = 'lambda 3'
        self.unit = 'a.u.'
        self.set_kw()
        self.comp_det = comp_det

    def set_parameter(self, val):
        old_lambda3 = self.comp_det.lambda3
        if old_lambda3 != val:
            self.comp_det.lambda3 = val
            self.comp_det.pars_changed = True
