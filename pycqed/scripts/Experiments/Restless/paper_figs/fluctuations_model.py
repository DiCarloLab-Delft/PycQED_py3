# Some functions to help with the conversion based on the model in the SOM
import numpy as np
from uncertainties import ufloat
from uncertainties.umath import log


def assymetric_erorrs(p0, p1):
    P = (p0*(1-p1)+p1*(1-p0))/ ((1-p0)+(1-p1))
    return P

def inverse_assymetric_errors(P_total, p0):
    p1 = -(P_total*(p0-2)+p0)/(P_total-2*p0 + 1)
    return p1

print(inverse_assymetric_errors(assymetric_erorrs(0, 0.2), 0))

t_rest = 4.25e-6
T1_PSD = ufloat(22.7e-6, 3e-6)
# T1 = ufloat(19.3e-6, 2.1825e-6)
p0 = 0.0
p1 = 1-np.e**(-t_rest/T1_PSD)
# p1-1 = -e**()
# 1- p1 = e**()
# log(1-p1) = -t_rest/T1_PSD
p_naive = 1-np.e**(-t_rest/(2*T1_PSD))

print('P_assymetric: {:.4f}'.format(assymetric_erorrs(p0=p0, p1=p1)*100))

P_total = ufloat(0.117, 0.01)
# P_total = .117
p1 = inverse_assymetric_errors(P_total=P_total, p0=p0)
T1 = -t_rest/(log(1-p1))
print('PSD T1: {:.4f} us, CV: {:.4f}'.format(T1_PSD*1e6, T1_PSD.std_dev/T1_PSD.nominal_value))
print('expected T1 from model fit: {:.4f} us, CV:{:.4f}'.format(T1*1e6, T1.std_dev/T1.nominal_value))
