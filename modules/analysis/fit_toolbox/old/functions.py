from numpy import *
import numpy
import numpy as np
from . import hamil2 as h
import scipy

def sloped_lorentzian(x,p):
    '''

    x0 is center of peak
    A is amplitude of peak
    w is Full Width Half Maximum
    h is height of baselevel
    slope is slope of lorentz
    '''
    x0,A,w,h, slope = p[0],p[1],p[2],p[3], p[4]
    y = slope*x + lorentzian(x,p[:-1])
    return y

def lorentzian(x,p):
    '''
    x0 is center of peak
    A is amplitude of peak
    w is Full Width Half Maximum
    h is height of baselevel
    '''
    x0,A,w,h = p[0],p[1],p[2],p[3]
    y =h + A*(w/2.0)**2/ ((w/2.0)**2 + ((x - x0))**2)
    return y

def lorentzian(*xp):
    '''
    x0 is center of peak
    A is amplitude of peak
    w is Full Width Half Maximum
    h is height of baselevel
    '''
    x,x0,A,w,h = xp
    y =h + A*(w/2.0)**2/ ((w/2.0)**2 + ((x - x0))**2)
    return y


def square_root_lorentzian(x,p):
    '''
    x0 is center of peak
    A is amplitude of peak
    w is Full Width Half Maximum
    h is height of baselevel
    '''
    x0,A,w,h = p[0],p[1],p[2],p[3]
    y = (h + A*(w/2.0)**2/ ((w/2.0)**2 + ((x - x0))**2))**0.5
    return y    

def double_lorentzian(x,xc1, xc2,A1, A2, w1, w2 ,h):
    '''
    x0 is center of peak
    A is amplitude of peak
    w is Full Width Half Maximum
    h is height of baselevel
    '''
    y =h + A1 / (1.0 + ((x - xc1)/(w1/2.0))**2) + A2 / (1.0 + ((x - xc2)/(w2/2.0))**2)

    return y

def HF_split_lorentzian(f, P ,**kw):
    '''
    f0 is center of Lorentz
    A is amplitude of peak
    w is Full Width Half Maximum
    y0 is height of baselevel
    fhf= hyperfine splitting
    f = independent value
    '''
    #print P
    f0, A, w, y0 = P[0], P[1], P[2], P[3]
    fhf = kw.pop('fhf',0.00216)
    y =y0 + A*(1/ (1.0 + ((f - f0-fhf)/(w/2.0))**2) \
            +  1/ (1.0 + ((f - f0    )/(w/2.0))**2) \
            +  1/ (1.0 + ((f - f0+fhf)/(w/2.0))**2))
    return y

def HF_split_gaussian(f, P ,**kw):
    '''
    P = [mu (GHz), sigma(GHz), y0 (cnts), A (cnts)]
    '''
    #print P
    mu, A, sigma, y0 = P[0], P[1], P[2], P[3]
    fhf = kw.pop('fhf',0.00216)
    y =y0 + A*(gauss(f,[mu,sigma]) +
            gauss(f,[mu-fhf,sigma]) +
                gauss(f,[mu+fhf,sigma]))
    return y
def gauss(x, P, **kw):
    '''
    P = [mu, sigma]
    '''
    mu, sigma = P[0],P[1]
    return exp((-((x-mu)/sigma)**2)/2)

def gauss_echo(*xP, **kw):
    '''
    P = (x, y0,A,T2)
    '''
    x, y0,A,T2 = xP
    return y0+A*exp((-(x/T2)**2))
    
def exp_echo(*xP, **kw):
    '''
    P = (x, y0,A,T2)
    '''
    x, y0,A,T2 = xP
    return y0+A*exp((-(x/T2)))

def comb_echo(*xP, **kw):
    '''
    P = (x, y0,A,T21,T22)
    '''
    x, T22,T21,y0,A = xP
    return y0+A*exp(-(x/T21)-((x/T22))**2)

    
def HF_NC_split_gaussian(f, P ,**kw):
    '''
    P = [mu (MHz), sigma(MHz), y0 (cnts), A (cnts), fhf2 (MHz)]
    fhf2 is hyperfine splitting from carbon
    '''
    #print P
    mu, A, sigma, y0, fhf2 = P[0], P[1], P[2], P[3], P[4]
    fhf = kw.pop('fhf',2.16)
    y =y0 + A*(gauss(f,[mu-fhf2,sigma]) +
            gauss(f,[mu+fhf2,sigma]) +
            gauss(f,[mu-fhf-fhf2,sigma]) +
            gauss(f,[mu-fhf+fhf2,sigma]) +
            gauss(f,[mu+fhf-fhf2,sigma]) +
                gauss(f,[mu+fhf+fhf2,sigma]))
    return y

def cavity_anticrossing(f, P,**kw):
    '''
    P = [g, gamma, fr, kc, ki]
    g = coupling strength
    gamma = spin line width
    fr = cavity resonance frequency
    kc = cavity external dissipation rate
    ki = cavity internal dissipation rate
    '''
    alfa = pi - 2 * arcsin(1 / sqrt(3))
    B = matrix([[sin(alfa/2)],[0],[-cos(alfa/2)]])
    B_ampl = kw.pop('Bampl', 266.239) # should be given in mT
    HT = h.hamilB(B_ampl*B)
    freq = h.evalcalc(HT)
    lvls = h.evallvls(freq)


    g, gamma, fr,kc, ki = P[0], P[1], P[2], P[3], P[4]

    cplterm = abs(g)**2 *(1 / (1j * (f - lvls[0,0]/1000) - gamma / 2) +
            1 / (1j * (f - lvls[1,0]/1000) - gamma / 2) +
            1 / (1j * (f - lvls[2,0]/1000) - gamma / 2))
    y = abs(1 + kc / (1j * (f - fr) - (kc + ki) + cplterm))**2

def cavity_anticrossing(f, P,**kw):
    '''
    P = [g, gamma, fr, kc, ki]
    g = coupling strength
    gamma = spin line width
    fr = cavity resonance frequency
    kc = cavity external dissipation rate
    ki = cavity internal dissipation rate
    '''
    alfa = pi - 2 * arcsin(1 / sqrt(3))
    B = matrix([[sin(alfa/2)],[0],[-cos(alfa/2)]])
    B_ampl = kw.pop('Bampl', 266.239) # should be given in mT
    HT = h.hamilB(B_ampl*B)
    freq = h.evalcalc(HT)
    lvls = h.evallvls(freq)


    g, gamma, fr,kc, ki = P[0], P[1], P[2], P[3], P[4]

    cplterm = abs(g)**2 *(1 / (1j * (f - lvls[0,0]/1000) - gamma / 2) +
            1 / (1j * (f - lvls[1,0]/1000) - gamma / 2) +
            1 / (1j * (f - lvls[2,0]/1000) - gamma / 2))
    y = abs(1 + kc / (1j * (f - fr) - (kc + ki) + cplterm))**2
    return y 

def resonator_ensemble_B(f,B, P,**kw):
    '''
    S21 for cavity coupled to P1 spin esnemble
    P = (fr, ki,kc,g,gamma)
    '''
    
    alfa = pi - 2 * arcsin(1 / sqrt(3))
    dirB = matrix([[sin(alfa/2)],[0],[-cos(alfa/2)]])
    
    HT = h.hamilB(B*dirB)
    freq = h.evalcalc(HT)
    lvls = h.evallvls(freq)
    lvls0=lvls[0,0]
    lvls1=lvls[1,0]
    lvls2=lvls[2,0]
    
    fr, ki,kc,g,gamma = P
    
    cplterm = abs(g)**2 *(1 / (1.j * (f - lvls0/1000.) - gamma / 2) +
            1 / (1.j * (f - lvls1/1000.) - gamma / 2) +
            1 / (1.j * (f - lvls2/1000.) - gamma / 2))
    y = abs(1 + kc / (1.j * (f - fr) - (kc + ki) + cplterm))**2


    return y 



def cavity_lorentzian2(f, P,**kw):
    '''
    lorentzian fit for cavity resonance width dispersion parameter
    P = [fr, kc, ki]
    fr = cavity resonance frequency
    kc = cavity external dissipation rate
    ki = cavity internal dissipation rate
    '''
    alfa = pi - 2 * arcsin(1 / sqrt(3))
    B = matrix([[sin(alfa/2)],[0],[-cos(alfa/2)]])
    B_ampl = kw.pop('Bampl', 266.239) # should be given in mT
    HT = h.hamilB(B_ampl*B)
    freq = h.evalcalc(HT)
    lvls = h.evallvls(freq)

    fr,kc, ki, theta = P[0], P[1], P[2], P[3]
    y = abs(1 + e**(1j * theta) * kc / (1j * (f - fr) - (kc + ki)))**2


    return y 

def cavity_anticrossing2(f, P,**kw):
    '''
    P = [g, gamma, fr]
    g = coupling strength
    gamma = spin line width
    fr = cavity resonance frequency

    take cavity linewidths as constants
    '''
    alfa = pi - 2 * arcsin(1 / sqrt(3))
    B = matrix([[sin(alfa/2)],[0],[-cos(alfa/2)]])
    B_ampl = kw.pop('Bampl', 266.239) # should be given in mT
    HT = h.hamilB(B_ampl*B)
    freq = h.evalcalc(HT)
    lvls = h.evallvls(freq)
    ki = 9.5e-4
    kc = 4.2e-5
    g, gamma, fr= P[0], P[1], P[2]

    cplterm = abs(g)**2 *(1 / (1j * (f - lvls[0,0]/1000) - gamma / 2) +
            1 / (1j * (f - lvls[1,0]/1000) - gamma / 2) +
            1 / (1j * (f - lvls[2,0]/1000) - gamma / 2))
    y = abs(1 + kc / (1j * (f - fr) - (kc + ki) + cplterm))**2


    return y

def cavity_spin_resonance(f,p):
    '''
    p = [fs1, fs2, fs3, y, g, fc, kc, ki, theta]
    '''
    [fs1, fs2, fs3, y, g, fc, kc, ki, theta] = p
    abs(g)**2 *(1 / (1j * (f - fs2) - y / 2) +
            1 / (1j * (f - fs2) - y / 2) +
            1 / (1j * (f - fs3) - y / 2))
    y = abs(1 + e**(1j * theta)*kc / (1j * (f - fc) - (kc + ki) + cplterm))
    return y

def bare_cavity_resonance(*fP,**kw):
    '''
    cavity resonance with dispersion parameter
    P = [fr, kc, ki, theta, A]
    fr = cavity resonance frequency
    kc = cavity external dissipation rate
    ki = cavity internal dissipation rate
    '''
    

    f,fr,kc, ki, theta,A = fP
    y = A*sqrt(abs(1 + e**(1j * theta) * kc / (1j * (f - fr) - (kc + ki)))**2)
    return y

def diff_cavity_resonance(f,P,**kw):
    '''
    returns the difference of cavity resonance
    '''
    return scipy.diff(bare_cavity_resonance(f, P, **kw)) 

def bare_cavity_resonance_sloped(f, P,**kw):
    '''
    cavity resonance with dispersion parameter
    P = [fr, kc, ki, theta, A, dx]
    fr = cavity resonance frequency
    kc = cavity external dissipation rate
    ki = cavity internal dissipation rate
    theta= to account for standing waves
    A = amplitude off resonance
    dx = slope of baseline
    
    '''
    

    fr,kc, ki, theta,A, dx = abs(P[0]), abs(P[1]), abs(P[2]), P[3], abs(P[4]), P[5]
    y = A*(sqrt(abs(1 + e**(1j * theta) * kc / (1j * (f - fr) - (kc + ki)))**2)+dx*(f-fr))
    return y

def lorentzian_sloped(x,p):
    '''
    x0 is center of peak
    A is amplitude of peak
    w is Full Width Half Maximum
    h is height of baselevel
    dx is slope
    '''
    x0,A,w,h, dx = p[0],p[1],p[2],p[3],p[4]
    y =h + A*(w/2.0)**2/ ((w/2.0)**2 + ((x - x0))**2)+dx*(x-x0)
    return y

def lorentzian_sloped_disp(*xp):
    '''
    x0 is center of peak
    A is amplitude of peak
    w is Full Width Half Maximum
    h is height of baselevel
    dx is slope
    theta = dispersion parameter
    '''
    x,x0,A,w,h, dx ,theta= xp
    y =h + A*(cos(theta)*((w/2.0)**2/ ((w/2.0)**2 +((x - x0))**2))+ sin(theta)*((x -x0)/ ((w/2.0)**2 +((x -x0))**2)))+ dx*(x-x0)
    return y

def exponential_o(x,P):
    '''
    fits an exponential, returns 
    '''
    A,B,C = P
    y = A + B*exp(-x/C)
    return y
def gaussian(*xP):
    x,A,B,C,D = xP
    y = A + B*exp(-((x-D)/C)**2)
    return y
def exponential(*xP):
    '''
    fits an exponential, returns 
    '''
    x,A,B,C = xP
    y = A + B*exp(-x/C)
    return y
    
def cosine(*xP):
    '''
    cos
    x,y0,A,f,phi=xP
    '''
    
    x,y0,A,f,phi=xP
    return y0+A*numpy.cos(2*pi*f*x+phi)
def cosine_fixed_f(*xP):
    '''
    cos
    x,f,y0,A,phi=xP
    '''
    x,f,y0,A,phi=xP
    return y0+abs(A)*cos(2*pi*f*x+phi)

def damped_cos(*xP):
    '''
    exponentially decayig cos
    '''
    x,y_avg,A,f,phi,T2s = xP
    return y_avg+A*exp(-abs(x/T2s))*cos(2*pi*f*x+phi)

def cavity_traj(dt,eps,df,Xi,kappa):
    '''
    Simulates the trajectory of a cavity driven by a measurement pulse
    dt = time step
    eps = array with drive pulse amplitude
    df is detuning of drive relative to cavity resonance
    '''
    alpha_m = zeros(len(eps)+1)+0.j
    alpha_p = zeros(len(eps)+1)+0.j
    #print alpha_p
    k=1
    print(2.*pi*df+2.*pi*Xi)
    print(2.*pi*df-2.*pi*Xi)
    for epst in eps:
        #print epst
        alpha_p[k] = (-1.j*epst - 
                        (1.j*(2.*pi*df+2.*pi*Xi) + 2.*pi*kappa/2.)*alpha_p[k-1]
                        )*dt+alpha_p[k-1]
                       
        alpha_m[k] = (-1.j*epst - 
                        (1.j*(2.*pi*df-2.*pi*Xi) + 2.*pi*kappa/2)*alpha_m[k-1]
                        )*dt+alpha_m[k-1]
        k+=1
    return [alpha_p,alpha_m]    

def MID_epsd(*xP):
    #print xP
    x,r0,Gamma_d_eps = xP
    r = r0*exp(-(x)**2*abs(Gamma_d_eps))
    return r

import numpy as np
  
def disp_hanger_S21_complex(*xP):
    '''
    THis is THE hanger fitting function for fitting normalized hanger  (no offset)
    x,x0,Q,Qe,A,theta
    S21 = A*(1-Q/Q_c*exp(1.j*theta)/(1+2.j*Q(x-x0)/x0)
    where 1/Qi = 1/Q - Re{exp(1.j*theta)/Qc}
    '''
    x,x0,Q,Qe,A,theta = xP
    S21 = A*(1-Q/Qe*np.exp(1.j*theta)/(1+2.j*Q*(x-x0)/x0))
    return S21
def disp_hanger_S21_power(*xP):
    '''
    see disp_hanger_S21_complex(*xP)
    '''
    S21 = disp_hanger_S21_complex(*xP)
    return np.abs(S21)**2
def disp_hanger_S21_amplitude(*xP):
    '''
    see disp_hanger_S21_complex(*xP)
    '''
    S21 = disp_hanger_S21_complex(*xP)
    return np.abs(S21)

def disp_hanger_S21_amplitude_sloped(*xP):
    '''
    see disp_hanger_S21_complex(*xP)
    '''
    
    x,x0,Q,Qe,A,theta,dx = xP
    xPs = (x,x0,Q,Qe,A,theta)
    S21 = disp_hanger_S21_complex(*xPs)
    return np.abs(S21)+dx*(x-x0)
def S21_resonance_sloped(*fP,**kw):
    '''
    cavity resonance with dispersion parameter
    f,A,f0,Q,Qca,slope,theta = fP
    
    '''
    
    f,A,f0,Q,Qca,slope,theta = fP
    xP = (f,f0,Q,Qca,A,theta,slope)
    print('plot 1')
    y1 = 0#disp_hanger_S21_amplitude_sloped(*xP)
    df=f-f0
    y = A*(abs(
        (( (1-abs(1.*Q)/abs(1.*Qca)*(e**(1.j * theta)))+1.j*2*abs(1.*Q)*df/f0)/(1+1.j*2*abs(1.*Q)*df/f0)))
            +slope*(f-f0))

    return y-y1
    
def S21_resonance_sloped(*xP):
    '''
    f,A,f0,Q,Qca,slope,theta = fP
    '''
    print('plot 2')
    f,A,f0,Q,Qca,slope,theta = xP
    xP = (f,f0,Q,Qca,A,theta,slope)
    return disp_hanger_S21_amplitude_sloped(*xP)
    
    
def get_eta(r_off,r_ol,r_cl, u_r_off = 0.003, u_r_ol = 0.003, u_r_cl = 0.003):
    eta = log(r_ol/r_cl)/log(r_ol/r_off)
    deta_dr_ol = 1./r_ol*log(r_cl/r_off)/(log(r_ol/r_off))**2
    deta_dr_cl = -1./log(r_ol/r_off)*1./r_cl
    deta_dr_off = eta/log(r_ol/r_off)*1./r_off
    u_eta = np.sqrt((deta_dr_ol*u_r_ol)**2 +(deta_dr_off*u_r_off)**2+(deta_dr_cl*u_r_cl)**2)
    return eta, u_eta