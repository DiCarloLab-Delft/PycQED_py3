import numpy as np
import scipy
from lmfit import minimize, Parameters, Parameter, report_fit


def cavity_traj(dt, eps, df, Xi, kappa):
    '''
    Simulates the trajectory of a cavity driven by a measurement pulse
    dt = time step
    eps = array with drive pulse amplitude
    df is detuning of drive relative to cavity resonance
    '''
    alpha_m = np.zeros(len(eps)+1)+0.j
    alpha_p = np.zeros(len(eps)+1)+0.j
    #print alpha_p
    k=1
    print(2.*np.pi*df+2.*np.pi*Xi)
    print(2.*np.pi*df-2.*np.pi*Xi)
    for epst in eps:
        #print epst
        alpha_p[k] = (-1.j*epst -
                        (1.j*(2.*np.pi*df+2.*np.pi*Xi) + 2.*np.pi*kappa/2.)*alpha_p[k-1]
                        )*dt+alpha_p[k-1]

        alpha_m[k] = (-1.j*epst -
                        (1.j*(2.*np.pi*df-2.*np.pi*Xi) + 2.*np.pi*kappa/2)*alpha_m[k-1]
                        )*dt+alpha_m[k-1]
        k+=1
    return [alpha_p,alpha_m]

def MID_epsd(*xP):
    #print xP
    x, r0, Gamma_d_eps = xP
    r = r0*np.exp(-(x)**2*abs(Gamma_d_eps))
    return r


def disp_hanger_S21_complex_sloped(*xP):
    '''
    '''
    if type(xP[0])== type(Parameters()):
        #print xP
        param = xP[0]
        (x,x0,Q,Qe,A,theta,phi_v,phi_0,dx) = (param['f'].value,
                            param['f0'].value,
                            param['Q'].value,
                            param['Qe'].value,
                            param['A'].value,
                            param['theta'].value,
                            param['phi_v'].value,
                            param['phi_0'].value,
                            param['df'].value)
    else:
        x,x0,Q,Qe,A,theta,phi_v, phi_0,dx = xP

    S21=(1.+dx*(x-x0)/x0)*np.exp(1.j*(phi_v*x+phi_0-phi_v*x[0]))*disp_hanger_S21_complex(*(x,x0,Q,Qe,A,theta))
    return S21


def disp_hanger_S21_complex(*xP):
    '''
    THis is THE hanger fitting function for fitting normalized hanger  (no offset)
    x,x0,Q,Qe,A,theta
    S21 = A*(1-Q/Q_c*exp(1.j*theta)/(1+2.j*Q(x-x0)/x0)
    where 1/Qi = 1/Q - Re{exp(1.j*theta)/Qc}
    '''
    #print type(xP[0])
    if type(xP[0])== type(Parameters()):
        param = xP[0]
        (x,x0,Q,Qe,A,theta) = (param['f'].value,
                            param['f0'].value,
                            param['Q'].value,
                            param['Qe'].value,
                            param['A'].value,
                            param['theta'].value)
    else:
        x,x0,Q,Qe,A,theta = xP
    S21 = A*(1.-Q/Qe*np.exp(1.j*theta)/(1.+2.j*Q*(x-x0)/x0))
    return S21
def disp_hanger_S21_power(*xP):
    '''
    see disp_hanger_S21_complex(*xP)
    '''
    x,x0,Q,Qe,A,theta = xP
    S21 = disp_hanger_S21_complex(*xP)
    return np.abs(S21)**2/A
def disp_hanger_S21_amplitude(*xP):
    '''
    see disp_hanger_S21_complex(*xP)
    '''
    S21 = disp_hanger_S21_complex(*xP)
    return np.abs(S21)

def disp_hanger_S21_amplitude_sloped(*xP):
    '''
    see disp_hanger_S21_complex(*xP)
    added param is slope "dx"
    '''
    if type(xP[0])== type(Parameters()):
        params = xP[0]
        dx,x,x0,A = params['df'].value,params['f'].value,params['f0'].value,params['A'].value
        xPs = xP
    else:
        x,x0,Q,Qe,A,theta,dx = xP
        xPs = (x,x0,Q,Qe,A,theta)
    S21 = disp_hanger_S21_complex(*xPs)
    return np.abs(S21)*(1.+dx*(x-x0)/x0)
def S21_resonance_sloped(*xP):
    '''
    f,A,f0,Q,Qe,slope,theta = fP
    '''
    #print '1'
    if type(xP[0])== type(Parameters()):
        pass
    else:
        f,A,f0,Q,Qca,slope,theta = xP
        xP = (f,f0,Q,Qca,A,theta,slope)
    return disp_hanger_S21_amplitude_sloped(*xP)

def Lorentzian(*xP):
    params = xP[0]
    dx,x,x0,A,y0 = params['df'].value,params['f'].value,params['f0'].value,params['A'].value,params['y0'].value
    return A*dx**2/4./((x-x0)**2+dx**2/4.)+y0
def Double_Lorentzian(*xP):
    params = xP[0]
    (x,y0,A1,x01,dx1,A2,x02,dx2) = (params['f'].value,
                    params['y0'].value,
                    params['A1'].value,
                    params['f01'].value,
                    params['df1'].value,
                     params['A2'].value,
                    params['f02'].value,
                    params['df2'].value)
    return A1*dx1**2/4./((x-x01)**2+dx1**2/4.)+A2*dx2**2/4./((x-x02)**2+dx2**2/4.)+y0

def Lorentz_params(p0):
    pars = Parameters()
    f,A,f0,df,y0 = p0[0],p0[1],p0[2],p0[3],p0[4]
    pars.add('A',value = A)
    pars.add('f',value = f.tolist(), vary = False)
    pars.add('f0',value = f0)
    pars.add('df', value = df, min = 0.)
    pars.add('y0', value = y0)
    return pars

def Exp_decay(*xP):
    params = xP[0]
    y0,A,T1,t = params['y0'].value,params['A'].value,params['T1'].value,params['t'].value
    return y0+A*np.exp(-t/T1)
def S21p0i2params(f,p0i):
    pars = Parameters()
    (A,f0,Q,Qe,dx,theta) = p0i
    pars.add('A',value = 1.e-3)
    pars.add('f',value = f,vary=False)
    pars.add('f0',value = 6.28, min= 0.)
    pars.add('Q',value = Q, min= 0.)
    pars.add('Qe',value = Qe, min= 0.)
    pars.add('Qi',expr = '1./(1./Q-1./Qe*cos(theta))', vary=False)
    pars.add('Qc',expr = 'Qe/cos(theta)', vary=False)
    pars.add('df',value = dx)
    pars.add('theta',value = theta,min=-np.pi/2,max=np.pi/2)
    return pars

def norm_S21(rqcqi):
    f0=1000.
    Qi= 1000.
    Qe = rqcqi*Qi
    Q = 1./(1./Qi+1./Qe)
    df = f0/Q
    f = np.linspace(f0-5*df,f0+5*df,10001)
    p0 = (1.,f0,Q,Qe,df,1.)
    param = S21p0i2params(f,p0)

    return np.abs(disp_hanger_S21_complex(param))

def norm_S21_theta(theta, rqcqi):
    f0=1000.
    Qi= 1000.
    Qe = rqcqi*Qi
    Q = 1./(1./Qi+1./Qe)
    df = f0/Q
    f = np.linspace(f0-5*df,f0+5*df,10001)
    p0 = (1.,f0,Q,Qe,df,theta)
    param = S21p0i2params(f,p0)

    return np.abs(disp_hanger_S21_complex(param))


def get_eta(r_off,r_ol,r_cl, u_r_off = 0.003, u_r_ol = 0.003, u_r_cl = 0.003):
    eta = log(r_ol/r_cl)/log(r_ol/r_off)
    deta_dr_ol = 1./r_ol*log(r_cl/r_off)/(log(r_ol/r_off))**2
    deta_dr_cl = -1./log(r_ol/r_off)*1./r_cl
    deta_dr_off = eta/log(r_ol/r_off)*1./r_off
    u_eta = np.sqrt((deta_dr_ol*u_r_ol)**2 +(deta_dr_off*u_r_off)**2+(deta_dr_cl*u_r_cl)**2)
    return eta, u_eta


def PSD(array, time_step):
    """
    PSD function by Niels
    """
    f_axis = np.fft.fftfreq(len(array), time_step)
    idx = np.argsort(f_axis)
    f_axis = f_axis[idx]
    period = time_step*len(array)
    psd = time_step*time_step/period*(np.abs(np.fft.fft(array)))**2
    psd = psd[idx]
    return f_axis[len(psd)//2:], psd[len(psd)//2:]
