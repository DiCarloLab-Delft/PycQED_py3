import numpy as np
from fit_toolbox import fit
from fit_toolbox import functions as fn
from fit_toolbox import guess_initial_values as giv
import matplotlib.pyplot as plt
from time import time
from qt import Data
import qt
#from numpy import *
import os
from os import path
import win32api, _thread, ctypes
import cmath

import numpy as np
def store_scripts(filepath):
    from shutil import copytree
    import os
    if not os.path.exists(os.path.join(filepath,'scripts')):
        #fol,fil = os.path.split(fp)
        copytree('D:\\qtlab\\scripts',os.path.join(filepath,'scripts'))
    else: 
        print('scripts already stored')
## resonator
def find_datestr(tstr):
    
    lookfor = tstr
    print(lookfor)
    for root, dirs, files in os.walk('D:\qtlab\data'):
        #print "searching", root
        for dir in dirs:
            if lookfor in dir:
                datestr = root
                print("found: %s at %s" % (lookfor,datestr))
                break
    return datestr[-8:]
    
def littleEndian(data, bytes=4):
    return [(data >> ofs) & 0xFF for ofs in (0, 8, 16, 24)[:bytes]]

def fit_S21_resonance_complex(*args,**kw):
    '''
    input:
    x,y
    returns fitresults with:

    fr[0]=(A,f0,Q,s21min,slope,theta)
    '''
    #print 'no slope'
    if type(args[0])==Data:
        dat = args[0]
        xy = dat.get_data()
        x= xy[:,0]
        y= xy[:,1]
    else:
        x=args[0]
        y=args[1]
    silent = kw.pop('silent',True)
    p0i=param_estimator_complex(x,y)

    if not silent:
        print('init guess: ',p0i)
    pars = S21p0i2paramsc(x,p0i)
    fr = fit.fitxy(pars,y, fn.disp_hanger_S21_complex_sloped, ret_fit=True)   
    if not silent:
        fit.print_fitres(pars)
    fitpars=params2tuple(pars)
    fitparserr = paramserr2tuple(pars)

    output = {'fitpars':fitpars,'fitparserr':fitparserr,'fit':fr[1],'fitparsobj':pars,'parsinit':p0i}
    return output
    #return pfit,errp,fr[1],fr[0],pars,p0i

def param_estimator_complex(x,y):
    #Temps to make param estimations
    ya = np.abs(y[:])
    phi_0 = cmath.phase(y[0])
    phi_v = 0
    s21min = np.abs(min(ya)/max(ya))
    
    y_rotated = y*np.exp(-1.j*(phi_v*x+phi_0-phi_v*x[0]))
    y_min = np.argmin(y_rotated.imag)
    y_max = np.argmax(y_rotated.imag)

    #Parameters
    A = (ya[0]+ya[-1])/2.
    f0 = x[ya[:].tolist().index(min(ya[:]))]
    Q = f0/(abs(x[y_min]-x[y_max]))
    Qca = np.abs(Q/np.abs(1-s21min))
    #-440.
    phi_0 = phi_0
    p0i = (A,f0,Q,Qca,phi_0,phi_v)
    return p0i

    
    
    
    
def S21p0i2paramsc(f,p0i, **kw):
    pars = fit.Parameters()
    (A,f0,Q,Qe, phi_0,phi_v) = p0i
    pars.add('A',value = A)
    pars.add('f',value = f,vary=False)
    pars.add('f0',value = f0)
    pars.add('Q',value = Q)
    pars.add('Qe',value = Qe)
    pars.add('Qi',expr = '1./(1./Q-1./Qe*cos(theta))', vary=False)
    pars.add('Qc',expr = 'Qe/cos(theta)', vary=False)
    pars.add('df',value = 0.,vary=True)  
    pars.add('theta',value = 0.)
    pars.add('phi_v',value=phi_v)
    pars.add('phi_0',value=phi_0)#,min=-np.pi,max=np.pi)
    return pars

    
def fit_S21_resonance(*args,**kw):
    '''
    input:
    x,y
    returns fitresults with:

    fr[0]=(A,f0,Q,s21min,slope,theta)
    '''
    #print 'no slope'
    if type(args[0])==Data:
        dat = args[0]
        xy = dat.get_data()
        x= xy[:,0]
        y= xy[:,1]
        
    else:
        x=args[0]
        y=np.abs(args[1])

    p0i = param_estimator(x,y)
    silent = kw.pop('silent',True)
    if not silent:
        print('init guess: ',p0i)
    pars = S21p0i2params(x,p0i)
    #print 'init_guess', p0i
    fr= fit.fitxy(pars,y,  fn.S21_resonance_sloped, ret_fit=True)   
    if not silent:
        fit.print_fitres(pars)
    fitpars=params2tuple(pars)
    fitparserr = paramserr2tuple(pars)
    output = {'fitpars':fitpars,'parserr':fitparserr,'fit':fr[1],'fitparsobj':pars,'parsinit':p0i}
    return output
    #return pfit,errp,fr[1],fr[0],pars,p0i
    
def param_estimator(x,y):
    #Temps to make param estimations
    ya = np.abs(y[:])
    s21min = np.abs(min(ya)/max(ya))
    y_min = ya.tolist().index(min(ya[:]))
    y_max = ya.tolist().index(max(ya[:]))
    #Parameters
    slope = 0.
    theta = 0.
    A = (ya[0]+ya[-1])/2.
    f0 = x[y_min]
    Q = f0/(abs(x[y_min]-x[y_max]))
    Qca = np.abs(Q/np.abs(1-s21min))
    p0i = (A,f0,Q,Qca,slope,theta)
    return p0i
    
def S21p0i2params(f,p0i):
    pars = fit.Parameters()
    (A,f0,Q,Qe,dx,theta) = p0i
    pars.add('A',value = A, min= 0.)
    pars.add('f',value = f,vary=False)
    pars.add('f0',value = f0, min= 0.)
    pars.add('Q',value = Q, min= 0.)
    pars.add('Qe',value = Qe, min= 0.)
    pars.add('Qi',expr = '1./(1./Q-1./Qe*cos(theta))', vary=False)
    pars.add('Qc',expr = 'Qe/cos(theta)', vary=False)
    pars.add('df',value = 0.)  
    pars.add('theta',value = 0.,min=-np.pi/2,max=np.pi/2)     
    return pars

   
def params2tuple(pars, *args):
    '''
    p = (A,f0,Q,Qe,df,theta)  
    '''
    if len(args) == 1:
        keys = args[0]
    else:
        keys = ['A','f0','Q','Qe','df','theta']

    p = ()
    for key in keys:
        p += (pars[key].value,)   
    return p

def paramserr2tuple(pars, *args):
    '''
    p = (A,f0,Q,Qe,df,theta)  
    '''
    if len(args) == 1:
        keys = args[0]
    else:
        keys = ['A','f0','Q','Qe','df','theta']

    errp = ()
    for key in keys:
        errp += (pars[key].stderr,)                  
    return errp

def fit_lorentzian(x,y, plot=True):
    y0 = (y[0]+y[-1])/2.
    peak = np.abs((np.max(y)-y0)) > np.abs((np.min(y)-y0))
    print('peak = ',peak, y0, np.max(y),np.min(y))
    
    if peak:
        A = np.max(y)-y0
        f0 = x[y[:].tolist().index(max(y[:]))]
        find_y2 = (y-y0)>A/2
    else:
        A = np.min(y)-y0
        f0 = x[y[:].tolist().index(min(y[:]))]
        find_y2 = (y-y0)<A/2
    ind_df = find_y2.tolist().index(True)
    df = 2*np.abs(x[ind_df] - f0)
    pars = fit.Parameters()
    pars.add('f',value = x, vary = False)
    pars.add('A',value = A)
    pars.add('f0',value = f0)
    pars.add('df',value = df)
    pars.add('y0',value = y0)
    fit.print_fitres(pars)
    fit.fitxy(pars,y,fn.Lorentzian)
    fit.print_fitres(pars)
    yfit = fn.Lorentzian(pars)
    if plot:
        plt.plot(x,yfit, label = 'fit: f0 = %.4f, df = %.1f'%(pars['f0'].value,1e3*(pars['df'].value)))
        plt.legend()
    return pars, fn.Lorentzian(pars)
    
### misc    


def notetoself(mes):
    f= open('D:\\qtlab\\user\\user_scripts\\TODO.txt','a')
    f.write('- '+mes+'\n\n')
    f.close()
def shownotestoself():
    f= open('D:\\qtlab\\user\\user_scripts\\TODO.txt','r')
    mes = f.readlines()
    f.close()
    for m in mes:
        print(m[:-1])

### Conversions

pi,log = np.pi, np.log
e_ = 1.6e-19
h = 6.62e-34
phi_0 = 2.e-15
hbar = h/2/np.pi
k_B = 1.38e-23
eps_0 = 8.854e-12
mu_0 = 4*np.pi*1e-7



def VtodBm(V):
    '''
    Converts voltage in 50 Ohm to dBm
    '''
    P=(V**2)/50/np.sqrt(2.)
    return 10*np.log(P/1e-3)/np.log(10)

def photon_number(eps, kappa):
    '''
    steady state photon number from drive strenghth eps_d and kappa 
    '''
    n_bar = (2*eps/kappa)**2 #kappa=2pi * decay rate
    return n_bar

def dBm2nbar(P, f0, Ql, Qi):
    '''
    Calculates n_bar for hangers form the power in the feedline
    '''
    Pin = 10**(P/10.)*1e-3
    
    Pres = (1-(1.*Ql/Qi)**2)*Pin
    print(Pres)
    n_in = Pres/(h*f0)/(f0/Ql)
    return n_in
    
def C_to_Ec(C):
    '''
    returns Ec in GHz from the capacitance
    '''
    return e_**2/2/C/h/1e9

def Ic_to_Ej(Ic):
    '''
    returns Ej in GHz from Ic 
    '''
    return Ic*phi_0/2/pi/h/1e9

def EjEc_to_f(Ej,Ec): 
    '''
    Calculates transmon f_ge from Ec and Ej
    '''
    return np.sqrt(8*Ej*Ec)
def IctoLj(Ic):
    return phi_0/2/np.pi/Ic
    
def IcC_to_f(Ic,C):
    '''
    Calculates transmon f_ge from Ic and C
    '''
    Ec = C_to_Ec(C)
    Ej = Ic_to_Ej(Ic)
    return EjEc_to_f(Ej,Ec)

    
def allclose():
    while not not plt.get_fignums():
        plt.close()

from instrument import Instrument
import types


# class VNA_ATT:

#     def __init__(self, VNA_ins = None, VAtt_ins = None):
#         #Instrument.__init__(self, name, tags=['virtual'])
#         self.VNA = VNA_ins
#         self.VAtt = VAtt_ins
#         self.get_funcs()
#         def set_power(pow):
#             if pow<-60:
#                 self.VAtt.set_ch1_attenuation(60)
#                 qt.msleep(0.2)
#                 self.VAtt.set_ch2_attenuation(-pow-60)
#                 qt.msleep(0.2)
#             else:
#                 self.VAtt.set_ch2_attenuation(0)
#                 qt.msleep(0.2)
#                 self.VAtt.set_ch1_attenuation(-pow)
#                 qt.msleep(0.2)
                
            
#         self.set_power = set_power
#         def prepare_sweep(fsta,fsto, npoints, tint, pow,navg):
#             self.set_power(pow)
#             self.VNA.prepare_sweep(fsta,fsto, npoints, tint, 0,navg)
#         self.prepare_sweep = prepare_sweep
#     def get_funcs(self):
#         pars = self.VNA.get_parameters()
#         for par in pars.keys():
            
#             try:    
#                 setattr(self,'set_%s'%par,pars[par]["set_func"])
#             except:
#                 pass
#             try:
#                 setattr(self,'get_%s'%par,pars[par]["get_func"])
#             except:
#                 pass
#         pars = self.VNA.get_functions()
#         for par in pars.keys():
#             setattr(self,'%s'%par,getattr(self.VNA, par)  )
#         pars = self.VAtt.get_parameters() 
#         for par in pars.keys():
#             print par
#             try: 
                
#                 setattr(self,'set_%s'%par,getattr(self.VAtt, 'set_%s'%par))
#             except:
#                 print 'no set_%s'%par
#                 pass
#             try:
#                 setattr(self,'get_%s'%par,getattr(self.VAtt, 'set_%s'%par))
#             except:
#                 'no get_%s'%par
#                 pass        
