import numpy as np
import scipy
import scipy.linalg.basic as slb
from  scipy.optimize import leastsq
from . import functions
import qt
from qt import plot as plot

def _residuals(p0, x, y, fit_func):
    return y-fit_func(x,*p0)

def _residuals_m(p0, fit_func, *x):
    '''
    residual for fnctions y= f(u,v,w,...)
    x = variable length tuple with one dependent variable at the end
    x=(u,w,v,y)
    u,v,w,y all have the same length
    '''
    p0=tuple(p0)
    print(p0)
    qt.msleep()
    
    return x[-1]-fit_func(*(x[:-1]+p0))

def fit_func_example(x1,x2,p1,p2,p3,p4):
    y=(x+p1)*p2**p3/p4
    return y

#fit1D((x1,x2,p1,p2),y,(p3,p4))

def fit_func_ex(x1,x2,p1,p2,p4,p3):
    return fit_func_example(x1,x2,p1,p2,p3,p4)

def fit_func_examplep1(x,p1,p2,p3,p4):
    y=fit_func_example(p1,x,p2,p3,p4)
    return y
def _residuals2(p0, x1,x2, y, fit_func):
    return y-fit_func(x1,x2,p0)


def fit1D(x,y,fit_func, **kw):
    '''
    parameters
    x : tuple of indep. variables
    y : dep. var. (data)
    fit_func : function to be fitted structure fit_fun(x, p0) with p0 parameters
    
    known KW
    init_guess : vector of initial estimates for p0
    guess_func : function that guesses the initial parameters p0
    full_output : True by default
    onscreen : print reuslts on screen
    ret_fit : return an array fit_func(x, Pfit)
    plot_results=False
    '''
    full_output = kw.pop('full_output',True)
    onscreen = kw.pop('onscreen',True)
    kw_c=kw.copy()
    qt.mstart()
    
    try:
        p0 = kw_c.pop('init_guess')
    except KeyError:
        print('No initial guess given, trying guess func instead')
        p0 = kw_c.pop('guess_func')(x,y) 
    if type(x)==type((0,)):
        var = x+(y,)
    else :
        x=(x,)
        var=x+(y,)

    if type(p0)==type((0,)):
        pass
    else :
        p0=tuple(p0)

    plres=kw.pop('plot_results', False)
    if plres:
        pltr=plot(name='fit monitor')
        pltr.clear()
        init_guess_arr = fit_func(*(x+tuple(p0)))
        plot(x[0], init_guess_arr, title = 'initial guess',name='fit monitor')
        plot(x[0], y, title = 'data',name='fit monitor')
        



    plsq,cov,info,mesg,success = leastsq(_residuals_m, \
            p0, args=(fit_func,)+var, full_output = 1, maxfev=10000,xtol=kw.pop('xtol',1.e-10))
    try :
        len(plsq)
    except TypeError :
        plsq=[plsq]
    if onscreen:
        print('plsq',plsq)
        print('mesg',mesg)
        print(cov)
        #print 'info: ',info
    dof = max(np.shape(y)) - len(plsq)
    errors = estimate_errors(plsq,info,cov,dof)
    k=0
    if plres:
        
        fresarr = fit_func(*(x+tuple(plsq)))
        plot(x[0],fresarr , title = 'fit',name='fit monitor')
    if onscreen:
        for p in plsq:
            print('p%s = %s +/- %s'%(k,p,errors[k]))
            k+=1
    if kw.pop('ret_fit',False):
        return plsq, errors, fit_func(*(x+tuple(plsq))),cov,info
    else:
        return plsq, errors
    qt.mstop()

def estimate_errors(plsq, fit_info, cov, dof):
    '''
    plsq = fitparams
    fit_info = full_output of leastsq
    cov = covariance matrix
    dof = degrees of freedom (or len(x_data) - len(plsq)) 
    '''
    error =  len(plsq)*[0]
    chisq=sum(fit_info["fvec"]*fit_info["fvec"])
    for i in np.arange(len(plsq)):
        print('cov: ',cov[i,i])
        error[i] = np.sqrt(cov[i,i])*np.sqrt(chisq/dof)
    return error
    


