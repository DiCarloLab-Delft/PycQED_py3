import numpy
from numpy import *
from scipy import *
from fit_toolbox import fit, functions
from data_processing_toolbox import movingaverage as mva
from qt import plot

    
def find_initial_values_lorentzian(x_data,y_data,moving_average = 2, plot_res = False):
    '''
    finds the initial values for a lorentzian dip or peak
    p[0] = x0
    p[1] = 2*HM dip height
    p[2] = dx width
    p[3] = y0 offset
    '''
    p=4*[0]
    diffy = mva.movingaverage(diff(y_data),moving_average)
    if plot_res:
        pl = plot(name='find_resonance')
        pl.clear()
        plot(diffy,title='moving_average of diff_y',name='find_resonance')
        plot(y_data,title='y_data',name='find_resonance')
    dymax = max(diffy)
    i_dymax = diffy.tolist().index(dymax)
    dymin = min(diffy)
    i_dymin = diffy.tolist().index(dymin)
    if i_dymin>i_dymax:
        '''
        means its a peak
        '''
        peak = 1
        y0 = min(y_data)
    else:
        '''
        means its a dip
        '''
        peak = -1
        y0 = max(y_data)
    dx = abs(x_data[i_dymin] - x_data[i_dymax])
    x0 = (x_data[i_dymin] + x_data[i_dymax])/2
    HM = peak*abs(y_data[i_dymax] - y_data[floor((i_dymin+i_dymax)/2)])
    p[0] = x0
    p[1] = 2*HM
    p[2] = dx
    p[3] = y0
    return p



def find_initial_values_lorentzian_dip(x_data, y_data):
    '''
    finds the inital estimate for a lorentzian
    this function is for compatibillity
    use find_lorentzian instead (also finds lorentz peaks)
    '''
    p=4*[0]
    y_min = min(y_data)
    index_y_min = y_data.tolist().index(y_min)
    x_min = x_data[index_y_min]
    y_max = max(y_data)
    index_y_max = y_data.tolist().index(y_max)
    y_mean = y_data.mean()
    HM = (y_max - y_min)/2
    #print 'check 3'
    #print x_min
    #print index_y_max
    HM_index = index_y_min
    #print HM_index
    value_found = False
    index_array = numpy.linspace(index_y_min, index_y_max, 
            abs(index_y_max-index_y_min)+1)
    
    if sign(index_y_min-index_y_max)>0:
        index_array_2 = numpy.linspace(index_y_min, size(y_data), 
                abs(index_y_min-size(y_data))+1)
    else:
        index_array_2 = numpy.linspace(index_y_min, 0, abs(index_y_min))
    #print 'check 4'
    #print index_array
    #print index_array_2

    index_i = 0
    while (not value_found):
        try:
            index1 = int(index_array[index_i])
            
            
        except IndexError:
            print(('index1 out of bounds index: %s'%index_i + 
            '\n while len array = %s'%len(index_array)))
            value_found= True
            HM_index = index_y_min-1
            pass
        try:
            index2 = int(index_array_2[index_i])
        except IndexError:
            print(('index1 out of bounds index: %s'%index_i + 
            '\n while len array = %s'%len(index_array_2)))
            value_found= True
            HM_index = index_y_min-1
            pass

        try:
            if y_data[index1] > (y_max-HM):
                
                HM_index = index1
                #print 'check 2'
                #print HM_index
                value_found = True
            elif y_data[index2] > (y_max-HM):
                   
                HM_index = index2
                #print 'check 2'
                #print HM_index
                value_found = True
        except IndexError:
            HM_index = index_y_min-1


        index_i+=1
    
    HWHM = abs(x_data[HM_index] - x_min)
    #print 'check 1'
    #print 2*HWHM
    FWHM = abs(2*HWHM)
    p[0] = x_min
    p[1] = -2*HM
    p[2] = FWHM
    p[3] = y_max
    #print p
    return p

def find_initial_values_cos(x,y):
    '''
    finds cos init pars
    par order y0,A,f,phi=P
    '''
    y_avg = mean(y)
    fty = abs(fft(y-y_avg))
    A = max(fty)/len(fty)*2*pi
    hl= int(len(fty)/2)
    f_ind =  ((fty.tolist()).index(max(fty[:hl]))-1)
    f =f_ind/max(x)
    phi = arctan(imag(fty[f_ind])/real(fty[f_ind]))


#    y0=mean(y_data)
#    y_avg=y0
#    may = max(y_data)
#    miy = min(y_data)
#    if type(x_data)==type([]):
#        xl = x_data
#    else:
#        xl = x_data.tolist()
#    if type(y_data)==type([]):
#        yl = y_data
#    else:
#        yl = y_data.tolist()
#    
#    mix = xl[(yl.index(miy))]
#    mxx = xl[(yl.index(may))]
#    
#    fftIp = abs(numpy.fft.fft(y_data-y0))
#    fIpmax = max(fftIp[0:int(len(fftIp)/2.)])
#    fftIpl= fftIp[0:int(len(fftIp)/2.)].tolist()
#    print fIpmax
#    f = fftIpl.index(fIpmax)/abs((x_data[-1]-x_data[0]))
#    
#    x0 = min(abs(x_data))
#    x0i = xl.index(x0)
#    A=abs(may-miy)/2.
#    phi = -arccos((1.*(yl[x0i]-y0)/A))
    return y_avg,A,f,phi
    

def convert_lorentz_to_cavity_paramters(p):
    '''
    Converts Lorentz params of a lorentz from f0, FWHM, 
    A y_max to f0 kc, ki 
    '''
    return p[0], p[2]*(abs(p[1])/p[3]), p[2]*(1-abs(p[1]/p[3]))

def convert_norm_lorentz_to_cavity_paramters(p):
    '''
    Converts Lorentz params of a lorentz normed to 0,1 from f0, FWHM, 
    A to f0 kc, ki 
    '''
    return p[0], p[2]*(abs(p[1])), p[2]*(1-abs(p[1])) 

def find_bare_cavity_parameters(f_vals, S21):
    '''
    Finds k_i and k_c and f_c of normalized S21 data
    '''
    p=find_initial_values_lorentzian_dip(f_vals, S21)
    f0, kc, ki = convert_norm_lorentz_to_cavity_paramters(p) 
    fit_result = fit.fit1D(f_vals, S21, functions.bare_cavity_resonance, 
            init_guess = [f0,kc,ki,0,1], ret_fit=True)
    return fit_result

def find_resonance(x, y, **kw):
    '''
    find resonance by taking the derivative
    maxdf : maximum allowable fwhm in units of the x-axis 
    '''
    dx = diff(x)
    maxdf = kw.pop('maxdf', 1e6)
    maxdfn = maxdf/(abs(dx[0]))
    print(maxdfn)
    
    dy = diff(y)
    
    dyx = dy/dx
    dyx = dyx - (sum(dyx)/len(dyx))

    madyx = max(dyx)
    midyx = min(dyx)
    A = madyx - midyx
    dm = madyx + midyx
    may_i = (dyx.tolist()).index(madyx)
    miy_i = (dyx.tolist()).index(midyx)
    if abs(miy_i-may_i)>maxdfn:
        x0_i = dyx.tolist().index(max([madyx,midyx]))
        FWHM_i = maxdfn
    else:
        x0_i = int((may_i+miy_i)/2.)
        FWHM_i = abs(may_i-miy_i)
 
    #eta = numpy.sign(mixv-maxv)*2*numpy.arctan(A/dm)*numpy.sign()
    return x0_i, FWHM_i 


def find_initial_values_Ramsey(x,y,**kw):
    '''
    find s initial values for gaussian decaying Ramsey
    '''
    y_avg = mean(y)
    fty = abs(fft(y-y_avg))
    A = max(fty)/len(fty)*2*pi
    hl= int(len(fty)/2)
    f_ind =  ((fty.tolist()).index(max(fty[:hl]))-1)
    f =f_ind/max(x)
    phi = arctan(imag(fty[f_ind])/real(fty[f_ind]))
    flh = abs(fty[:hl]) > (0.45*max(abs(fty[:hl])))
    ind1 = (flh.tolist()).index(True)
    k=0
    while flh[ind1+k]:
        k+=1

    sigm = (1.*k)/max(x)/2
    #print sigm
    T2s = 1./(pi*sigm)
    #print T2s
    
    return y_avg,A,f,phi,T2s



