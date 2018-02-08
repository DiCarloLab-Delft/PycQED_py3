import numpy as np
import scipy.signal as signal
import logging

def import_iir(filename):
    '''
    imports csv files generated with Mathematica notebooks of the form
    a1_0,b0_0,b1_0
    a1_1,b0_1,b1_1
    a1_2,b0_2,b1_2
    .
    .
    .

    args:
        filename : string containging to full path of the file (or only the filename if in same directory)

    returns:
        [aIIRfilterLis,bIIRfilterList] : list of two numpy arrays compatable for use
        with the scipy.signal.lfilter() function
        used by filterIIR() function

    '''
    IIRfilterList = np.loadtxt(filename,
                               delimiter=',')
    aIIRfilterList = np.transpose(np.vstack((np.ones(len(IIRfilterList)),
                                             -IIRfilterList[:,0])))
    bIIRfilterList = IIRfilterList[:,1:]

    return [aIIRfilterList,bIIRfilterList]


def filter_fir(kernel,x):
    '''
    function to apply a FIR filter to a dataset

    args:
        kernel: FIR filter kernel
        x:      data set
    return:
        y :     data convoluted with kernel, aligned such that pulses do not
                shift (expects kernel to have a impulse like peak)
    '''
    iMax = kernel.argmax()
    y = np.convolve(x,kernel,mode='full')[iMax:(len(x)+iMax)]
    return y


def filter_iir(aIIRfilterList,bIIRfilterList,x):
    '''
    applies IIR filter to the data x (aIIRfilterList and bIIRfilterList are load by the importIIR() function)

    args:
        aIIRfilterList : array containing the a coefficients of the IIR filters
                         (one row per IIR filter with coefficients 1,-a1,-a2,.. in the form required by scipy.signal.lfilter)
        bIIRfilterList : array containing the b coefficients of the IIR filters
                         (one row per IIR filter with coefficients b0, b1, b2,.. in the form required by scipy.signal.lfilter)
        x : data array to be filtered

    returns:
        y : filtered data array
    '''
    y = x
    for a,b in zip(aIIRfilterList,bIIRfilterList):
        y = signal.lfilter(b,a,y)
    return y




def distort_qudev(element, distortion_dict):
    """
    Distorts an element using the contenst of a distortion dictionary.
    The distortion dictionary should be formatted as follows.

    distortion_dict = {'ch_list': ['chx', 'chy', ...],
              'chx': filter_dict,
              'chy': filter_dict,
              ...
              }
    with filter_dict = {'FIR' : [filter_kernel1,filter_kernel2,..], 'IIR':  [aIIRfilterLis,bIIRfilterList]}

    args:
        element : element instance of the Element class in element.py module
        distortion_dict : distortion dictionary (format see above)

    returns : element with distorted waveforms attached (element.distorted_wfs)
    """
    t_vals, wfs_dict = element.waveforms()
    for ch in distortion_dict['ch_list']:
        element.chan_distorted[ch] = True
        kernelvecs = distortion_dict[ch]['FIR']
        wf_dist = wfs_dict[ch]
        if kernelvecs is not None and len(kernelvecs) > 0:
            if not hasattr(kernelvecs[0], '__iter__'):
                wf_dist = filter_fir(kernelvecs, wf_dist)
            else:
                for kernelvec in kernelvecs:
                    wf_dist = filter_fir(kernelvec, wf_dist)
        if distortion_dict[ch]['IIR'] is not None:
            aIIRfilterList,bIIRfilterList = distortion_dict[ch]['IIR']
            wf_dist = filter_iir(aIIRfilterList,bIIRfilterList,wf_dist)

        wf_dist[-1] = 0.
        wf_dist[0] = 0.
        element.distorted_wfs[ch] = wf_dist
    return element

def gaussian_filter_kernel(sigma,nr_sigma,dt):
    '''
    function to generate a Gaussian filter kernel with specified sigma and
    filter kernel width (nr_sigma).

    Args:
        sigma (float): width of the Gaussian
        nr_sigma (int): specifies the length of the filter kernel
        dt (float): AWG sampling period

    Returns:
        kernel (numpy array): Gaussian filter kernel
    '''
    nr_samples = int(nr_sigma*sigma/dt)
    if nr_samples == 0:
        logging.warning('sigma too small (much smaller than sampling rate).')
        return np.array([1])
    gauss_kernel = signal.gaussian(nr_samples, sigma/dt, sym=False)
    gauss_kernel = gauss_kernel/np.sum(gauss_kernel)
    return np.array(gauss_kernel)







