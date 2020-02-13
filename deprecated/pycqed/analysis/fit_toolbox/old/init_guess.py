import numpy
from scipy import *

def lorentzian(x_data, y_data):
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
    index_array = numpy.linspace(index_y_min, index_y_max, abs(index_y_max-index_y_min)+1)
    if sign(index_y_min-index_y_max)>0:
        index_array_2 = numpy.linspace(index_y_min, size(y_data), abs(index_y_min-size(y_data))+1)
    else:
        index_array_2 = numpy.linspace(index_y_min, 0, abs(index_y_min)+1)
    #print 'check 4'
    #print index_array
    #print index_array_2

    index_i = 0
    while (not value_found):
        index1 = index_array[index_i]
        index2 = index_array_2[index_i]
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

        index_i+=1
    
    HWHM = abs(x_data[HM_index] - x_min)
    #print 'check 1'
    #print 2*HWHM
    FWHM = 2*HWHM
    p[0] = x_min
    p[1] = -2*HM
    p[2] = FWHM
    p[3] = y_max
    #print p
    return p
