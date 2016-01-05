from . import LabVIEW as lb
import numpy as np
class LVmon:
    def __init__(self):
        self.lb = lb
        self.lb.connect()
    def plot2D_monitor(self, mon, data):
        shp_dat = np.shape(data)
        if len(shp_dat) == 1:
            '''
            only y values
            '''
            x = np.arange(shp_dat[0])
            buf = np.array([x, data]).tostring()
            shp_buf = (2, shp_dat[0])
        else:
            buf = np.array(data).tostring()
            shp_buf = shp_dat
        command = 'shape[%i,%i]' % shp_buf+buf
        #self.lb.connect()
        exec("self.lb.Plot2D.Monitor%i(command)" % mon)
        #self.lb.disconnect()

    def plot3D_monitor(self,mon, data, axis = (0.,1.,0.,1.)):
        shp_dat = np.shape(data)

        buf = np.array(data).tostring()
        shp_buf = shp_dat
        command = 'axis[%f,%f,%f,%f]shape[%i,%i]'%(axis+shp_buf)+buf
        #self.lb.connect()
        exec("self.lb.Plot3D.Monitor%i(command)"%mon)
        #self.lb.disconnect()