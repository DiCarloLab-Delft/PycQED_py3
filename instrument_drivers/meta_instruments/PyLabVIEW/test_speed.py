import numpy as np
import time
from . import lv_monitor
from PIL import Image
from fit_toolbox import functions as fn

lvmon = lv_monitor.LVmon()
f = Image.open('D:\\PyLabVIEW\\Leo-DC-klein-145x150.jpg')
fpy = Image.open('D:\\PyLabVIEW\\python.jpg')
farr = np.average(np.asarray(f),2).transpose()[:,::-1]
fpyr = np.average(np.asarray(fpy),2).transpose()[:,::-1]
shp_farr = np.shape(farr)
shp_fpyr = np.shape(fpyr)
noise=10000
sdat = farr+noise*np.random.randn(*shp_farr)
pydat = fpyr+noise*np.random.randn(*shp_fpyr)
x= np.arange(1000)
y1 =  np.sin(2*np.pi/190.*x)
y2 =  fn.disp_hanger_S21_amplitude(x,len(x)/2.,20.,20.,2.8,0.1)
y3 =  fn.disp_hanger_S21_amplitude(x,len(x)/2.+40,20.,20.,2.8,0.1)
t0 = time.time()

skip = 1
speed = (len(x)/skip)*[0]
for kk in x[1:len(x)/skip].tolist():
    ind = skip*kk
    lvmon.plot2D_monitor(2,[(x[:ind]+6e3)*1e6,y2[:ind]])
    qt.msleep(0.001)
    print(np.shape([x[:ind],y1[:ind]]))
    lvmon.plot2D_monitor(1,[x[:ind],y1[:ind]])
    qt.msleep(0.02)
    lvmon.plot2D_monitor(3,[(x[:ind]+6e3)*1e6,y3[:ind]])
    if (kk/10==kk/10.) and (kk>10):
        sdat += noise*np.random.randn(*shp_farr)+kk**3*farr/(1000**2)
        pydat +=noise*np.random.randn(*shp_fpyr)+kk**3*fpyr/(1000**2)
        lvmon.plot3D_monitor(1,sdat/2)
        qt.msleep(0.02)
        lvmon.plot3D_monitor(2,pydat/2)
        #lvmon.plot3D_monitor(2,yarr**2)
        #qt.msleep(0.02)
        
        print(kk)
    speed[kk] = time.time()-t0
    t0=time.time()
