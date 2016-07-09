import pandas as pd
import glob

#Temp = pd.read_csv('2014-06-10_12-53_Crystec-LiTaO3-05mm_TEMP-t-Tpelt-Tsoll-Tsample.log',delimiter=' ',skiprows=10,names=['time','Tdown','Tsoll','Ttop'],index_col='time')

filelist = glob.glob('*.log')
frame = pd.DataFrame()
list_ = []
for file_ in filelist:
	if file_.endswith('_TEMP-t-Tpelt-Tsoll-Tsample.log'):
		df= pd.read_csv(file_,delimiter=' ',skiprows=10,names=['time','Tdown','Tsoll','Ttop'],index_col='time')
		list_.append(df)
	if file_.endswith('_ELT-Curr-t-I.log'):
		df = pd.read_csv(file_,delimiter=' ',names=['time','current'],index_col='time')
		list_.append(df)
	if file_.endswith('_PWR-t-I-U.log'):
		df = pd.read_csv(file_,delimiter=' ',names=['time','Pelt_curr','Pelt_volt'],index_col='time')
		list_.append(df)
data = pd.concat(list_,axis=1)

data_interpolated = data.interpolate(method='slinear')

# example plot

from matplotlib.pyplot import *

f = figure('Testplot')
ax = f.add_subplot(111)

ax.plot(data.index,data.current,'ro')
ax.plot(data_interpolated.index,data_interpolated.current,'b-')

ax.set_xlim(60,70)
ax.set_ylim(-0.2e-8,0.2e-8)
ax.grid()

f.show()