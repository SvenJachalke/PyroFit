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
data = pd.concat(list_,axis=1)