#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create baseline, which has to be substracted from small current measurements
@author: svenjachalke
"""

from pylab import *
import pandas as pd
from scipy.interpolate import interp1d, UnivariateSpline
from tubafcdpy import *

from scipy.signal import savgol_filter

# -------------------------------
# FUNCTIONS

def set_interpolation_range(a,b):
	"""
	Function to find die interpolation range of two variables, e.g. temperature and current
	"""
	boundries = [0.0,0.0]
	if min(a) <= min(b):
		boundries[0] = min(b)
	else:
		boundries[0] = min(a)

	if max(a) >= max(b):
		boundries[1] = max(b)
	else:
		boundries[1] = max(a)
	return boundries

def interpolate_data(temp_array, curr_array, steps):
	boundries = set_interpolation_range(curr_array[:,0],temp_array[:,0])	 #find interpolation range
	tnew = arange(boundries[0], boundries[1], steps)					 #arange new time axis in 0.5s steps

	#Temperature
	Tinterpol_down = interp1d(temp_array[:,0],temp_array[:,1])			 #interpolation of lower temperature
	Tnew_down = Tinterpol_down(tnew)
	Tnew_top = zeros(len(Tnew_down))
	Tnew = vstack([Tnew_down,Tnew_top]).T
		
	#Interpolation current data																	#same for current
	Iinterpol = interp1d(curr_array[:,0],curr_array[:,1])
	Inew = Iinterpol(tnew)
	
	return tnew, Tnew, Inew

# files ----------- -----------------------------------------------------------------
SamplefileI = "2018-06-23_12-02_AlHfO2-CR-061-05-1-D05_ELT-Curr-t-I-VB.log"
SamplefileT = "2018-06-23_12-02_AlHfO2-CR-061-05-1-D05_TEMP-t-Tpelt-Tsoll-Tsample.log"

Baseline1fileI = "2018-06-24_09-43_Baseline1_ELT-Curr-t-I-VB.log"
Baseline1fileT = "2018-06-24_09-43_Baseline1_TEMP-t-Tpelt-Tsoll-Tsample.log"

#Baseline2fileI = "2018-06-25_00-12_Baseline2_ELT-Curr-t-I-VB.log"
#Baseline2fileT = "2018-06-25_00-12_Baseline2_TEMP-t-Tpelt-Tsoll-Tsample.log"

# read file content -----------------------------------------------------------------
SI = pd.read_csv(SamplefileI, names=['time','curr','VB'], delimiter=' ')
ST = pd.read_csv(SamplefileT, names=['time','temp','temp_soll'], skiprows=10, delimiter=' ')
tS, TS, IS = interpolate_data(array(ST),array(SI),0.5)

BL1I = pd.read_csv(Baseline1fileI, names=['time','curr','VB'], delimiter=' ')
BL1T = pd.read_csv(Baseline1fileT, names=['time','temp','temp_soll'], skiprows=10, delimiter=' ')
tB1, TB1, IB1 = interpolate_data(array(BL1T),array(BL2I),0.5)

#BL2I = pd.read_csv(Baseline2fileI, names=['time','curr','VB'], delimiter=' ')
#BL2T = pd.read_csv(Baseline2fileT, names=['time','temp','temp_soll'], skiprows=10, delimiter=' ')

timeshift = 0.21		#s

BL = pd.DataFrame()
BL['time'] = tB1+timeshift
BL['curr'] = IS-IB1

#
f = figure('BaselineSubstraction',figsize=(8,6))
axT = f.add_subplot(111)
axI = axT.twinx()

axT.set_ylabel('$T$ (\si{\celsius})')
axT.set_xlabel('$t$ (s)')
axI.set_ylabel('$I$ (pA)')
#
#axT.plot(ST.time,ST.temp)
#axT.plot(BL1T.time+timeshift,BL1T.temp)
axT.plot(tS, TS[:,0])
axT.plot(tB1+timeshift, TB1[:,0])

axI.plot(tS, IS*1e12, label='Sample')
axI.plot(tB1+timeshift,IB1*1e12, label = 'Baseline')
axI.plot(BL.time,BL.curr*1e12, label = 'Difference')


axT.grid(linestyle=':')
axI.legend()

f.savefig('BaselineDifference.png',dpi=300)o
BL.to_csv('2018-06-25_Baseline-A2K-f10mHz-273K-441K-25Kh.txt')



#axT.set_xlim(9953.2, 9954.6)
#axT.set_ylim(342.2, 342.37)



