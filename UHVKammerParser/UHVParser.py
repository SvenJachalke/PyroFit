#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:58:40 2017
Data parser for UHV-Pyro-Chamber dataset to be compatible for PyroFit
@author: svenjachalke
"""

import pandas as pd
from glob import glob

def hh_mm_ss2seconds(hh_mm_ss):
	seconds = reduce(lambda acc, x: acc*60 + x, map(float, hh_mm_ss.split(':')))
	return seconds

UHVfile = glob('*.txt')
name = UHVfile[0].split('.')[0]
data = pd.read_csv(UHVfile[0],delimiter='\t',parse_dates = True,names=['Date','Time','Setpoint','Temperature','Voltage_Measured','Current_Measured','Current_Applied','Pressure','Loop_Time','PID_on','Ramp_on','Input_Error','Output_Error','Current_date','Current_time','Current'],converters={'Time': hh_mm_ss2seconds,'Current_time': hh_mm_ss2seconds})

time_temperature = data.Time-data.Time[0]
set_temperature = data.Setpoint+273.15
meas_temperature = data.Temperature+273.15
TemperatureData = pd.concat([time_temperature, set_temperature, meas_temperature], axis=1)
Temp_Log = open(name+'_TEMP-t-Tpelt-Tsoll-Tsample.log','w+')
Temp_Log.write('HV-Mode Off\nT-Waveform SineWave\nT-Amplitude[K] 1\nT-Frequency[Hz] 0.0100\nT-Offset[K] 273.15\nHeating-Rate[K/s] 0.0069444\nCooling-Rate[K/s] 0.0\nT-Limit-High[K] 441.15\nT-Limit-Low[K] 273.15')
TemperatureData.to_csv(Temp_Log,index=False,header=False,sep=' ')
Temp_Log.close

time_current = data.Current_time-data.Current_time[0]
meas_current = data.Current
CurrentData = pd.concat([time_current,meas_current], axis=1)
CurrentData.to_csv(name+'_ELT-Curr-t-I.log',index=False,sep=' ',header=False)

