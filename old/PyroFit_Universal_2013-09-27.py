# Universal Script for PyroData Evaluation
# (Use only for individual data records, e.g. all files are contained in one single folder!!!)
#---------------------------------------------------------------------------------------------------------------------------
# Author: 	Sven Jachalke
# Mail:		sven.jachalke@phyik.tu-freiberg.de
# Adress:	Institut fuer Experimentelle Physik
# 		Leipziger Strasse 23
#		09596 Freiberg
#---------------------------------------------------------------------------------------------------------------------------
#Necessary Python Packages:
# - numpy
# - scipy
# - pylab (matplotlib), etc.
# - lmfit (http://newville.github.io/lmfit-py/)
#---------------------------------------------------------------------------------------------------------------------------
###Changlog###--------------------------------------------------------------------------------------------------------------
# 2013-07-19:	-Creation date
# 2013-07-23:	-finishing implementation of SineWave, SineWave+LinRamp, AutoPol
#		-SineWave+LinRamp uses now a interpolation of the data to get all data on the save time grid
# 2013-07-25:	-enhanced console output for AutoPol, SineWave, SineWave+LinRamp
#		-correct filenames for figures
#		-correct filenames for txts
#		-exp. fit of AutoPol Current with graphical input
# 2013-07-28:	-activated ion(), ioff() when shell was not started with --pylab
# 2013-07-29:	-Triangle/SineWave+Triangle profiles plotting implemented
# 2013-08-02:	-implententation of OnPerm Modes (Thermostat, SineWave (for STO crystal))
# 2013-08-05:	-modifified Inp/Ip calculation in SineWave, .... to abs() in order to make amplitude postive and drawing of fits correct!
#		-Thermostat for HV Off section
# 2013-08-09:	-improved plot and fit for STO samples (right TSC phase with phase shifts of 360, 180 and 0 degree)
#		-LaTeX text in textbox SineWave (HVOff, HVOn)
#		-interpolation on SineWave data (HVOff, HVOn)
#		-correct filename for output in SineWave method
# 2013-08-14:	-partial implementation of SineWave+TriangHat method
#		-optimized SineWave+LinRamp routine (fit and plot beginning at start_index=100) + filtering of I>1muA before plotting!
#		-method for save figure
#		-copied new SineWave+LinRamp to "HV_on" part
#		-fixed phase for pyro current plot when phasediff >180
# 2013-08-16:	-fixed HV Comp readout!
# 2013-09-27:	-fixed reading error in HVlog for OnPerm measurements


# Import modules------------------------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from pylab import *
import glob
import math
import sys
import os
from lmfit import minimize, Parameters, Parameter, report_errors
matplotlib.rcParams['legend.fancybox'] = True

# Functions-----------------------------------------------------------------------------------------------------------------
# file functions -----------------------------------------------------------------------------------------------------------
# extract date from filename
def extract_date(filename):
	return filename[:16]

#extract samplename
def extract_samplename(filename):
	return filename.split("_")[2]

# extract from filename with data is contained in the specific file
def extract_datatype(filename):
	if filename.endswith("ELT-Curr-t-I.log"):
		return "Current"
	elif filename.endswith("ELT-Volt-t-V.log"):
		return "Voltage"
	elif filename.endswith("ELT-Char-t-Q.log"):
		return "Charge"
	elif filename.endswith("TEMP-t-Tpelt-Tsoll-Tsample.log"):
		return "Temperature"
	elif filename.endswith("PWR-t-I-U.log"):
		return "Powersupply"
	elif filename.endswith("VAC-t-pressure.log"):
		return "Vacuum"
	elif filename.endswith("HiV-t-HVsetVoltage-HVmeasVoltage.log"):
		return "HighVoltage"
	elif filename.endswith("GPIB-t-addr-errcount-cmd-data-atconvdat-STBerrque.log"):
		return "GBIP-Errors"
	else:
		return None

# extract which kind of measurement was performed
def extract_measurementmode(filename):
	if filename.endswith("TEMP-t-Tpelt-Tsoll-Tsample.log"):
		datei = open(filename, "r")
		line = datei.readline()								#first line = HV status
		hv_mode = (line.strip()).split(" ")[1]
		line = datei.readline()								#second line = waveform
		waveform = (line.strip()).split(" ")[1]
		datei.close()
		return hv_mode, waveform
	else:
		print "Could not find measurement type. Check temperature file!"

#extract temp stimulation data from file
def extract_T_stimulation_params(filename):
	if filename.endswith("TEMP-t-Tpelt-Tsoll-Tsample.log"):
		datei = open(filename, 'r')
		hv_mode = datei.readline().strip().split(" ")[1]
		waveform = datei.readline().strip().split(" ")[1]
		amp = datei.readline().strip().split(" ")[1]
		freq = datei.readline().strip().split(" ")[1]
		offs = datei.readline().strip().split(" ")[1]
		heat_rate = datei.readline().strip().split(" ")[1]
		cool_rate = datei.readline().strip().split(" ")[1]
		T_Limit_H = datei.readline().strip().split(" ")[1]
		T_Limit_L = datei.readline().strip().split(" ")[1]
		datei.close()
		return [float(amp), float(freq), float(offs), float(heat_rate), float(cool_rate), float(T_Limit_H), float(T_Limit_L)]
	
#extract HV parameters from file
def extract_HV_params(filename):
	if filename.endswith("HiV-t-HVsetVoltage-HVmeasVoltage.log"):
		datei = open(filename, 'r')
		HVmax = datei.readline().strip().split(" ")[4]
		zeile = datei.readline()
		if zeile!=' ' and zeile!='\r\n':
			HVcomp = zeile.strip().split(" ")[5]
			datei.close()
			return [float(HVmax), float(HVcomp)]
		datei.close()
		return [float(HVmax)]

# fit functions ---------------------------------------------------------------------------------------------------------------
# SineWave
def sinfunc(params, x, data=None):
	amp = params['amp'].value
	freq = params['freq'].value
	phase = params['phase'].value
	offs = params['offs'].value
	slope = params['slope'].value
	
	model = amp*sin(2*pi*freq*x+phase)+offs+slope*x
	
	if data==None:
		return model
	return model-data

# ExpDecay
def expdecay(params, x, data=None):
	A = params['factor'].value
	decay = params['decay'].value
	offs = params['offs'].value
	
	model = A*np.exp(-x/decay) + offs
	
	if data==None:
		return model
	return model-data

# Extraction of Params-Dict (put elements from dict to "fit" and "error"-lists)
def extract_fit_relerr_params(params, fit, err):
	fit[0]=params['amp'].value
	fit[1]=params['freq'].value
	fit[2]=params['phase'].value
	fit[3]=params['offs'].value
	fit[4]=params['slope'].value
	
	err[0]=abs(params['amp'].stderr)
	err[1]=abs(params['freq'].stderr)
	err[2]=abs(params['phase'].stderr)
	err[3]=abs(params['offs'].stderr)
	err[4]=abs(params['slope'].stderr)
	
	return None

#ListtoParameters-Fkt
def listtoparam(liste, parameterdic):
	parameterdic.add('amp', value=liste[0])
	parameterdic.add('freq', value=liste[1])
	parameterdic.add('phase', value=liste[2])
	parameterdic.add('offs', value=liste[3])
	parameterdic.add('slope', value=liste[4])

#calculates error p (for no index)
def p_error(Tfit, Terror, Ifit, Ierror, phasediff, area, area_error):
	err_A_I = (sin(phasediff)/(area * Tfit[0] * Tfit[1])) * Ierror[0]
	err_phi = (Ifit[0]*cos(phasediff)/(area * Tfit[0] * Tfit[1])) * (Ierror[1]+Terror[1])
	err_area = -(Ifit[0]*sin(phasediff)/((area**2)*Tfit[0] * Tfit[1])) * area_error
	err_A_T = -(Ifit[0]*sin(phasediff)/(area * (Tfit[0]**2) * Tfit[1])) * Terror[0]
	err_w_T = -(Ifit[0]*sin(phasediff)/(area * Tfit[0] * (Tfit[1]**2))) * Terror[1]

	p_ges_error = abs(err_A_I)+abs(err_phi)+abs(err_area)+abs(err_A_T)*abs(err_w_T)
	
	return p_ges_error	

#calculates error p (for index i)
def p_error_i(Tfit, Terror, Ifit, Ierror, phasediff, area, area_error, i):
	err_A_I = (sin(phasediff)/(area * Tfit[0] * Tfit[1])) * Ierror[i-1,0]
	err_phi = (Ifit[i-1,0]*cos(phasediff)/(area * Tfit[0] * Tfit[1])) * (Ierror[i-1,1]+Terror[1])
	err_area = -(Ifit[i-1,0]*sin(phasediff)/((area**2)*Tfit[0] * Tfit[1])) * area_error
	err_A_T = -(Ifit[i-1,0]*sin(phasediff)/(area * (Tfit[0]**2) * Tfit[1])) * Terror[0]
	err_w_T = -(Ifit[i-1,0]*sin(phasediff)/(area * Tfit[0] * (Tfit[1]**2))) * Terror[1]
	
	p_ges_error = abs(err_A_I)+abs(err_phi)+abs(err_area)+abs(err_A_T)*abs(err_w_T)
	
	return p_ges_error	

#saving figure method with individual filename
def saving_figure(filename):
	print "--------------------------------"
	print "...saving figure"
	savefig(filename)
	return None


# Main Program-----------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

ion()
print "--------------------------------"
print "PyroFit - UnivseralScript"
print "--------------------------------"


# File Reading-----------------------------------------------------------------------------------------------------------------
filelist = glob.glob('*.log')
filecounter = 0

#check folder for files and read files!
for filename in filelist:

	date=extract_date(filename)
	datatype=extract_datatype(filename)
	if datatype=="Temperature":
		HV_status, T_profile = extract_measurementmode(filename)
		start_parameters = extract_T_stimulation_params(filename)
		samplename = extract_samplename(filename)
		Tdata = loadtxt(filename, skiprows=9)
		filecounter = filecounter + 1
		sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
		sys.stdout.flush()
	elif datatype=="Current":
		Idata = loadtxt(filename)
		filecounter = filecounter + 1
		sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
		sys.stdout.flush()
	elif datatype=="Charge":
		Qdata = loadtxt(filename)
		filecounter = filecounter + 1
		sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
		sys.stdout.flush()
	elif datatype=="Voltage":
		Vdata = loadtxt(filename)
		filecounter = filecounter +1
		sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
		sys.stdout.flush()
	elif datatype=="HighVoltage":
		if os.path.getsize(filename)>4: 
			HV_set = extract_HV_params(filename)
			if os.path.getsize(filename)>32:
				HVdata = loadtxt(filename,skiprows=2)
			filecounter = filecounter + 1
			sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
			sys.stdout.flush()
	elif datatype=="Vacuum":
		Vacdata = loadtxt(filename)
		filecounter = filecounter + 1
		sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
		sys.stdout.flush()
	elif datatype=="Powersupply":
		Powerdata = loadtxt(filename)
		filecounter = filecounter + 1
		sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
		sys.stdout.flush()
	elif datatype=="GBIP-Errors":
		filecounter = filecounter + 1
		sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
		sys.stdout.flush()
	elif datatype==None:
		continue
print "\n--------------------------------"
#-------------------------------------------------------------------------------------------------------------------------------------
if filelist == []:
	print "No files in Folder!"
else:
#-------------------------------------------------------------------------------------------------------------------------------------
	#Routines for every measurement_type------------------------------------------------------------------------------------------
	#-----------------------------------------------------------------------------------------------------------------------------
	#-----------------------------------------------------------------------------------------------------------------------------
	#normal measurement routines without HV (SinWave, LinRamp, ...)
	if HV_status == "Off":
		#Thermostat Method
		#--------------------------------------------------------------------------------------------------------------------
		if T_profile == "Thermostat":
			print "Mode:\t\tThermostat"
			print "Temperature:\t%.1fK" % start_parameters[5]
			
			#Plotting of data
			print "--------------------------------"
			print "...plotting"
			head = date+"_"+samplename+"_Thermostat"
			bild = figure(head)
			ax1 = subplot(111)
			ax2 = ax1.twinx()
			title(samplename+"_Thermostat", size='15')
			
			start_index = 50
			
			#Plot Temperature
			ax1.set_xlabel('time [s]',size='20')
			ax1.set_ylabel('temperature [K]',color='b',size='20')
			ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
			ax1.plot(Tdata[start_index:,0], Tdata[start_index:,1], "b-", label='T-Down')
			ax1.set_ylim(start_parameters[5]-2, start_parameters[5]+2)
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				ax1.plot(Tdata[start_index::5,0], Tdata[start_index::5,3], "g-", label='T-Top')
			
			#Plot Current
			ax2.set_ylabel('current [A]',color='r',size='20')
			ax2.autoscale(enable=True, axis='y', tight=None)
			ax2.plot(Idata[start_index:,0], Idata[start_index:,1], 'r-', label='I')
			
			#text box
			box_text = "Temperature: "+str(start_parameters[5]) + "K"
			box = figtext(0.65,0.15,box_text,fontdict=None, bbox=dict(facecolor='white', alpha=0.5))
			
			show()
			
			saving_figure(date+'_'+samplename+'_Thermostat.png')
		
		#---------------------------------------------------------------------------------------------------------------------
		#LinearRamp Method
		elif T_profile == "LinRamp":
			print "Mode:\t\tLinRamp"
			print "Temperature:\t%.1fK\nSlope:\t%.1fK" % (start_parameters[5], start_parameters[3])
			
			#Plotting of data
			print "--------------------------------"
			print "...plotting"
			head = date+"_"+samplename+"_LinRamp"
			bild = figure(head)
			ax1 = subplot(111)
			ax2 = ax1.twinx()
			title(samplename+"_LinRamp", size='15')
			
			start_index = 0
			
			#Plot Temperature
			ax1.set_xlabel('time [s]',size='20')
			ax1.set_ylabel('temperature [K]',color='b',size='20')
			ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
			ax1.plot(Tdata[start_index:,0], Tdata[start_index:,1], "b-", label='T-Down')
			ax1.set_ylim(start_parameters[5]-2, start_parameters[5]+2)
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				ax1.plot(Tdata[start_index::5,0], Tdata[start_index::5,3], "g-", label='T-Top')
			
			#Plot Current
			ax2.set_ylabel('current [A]',color='r',size='20')
			ax2.autoscale(enable=True, axis='y', tight=None)
			ax2.plot(Idata[start_index:,0], Idata[start_index:,1], 'r-', label='I')
			
			#text box
			box_text = "Temperature: "+str(start_parameters[5]) + "K\nSlope: " + str(start_parameters[3]*3600) + "K/h"
			box = figtext(0.65,0.15,box_text,fontdict=None, bbox=dict(facecolor='white', alpha=0.5))
			
			show()
			
			saving_figure(date+'_'+samplename+'_LinRamp.png')
		
		#---------------------------------------------------------------------------------------------------------------------
		#SineWave Method
		elif T_profile == "SineWave":
			print "Mode:\t\tSineWave"
			print "Stimulation:\tA=%.1fK\n\t\tf=%.1fmHz\n\t\tO=%.1fK" % (start_parameters[0], start_parameters[1]*1000, start_parameters[2])
			
			#Interpolate data
			tnew = arange(min(Idata[:,0]),max(Tdata[:,0]),0.5)
			Tinterpol1 = interp1d(Tdata[:,0],Tdata[:,1])
			Tnew1 = Tinterpol1(tnew)
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				Tinterpol2 = interp1d(Tdata[::5,0],Tdata[::5,3])
				Tnew2 = Tinterpol2(tnew[:-5])
			Iinterpol = interp1d(Idata[:,0],Idata[:,1])
			Inew = Iinterpol(tnew)
			
			start_index = 100
			
			#Plotting of data
			print "--------------------------------"
			print "...plotting"
			head = date+"_"+samplename+"_SineWave"
			bild = figure(head)
			ax1 = subplot(111)
			ax2 = ax1.twinx()
			title(samplename+"_SineWave", size='15')
			
			#Plot Temperature
			ax1.set_xlabel('time [s]',size='20')
			ax1.set_ylabel('temperature [K]',color='b',size='20')
			ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
			ax1.plot(tnew[start_index:], Tnew1[start_index:], "bo", label='T-Down')
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				ax1.plot(tnew[start_index:-5], Tnew2[start_index:], "go", label='T-Top')
			
			#Plot Current
			ax2.set_ylabel('current [A]',color='r',size='20')
			ax2.autoscale(enable=True, axis='y', tight=None)
			ax2.plot(tnew[start_index:], Inew[start_index:], 'ro', label='I')
			
			show()
			
			#---------------------------------------------------------------------------------------------------------------
			input = raw_input("fit? (y/n)")
			if input == "y":
				
				print "--------------------------------"
				print "... fitting" 
				
				#Fit temperature----------------------------------------------------------------------------------------
				#intialize variables of fit
				Tfit = [0,0,0,0,0]	#bottom temperature
				Tfit2 = [0,0,0,0,0]	#top temperature
				Terror = [0,0,0,0,0]
				Terror2 = [0,0,0,0,0]
				
				#initialize parameters dict for lmfit (bottom temperature)
				Tparams = Parameters()
				Tparams.add('amp', value=start_parameters[0], min=0.1, max=40.0)
				Tparams.add('freq', value=start_parameters[1], min=1e-5, max=0.1, vary=False)
				Tparams.add('phase', value=0.1, min=-pi, max=pi)
				Tparams.add('offs', value=start_parameters[3], min=0.0)
				Tparams.add('slope', value=start_parameters[4])
				
				#fit of bottom temperature
				Tresults = minimize(sinfunc, Tparams, args=(tnew[start_index:], Tnew1[start_index:]), method="leastsq")
				#write fit params to Tfit-list
				extract_fit_relerr_params(Tparams, Tfit, Terror)
				#correction of phase < 0
				if Tfit[2]<0.0:
					Tfit[2] = Tfit[2]+(2*pi)
				
				#for top temperature
				if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
					#initialize parameters dict for lmfit (top temperature)
					Tparams2 = Parameters()
					Tparams2.add('amp', value=start_parameters[0], min=0.1, max=40.0)
					Tparams2.add('freq', value=start_parameters[1], min=1e-5, max=0.1, vary=False)
					Tparams2.add('phase', value=0.1, min=-pi, max=pi)
					Tparams2.add('offs', start_parameters[2], min=0.0)
					Tparams2.add('slope', value=start_parameters[3])
					
					#fit of top temperature
					Tresults2 = minimize(sinfunc, Tparams2, args=(tnew[start_index+(len(tnew)/2):-5], Tnew2[start_index+(len(tnew)/2):]), method="leastsq")
					extract_fit_relerr_params(Tparams, Tfit2, Terror2)
					
					#data corrections
					if Tfit2[0] < 0.0:
						Tfit2[0] = abs(Tfit2[0])
						Tfit2[2] = Tfit2[2] + 2*pi
					if Tfit2[2]<0.0:
						Tfit2[2] = Tfit2[2] + 2*pi
					if Tfit2[2] > 2*pi:
						Tfit2[2] = Tfit2[2] - 2*pi
				
				#plot of fits
				ax1.plot(tnew[start_index:], sinfunc(Tparams, tnew[start_index:]), 'b-', label='T-Fit-Bottom')
				if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
					ax1.plot(tnew[start_index:-5], sinfunc(Tparams2, tnew[start_index:-5]), 'g-', label='T-Fit-Top')
				

				#Fit current ---------------------------------------------------------------------------------------------
				#start_index=start_index+50
				Ifit = [0,0,0,0,0]
				Ierror = [0,0,0,0,0]

				#initialize parameters dict for current fit
				Iparams = Parameters()
				Iparams.add('amp', value=1e-9)#, min=1e-13, max=1e-7)
				Iparams.add('freq', value=start_parameters[1], min=1e-5, max=0.1, vary=False)
				Iparams.add('phase', value=0.1, min=-pi, max=pi)
				Iparams.add('offs', value=1e-10)
				Iparams.add('slope', value=1e-10)
				
				#current fit
				Iresults = minimize(sinfunc, Iparams, args=(tnew[start_index:], Inew[start_index:]), method="leastsq")
				#extract params dict
				extract_fit_relerr_params(Iparams, Ifit, Ierror)
				#fit corrections
				if Ifit[0]<0.0:				#if amplitude is negative (make positive + pi in phase)
					Ifit[0] = -1*Ifit[0]
					Ifit[2] = Ifit[2]+pi
				if Ifit[2]<0.0:				#if phase is negative (add 2*pi)
					Ifit[2] = Ifit[2]+(2*pi)

				#plot current fit
				ax2.plot(tnew[start_index:], sinfunc(Iparams, tnew[start_index:]), "r-", label='I-Fit')
				
				#calculating pyroelectric coefficient------------------------------------------------------------------------
				input = raw_input("Area [m2]?:")
				if input is "A":				#for PVDF (d=13,...mm)
					flaeche = 1.3994e-4
				elif input is "B":				#for cystalls (d=15mm)
					flaeche = 1.761e-4
				elif input is "C":				#STO Kristall M114
					flaeche = 1.4668e-5
				else:
					flaeche = float(input)
				area_error = 0.0082*flaeche
				
				#corrections
				phasediff = Tfit[2]-Ifit[2]
				if phasediff < 0.0:
					phasediff = phasediff + 2*pi
				if phasediff > 2*pi:
					phasediff = phasediff - 2*pi
				
				#NonPyroStrom
				#m=magenta (TSC)
				Inp = abs(Ifit[0]*cos(phasediff))
				nonpyroparams = Parameters()
				np_params = [Inp, Tfit[1], Tfit[2], Ifit[3], Ifit[4]]
				listtoparam(np_params, nonpyroparams)
				ax2.plot(tnew[start_index:], sinfunc(nonpyroparams, tnew[start_index:]), 'm-', label='I-np')

				#Pyrostrom
				#c=cyan (Pyro)
				Ip = abs(Ifit[0]*sin(phasediff))
				pyroparams = Parameters()
				if phasediff > pi:
					p_params = [Ip, Tfit[1], Tfit[2]+pi/2, Ifit[3], Ifit[4]]
				else:
					p_params = [Ip, Tfit[1], Tfit[2]-pi/2, Ifit[3], Ifit[4]]
				listtoparam(p_params, pyroparams)
				ax2.plot(tnew[start_index:], sinfunc(pyroparams, tnew[start_index:]), 'c-', label='I-p')

				pyro_koeff = (Ip)/(flaeche*Tfit[0]*2*pi*Ifit[1])
				perror = p_error(Tfit, Terror, Ifit, Ierror, phasediff, flaeche, area_error)
				
				#console output ------------------------------------------------------------------------------------------
				print 'Area:\t\t', flaeche, 'm2'
				print 'I_pyro:\t\t', fabs(Ip), 'A'
				print 'I_TSC:\t\t', fabs(Inp), 'A'
				print 'phase T-I:\t',(phasediff*180/pi)
				print 'p:\t\t', pyro_koeff*1e6,'yC/Km2'
				print '\t\t(+-', perror*1e6,'yC/Km2)'
				print 'B_T:\t\t%f nA/K' % (fabs(Inp/Tfit[1])*1e9)
				
				#file output ---------------------------------------------------------------------------------------------
				log_name = date + "_"+ samplename+"_Auswertung" + ".txt"
				log = open(log_name, "w+")
				log.write("Auswertung\n----------\n")
				log.write("T-Fit Down:\tA=%f K\t\tf=%f Hz\tp=%f\t\tOffs=%f K\t\tb=%f K/s\n" % (Tfit[0],Tfit[1],Tfit[2],Tfit[3],Tfit[4]))
				#log.write("T-Fit Top: \tA=%f K\t\tf=%f Hz\tp=%f\t\tOffs=%f K\t\t\tb=%f K/s\n" % (Tfit2[0],Tfit2[1],Tfit2[2],Tfit2[3],Tfit2[4]))
				log.write("I-Fit:\t\tA=%e K\tf=%f Hz\tp=%f\t\tOffs=%e K\t\tb=%e K/s\n" % (Ifit[0],Ifit[1],Ifit[2],Ifit[3],Ifit[4]))
				log.write('------------------------------------\n')
				log.write('Area:\t\t%fm2 \n' % (flaeche))
				log.write('T-Amp:\t\t%f K\n' %  (Tfit[0]))
				log.write('I_pyro:\t\t%eA \n' %(fabs(Ip)))
				log.write('I_TSC:\t\t%eA \n' % (fabs(Inp)))
				log.write('phase T-I:\t%f \n' % (phasediff*180/pi))
				log.write('p:\t\t\t%f yC/Km2\n' %(pyro_koeff*1e6))
				log.write('\t\t\t(+-%f yC/Km2)\n' %(perror*1e6))
				log.write('B_T:\t\t%f nA/K\n' % (fabs(Inp/Tfit[1])*1e9))
				log.close()
				
				#box, save figure, legend, ...
				ax1.legend(loc=1)
				ax2.legend(loc=4)
		
				box_text = r"Area: "+str(flaeche)+ r" $\mathrm{m^2}$"+"\n"+ r"I-Amp: "+str(Ifit[0])+r" A"+"\n"+ r"T-Amp: "+str(Tfit[0])+r" K"+"\n"+r"Phase-Diff.: "+str(phasediff*(180/pi))+"$^{\circ}$"+"\n"+r"pyroel. Coeff.: "+str(pyro_koeff*1e6)+r" $\mathrm{\mu C/Km^2}$"
				figtext(0.15,0.12,box_text,fontdict=None,bbox=dict(facecolor='white', alpha=0.5))

			else:
				pass
			
			draw()

			#save fig
			saving_figure(date+'_'+samplename+"_SineWave.png")
			
		#---------------------------------------------------------------------------------------------------------------------
		#SineWave+LinearRamp Method
		elif T_profile == "SineWave+LinRamp":
			print "Mode:\t\tSineWave+LinRamp"
			print "Stimulation:\tA=%.1fK\n\t\tf=%.1fmHz\n\t\tO=%.1f-%.1fK\n\t\tb=%.2fK/h" % (start_parameters[0], start_parameters[1]*1000, start_parameters[2],start_parameters[5], start_parameters[3]*3600)
			
			perioden = 2
			start_index = 100
			
			#Interpolate data
			tnew = arange(min(Idata[:,0]),max(Tdata[:,0]),0.5)
			Tinterpol1 = interp1d(Tdata[:,0],Tdata[:,1])
			Tnew1 = Tinterpol1(tnew)
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				Tinterpol2 = interp1d(Tdata[::5,0],Tdata[::5,3])
				Tnew2 = Tinterpol2(tnew[:-5])
			Iinterpol = interp1d(Idata[:,0],Idata[:,1])
			Inew = Iinterpol(tnew)
			
			#filtering of data for currente values larger than 1muA
			ausreisserliste = []
			for i in arange(0,len(Inew)-1):
				if Inew[i]>=1e-6:
					ausreisserliste.append(i)
			Inew = delete(Inew,ausreisserliste,0)
			tnew = delete(tnew,ausreisserliste,0)
			Tnew1 = delete(Tnew1,ausreisserliste,0)
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				Tnew2 = delete(Tnew2,ausreisserliste,0)
			
			#check when ramp runs into T_Limit_H
			for i in arange(0,len(Tnew1)-1):
				if Tnew1[i] < start_parameters[5]-1:
					limit = i
					
			#important calculations ;)
			max_Temp = tnew[limit]*start_parameters[3]+start_parameters[2]
			T_perioden = int(tnew[limit]/(1/start_parameters[1]))
			
			tmax = tnew[limit]
			satzlaenge = (limit-start_index)/T_perioden
			
			#Plotting of data
			print "--------------------------------"
			print "...plotting"
			head = date+"_"+samplename+"_SineWave+LinRamp"
			bild1 = figure(head)
			ax1 = subplot(111)
			ax2 = ax1.twinx()
			title(samplename+"_SineWave+LinRamp", size='15')
			
			#Plot Temperature
			ax1.set_xlabel('time [s]',size='20')
			ax1.set_ylabel('temperature [K]',color='b',size='20')
			ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
			ax1.autoscale(enable=True, axis='y', tight=None)
			ax1.plot(tnew[start_index:], Tnew1[start_index:], "bo", label='T-Down')	
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				ax1.plot(tnew[start_index:-5], Tnew2[start_index:], "go", label='T-Top')
				ax1.autoscale(enable=True, axis='y', tight=None)
			
			#Plot Current
			ax2.set_ylabel('current [A]',color='r',size='20')
			ax2.autoscale(enable=True, axis='y', tight=None)
			ax2.plot(tnew[start_index:], Inew[start_index:], 'ro', label='I')
			
			#Legend
			T_top = Rectangle((0,0), 1,1, fc="g")
			T_down = Rectangle((0,0), 1,1, fc ="b")
			ax1.legend([T_top, T_down], ["top", "down"], title="temperature",loc="upper left")
			
			show()
			
			input = raw_input("fit (y/n)?")
			if input == "y":
				
				print "--------------------------------"
				print "...fitting"
				
				#Temperature Fit -------------------------------------------------------------------------------------
				#initialize list and dicts for fit
				Tfit = [0,0,0,0,0]
				Terror = [0,0,0,0,0]
				Tparams = Parameters()
				Tparams.add('amp', value=start_parameters[0], min=0.1, max=40.0)
				Tparams.add('freq', value=start_parameters[1], min=1e-5, max=0.1, vary=False)
				Tparams.add('phase', value=0.1, min=-pi, max=pi)
				Tparams.add('offs', value=start_parameters[3], min=0.0)
				Tparams.add('slope', value=start_parameters[4])

				#perform fit
				Tresults = minimize(sinfunc, Tparams, args=(tnew[start_index:limit], Tnew1[start_index:limit]), method="leastsq")
				
				#extract params dict to lists
				extract_fit_relerr_params(Tparams, Tfit, Terror)
				
				#data correction
				if Tfit[0]<0.0:
					Tfit[0]=abs(Tfit[0])
					Tfit[2]=Tfit[2]+pi
				if Tfit[2]<0.0:
					Tfit[2] = Tfit[2] + 2*pi
				if Tfit[2]>2*pi:
					Tfit[2] = Tfit[2] - 2*pi
					
				#Fit-Plot
				ax1.plot(tnew[start_index:limit], sinfunc(Tparams, tnew[start_index:limit]), 'b-')
				draw()
				
				#calculation of maximum T error
				Terror = [Tparams['amp'].stderr, Tparams['freq'].stderr, Tparams['phase'].stderr, Tparams['offs'].stderr, Tparams['slope'].stderr]
				T_error = abs(Tparams['amp'].stderr/Tparams['amp'].value)+abs(Tparams['phase'].stderr/Tparams['phase'].value)+abs(Tparams['freq'].stderr/Tparams['freq'].value)+abs(Tparams['offs'].stderr/Tparams['offs'].value)+abs(Tparams['slope'].stderr/Tparams['slope'].value)
				
				#console output
				print "-->T-Fit:\tA=%fK\n\t\tf=%fmHz\n\t\tO=%d-%dK\n\t\tb=%.2fK/h\n\t\tError:%f" % (Tfit[0], Tfit[1]*1000, start_parameters[2],max_Temp,Tfit[4]*3600, T_error)
				
				#file output
				log = open(date+"_"+samplename+"_SineWave+LinRamp_I-T-Fits.txt", 'w+')
				log.write("#Temperature Fit Data\n----------\n\n")
				log.write("#Amp [K]\tAmp_Error\tFreq [Hz]\tFreq_Error\tPhase\t\tPhase_Error\tOffset [K]\tOffset_Error\tSlope [K/s]\tSlope_Error\n")
				log.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n\n"% (Tfit[0],Terror[0],Tfit[1],Terror[1],Tfit[2],Terror[2],Tfit[3],Terror[3],Tfit[4],Terror[4]))
				
				
				#Current Fit -----------------------------------------------------------------------------------------
				print "-->I-Fit"

				#initialize fit variables
				I_perioden = int(tnew[limit]/(perioden/start_parameters[1]))
				satzlaenge = limit/I_perioden
				
				Ifit = zeros((I_perioden-1,6))
				Ierror = zeros((I_perioden-1,5))
				Iparams = Parameters()
				Iparams.add('amp', value=1e-11)#, min=1e-13, max=1e-7)
				Iparams.add('freq', value=start_parameters[1], min=1e-5, max=0.1, vary=False)
				Iparams.add('phase', value=0.1, min=-pi, max=pi)
				Iparams.add('offs', value=1e-10)
				Iparams.add('slope', value=1e-10)
				
				#initialize file output
				log.write("#Current-Fit Data\n----------\n\n")
				log.write("#Amp [I]\t\tAmp_Error\t\tFreq [Hz]\t\tPhase\t\tPhase_Error\tOffset [A]\tOffset_Error\tSlope [A/s]\tSlope_Error\n")
				
				#perform partial fits
				for i in arange(1,I_perioden):
					start = start_index+int((i*satzlaenge)-satzlaenge)
					ende = start_index+int(i*satzlaenge)
					Iresults = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					ax2.plot(tnew[start:ende], sinfunc(Iparams, tnew[start:ende]), 'r-')
					
					#extrac fit parameters from dict to Ifit array
					Ifit[i-1,0] = Iparams['amp'].value
					Ifit[i-1,1] = Iparams['freq'].value
					Ifit[i-1,2] = Iparams['phase'].value
					Ifit[i-1,3] = Iparams['offs'].value
					Ifit[i-1,4] = Iparams['slope'].value
					Ierror[i-1,0] = Iparams['amp'].stderr
					Ierror[i-1,1] = Iparams['freq'].stderr
					Ierror[i-1,2] = Iparams['phase'].stderr
					Ierror[i-1,3] = Iparams['offs'].stderr
					Ierror[i-1,4] = Iparams['slope'].stderr
					
					#data correction
					if Ifit[i-1,0] < 0.0:
						Ifit[i-1,0] = -1*Ifit[i-1,0]
						Ifit[i-1,2] = Ifit[i-1,2]+pi
					if Ifit[i-1,2] < 0.0:
						Ifit[i-1,2] = Ifit[i-1,2]+2*pi
					if Ifit[i-1,2] > 2*pi:
						Ifit[i-1,2] = Ifit[i-1,2]-2*pi
					
					Ifit[i-1,5] = mean(Idata[start:ende,1])
					
					#calculate phase difference
					phasediff = Tfit[2]-Ifit[i-1,2]
					if phasediff < 0.0:
						phasediff = phasediff+2*pi
					if phasediff > 2*pi:
						phasefiff = phasediff-2*pi
					
					#NonPyroStrom
					#m=magenta (TSC-Strom)
					Inp = abs(Ifit[i-1,0]*cos(phasediff))
					nonpyroparams = Parameters()
					nonpyroparams.add('amp', value=Inp)
					nonpyroparams.add('freq', value=Tfit[1])
					nonpyroparams.add('phase', value=Tfit[2])
					nonpyroparams.add('offs', value=Ifit[i-1,3])
					nonpyroparams.add('slope', value=Ifit[i-1,4])
					ax2.plot(tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende]), 'm-')
					
					#Pyrostrom
					#c=cyan (Pyrostrom)
					Ip = abs(Ifit[i-1,0]*sin(phasediff))
					pyroparams = Parameters()
					pyroparams.add('amp', value=Ip)
					pyroparams.add('freq', value=Tfit[1])
					if phasediff > pi:
						pyroparams.add('phase', value=(Tfit[2]+pi/2))
					else:
						pyroparams.add('phase', value=(Tfit[2]-pi/2))
					pyroparams.add('offs', value=Ifit[i-1,3])
					pyroparams.add('slope', value=Ifit[i-1,4])
					ax2.plot(tnew[start:ende], sinfunc(pyroparams, tnew[start:ende]), 'c-')
				
				#Legend for Current Plots
				I_plot = Rectangle((0, 0), 1, 1, fc="r")
				I_TSC = Rectangle((0,0), 1,1, fc="m")
				I_p = Rectangle((0,0), 1,1, fc ="c")
				ax2.legend([I_plot, I_TSC, I_p], ["data", "TSC", "pyro"], title="current",loc="lower right")
				draw()
				
				#Bereinigung von Ifit and Ierrors
				#Ifit=vstack((trim_zeros(Ifit[:,0]),trim_zeros(Ifit[:,1]),trim_zeros(Ifit[:,2]),trim_zeros(Ifit[:,3]),trim_zeros(Ifit[:,4]), trim_zeros(Ifit[:,5])))
				#Ifit = Ifit.transpose()
				#Ierror=vstack((trim_zeros(Ierror[:,0]),trim_zeros(Ierror[:,2]),trim_zeros(Ierror[:,3]),trim_zeros(Ierror[:,4])))	#attention! freq.error column gets lost!
				#Ierror = Ierror.transpose()
				
				#file output
				for i in range(1,len(Ifit)):
					log.write("%e\t%e\t%f\t%f\t%f\t%e\t%e\t%e\t%e\n"%(Ifit[i-1,0],Ierror[i-1,0],Ifit[i-1,1],(Ifit[i-1,2]*180/pi),(Ierror[i-1,1]*180/pi),Ifit[i-1,3],Ierror[i-1,2],Ifit[i-1,4],Ierror[i-1,3]))
				log.close()
				
				#Calculating p ---------------------------------------------------------------------------------------
				print "-->p-Calculation"
				
				#for length of p array
				globale_intervalle = len(Ifit)														
		
				#area for pyroel. coefficent
				input = raw_input("    Area [m2]?:")
				if input is "A":				#for PVDF (d=13,... mm)
					flaeche = 1.3994e-4
				elif input is "B":				#for cystalls (d=15mm)
					flaeche = 1.761e-4
				elif input is "C":				#STO Kristall M114
					flaeche = 1.4668e-5
				else:
					flaeche = float(input)
				
				area_error = 0.0082*flaeche
				
				p = zeros((globale_intervalle,6))
				perror = zeros((globale_intervalle,1))
				
				for i in range(1,globale_intervalle):
					phasediff = Tfit[2]-Ifit[i-1,2]
					if phasediff < 0.0:
						phasediff = phasediff+2*pi
					if phasediff > 2*pi:
						phasediff = phasediff-2*pi
					
					p[i-1,0] = (tnew[start_index+((i-1)*satzlaenge)]*Tfit[4])+(((tnew[start_index+((i-1)*satzlaenge)]-tnew[start_index+(i*satzlaenge)])/2)*Tfit[4])+Tfit[3]	#Spalte1 - Temp
					p[i-1,1] = ((Ifit[i-1,0]*sin(phasediff))/(flaeche*Tfit[0]*2*pi*Tfit[1]))							#Spalte2 - p (Sharp-Garn)
					p[i-1,2] = (abs(Ifit[i-1,5])/(flaeche*Tfit[4]))											#Spalte3 - p (Glass-Lang-Steckel)
					p[i-1,3] = phasediff * 180/pi													#Spalte4 - Phasediff.
					p[i-1,4] = abs((Ifit[i-1,0]*sin(phasediff))/(Ifit[i-1,0]*cos(phasediff)))							#Spalte5 - ratio Pyro/TSC
					p[i-1,5] = Ifit[i-1,5]
					
					perror[i-1,0] = p_error_i(Tfit, Terror, Ifit, Ierror, phasediff, flaeche, area_error, i)
				
				#Remove zeros from array
				p_new=vstack((trim_zeros(p[:,0]),trim_zeros(p[:,1]),trim_zeros(p[:,2]),trim_zeros(p[:,3]),trim_zeros(p[:,4]), trim_zeros(p[:,5])))
				p = p_new.transpose()
				perror = trim_zeros((perror))
				
				#Print des Pyro-koeff. ueber Temperatur
				bild2=figure(date+"_"+samplename+'_Pyro')
				Tticks = arange(270,430,10)
		
				ax3=subplot(311)
				ax3.set_autoscale_on(True)
				ax3.set_xlim(270,420)
				ax3.set_ylim(min(p[:,1])*1e6-50, max(p[:,1])*1e6+50)
				ax3.set_xlabel('Temp [K]',size='20')
				ax3.set_ylabel(r"pyroel. coefficient $\mathrm{\lbrack\frac{\mu C}{Km^{2}}\rbrack}$",color='b',size='20')
				xticks(Tticks)
				ax3.grid(b=None, which='major', axis='both', color='grey')
				#ax3.errorbar(p[3:-2,0],(p[3:-2,1]*1e6), yerr=perror[3:-2,0]*1e6, fmt='b.', elinewidth=None, capsize=3, label='Pyro-koeff-SG')
				ax3.errorbar(p[:,0],(p[:,1]*1e6), yerr=perror[:,0]*1e6, fmt='b.', elinewidth=None, capsize=3, label='Pyro-koeff-SG')
				ax3.plot(p[:,0],(p[:,1]*1e6), "b.", label='Pyro-koeff-SG')
				#ax3.plot(p[3:-2,0],(p[3:-2,2]*1e6), "r.", label='Pyro-koeff-GLS')
		
				ax5=subplot(312)
				ax5.set_autoscale_on(True)#
				ax5.set_xlim(270,420)
				xticks(Tticks)
				ax5.grid(b=None, which='major', axis='both', color='grey')
				ax5.set_xlabel('Temp [K]',size='20')
				ax5.set_ylabel(r'$\mathrm{\frac{I_{p}}{I_{TSC}}}$',color='g',size='20')
				#ax5.semilogy(p[3:-2,0], p[3:-2,4], "go", label="Pyro-Curr/Non-Pyro-Curr")
				ax5.semilogy(p[:,0], p[:,4], "go", label="Pyro-Curr/Non-Pyro-Curr")

				show()
				
				#Calculating p ---------------------------------------------------------------------------------------
				print "-->P-calculation"
				PS_plot = raw_input("    T_C? (y/n): ")
				if PS_plot == "y":
					print "    ... select T_C from the p(T) or I_TSC/I_p plot!"
					#---- Berechnen der Polarisation anhand von T_C (dort ist P = 0)
					#Finden von TC
					T_C = ginput()
					T_C = T_C[0][0]
					print "    T_C: %.2f" % T_C

					#getting index of T_C in p array
					for i in arange(0, len(p)-1):
						if p[i,0] > T_C:
							T_C_start_index = i
							break

					#Berechnen des Polarisationsverlaufs
					#Initialsieren des P-arrays
					P_len = T_C_start_index
					P = zeros((P_len,2))
					
					#P bei T_C auf 0 setzten (und gleich T_C zuordnen)
					Pindex = P_len-1
					P[Pindex,0] = p[T_C_start_index,0]
					P[Pindex,1] = 0.0

					#Aufsumiereung in Array
					for i in range(0,Pindex):
						if i < Pindex:
							P[Pindex-i-1,0] = p[Pindex-i,0]								#Temperatur zuweisen
							P[Pindex-i-1,1] = P[Pindex-i,1]+abs(p[Pindex+1-i,1]*(p[Pindex+1-i,0]-p[Pindex-i,0]))	#Polarisation immer vom Vorgaenger hinzuaddieren

					#Plot
					ax6=subplot(313)
					ax6.set_autoscale_on(True)
					ax6.set_xlim(270,420)
					xticks(Tticks)
					ax6.grid(b=None, which='major', axis='both', color='grey')
					ax6.set_xlabel('Temp [K]',size='20')
					ax6.set_ylabel(r'Polarization $\mathrm{\lbrack\frac{mC}{m^{2}}\rbrack}$',color='k',size='20')
					ax6.plot(P[:,0], P[:,1]*1e3, "ko", label="Polarization")
				
				#Saving results and figs------------------------------------------------------------------------------
				print "...saving figures"
				name = date+samplename
				bild1.savefig(date+"_"+samplename+"_SineWave+LinRamp_I-T-Fits.png")
				bild2.savefig(date+"_"+samplename+"_SineWave+LinRamp_p-P.png")
				
				print "...writing log files"
				log_name2 = date+"_"+samplename+"_SineWave+LinRamp_p-Fits.txt"
				log = open(log_name2, "w+")
				log.write("#Berechnete Daten\n")
				log.write("#----------------\n")
				log.write("#Flaeche:\t%e m2\n" % flaeche)
				log.write("#----------------\n")
				log.write("#Temp\tPyro-Koeff(S-G)\t(Error)\tPyro-Koeff(L-S)\tPhasediff\tPolarization\n")
				log.write("#[K]\t[C/K*m2]\tC/K*m2]\t[C/m2]\n")
				try:
					for i in range(0,len(p)-1):
						if i>0 and i<len(P):
							log.write("%f\t%e\t%e\t%e\t%e\t%f\t%f\n" % (p[i,0],p[i,1],perror[i],p[i,4],p[i,2],p[i,3],P[i,1]))
						else:
							log.write("%f\t%e\t%e\t%e\t%e\t%f\n" % (p[i,0],p[i,1],perror[i],p[i,4],p[i,2],p[i,3]))
				except NameError:
					for i in range(0,len(p)-1):
						log.write("%f\t%e\t%e\t%e\t%e\t%f\n" % (p[i,0],p[i,1],perror[i],p[i,4],p[i,2],p[i,3]))
				log.close()
				
			else:
				pass
		
		#---------------------------------------------------------------------------------------------------------------------
		elif T_profile == "TriangleHat":
			print "Mode:\t\tTriangle"
			print "Stimulation:\tO1=%.1fK\n\t\tTm=%.1fK\n\t\tO2=%.1fK\n\t\tHR=%.1fK/h\n\t\tCR=%.1fK/h" % (start_parameters[2], start_parameters[5], start_parameters[2], start_parameters[3]*3600, start_parameters[4]*3600)
			
			#Plotting of data
			print "...plotting"
			head = date+"_"+samplename+"_Triangle"
			bild = figure(head)
			ax1 = subplot(111)
			ax2 = ax1.twinx()
			title(samplename+"_Triangle", size='15')
			
			start_index = 100
			
			#Plot Temperature
			ax1.set_xlabel('time [s]',size='20')
			ax1.set_ylabel('temperature [K]',color='b',size='20')
			ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
			ax1.plot(Tdata[start_index:,0], Tdata[start_index:,1], "bo", label='T-Down')	
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				ax1.plot(Tdata[start_index::5,0], Tdata[start_index::5,3], "go", label='T-Top')
			
			#Plot Current
			ax2.set_ylabel('current [A]',color='r',size='20')
			ax2.autoscale(enable=True, axis='y', tight=None)
			ax2.plot(Idata[start_index:,0], Idata[start_index:,1], 'ro', label='I')
			
			show()
			
			#save figure
			print "--------------------------------"
			print "...saving figure"
			savefig(date+'_'+samplename+'_Triangle.png')
			
		#SineWave+TriangleHat----------------------------------------------------------------------------------------------------
		elif T_profile == "SineWave+TriangleHat":
			print "Mode:\t\tSineWave+Triang"
			print "Stimulation:\tO1=%.1fK\n\t\tTm=%.1fK\n\t\tO2=%.1fK\n\t\tHR=%.1fK/h\n\t\tCR=%.1fK/h\n\t\tA=%.1fK\n\t\tf=%.1fmHz" % (start_parameters[2], start_parameters[5], start_parameters[2], start_parameters[3]*3600, start_parameters[4]*3600, start_parameters[0], start_parameters[1]*1000)
			
			#Interpolate data
			tnew = arange(min(Idata[:,0]),max(Tdata[:,0]),0.5)
			Tinterpol1 = interp1d(Tdata[:,0],Tdata[:,1])
			Tnew1 = Tinterpol1(tnew)
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				Tinterpol2 = interp1d(Tdata[::5,0],Tdata[::5,3])
				Tnew2 = Tinterpol2(tnew[:-5])
			Iinterpol = interp1d(Idata[:,0],Idata[:,1])
			Inew = Iinterpol(tnew)
			start_index = 100
			
			#extraction of maximum temperature
			turning_point_index = argmax(Tnew1)
			
			#Plotting of data
			print "...plotting"
			head = date+"_"+samplename+"_SineWave+Triangle"
			bild = figure(head)
			ax1 = subplot(111)
			ax2 = ax1.twinx()
			title(samplename+"_SineWave+Triangle", size='15')
			
			#Plot Temperature
			ax1.set_xlabel('time [s]',size='20')
			ax1.set_ylabel('temperature [K]',color='b',size='20')
			ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
			ax1.plot(tnew[start_index:], Tnew1[start_index:], "bo", label='T-Down')	
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				ax1.plot(tnew[start_index:-5], Tnew2[start_index:], "go", label='T-Top')
			
			#Plot Current
			ax2.set_ylabel('current [A]',color='r',size='20')
			ax2.autoscale(enable=True, axis='y', tight=None)
			ax2.plot(tnew[start_index:], Inew[start_index:], 'ro', label='I')
			show()
			
			#Legend
			T_top = Rectangle((0,0), 1,1, fc="g")
			T_down = Rectangle((0,0), 1,1, fc ="b")
			ax1.legend([T_top, T_down], ["top", "down"], title="temperature",loc="upper left")
			
			input = raw_input("fit (y/n)?")
			if input == "y":
				
				print "--------------------------------"
				print "...fitting"
				perioden = 1
				
				#Temperature Fit -------------------------------------------------------------------------------------
				#initialize list and dicts for fit
				Tfit_heat = [0,0,0,0,0]
				Terror_heat = [0,0,0,0,0]
				Tfit_cool = [0,0,0,0,0]
				Terror_cool = [0,0,0,0,0]
				Tparams = Parameters()
				Tparams.add('amp', value=start_parameters[0], min=0.1, max=40.0)
				Tparams.add('freq', value=start_parameters[1], min=1e-5, max=0.1, vary=False)
				Tparams.add('phase', value=0.1, min=-pi, max=pi)
				Tparams.add('offs', value=start_parameters[3], min=0.0)
				Tparams.add('slope', value=start_parameters[4])

				#perform fit/plot for heating
				Tresults_heat = minimize(sinfunc, Tparams, args=(tnew[start_index:turning_point_index], Tnew1[start_index:turning_point_index]), method="leastsq")
				extract_fit_relerr_params(Tparams, Tfit_heat, Terror_heat)
				Terror_heat = [Tparams['amp'].stderr, Tparams['freq'].stderr, Tparams['phase'].stderr, Tparams['offs'].stderr, Tparams['slope'].stderr]
				T_error_heat = abs(Tparams['amp'].stderr/Tparams['amp'].value)+abs(Tparams['phase'].stderr/Tparams['phase'].value)+abs(Tparams['freq'].stderr/Tparams['freq'].value)+abs(Tparams['offs'].stderr/Tparams['offs'].value)+abs(Tparams['slope'].stderr/Tparams['slope'].value)
				ax1.plot(tnew[start_index:turning_point_index], sinfunc(Tparams, tnew[start_index:turning_point_index]), 'b-')
				
				#perform fit/plot for cooling
				Tparams['slope'].value = - start_parameters[3]
				Tresults_heat = minimize(sinfunc, Tparams, args=(tnew[turning_point_index:], Tnew1[turning_point_index:]), method="leastsq")
				extract_fit_relerr_params(Tparams, Tfit_cool, Terror_cool)
				Terror_cool = [Tparams['amp'].stderr, Tparams['freq'].stderr, Tparams['phase'].stderr, Tparams['offs'].stderr, Tparams['slope'].stderr]
				T_error_cool = abs(Tparams['amp'].stderr/Tparams['amp'].value)+abs(Tparams['phase'].stderr/Tparams['phase'].value)+abs(Tparams['freq'].stderr/Tparams['freq'].value)+abs(Tparams['offs'].stderr/Tparams['offs'].value)+abs(Tparams['slope'].stderr/Tparams['slope'].value)
				ax1.plot(tnew[turning_point_index:], sinfunc(Tparams, tnew[turning_point_index:]), 'b-')

				draw()
				
				#fit data corrections:
				#heating:
				if Tfit_heat[0] < 0.0:							#for negative amplitude
					Tfit_heat[0] = abs(Tfit_heat[0])				#abs amplitude and ...
					Tfit_heat[2] = Tfit_heat[2] + pi				#...add pi to phase
				if Tfit_heat[2] < 0.0:							#for negative phase
					Tfit_heat[2] = Tfit_heat[2] + (2*pi)
				if Tfit_heat[2] > 2*pi:							#for phase larger than 2*pi
					Tfit_heat[2] = Tfit_heat[2] - (2*pi)
				#cooling:
				if Tfit_cool[0] < 0.0:							#for negative amplitude
					Tfit_cool[0] = abs(Tfit_cool[0])				#abs amplitude and ...
					Tfit_cool[2] = Tfit_cool[2] + pi				#...add pi to phase
				if Tfit_cool[2] < 0.0:							#for negative phase
					Tfit_cool[2] = Tfit_cool[2] + (2*pi)
				if Tfit_cool[2] > 2*pi:							#for phase larger than 2*pi
					Tfit_cool[2] = Tfit_cool[2] - (2*pi)
				
				#console output
				print "-->T-Fit:\nheating:\tA=%fK\n\t\tf=%fmHz\n\t\tb=%.2fK/h\n\t\tError:%f" % (Tfit_heat[0], Tfit_heat[1]*1000, Tfit_heat[4]*3600, T_error_heat)
				print "cooling:\tA=%fK\n\t\tf=%fmHz\n\t\tb=%.2fK/h\n\t\tError:%f" % (Tfit_cool[0], Tfit_cool[1]*1000, Tfit_cool[4]*3600, T_error_cool)
				
				#file output
				log = open(date+"_"+samplename+"_SineWave+Triang_I-T-Fits.txt", 'w+')
				log.write("#Temperature Fit Data\n----------\n\n")
				log.write("#Heating: %.2f K-%.2f K\n" % (start_parameters[2], start_parameters[5]))
				log.write("#Amp [K]\tAmp_Error\tFreq [Hz]\tFreq_Error\tPhase\t\tPhase_Error\tOffset [K]\tOffset_Error\tSlope [K/s]\tSlope_Error\n")
				log.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n\n"% (Tfit_heat[0],Terror_heat[0],Tfit_heat[1],Terror_heat[1],Tfit_heat[2],Terror_heat[2],Tfit_heat[3],Terror_heat[3],Tfit_heat[4],Terror_heat[4]))
				log.write("#Heating: %.2f K-%.2f K\n" % (start_parameters[2], start_parameters[5]))
				log.write("#Amp [K]\tAmp_Error\tFreq [Hz]\tFreq_Error\tPhase\t\tPhase_Error\tOffset [K]\tOffset_Error\tSlope [K/s]\tSlope_Error\n")
				log.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n\n"% (Tfit_cool[0],Terror_cool[0],Tfit_cool[1],Terror_cool[1],Tfit_cool[2],Terror_cool[2],Tfit_cool[3],Terror_cool[3],Tfit_cool[4],Terror_cool[4]))
				
				
				#Current Fit -------------------------------------------------------------------------------------
				#filtering of runaways
				print "-->I-Fit"
				ausreisserliste = []
				for i in arange(0,len(Inew)-1):
					if Inew[i]>1e-6:		#everything bigger than 1muA
						ausreisserliste.append(i)
				Inew = delete(Inew,ausreisserliste,0)
				
				#fit for heating ramp-----------------------------------------------------------------------------
				I_perioden_heat = int((tnew[turning_point_index]-tnew[start_index])/(perioden/start_parameters[1]))
				satzlaenge_heat = len(tnew[start_index:turning_point_index])/I_perioden_heat
				
				#Initialize Variables
				Ifit_heat = zeros((I_perioden_heat-1,6))
				Ierror_heat = zeros((I_perioden_heat-1,5))
				Iparams = Parameters()
				Iparams.add('amp', value=1e-11)#, min=1e-13, max=1e-7)
				Iparams.add('freq', value=start_parameters[1], min=1e-5, max=0.1, vary=False)
				Iparams.add('phase', value=0.1, min=-pi, max=pi)
				Iparams.add('offs', value=1e-10)
				Iparams.add('slope', value=1e-10)
				
				#initialize file output
				log.write("#Current-Fit Data\n----------\n\n")
				log.write("#Heating:\n")
				log.write("#Amp [I]\t\tAmp_Error\t\tFreq [Hz]\t\tPhase\t\tPhase_Error\tOffset [A]\tOffset_Error\tSlope [A/s]\tSlope_Error\n")
				
				for i in arange(1,I_perioden_heat):
					start = start_index+(int((i*satzlaenge_heat)-satzlaenge_heat))
					ende = start_index+(int(i*satzlaenge_heat))
					Iresults = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					ax2.plot(tnew[start:ende], sinfunc(Iparams, tnew[start:ende]), 'r-')
					
					#data extraction
					Ifit_heat[i-1,0] = Iparams['amp'].value
					Ifit_heat[i-1,1] = Iparams['freq'].value
					Ifit_heat[i-1,2] = Iparams['phase'].value
					Ifit_heat[i-1,3] = Iparams['offs'].value
					Ifit_heat[i-1,4] = Iparams['slope'].value
					Ierror_heat[i-1,0] = Iparams['amp'].stderr
					Ierror_heat[i-1,1] = Iparams['freq'].stderr
					Ierror_heat[i-1,2] = Iparams['phase'].stderr
					Ierror_heat[i-1,3] = Iparams['offs'].stderr
					Ierror_heat[i-1,4] = Iparams['slope'].stderr
					
					#correction of data
					if Ifit_heat[i-1,0] < 0.0:					#for negative amplitude
						Ifit_heat[i-1,0] = abs(Ifit_heat[i-1,0])		#make positive ...
						Ifit_heat[i-1,2] = Ifit_heat[i-1,2]+pi			#and add pi to phase
					if Ifit_heat[i-1,2] < 0.0:
						Ifit_heat[i-1,2] = Ifit_heat[i-1,2] + 2+pi
					if Ifit_heat[i-1,2] > 2*pi:
						Ifit_heat[i-1,2] = Ifit_heat[i-1,2] + 2+pi
						
					#calculate phase difference, mean, ...
					phasediff = Tfit_heat[2]-Ifit_heat[i-1,2]
					if phasediff < 0.0:
						phasediff = phasediff + 2*pi
					if phasediff > 2*pi:
						phasediff = phasediff - 2*pi
					Ifit_heat[i-1,5] = mean(Inew[start:ende])
					
					#plot TSC and pyro-current
					#m=magenta (TSC-Strom)
					Inp = abs(Ifit_heat[i-1,0]*cos(phasediff))
					nonpyroparams = Parameters()
					nonpyroparams.add('amp', value=Inp)
					nonpyroparams.add('freq', value=Tfit_heat[1])
					nonpyroparams.add('phase', value=Tfit_heat[2])
					nonpyroparams.add('offs', value=Ifit_heat[i-1,3])
					nonpyroparams.add('slope', value=Ifit_heat[i-1,4])
					ax2.plot(tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende]), 'm-')
					
					#c=cyan (Pyrostrom)
					Ip = abs(Ifit_heat[i-1,0]*sin(phasediff))
					pyroparams = Parameters()
					pyroparams.add('amp', value=Ip)
					pyroparams.add('freq', value=Tfit_heat[1])
					if phasediff > pi:
						pyroparams.add('phase', value=(Tfit_heat[2]+pi/2))
					else:
						pyroparams.add('phase', value=(Tfit_heat[2]-pi/2))
					pyroparams.add('offs', value=Ifit_heat[i-1,3])
					pyroparams.add('slope', value=Ifit_heat[i-1,4])
					ax2.plot(tnew[start:ende], sinfunc(pyroparams, tnew[start:ende]), 'c-')
				
				draw()
				
				#fit for cooling ramp-----------------------------------------------------------------------------
				I_perioden_cool = int((tnew[-1]-tnew[turning_point_index])/(perioden/start_parameters[1]))
				satzlaenge_cool = len(tnew[turning_point_index:])/I_perioden_cool
				
				#Initialize Variables
				Ifit_cool = zeros((I_perioden_cool-1,6))
				Ierror_cool = zeros((I_perioden_cool-1,5))
				
				#initialize file output for cooling
				log.write("#Cooling:\n")
				log.write("#Amp [I]\t\tAmp_Error\t\tFreq [Hz]\t\tPhase\t\tPhase_Error\tOffset [A]\tOffset_Error\tSlope [A/s]\tSlope_Error\n")
				
				for i in arange(1,I_perioden_cool):
					start = turning_point_index+(int((i*satzlaenge_cool)-satzlaenge_cool))
					ende = turning_point_index+(int(i*satzlaenge_cool))
					Iresults = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					ax2.plot(tnew[start:ende], sinfunc(Iparams, tnew[start:ende]), 'r-')
					
					#data extraction
					Ifit_cool[i-1,0] = Iparams['amp'].value
					Ifit_cool[i-1,1] = Iparams['freq'].value
					Ifit_cool[i-1,2] = Iparams['phase'].value
					Ifit_cool[i-1,3] = Iparams['offs'].value
					Ifit_cool[i-1,4] = Iparams['slope'].value
					Ierror_cool[i-1,0] = Iparams['amp'].stderr
					Ierror_cool[i-1,1] = Iparams['freq'].stderr
					Ierror_cool[i-1,2] = Iparams['phase'].stderr
					Ierror_cool[i-1,3] = Iparams['offs'].stderr
					Ierror_cool[i-1,4] = Iparams['slope'].stderr
					
					#correction of data
					if Ifit_cool[i-1,0] < 0.0:					#for negative amplitude
						Ifit_cool[i-1,0] = abs(Ifit_cool[i-1,0])		#make positive ...
						Ifit_cool[i-1,2] = Ifit_cool[i-1,2]+pi			#and add pi to phase
					if Ifit_cool[i-1,2] < 0.0:
						Ifit_cool[i-1,2] = Ifit_cool[i-1,2] + 2+pi
					if Ifit_cool[i-1,2] > 2*pi:
						Ifit_cool[i-1,2] = Ifit_cool[i-1,2] + 2+pi
						
					#calculate phase difference, mean, ...
					phasediff = Tfit_cool[2]-Ifit_cool[i-1,2]
					if phasediff < 0.0:
						phasediff = phasediff + 2*pi
					if phasediff > 2*pi:
						phasediff = phasediff - 2*pi
					Ifit_cool[i-1,5] = mean(Inew[start:ende])
					
					#plot TSC and pyro-current
					#m=magenta (TSC-Strom)
					Inp = abs(Ifit_cool[i-1,0]*cos(phasediff))
					nonpyroparams = Parameters()
					nonpyroparams.add('amp', value=Inp)
					nonpyroparams.add('freq', value=Tfit_cool[1])
					nonpyroparams.add('phase', value=Tfit_cool[2])
					nonpyroparams.add('offs', value=Ifit_cool[i-1,3])
					nonpyroparams.add('slope', value=Ifit_cool[i-1,4])
					ax2.plot(tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende]), 'm-')
					
					#c=cyan (Pyrostrom)
					Ip = abs(Ifit_cool[i-1,0]*sin(phasediff))
					pyroparams = Parameters()
					pyroparams.add('amp', value=Ip)
					pyroparams.add('freq', value=Tfit_cool[1])
					if phasediff > pi:
						pyroparams.add('phase', value=(Tfit_cool[2]+pi/2))
					else:
						pyroparams.add('phase', value=(Tfit_cool[2]-pi/2))
					pyroparams.add('offs', value=Ifit_cool[i-1,3])
					pyroparams.add('slope', value=Ifit_cool[i-1,4])
					ax2.plot(tnew[start:ende], sinfunc(pyroparams, tnew[start:ende]), 'c-')
					
				draw()
				
				#Legend for Current Plots
				I_plot = Rectangle((0, 0), 1, 1, fc="r")
				I_TSC = Rectangle((0,0), 1,1, fc="m")
				I_p = Rectangle((0,0), 1,1, fc ="c")
				ax2.legend([I_plot, I_TSC, I_p], ["data", "TSC", "pyro"], title="current",loc="lower right")
				draw()
				
				#file output
				for i in range(1,len(Ifit_heat)):
					log.write("%e\t%e\t%f\t%f\t%f\t%e\t%e\t%e\t%e\n"%(Ifit_heat[i-1,0],Ierror_heat[i-1,0],Ifit_heat[i-1,1],(Ifit_heat[i-1,2]*180/pi),(Ierror_heat[i-1,1]*180/pi),Ifit_heat[i-1,3],Ierror_heat[i-1,2],Ifit_heat[i-1,4],Ierror_heat[i-1,3]))
				for i in range(1,len(Ifit_cool)):
					log.write("%e\t%e\t%f\t%f\t%f\t%e\t%e\t%e\t%e\n"%(Ifit_cool[i-1,0],Ierror_cool[i-1,0],Ifit_cool[i-1,1],(Ifit_cool[i-1,2]*180/pi),(Ierror_cool[i-1,1]*180/pi),Ifit_cool[i-1,3],Ierror_cool[i-1,2],Ifit_cool[i-1,4],Ierror_cool[i-1,3]))
				log.close()
				
				#Calculating p ---------------------------------------------------------------------------------------
				print "-->p-Calculation"
				
				#area for pyroel. coefficent
				input = raw_input("    Area [m2]?:")
				if input is "A":				#for PVDF (d=13,... mm)
					flaeche = 1.3994e-4
				elif input is "B":				#for cystalls (d=15mm)
					flaeche = 1.761e-4
				elif input is "C":				#STO Kristall M114
					flaeche = 1.4668e-5
				else:
					flaeche = float(input)
				
				#for length of p array
				number_p_heat = len(Ifit_heat)
				number_p_cool = len(Ifit_cool)
				area_error = 0.0082*flaeche
				p_heat = zeros((number_p_heat,6))
				p_cool = zeros((number_p_cool,6))
				perror_heat = zeros((number_p_heat,1))
				perror_cool = zeros((number_p_cool,1))
				
				for i in arange(1,number_p_heat):
					phasediff = Tfit_heat[2]-Ifit_heat[i-1,2]
					if phasediff < 0.0:
						phasediff = phasediff+2*pi
					if phasediff > 2*pi:
						phasediff = phasediff-2*pi
					
					p_heat[i-1,0] = (tnew[start_index+((i-1)*satzlaenge_heat)]*Tfit_heat[4])+(((tnew[start_index+((i-1)*satzlaenge_heat)]-tnew[start_index+(i*satzlaenge_heat)])/2)*Tfit_heat[4])+Tfit_heat[3]			#Spalte1 - Temperatur
					p_heat[i-1,1] = (Ifit_heat[i-1,0]*sin(phasediff))/(flaeche*Tfit_heat[0]*2*pi*Tfit_heat[1])							#Spalte2 - Pyrokoeff. berechnen (Sharp-Garn)
					p_heat[i-1,2] = (abs(Ifit_heat[i-1,5])/(flaeche*Tfit_heat[4]))											#Spalte3 - Pyrokoeff. berechnen (Glass-Lang-Steckel)
					p_heat[i-1,3] = phasediff * 180/pi													#Spalte4 - Phasediff.
					p_heat[i-1,4] = abs((Ifit_heat[i-1,0]*sin(phasediff))/(Ifit_heat[i-1,0]*cos(phasediff)))							#Verhaeltnis Pyro/nichPyroStrom
					p_heat[i-1,5] = Ifit_heat[i-1,5]
					
					perror_heat[i-1,0] = p_error_i(Tfit_heat, Terror_heat, Ifit_heat, Ierror_heat, phasediff, flaeche, area_error, i)
				

				
				##Remove zeros from array
				#p_new=vstack((trim_zeros(p[:,0]),trim_zeros(p[:,1]),trim_zeros(p[:,2]),trim_zeros(p[:,3]),trim_zeros(p[:,4]), trim_zeros(p[:,5])))
				#p = p_new.transpose()
				#perror = trim_zeros((perror))
				
				##Print des Pyro-koeff. ueber Temperatur
				#bild2=figure(date+"_"+samplename+'_Pyro')
				#Tticks = arange(270,430,10)
		
				#ax3=subplot(311)
				#ax3.set_autoscale_on(True)
				#ax3.set_xlim(270,420)
				#ax3.set_ylim(min(p[:,1])*1e6-50, max(p[:,1])*1e6+50)
				#ax3.set_xlabel('Temp [K]',size='20')
				#ax3.set_ylabel(r"pyroel. coefficient $\mathrm{\lbrack\frac{\mu C}{Km^{2}}\rbrack}$",color='b',size='20')
				#xticks(Tticks)
				#ax3.grid(b=None, which='major', axis='both', color='grey')
				##ax3.errorbar(p[3:-2,0],(p[3:-2,1]*1e6), yerr=perror[3:-2,0]*1e6, fmt='b.', elinewidth=None, capsize=3, label='Pyro-koeff-SG')
				#ax3.errorbar(p[:,0],(p[:,1]*1e6), yerr=perror[:,0]*1e6, fmt='b.', elinewidth=None, capsize=3, label='Pyro-koeff-SG')
				#ax3.plot(p[:,0],(p[:,1]*1e6), "b.", label='Pyro-koeff-SG')
				##ax3.plot(p[3:-2,0],(p[3:-2,2]*1e6), "r.", label='Pyro-koeff-GLS')
		
				#ax5=subplot(312)
				#ax5.set_autoscale_on(True)#
				#ax5.set_xlim(270,420)
				#xticks(Tticks)
				#ax5.grid(b=None, which='major', axis='both', color='grey')
				#ax5.set_xlabel('Temp [K]',size='20')
				#ax5.set_ylabel(r'$\mathrm{\frac{I_{p}}{I_{TSC}}}$',color='g',size='20')
				##ax5.semilogy(p[3:-2,0], p[3:-2,4], "go", label="Pyro-Curr/Non-Pyro-Curr")
				#ax5.semilogy(p[:,0], p[:,4], "go", label="Pyro-Curr/Non-Pyro-Curr")

				#show()
				


				
				
				
			
		
			#save figure
			print "--------------------------------"
			print "...saving figure"
			savefig(date+'_'+samplename+'_SineWave+Triangle.png')
		else:
			pass
		
		print "--------------------------------"
		print "DONE!"
		
	#-----------------------------------------------------------------------------------------------------------------------------
	#AutoPol
	elif HV_status == "Polarize":
		print "Mode:\t\tAutoPolarization"
		print "Temperature:\t%.2f K" % start_parameters[5]
		print "max. Voltage:\t%.2f V" % max(HVdata[:,1])
		print "Compliance:\t%.2e A" % HV_set[1]
		print "--------------------------------"
		
		#Plotting of Data
		print "...plotting"
		head = date+"_"+samplename + "_AutoPol"
		bild = figure(head)
		ax1 = subplot(111)
		ax2 = ax1.twinx()
		title(samplename+"_AutoPol",size='15')

		#Plot Voltage
		ax1.set_xlabel('time [s]',size='20')
		ax1.set_ylabel('voltage [A]',color='g',size='20')
		ax1.grid(b=None, which='major', color='grey', linewidth=1)
		ax1.autoscale(enable=True, axis='y', tight=None)
		ax1.plot(HVdata[:,0], HVdata[:,1], "g.", label='Voltage')

		#Plot Current
		ax2.autoscale(enable=True, axis='y', tight=None)
		ax2.set_ylabel('current [A]',color='r',size='20')
		ax2.plot(Idata[:,0], Idata[:,1], "r.", label='Current')
		
		show()
		
		#Fit exponential decay
		input = raw_input("fit exponential decay? (y/n)")
		if input == "y":
			#console output and graphical input
			print "...fitting"
			print "-->select start of fit from graph."
			input = ginput()
			
			#getting starting time
			start_time = input[0][0]
			
			#interpolation
			HVinterpol = interp1d(HVdata[:,0], HVdata[:,1])
			Iinterpol = interp1d(Idata[:,0], Idata[:,1])
			tnew = arange(min(Idata[:,0]),max(Idata[:,0]), 0.25)
			HVnew = HVinterpol(tnew)
			Inew = Iinterpol(tnew)
			
			count = 0
			for i in arange(0,len(tnew)-1):
				if tnew[i] < start_time:
					count = count+1
			
			start = count
			end = len(tnew)-20
			
			#fit
			expparams = Parameters()
			expparams.add('factor', value=1e-9)
			expparams.add('decay', value=2000, min=0.0)
			expparams.add('offs', value=45e-9)#, min=0.0)
			HVresults = minimize(expdecay, expparams, args=(tnew[start:end], Inew[start:end]), method="leastsq")

			#plot
			ax2.plot(tnew[start:end], expdecay(expparams, tnew[start:end]), 'k-')
			box_text = "Temperature: "+str(start_parameters[5])+" K\nmax.Voltage: "+str(max(HVdata[:,1]))+" V\nCompliance: "+str(HV_set[1])+" A\nA: " + str(expparams['factor'].value) + "\nt0: "+ str(expparams['decay'].value) + "s\nIoffs: " + str(expparams['offs'].value) + "A"
			box = figtext(0.55,0.15,box_text,fontdict=None, bbox=dict(facecolor='white', alpha=0.5))
			draw()
			
			#console output
			print "-->Exp-Fit:\tA=%e\n\t\tt0=%ds\n\t\tO=%eA" % (expparams['factor'].value, expparams['decay'].value, expparams['offs'].value)
			
		else:
			box_text = "Temperature: "+str(start_parameters[5])+" K\nmax.Voltage: "+str(max(HVdata[:,1]))+" V\nCompliance: "+str(HV_set[1])+" A"
			box = figtext(0.65,0.15,box_text,fontdict=None, bbox=dict(facecolor='white', alpha=0.5))
			
		draw()
			
		#save figure
		print "--------------------------------"
		print "...saving figure"
		savefig(date+'_'+samplename+'_Polarize.png')
		print "DONE!"

	#-----------------------------------------------------------------------------------------------------------------------------
	#HighVoltage always on
	elif HV_status == "On":
		#---------------------------------------------------------------------------------------------------------------------
		if T_profile == "Thermostat":
			print "Mode:\t\tHV_on+Thermostat"
			print "Voltage:\t%.1fV" % HV_set[0]	#for future use
			print "Temperature:\t%.1fK" % start_parameters[5]
			
			#Plotting of data
			print "--------------------------------"
			print "...plotting"
			head = date+"_"+samplename+"_HVon+Thermostat"
			bild = figure(head)
			ax1 = subplot(111)
			ax2 = ax1.twinx()
			title(samplename+"_HVon+Thermostat", size='15')
			
			start_index = 0
			
			#Plot Temperature
			ax1.set_xlabel('time [s]',size='20')
			ax1.set_ylabel('temperature [K]',color='b',size='20')
			ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
			ax1.plot(Tdata[start_index:,0], Tdata[start_index:,1], "b-", label='T-Down')
			ax1.set_ylim(start_parameters[5]-2, start_parameters[5]+2)
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				ax1.plot(Tdata[start_index::5,0], Tdata[start_index::5,3], "g-", label='T-Top')
			
			#Plot Current
			ax2.set_ylabel('current [A]',color='r',size='20')
			ax2.autoscale(enable=True, axis='y', tight=None)
			ax2.plot(Idata[start_index:,0], Idata[start_index:,1], 'r-', label='I')
			
			#text box
			box_text = "Temperature: "+str(start_parameters[5])+" K\nVoltage: 100V"
			#box_text = "Temperature: "+str(start_parameters[5])+" K\nVoltage: "+str(HV_set[0])+" V"
			box = figtext(0.65,0.15,box_text,fontdict=None, bbox=dict(facecolor='white', alpha=0.5))
			
			show()
			
			#save figure
			print "--------------------------------"
			print "...saving figure"
			savefig(date+'_'+samplename+'_HVon+Thermostat.png')
		
		#---------------------------------------------------------------------------------------------------------------------
		elif T_profile == "SineWave":
			print "Mode:\t\tHV_on+SineWave"
			print "Voltage:\t%.1fV" % HV_set[0]	#for future use
			print "Stimulation:\tA=%.1fK\n\t\tf=%.1fmHz\n\t\tO=%.1fK" % (start_parameters[0], start_parameters[1]*1000, start_parameters[2])
		
		
			#Interpolate data
			tnew = arange(min(Idata[:,0])+1,max(Tdata[:,0]-1),0.5)
			Tinterpol1 = interp1d(Tdata[:,0],Tdata[:,1])
			Tnew1 = Tinterpol1(tnew)
			#if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
			#	Tinterpol2 = interp1d(Tdata[::5,0],Tdata[::5,3])
			#	Tnew2 = Tinterpol2(tnew)
			Iinterpol = interp1d(Idata[:,0],Idata[:,1])
			Inew = Iinterpol(tnew)
			
			start_index = 50
			
			#Plotting of data
			print "--------------------------------"
			print "...plotting"
			head = date+"_"+samplename+"_HVon+SineWave"
			bild = figure(head)
			ax1 = subplot(111)
			ax2 = ax1.twinx()
			title(samplename+"_HVon+SineWave", size='15')
			
			#Plot Temperature
			ax1.set_xlabel('time [s]',size='20')
			ax1.set_ylabel('temperature [K]',color='b',size='20')
			ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
			ax1.plot(tnew[start_index:], Tnew1[start_index:], "bo", label='T-Down')	
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				ax1.plot(Tdata[start_index::5,0], Tdata[start_index::5,3], "go", label='T-Top')
			
			#Plot Current
			ax2.set_ylabel('current [A]',color='r',size='20')
			ax2.autoscale(enable=True, axis='y', tight=None)
			ax2.plot(tnew[start_index:], Inew[start_index:], 'ro', label='I')
			
			show()
			
			#---------------------------------------------------------------------------------------------------------------
			input = raw_input("fit? (y/n)")
			if input == "y":
				
				print "--------------------------------"
				print "... fitting" 
				
				#Fit temperature----------------------------------------------------------------------------------------
				#intialize variables of fit
				Tfit = [0,0,0,0,0]	#bottom temperature
				Tfit2 = [0,0,0,0,0]	#top temperature
				Terror = [0,0,0,0,0]
				Terror2 = [0,0,0,0,0]
				
				#initialize parameters dict for lmfit (bottom temperature)
				Tparams = Parameters()
				Tparams.add('amp', value=start_parameters[0], min=0.1, max=40.0)
				Tparams.add('freq', value=start_parameters[1], min=1e-5, max=0.05, vary=False)
				Tparams.add('phase', value=0.1, min=-pi, max=pi)
				Tparams.add('offs', value=start_parameters[3], min=0.0)
				Tparams.add('slope', value=start_parameters[4])
				
				#fit of bottom temperature
				Tresults = minimize(sinfunc, Tparams, args=(tnew[start_index:], Tnew1[start_index:]), method="leastsq")
				#write fit params to Tfit-list
				extract_fit_relerr_params(Tparams, Tfit, Terror)
				#correction of phase < 0
				if Tfit[2]<0.0:
					Tfit[2] = Tfit[2]+(2*pi)
				
				#for top temperature
				if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
					#initialize parameters dict for lmfit (top temperature)
					Tparams2 = Parameters()
					Tparams2.add('amp', value=start_parameters[0], min=0.1, max=40.0)
					Tparams2.add('freq', value=start_parameters[1], min=1e-5, max=0.05, vary=False)
					Tparams2.add('phase', value=0.1, min=-pi, max=pi)
					Tparams2.add('offs', start_parameters[2], min=0.0)
					Tparams2.add('slope', value=start_parameters[3])
					
					#fit of top temperature
					Tresults2 = minimize(sinfunc, Tparams2, args=(Tdata[start_index::5,0], Tdata[start_index::5,3]), method="leastsq")
					#write fit params to Tfit2-list
					extract_fit_relerr_params(Tparams, Tfit2, Terror2)
					#correction of phase < 0
					if Tfit2[2]<0.0:
						Tfit2[2] = Tfit2[2]+(2*pi)
				
				#plot of fits
				ax1.plot(tnew[start_index:], sinfunc(Tparams, tnew[start_index:]), 'b-', label='T-Fit-Bottom')
				if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
					ax1.plot(Tdata[start_index:,0], sinfunc(Tparams2, Tdata[start_index:,0]), 'g-', label='T-Fit-Top')
				

				#Fit current ---------------------------------------------------------------------------------------------
				#start_index=start_index+50
				Ifit = [0,0,0,0,0]
				Ierror = [0,0,0,0,0]

				#initialize parameters dict for current fit
				Iparams = Parameters()
				Iparams.add('amp', value=1e-11)#, min=1e-13, max=1e-7)
				Iparams.add('freq', value=start_parameters[1], min=1e-5, max=0.05, vary=False)
				Iparams.add('phase', value=0.1, min=-pi, max=pi)
				Iparams.add('offs', value=1e-10)
				Iparams.add('slope', value=1e-10)
				
				#current fit
				Iresults = minimize(sinfunc, Iparams, args=(tnew[start_index:], Inew[start_index:]), method="leastsq")
				#extract params dict
				extract_fit_relerr_params(Iparams, Ifit, Ierror)
				#fit corrections
				if Ifit[0]<0.0:				#if amplitude is negative (make positive + pi in phase)
					Ifit[0] = -1*Ifit[0]
					Ifit[2] = Ifit[2]+pi
				if Ifit[2]<0.0:				#if phase is negative (add 2*pi)
					Ifit[2] = Ifit[2]+(2*pi)

				#plot current fit
				ax2.plot(tnew[start_index:], sinfunc(Iparams, tnew[start_index:]), "r-", label='I-Fit')
				
				#calculating pyroelectric coefficient------------------------------------------------------------------------
				input = raw_input("Area [m2]?:")
				if input is "A":				#for PVDF (d=13,...mm)
					flaeche = 1.3994e-4
				elif input is "B":				#for cystalls (d=15mm)
					flaeche = 1.761e-4
				elif input is "C":				#STO Kristall M114
					flaeche = 1.4668e-5
				else:
					flaeche = float(input)
				
				area_error = 0.0082*flaeche
				phasediff = Tfit[2]-Ifit[2]
				if phasediff < 0.0:
					phasediff = phasediff + 2*pi
				
				#NonPyroStrom
				#m=magenta (TSC)
				Inp = abs(Ifit[0])*cos(phasediff)
				nonpyroparams = Parameters()
				np_params = [Inp, Tfit[1], Tfit[2], Ifit[3], Ifit[4]]
				listtoparam(np_params, nonpyroparams)
				ax2.plot(tnew[start_index:], sinfunc(nonpyroparams, tnew[start_index:]), 'm-', label='I-np')

				#Pyrostrom
				#c=cyan (Pyro)
				Ip = abs(Ifit[0])*sin(phasediff)
				pyroparams = Parameters()
				if phasediff > pi:
					p_params = [Ip, Tfit[1], Tfit[2]+pi/2, Ifit[3], Ifit[4]]
				else:
					p_params = [Ip, Tfit[1], Tfit[2]-pi/2, Ifit[3], Ifit[4]]
				listtoparam(p_params, pyroparams)
				ax2.plot(tnew[start_index:], sinfunc(pyroparams, tnew[start_index:]), 'c-', label='I-p')

				pyro_koeff = (Ip)/(flaeche*Tfit[0]*2*pi*Ifit[1])
				perror = p_error(Tfit, Terror, Ifit, Ierror, phasediff, flaeche, area_error)
				
				#console output ------------------------------------------------------------------------------------------
				print 'Area:\t\t', flaeche, 'm2'
				print 'I_pyro:\t\t', fabs(Ip), 'A'
				print 'I_TSC:\t\t', fabs(Inp), 'A'
				print 'phase T-I:\t',(phasediff*180/pi)
				print 'p:\t\t', pyro_koeff*1e6,'yC/Km2'
				print '\t\t(+-', perror*1e6,'yC/Km2)'
				print 'B_T:\t\t%f nA/K' % (fabs(Inp/Tfit[1])*1e9)
				
				#file output ---------------------------------------------------------------------------------------------
				log_name = date + "_Auswertung" + ".txt"
				log = open(log_name, "w+")
				log.write("Auswertung\n----------\n")
				log.write("T-Fit Down:\tA=%f K\t\tf=%f Hz\tp=%f\t\tOffs=%f K\t\tb=%f K/s\n" % (Tfit[0],Tfit[1],Tfit[2],Tfit[3],Tfit[4]))
				log.write("T-Fit Top: \tA=%f K\t\tf=%f Hz\tp=%f\t\tOffs=%f K\t\t\tb=%f K/s\n" % (Tfit2[0],Tfit2[1],Tfit2[2],Tfit2[3],Tfit2[4]))
				log.write("I-Fit:\t\tA=%e K\tf=%f Hz\tp=%f\t\tOffs=%e K\t\tb=%e K/s\n" % (Ifit[0],Ifit[1],Ifit[2],Ifit[3],Ifit[4]))
				log.write('------------------------------------\n')
				log.write('Area:\t\t%fm2 \n' % (flaeche))
				log.write('T-Amp:\t\t%f K\n' %  (Tfit[0]))
				log.write('I_pyro:\t\t%eA \n' %(fabs(Ip)))
				log.write('I_TSC:\t\t%eA \n' % (fabs(Inp)))
				log.write('phase T-I:\t%f \n' % (phasediff*180/pi))
				log.write('p:\t\t\t%f yC/Km2\n' %(pyro_koeff*1e6))
				log.write('\t\t\t(+-%f yC/Km2)\n' %(perror*1e6))
				log.write('B_T:\t\t%f nA/K\n' % (fabs(Inp/Tfit[1])*1e9))
				log.close()
				
				#box, save figure, legend, ...
				ax1.legend(loc=1)
				ax2.legend(loc=4)
		
				#box_text = "Area: "+str(flaeche)+" m2\n"+"I-Amp: "+str(Ifit[0])+" A\n"+"T-Amp: "+str(Tfit[0])+" K\n"+"Phase-Diff.: "+str(phasediff*(180/pi))+"\n"+"pyroel. Coeff.: "+str(pyro_koeff*1e6)+" yC/Km2\nVoltage: %.1f" + str(HV_set[0]) + " V"
				box_text = r"Area: "+str(flaeche)+ r" $\mathrm{m^2}$"+"\n"+ r"I-Amp: "+str(Ifit[0])+r" A"+"\n"+ r"T-Amp: "+str(Tfit[0])+r" K"+"\n"+r"Phase-Diff.: "+str(phasediff*(180/pi))+"$^{\circ}$"+"\n"+r"pyroel. Coeff.: "+str(pyro_koeff*1e6)+r" $\mathrm{\mu C/Km^2}$"+"\n"+r"-100 V"

				figtext(0.15,0.12,box_text,fontdict=None, bbox=dict(facecolor='white', alpha=0.5))
			
			#save figure
			print "--------------------------------"
			print "...saving figure"
			savefig(date+'_'+samplename+'_HVon+SineWave.png')
		elif T_profile == "SineWave+LinRamp":
			print "Mode:\t\tHV_on+SineWave+LinRamp"
			print "Voltage:\t%.1fV" % HV_set[0]	#for future use
			print "Stimulation:\tA=%.1fK\n\t\tf=%.1fmHz\n\t\tO=%.1fK\n\t\tb=%.1fK/h" % (start_parameters[0], start_parameters[1]*1000, start_parameters[2], start_parameters[3]*3600)
			
						
			perioden = 2
			start_index = 100
			
			#Interpolate data
			tnew = arange(min(Idata[:,0]),max(Tdata[:,0]),0.5)
			Tinterpol1 = interp1d(Tdata[:,0],Tdata[:,1])
			Tnew1 = Tinterpol1(tnew)
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				Tinterpol2 = interp1d(Tdata[::5,0],Tdata[::5,3])
				Tnew2 = Tinterpol2(tnew)
			Iinterpol = interp1d(Idata[:,0],Idata[:,1])
			Inew = Iinterpol(tnew)
			
			#filtering of data for currente values larger than 1muA
			ausreisserliste = []
			for i in arange(0,len(Inew)-1):
				if Inew[i]>=1e-6:
					ausreisserliste.append(i)
			Inew = delete(Inew,ausreisserliste,0)
			tnew = delete(tnew,ausreisserliste,0)
			Tnew1 = delete(Tnew1,ausreisserliste,0)
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				Tnew2 = delete(Tnew2,ausreisserliste,0)
			
			#check when ramp runs into T_Limit_H
			for i in arange(0,len(Tnew1)-1):
				if Tnew1[i] < start_parameters[5]-1:
					limit = i
					
			#important calculations ;)
			max_Temp = tnew[limit]*start_parameters[3]+start_parameters[2]
			T_perioden = int(tnew[limit]/(1/start_parameters[1]))
			
			tmax = tnew[limit]
			satzlaenge = (limit-start_index)/T_perioden
			
			#Plotting of data
			print "--------------------------------"
			print "...plotting"
			head = date+"_"+samplename+"_HV_on+SineWave+LinRamp"
			bild1 = figure(head)
			ax1 = subplot(111)
			ax2 = ax1.twinx()
			title(samplename+"_HV_on+SineWave+LinRamp", size='15')
			
			#Plot Temperature
			ax1.set_xlabel('time [s]',size='20')
			ax1.set_ylabel('temperature [K]',color='b',size='20')
			ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
			ax1.autoscale(enable=True, axis='y', tight=None)
			ax1.plot(tnew[start_index:], Tnew1[start_index:], "bo", label='T-Down')	
			if shape(Tdata)[1]>=4 and max(Tdata[:,3])!=9.9e+37:
				ax1.plot(tnew[start_index:], Tnew2[start_index:], "go", label='T-Top')
				ax1.autoscale(enable=True, axis='y', tight=None)
			
			#Plot Current
			ax2.set_ylabel('current [A]',color='r',size='20')
			ax2.autoscale(enable=True, axis='y', tight=None)
			ax2.plot(tnew[start_index:], Inew[start_index:], 'ro', label='I')
			
			#Legend
			T_top = Rectangle((0,0), 1,1, fc="g")
			T_down = Rectangle((0,0), 1,1, fc ="b")
			ax1.legend([T_top, T_down], ["top", "down"], title="temperature",loc="upper left")
			
			show()
			
			input = raw_input("fit (y/n)?")
			if input == "y":
				
				print "--------------------------------"
				print "...fitting"
				
				#Temperature Fit -------------------------------------------------------------------------------------
				#initialize list and dicts for fit
				Tfit = [0,0,0,0,0]
				Terror = [0,0,0,0,0]
				Tparams = Parameters()
				Tparams.add('amp', value=start_parameters[0], min=0.1, max=40.0)
				Tparams.add('freq', value=start_parameters[1], min=1e-5, max=0.1, vary=False)
				Tparams.add('phase', value=0.1, min=-pi, max=pi)
				Tparams.add('offs', value=start_parameters[3], min=0.0)
				Tparams.add('slope', value=start_parameters[4])

				#perform fit
				Tresults = minimize(sinfunc, Tparams, args=(tnew[start_index:limit], Tnew1[start_index:limit]), method="leastsq")
				
				#extract params dict to lists
				extract_fit_relerr_params(Tparams, Tfit, Terror)
				
				#data correction
				if Tfit[0]<0.0:
					Tfit[0]=abs(Tfit[0])
					Tfit[2]=Tfit[2]+pi
				if Tfit[2]<0.0:
					Tfit[2] = Tfit[2] + 2*pi
				if Tfit[2]>2*pi:
					Tfit[2] = Tfit[2] - 2*pi
					
				#Fit-Plot
				ax1.plot(tnew[start_index:limit], sinfunc(Tparams, tnew[start_index:limit]), 'b-')
				draw()
				
				#calculation of maximum T error
				Terror = [Tparams['amp'].stderr, Tparams['freq'].stderr, Tparams['phase'].stderr, Tparams['offs'].stderr, Tparams['slope'].stderr]
				T_error = abs(Tparams['amp'].stderr/Tparams['amp'].value)+abs(Tparams['phase'].stderr/Tparams['phase'].value)+abs(Tparams['freq'].stderr/Tparams['freq'].value)+abs(Tparams['offs'].stderr/Tparams['offs'].value)+abs(Tparams['slope'].stderr/Tparams['slope'].value)
				
				#console output
				print "-->T-Fit:\tA=%fK\n\t\tf=%fmHz\n\t\tO=%d-%dK\n\t\tb=%.2fK/h\n\t\tError:%f" % (Tfit[0], Tfit[1]*1000, start_parameters[2],max_Temp,Tfit[4]*3600, T_error)
				
				#file output
				log = open(date+"_"+samplename+"_HV_on+SineWave+LinRamp_I-T-Fits.txt", 'w+')
				log.write("#Temperature Fit Data\n----------\n\n")
				log.write("#Amp [K]\tAmp_Error\tFreq [Hz]\tFreq_Error\tPhase\t\tPhase_Error\tOffset [K]\tOffset_Error\tSlope [K/s]\tSlope_Error\n")
				log.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n\n"% (Tfit[0],Terror[0],Tfit[1],Terror[1],Tfit[2],Terror[2],Tfit[3],Terror[3],Tfit[4],Terror[4]))
				
				
				#Current Fit -----------------------------------------------------------------------------------------
				print "-->I-Fit"

				#initialize fit variables
				I_perioden = int(tnew[limit]/(perioden/start_parameters[1]))
				satzlaenge = limit/I_perioden
				
				Ifit = zeros((I_perioden-1,6))
				Ierror = zeros((I_perioden-1,5))
				Iparams = Parameters()
				Iparams.add('amp', value=1e-11)#, min=1e-13, max=1e-7)
				Iparams.add('freq', value=start_parameters[1], min=1e-5, max=0.1, vary=False)
				Iparams.add('phase', value=0.1, min=-pi, max=pi)
				Iparams.add('offs', value=1e-10)
				Iparams.add('slope', value=1e-10)
				
				#initialize file output
				log.write("#Current-Fit Data\n----------\n\n")
				log.write("#Amp [I]\t\tAmp_Error\t\tFreq [Hz]\t\tPhase\t\tPhase_Error\tOffset [A]\tOffset_Error\tSlope [A/s]\tSlope_Error\n")
				
				#perform partial fits
				for i in arange(1,I_perioden):
					start = start_index+int((i*satzlaenge)-satzlaenge)
					ende = start_index+int(i*satzlaenge)
					Iresults = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					ax2.plot(tnew[start:ende], sinfunc(Iparams, tnew[start:ende]), 'r-')
					
					#extrac fit parameters from dict to Ifit array
					Ifit[i-1,0] = Iparams['amp'].value
					Ifit[i-1,1] = Iparams['freq'].value
					Ifit[i-1,2] = Iparams['phase'].value
					Ifit[i-1,3] = Iparams['offs'].value
					Ifit[i-1,4] = Iparams['slope'].value
					Ierror[i-1,0] = Iparams['amp'].stderr
					Ierror[i-1,1] = Iparams['freq'].stderr
					Ierror[i-1,2] = Iparams['phase'].stderr
					Ierror[i-1,3] = Iparams['offs'].stderr
					Ierror[i-1,4] = Iparams['slope'].stderr
					
					#data correction
					if Ifit[i-1,0] < 0.0:
						Ifit[i-1,0] = -1*Ifit[i-1,0]
						Ifit[i-1,2] = Ifit[i-1,2]+pi
					if Ifit[i-1,2] < 0.0:
						Ifit[i-1,2] = Ifit[i-1,2]+2*pi
					if Ifit[i-1,2] > 2*pi:
						Ifit[i-1,2] = Ifit[i-1,2]-2*pi
					
					Ifit[i-1,5] = mean(Idata[start:ende,1])
					
					#calculate phase difference
					phasediff = Tfit[2]-Ifit[i-1,2]
					if phasediff < 0.0:
						phasediff = phasediff+2*pi
					elif phasediff > 2*pi:
						phasefiff = phasediff-2*pi
					
					#NonPyroStrom
					#m=magenta (TSC-Strom)
					Inp = abs(Ifit[i-1,0]*cos(phasediff))
					nonpyroparams = Parameters()
					nonpyroparams.add('amp', value=Inp)
					nonpyroparams.add('freq', value=Tfit[1])
					nonpyroparams.add('phase', value=Tfit[2])
					nonpyroparams.add('offs', value=Ifit[i-1,3])
					nonpyroparams.add('slope', value=Ifit[i-1,4])
					ax2.plot(tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende]), 'm-')
					
					#Pyrostrom
					#c=cyan (Pyrostrom)
					Ip = abs(Ifit[i-1,0]*sin(phasediff))
					pyroparams = Parameters()
					pyroparams.add('amp', value=Ip)
					pyroparams.add('freq', value=Tfit[1])
					if phasediff > pi:
						pyroparams.add('phase', value=(Tfit[2]+pi/2))
					else:
						pyroparams.add('phase', value=(Tfit[2]-pi/2))
					pyroparams.add('offs', value=Ifit[i-1,3])
					pyroparams.add('slope', value=Ifit[i-1,4])
					ax2.plot(tnew[start:ende], sinfunc(pyroparams, tnew[start:ende]), 'c-')
				
				#Legend for Current Plots
				I_plot = Rectangle((0, 0), 1, 1, fc="r")
				I_TSC = Rectangle((0,0), 1,1, fc="m")
				I_p = Rectangle((0,0), 1,1, fc ="c")
				ax2.legend([I_plot, I_TSC, I_p], ["data", "TSC", "pyro"], title="current",loc="lower right")
				draw()
				
				#Bereinigung von Ifit and Ierrors
				#Ifit=vstack((trim_zeros(Ifit[:,0]),trim_zeros(Ifit[:,1]),trim_zeros(Ifit[:,2]),trim_zeros(Ifit[:,3]),trim_zeros(Ifit[:,4]), trim_zeros(Ifit[:,5])))
				#Ifit = Ifit.transpose()
				#Ierror=vstack((trim_zeros(Ierror[:,0]),trim_zeros(Ierror[:,2]),trim_zeros(Ierror[:,3]),trim_zeros(Ierror[:,4])))	#attention! freq.error column gets lost!
				#Ierror = Ierror.transpose()
				
				#file output
				for i in range(1,len(Ifit)):
					log.write("%e\t%e\t%f\t%f\t%f\t%e\t%e\t%e\t%e\n"%(Ifit[i-1,0],Ierror[i-1,0],Ifit[i-1,1],(Ifit[i-1,2]*180/pi),(Ierror[i-1,1]*180/pi),Ifit[i-1,3],Ierror[i-1,2],Ifit[i-1,4],Ierror[i-1,3]))
				log.close()
				
				#Calculating p ---------------------------------------------------------------------------------------
				print "-->p-Calculation"
				
				#for length of p array
				globale_intervalle = len(Ifit)														
		
				#area for pyroel. coefficent
				input = raw_input("    Area [m2]?:")
				if input is "A":				#for PVDF (d=13,... mm)
					flaeche = 1.3994e-4
				elif input is "B":				#for cystalls (d=15mm)
					flaeche = 1.761e-4
				elif input is "C":				#STO Kristall M114
					flaeche = 1.4668e-5
				else:
					flaeche = float(input)
				
				area_error = 0.0082*flaeche
				
				p = zeros((globale_intervalle,6))
				perror = zeros((globale_intervalle,1))
				
				for i in range(1,globale_intervalle):
					phasediff = Tfit[2]-Ifit[i-1,2]
					if phasediff < 0.0:
						phasediff = phasediff+2*pi
					p[i-1,0] = (tnew[start_index+((i-1)*satzlaenge)]*Tfit[4])+(((tnew[start_index+((i-1)*satzlaenge)]-tnew[start_index+(i*satzlaenge)])/2)*Tfit[4])+Tfit[3]	#Spalte1 - Temp
					p[i-1,1] = ((Ifit[i-1,0]*sin(phasediff))/(flaeche*Tfit[0]*2*pi*Tfit[1]))							#Spalte2 - p (Sharp-Garn)
					p[i-1,2] = (abs(Ifit[i-1,5])/(flaeche*Tfit[4]))											#Spalte3 - p (Glass-Lang-Steckel)
					p[i-1,3] = phasediff * 180/pi													#Spalte4 - Phasediff.
					p[i-1,4] = abs((Ifit[i-1,0]*sin(phasediff))/(Ifit[i-1,0]*cos(phasediff)))							#Spalte5 - ratio Pyro/TSC
					p[i-1,5] = Ifit[i-1,5]
					
					perror[i-1,0] = p_error_i(Tfit, Terror, Ifit, Ierror, phasediff, flaeche, area_error, i)
				
				#Remove zeros from array
				p_new=vstack((trim_zeros(p[:,0]),trim_zeros(p[:,1]),trim_zeros(p[:,2]),trim_zeros(p[:,3]),trim_zeros(p[:,4]), trim_zeros(p[:,5])))
				p = p_new.transpose()
				perror = trim_zeros((perror))
				
				#Print des Pyro-koeff. ueber Temperatur
				bild2=figure(date+"_"+samplename+'_Pyro')
				Tticks = arange(270,430,10)
		
				ax3=subplot(311)
				ax3.set_autoscale_on(True)
				ax3.set_xlim(270,420)
				ax3.set_ylim(min(p[:,1])*1e6-50, max(p[:,1])*1e6+50)
				ax3.set_xlabel('Temp [K]',size='20')
				ax3.set_ylabel(r"pyroel. coefficient $\mathrm{\lbrack\frac{\mu C}{Km^{2}}\rbrack}$",color='b',size='20')
				xticks(Tticks)
				ax3.grid(b=None, which='major', axis='both', color='grey')
				#ax3.errorbar(p[3:-2,0],(p[3:-2,1]*1e6), yerr=perror[3:-2,0]*1e6, fmt='b.', elinewidth=None, capsize=3, label='Pyro-koeff-SG')
				ax3.errorbar(p[:,0],(p[:,1]*1e6), yerr=perror[:,0]*1e6, fmt='b.', elinewidth=None, capsize=3, label='Pyro-koeff-SG')
				ax3.plot(p[:,0],(p[:,1]*1e6), "b.", label='Pyro-koeff-SG')
				#ax3.plot(p[3:-2,0],(p[3:-2,2]*1e6), "r.", label='Pyro-koeff-GLS')
		
				ax5=subplot(312)
				ax5.set_autoscale_on(True)#
				ax5.set_xlim(270,420)
				xticks(Tticks)
				ax5.grid(b=None, which='major', axis='both', color='grey')
				ax5.set_xlabel('Temp [K]',size='20')
				ax5.set_ylabel(r'$\mathrm{\frac{I_{p}}{I_{TSC}}}$',color='g',size='20')
				#ax5.semilogy(p[3:-2,0], p[3:-2,4], "go", label="Pyro-Curr/Non-Pyro-Curr")
				ax5.semilogy(p[:,0], p[:,4], "go", label="Pyro-Curr/Non-Pyro-Curr")

				show()
				
				#Calculating p ---------------------------------------------------------------------------------------
				print "-->P-calculation"
				PS_plot = raw_input("    T_C? (y/n): ")
				if PS_plot == "y":
					print "    ... select T_C from the p(T) or I_TSC/I_p plot!"
					#---- Berechnen der Polarisation anhand von T_C (dort ist P = 0)
					#Finden von TC
					T_C = ginput()
					T_C = T_C[0][0]
					print "    T_C: %.2f" % T_C

					#getting index of T_C in p array
					for i in arange(0, len(p)-1):
						if p[i,0] > T_C:
							T_C_start_index = i
							break

					#Berechnen des Polarisationsverlaufs
					#Initialsieren des P-arrays
					P_len = T_C_start_index
					P = zeros((P_len,2))
					
					#P bei T_C auf 0 setzten (und gleich T_C zuordnen)
					Pindex = P_len-1
					P[Pindex,0] = p[T_C_start_index,0]
					P[Pindex,1] = 0.0

					#Aufsumiereung in Array
					for i in range(0,Pindex):
						if i < Pindex:
							P[Pindex-i-1,0] = p[Pindex-i,0]								#Temperatur zuweisen
							P[Pindex-i-1,1] = P[Pindex-i,1]+abs(p[Pindex+1-i,1]*(p[Pindex+1-i,0]-p[Pindex-i,0]))	#Polarisation immer vom Vorgaenger hinzuaddieren

					#Plot
					ax6=subplot(313)
					ax6.set_autoscale_on(True)
					ax6.set_xlim(270,420)
					xticks(Tticks)
					ax6.grid(b=None, which='major', axis='both', color='grey')
					ax6.set_xlabel('Temp [K]',size='20')
					ax6.set_ylabel(r'Polarization $\mathrm{\lbrack\frac{mC}{m^{2}}\rbrack}$',color='k',size='20')
					ax6.plot(P[:,0], P[:,1]*1e3, "ko", label="Polarization")
				
				#Saving results and figs------------------------------------------------------------------------------
				print "...saving figures"
				name = date+samplename
				bild1.savefig(date+"_"+samplename+"_HV_on+SineWave+LinRamp_I-T-Fits.png")
				bild2.savefig(date+"_"+samplename+"_HV_on+SineWave+LinRamp_p-P.png")
				
				print "...writing log files"
				log_name2 = date+"_"+samplename+"_HV_on+SineWave+LinRamp_p-Fits.txt"
				log = open(log_name2, "w+")
				log.write("#Berechnete Daten\n")
				log.write("#----------------\n")
				log.write("#Flaeche:\t%e m2\n" % flaeche)
				log.write("#----------------\n")
				log.write("#Temp\tPyro-Koeff(S-G)\t(Error)\tPyro-Koeff(L-S)\tPhasediff\tPolarization\n")
				log.write("#[K]\t[C/K*m2]\tC/K*m2]\t[C/m2]\n")
				try:
					for i in range(0,len(p)-1):
						if i>0 and i<len(P):
							log.write("%f\t%e\t%e\t%e\t%e\t%f\t%f\n" % (p[i,0],p[i,1],perror[i],p[i,4],p[i,2],p[i,3],P[i,1]))
						else:
							log.write("%f\t%e\t%e\t%e\t%e\t%f\n" % (p[i,0],p[i,1],perror[i],p[i,4],p[i,2],p[i,3]))
				except NameError:
					for i in range(0,len(p)-1):
						log.write("%f\t%e\t%e\t%e\t%e\t%f\n" % (p[i,0],p[i,1],perror[i],p[i,4],p[i,2],p[i,3]))
				log.close()
				
			else:
				pass
		else:
			pass

		print "--------------------------------"
		print "DONE!"

	#-----------------------------------------------------------------------------------------------------------------------------
	#for every other
	else:
		pass
ioff()