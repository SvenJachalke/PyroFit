# -*- coding: utf-8 -*-
#
# Universal Script for PyroData Evaluation
# (Use only for individual data records -- all files are contained in one single folder!!!)
# Start with Version Numbers on: 03.09.2014 to distinguish between different versions in filename
#---------------------------------------------------------------------------------------------------------------------------
# Author:	Sven Jachalke
# Mail:		sven.jachalke@phyik.tu-freiberg.de
# Adress:	Institut fuer Experimentelle Physik
#			Leipziger Strasse 23
#			09596 Freiberg
#---------------------------------------------------------------------------------------------------------------------------
# Necessary Python Packages:
# - scipy.interpolate
# - pylab (matplotlib, numpy, scipy), etc.
# - lmfit (https://github.com/lmfit/lmfit-py)
# - tubafcdpy(https://github.com/SvenJachalke/tubafcdpy)
#---------------------------------------------------------------------------------------------------------------------------
 
# Import modules------------------------------------------------------------------------------------------------------------
from pylab import *
from tubafcdpy import *
from matplotlib import __version__
import glob, sys, os, datetime
from scipy.interpolate import interp1d
#from scipy.signal import argrelmax, argrelmin
#from warnings import filterwarnings
from lmfit import minimize, Parameters, report_errors, fit_report
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

version = '0.9.0'
ion()

# Operator Information------------------------------------------------------------------------------------------------------
now = datetime.datetime.now()
operator = {
			'name':'Sven Jachalke',
			'mail':'sven.jachalke@physik.tu-freiberg.de',
			'company':'TU Bergakademie Freiberg',
			'tel':'+49 (0) 3731 / 39-3787',
			'date': now.strftime('%Y-%m-%d')
			}

# Areas for pyroKoeff-------------------------------------------------------------------------------------------------------
area_d5 = pi/4.0*(5.385/1000)**2							# D -- for small Edwards shadow mask (d=5.385mm)
area_d13 = pi/4.0*(12.68/1000)**2							# A -- for big Edwards shadow mask (d=12.68mm)
area_d15 = pi/4.0*(15.0/1000)**2							# B -- for single crystals with d=15mm
area_a5 = 1.4668e-5											# C -- for 5x5mm samples, e.g. SrTiO3, ...
#areas from older skript versions
area_d13_old = 1.3994e-4									# Aold -- for large Edwards shadow mask (d=13,...mm), e.g. for PVDF, ...
area_d15_old = 1.761e-4										# Bold -- for single crystals with d=15mm
#costums area and error (in m2)								# CUSTOM -- 
custom = 2.49e-5
custom_error = 1e-10

# User Settings-------------------------------------------------------------------------------------------------------------
start_index = 200											#start index for fit/plot (100 = 50s, because 2 indices = 1s)
fit_periods = 2												#how many periods have to fitted with sine wave in SinLinRamp
sigma = 3													#error level

upper_I_lim = 1e-3											#limitation of current in plot and fit (for spikes, ...)
temp_filter_flag = True										#True = no plot/fit of second temperature (top PT100)
current_filter_flag = True
calculate_data_from_fit_flag = False						#True = saving fit as data points to txt file for I_pyro and I_TSC
PS_flag = False												#flag if PS should be calculated from p(T)
BR_flag = False												#Flag for ByerRoundy Plot (False=not plotting)
single_crystal = False										#for single crystals phase=90deg ... thermal contact correction

interpolation_step = 0.5									#time grid for interpolation (in sec)
Ifit_counter_limit = 5										#repeat number when I-fit insufficient
warnings.filterwarnings("ignore")							#ignores warnings

# Alternatives for calculations --------------------------------------------------------------------------------------------
Formation = False											#If TRUE and OnPerm / SineWave Method will be evaluated as SinLinRamp by p(t) instead of p(T)
															#Used for SrTiO3 Formation (under electric field)
															
Resistance = False 										 	#If True and OnPerm / Calculation of R(T)
															#Maybe needs some testing

PartWiseTFit = True											#If TRUE the temperature of a SineWave + LinRamp/TrangleHat will be fitted part wise
															#as the current (same interval!) and not over the whole range
															#In order to keep the increasing error low, it is recommended to use more than 1 fit period!

# Plot Settings-------------------------------------------------------------------------------------------------------------
# Check Matplotlib Version--------------------------------------------------------------------------------------------------
if int(__version__[0]) == 2:
	style.use('classic')									#get old mpl style, if installed also my 'science' style can be used
color_style = 'TUBAF'										#TUBAF = CD colors, Standard = Matplotlib standard colors

rcParams['legend.fancybox'] = True				 			#plot fancy box (round edges)
print_signature = True										#print the operators signature at the bottom of the plot
enable_title = True											#enable/disable title in plot

label_size = '18'											#font size of x,y labels in plot in standard style
title_size = '15'											#font size of the figure title in standard style
fontsize_box = '12'											#font size in the text box in standard style

fig_size = (12.0,9.0)										#size of figures (general aspect ratio 4:3!!!)
set_dpi = 150							 					#dpi for exporting figures as png
skip_points = 0												#number of points to skip in plotting to speed up plotting and zooming (not interpol, fit)
transparency_flag = False									#exporting figures with transparent background?
facecolor_legends = 'white'
colorlist = ['m','g', 'c', 'r']
linestylelist = ['x','*','o ', 'x']
temp_linestyle=['o','']										#[makerstyle, linestyle] for temperature
curr_linestyle = ['o','']									#[makerstyle, linestyle] for current
volt_linestyle = ['*','']									#[makerstyle, linestyle] for voltage					
line = "----------------------------------"

export_format = 'png'										#figure output format (png,jpeg,pdf,eps)

# Functions-----------------------------------------------------------------------------------------------------------------
# file functions -----------------------------------------------------------------------------------------------------------
def extract_date(filename):
	"""
	Returns the date of the filename, wich is located at the first 16 characters of the filename
	input: filename [str]
	output: date [str]
	"""
	return filename[:16]
def extract_samplename(filename):
	"""
	Returns samplename, which is located after the second "_"
	input: filename [str]
	output: samplename [str]
	"""
	return filename.split("_")[2]
def extract_datatype(filename):
	"""
	Check's ending of recorded files and returns string with data type of the file
	input:  filename [str]
	output: data type [str]:
		Current, Voltage, Charge, Temperature, Powersupply, Vaccum, HighVoltage, GBIP-Errors
	"""
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
def extract_measurementmode(filename):
	"""
	Opens temperature file and extract the Waveform and HV-Status
	input: filename [str]
	output: hv_mode, waveform [str]
	"""
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
def extract_T_stimulation_params(filename):
	"""
	Return set parameters of the temperature stimulation, which are located in the header of the Temperature file
	input:	filename [str]
	output:	T_stimulation_params_dict [dict] (contains key:value pair for each finding)
	"""
	if filename.endswith("TEMP-t-Tpelt-Tsoll-Tsample.log"):
		datei = open(filename, 'r')
		T_stimulation_params_dict = {}
		
		try:
			hv_mode = datei.readline().strip().split(" ")[1]
			T_stimulation_params_dict.update({"hv_mode":hv_mode})
		except:
			pass
		try:
			waveform = datei.readline().strip().split(" ")[1]
			T_stimulation_params_dict.update({"waveform":waveform})
		except:
			pass
		try:
			if T_stimulation_params_dict["waveform"]=="PWRSquareWave":
				I = datei.readline().strip().split(" ")[1]
				T_stimulation_params_dict.update({"I":float(I)})
			else:
				amp = datei.readline().strip().split(" ")[1]
				T_stimulation_params_dict.update({"amp":float(amp)})
		except:
			pass
		try:
			freq = datei.readline().strip().split(" ")[1]
			T_stimulation_params_dict.update({"freq":float(freq)})
		except:
			pass
		try:
			if T_stimulation_params_dict["waveform"]=="PWRSquareWave":
				V = datei.readline().strip().split(" ")[1]
				T_stimulation_params_dict.update({"V":float(V)})
			else:
				offs = datei.readline().strip().split(" ")[1]
				T_stimulation_params_dict.update({"offs":float(offs)})
		except:
			pass
		try:
			heat_rate = datei.readline().strip().split(" ")[1]
			T_stimulation_params_dict.update({"heat_rate":float(heat_rate)})
		except:
			pass
		try:
			cool_rate = datei.readline().strip().split(" ")[1]
			T_stimulation_params_dict.update({"cool_rate":float(cool_rate)})
		except:
			pass
		try:
			T_Limit_H = datei.readline().strip().split(" ")[1]
			T_stimulation_params_dict.update({"T_Limit_H":float(T_Limit_H)})
		except:
			pass
		try:
			T_Limit_L = datei.readline().strip().split(" ")[1]
			T_stimulation_params_dict.update({"T_Limit_L":float(T_Limit_L)})
		except:
			pass
		datei.close()
		return T_stimulation_params_dict
def extract_HV_params(filename):
	"""
	Returns a list with set HV-settings
	input: filename [str]
	output: list of HV parameters [[float],...]
	"""
	if filename.endswith("HiV-t-HVsetVoltage-HVmeasVoltage.log"):
		datei = open(filename, 'r')
		HVmax = datei.readline().strip().split(" ")[4]
		zeile = datei.readline()
		if zeile!='' and zeile!='\r\n' and zeile !='\n':
			HVcomp = zeile.strip().split(" ")[5]
			datei.close()
			return [float(HVmax), float(HVcomp)]
		datei.close()
		return [float(HVmax)]
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
def interpolate_data(temp_array, curr_array, steps, temp_filter_flag):
	"""
	interpolates current and temperature data for plotting and fitting
	input: temperature array [ndarray]
			current array [ndarray]
			steps [float]
			temp_filter_flag [bool]
	output: interpolated arrays
	"""
	boundries = set_interpolation_range(curr_array[:,0],temp_array[:,0])	#find interpolation range
	tnew = arange(boundries[0], boundries[1], steps)								#arange new time axis in 0.5s steps

	#Temperature
	Tinterpol_down = interp1d(temp_array[:,0],temp_array[:,1])				#interpolation of lower temperature
	Tnew_down = Tinterpol_down(tnew)
	Tnew_top = zeros(len(Tnew_down))
	if temp_filter_flag == False:
		Tinterpol_top = interp1d(temp_array[::5,0],temp_array[::5,3])
		Tnew_top = Tinterpol_top(tnew[:-5])
		#array ist zu kurz fuer vstack!!!! Was tun? was tun wenn soweit!
	Tnew = vstack([Tnew_down,Tnew_top]).T
		
	#Interpolation current data																	#same for current
	Iinterpol = interp1d(curr_array[:,0],curr_array[:,1])
	Inew = Iinterpol(tnew)
	
	return tnew, Tnew, Inew
def fileprint_fit(log, fit, name):
	"""
	Writes fit values into file
	Input:	log [filehandle] - previously generated
			fit [dicts] - Params dict (lmfit)
			name [str] - what was fitted? (Temp, Curr, ...)
	Output: None
	"""

	log.write("#%s fit data\n#----------\n" % name)
	log.write(fit_report(fit))
	log.write("\n#----------\n")

	return None
def consoleprint_fit(fit, name):
	"""
	Writes fit value in shell window
	Input:	fit [dict] - Parameters-dict from lmfit
	name [str] - what was fitted= (Temp, Curr, ...)
	"""

	print("---------------")
	print("Fit: %s"%name)
	print("---------------")

	report_errors(fit)

	return None

# plot functions ---------------------------------------------------------------------------------------------------------------
def set_skip_points():
	if len(tnew) < 10000:
		return 	1
	elif len(tnew) >= 10000 and len(tnew) <= 100000:
		return 3
	else:
		return 6
def plot_graph(tnew, Tnew, Inew, T_profile):
	head = date+"_"+samplename+"_"+T_profile
	bild = figure(head, figsize=fig_size)
	ax1 = subplot(111)
	ax2 = ax1.twinx()
	if enable_title == True:
		title(samplename, size=title_size)

	#Plot Temperature
	ax1.set_xlabel("Time (s)",size=label_size)
	ax1.set_ylabel("Temperature (K)",color=temp_color,size=label_size)
	ax1.set_xlim(tnew[0],tnew[-1])
	ax1.grid(b=None, which='major', axis='both', color='grey', linewidth=1)
	ax1.tick_params(axis='y', colors=temp_color)
	
	if temp_filter_flag == True:
		ax1.plot(tnew[start_index::set_skip_points()], Tnew[start_index::set_skip_points(),0], color=temp_color,marker=temp_linestyle[0],linestyle=temp_linestyle[1], label="data")
	else:
		ax1.plot(tnew[start_index:-5:skip_points], Tnew[start_index::skip_points,1],color=temp_color,marker=temp_linestyle[0],linestyle=temp_linestyle[1], label="data (top)")
	
	ax1.autoscale(enable=True, axis='y', tight=None)
	legT = ax1.legend(title="Temperatures", loc='upper right')
	ax1.set_xlim(tnew[start_index])
	ax1.locator_params(nbins=10)

	#Plot Current
	ax2.set_ylabel("Current (A)",color=curr_color,size=label_size)
	ax2.plot(tnew[start_index::set_skip_points()], Inew[start_index::set_skip_points()], marker=curr_linestyle[0], linestyle=curr_linestyle[1] ,color=curr_color, label="data")
	ax2.legend(title="Currents", loc='lower right')
	ax2.set_xlim(tnew[start_index])
	ax2.locator_params(nbins=10,axis = 'y')
	ax2.set_ylim(min(Inew[start_index:]),max(Inew[start_index:]))
	ax2.tick_params(axis='y', colors=curr_color)
	#ax1.set_zorder(+1)
	#ax2.autoscale(enable=True, axis='y', tight=None)
	#ax2.add_artist(legT)

	bild.tight_layout()
	show()
	pause(0.1)
	
	if print_signature == True:
		bild.subplots_adjust(bottom=0.125)
		signature = operator['name']+' | '+operator['mail']+' | ' + operator['tel'] + ' | ' +operator['company'] + ' | ' + operator['date']
		figtext(0.15,0.02,signature)
	
	return bild, ax1, ax2
def plot_textbox(boxtext):
	"""
	Plots anchored text box with measurement informations into the graph.
	Input:	boxtext [str]
	Output:	box [instance]
	"""
	#box = figtext(x,y,boxtext,fontdict=None, bbox=properties)
	box = AnchoredText(boxtext,
                  prop=dict(size=fontsize_box), frameon=True,
                  loc=3,
                  )
	box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

	return box
def current_custom_legend(ax,loc=4):
	np_line = Line2D(range(10), range(10), linestyle='-', marker='', color = np_color)
	p_line = Line2D(range(10), range(10), linestyle='-', marker='', color = p_color)
	Imeas_line = Line2D(range(10), range(10), linestyle='', marker='o', color = curr_color)
	Ifit_line = Line2D(range(10), range(10), linestyle='-', marker='', color = curr_color)
	
	legend = ax.legend((Imeas_line,Ifit_line,np_line,p_line), ('data','fit','non-pyro','pyro'),loc=loc,title="currents")
	return legend 
def temperature_custom_legend(ax,loc=1):
	Tmeas_line = Line2D(range(10),range(10), linestyle='', marker='o', color = temp_color)
	Tfit_line = Line2D(range(10),range(10), linestyle='-', marker='', color = temp_color)

	legend = ax.legend((Tmeas_line,Tfit_line),('data','fit'),loc=loc,title='temperatures')
	return legend
def saving_figure(bild, pbild=False):
	"""
	saves figure with individual filename composed of date, filename, T_profile and print on console
	input:	bild - figure instance
			pbild - bool, when True pyroelectric coefficient figure in SinLimRamp will be plotted
	return: None
	"""
	print("saving ...")
	if pbild == False:
		image_name = date+"_"+samplename+"_"+T_profile+"_T-I"
		print("...Temperature/Current Plot\n%s.%s" % (image_name,export_format))
		
	elif pbild == "Polarize":
		image_name = date+'_'+samplename+'_Polarize'
		print("...Temperature/Polarization Plot\n%s.%s" % (image_name,export_format))
	elif pbild == 'Resistance':
		image_name = date+'_'+samplename+'_Resistance'
		print("...Temperature/Resistance Plot\n%s.%s" % (image_name,export_format))
	else:
		image_name = date+"_"+samplename+"_"+T_profile+"_p"
		print("...Pyro Plot\n%s.%s" %(image_name,export_format))
	
	if export_format == 'png':
			bild.savefig(image_name+'.png', dpi=set_dpi, transparent=transparency_flag)
	elif export_format == 'pdf':
			bild.savefig(image_name+'.pdf')
	return None

# fit functions ---------------------------------------------------------------------------------------------------------------
def sinfunc(params, x, data=None):
	"""
	Model for sine function using the lmfit model
	input: Parameters Dict (lmfit)
	output: sine wave model
	"""
	amp = params['amp'].value
	freq = params['freq'].value
	phase = params['phase'].value
	offs = params['offs'].value
	slope = params['slope'].value

	model = amp*sin(2*pi*freq*x+phase)+offs+slope*x
	
	if data==None:
		return model
	return model-data
def expdecay(params, x, data=None):
	"""
	Model for exponetial decay function using the lmfit model
	input: Parameters Dict (lmfit)
	output: decay model
	"""
	model = params['factor'].value * exp(-x/ params['decay'].value) + params['offs'].value

	if data==None:
		return model
	return model-data
def linear(params, x, data=None):
	"""
	Model for linear function using lmfit module
	input: Parameter dict (lmfit)
	output: model
	"""
	a = params['a'].value
	b = params['b'].value
	model = a*x + b
	if data==None:
		return model
	return model-data

# misc functions --------------------------------------------------------------------------------------------------------------
def extract_fit_relerr_params(params):
	"""
	Extract the fitted parameters from the Paramters Dict and put it into lists (values and errors)
	input:  params [Params Dict]
	return: fit [list], err[list]
	"""
	fit = [params['amp'].value,params['freq'].value,params['phase'].value,params['offs'].value,params['slope'].value]
	err = [abs(params['amp'].stderr),abs(params['freq'].stderr),abs(params['phase'].stderr),abs(params['offs'].stderr),abs(params['slope'].stderr)]
	return fit, err
def listtoparam(liste, parameterdic):
	"""
	Adds a list of temperature parameters to the Parameters Dict of lmfit
	input:  liste [list]
	paramterdic [Parameters Dict]
	"""
	parameterdic.add('amp', value=liste[0])
	parameterdic.add('freq', value=liste[1])
	parameterdic.add('phase', value=liste[2])
	parameterdic.add('offs', value=liste[3])
	parameterdic.add('slope', value=liste[4])

	return None
def fit(x, y, start, end, slice, start_parameters, vary_freq=True, heating=True):
	"""
	Peforms fit for y(x) with start and end values (indices) and returns fit dictionary
	Input:	t [ndarray]
			T [ndarray]
			start [int]
			end [int]
			start_paramters [dict]
			vary_freq [bool]
			heating [bool], - heating rate (True) or cool rate (False) as start parameter
	Return:	results [minimize instance]
			Params [lmfit dict]
	"""

	#initialize list and dicts for fit
	Params = Parameters()
	Params.add('amp', value=start_parameters['amp'],min=0.0)
	Params.add('freq', value=start_parameters['freq'], min=1e-5, max=0.2, vary=vary_freq)
	Params.add('phase', value=0.1)
	Params.add('offs', value=start_parameters['offs'], min=0.0)
	if heating==True:
	  Params.add('slope', value=start_parameters['heat_rate'])
	else:
	  Params.add('slope', value=-start_parameters['cool_rate'])
	  
	#perform fit
	result = minimize(sinfunc, Params, args=(x[start:end:slice], y[start:end:slice]), method="leastsq")

	return result
def rel_err(Tfit, Terror, Ifit, Ierror, area, area_error, phasediff, Xsigma=1):
	"""
	Calculates relative maximum error 
	"""
	
	phasediff_error = Terror[2]+Ierror[2]
	# rel err = dp/p .... I_Amp	phi	A	T_Amp	f	
	rel_err = Xsigma*(abs(Ierror[0]/Ifit[0]) + abs(phasediff_error/(tan(phasediff))) + abs(area_error/area) + abs(Terror[0]/Tfit[0]) + abs(Terror[1]/Tfit[1]))
		
	return rel_err
def get_area():
	"""
		function to get the active area of several pyroelectric materials, depending which mask was used
		input: None
		output: area [float]
		"""
	input = raw_input("Area [m2]:")
	if input == "A":								#d13
		return area_d13, 0.0082*area_d13
	elif input == "Aold":
		return area_d13_old, 0.0082*area_d13_old
	elif input == "B":								#d15
		return area_d15, 0.0082*area_d15
	elif input == "Bold":
		return area_d15_old, 0.0082*area_d15_old
	elif input == "C":								#a5
		return area_a5, 0.0082*area_a5
	elif input == "D":								#d5
		return area_d5, 0.0082*area_d5
	elif input == "PMNPT":
		return 1.65e-4, 1.35292e-6			#area of PMN-PT samples with SputterShadowMaskContacts
	elif input == "CUSTOM":						#custom defined values
		return custom, custom_error
	else:
		return float(input), 0.0082*float(input)	#direct area input
def amp_phase_correction(fit_dict):
	"""
	Correction if neg. amplitudes and phases >< 2 pi are fitted
	Input:	Parameters dict
	Output:	Parameters dict
	"""
	
	if fit_dict['amp'].value < 0.0:
		fit_dict['amp'].value = abs(fit_dict['amp'].value)
		fit_dict['phase'].value = fit_dict['phase'].value + pi		#add value of pi if amplitude negativ 
	
	else:
		pass
	
	fit_dict['phase'].value = phase_correction(fit_dict['phase'].value)
	
	return fit_dict
def phase_correction(phase):
	"""
	Brings phase in the range of 0 to 2*pi
	"""
	
	if phase > 2*pi:
		while phase > +2*pi:
			phase = phase - 2*pi
	elif phase < 2*pi or phase<0.0:
		while phase < 0.0:
			phase = phase + 2*pi
	else:
		pass
		
	return phase

#Main Program------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

print(line)
print("PyroFit - UnivseralScript - V%s" % version)
print(line)

#Init Plot Colors-----------------------------------------------------------------------------------------------------------------
if color_style == 'TUBAF':
	other = 'k'
	temp_color = TUBAFblue()
	curr_color = TUBAFred()
	p_color = TUBAForange()
	np_color = TUBAFcyan()
	volt_color = TUBAFgreen()
elif color_style == 'Standard':
	other = 'k'
	temp_color = 'b'
	curr_color = 'r'
	p_color = 'c'
	np_color = 'm'
	volt_color = 'g'

# File Reading-----------------------------------------------------------------------------------------------------------------
filelist = glob.glob('*.log')
filecounter = 0

#check folder for files and read files!
for filename in filelist:

	date=extract_date(filename)
	datatype=extract_datatype(filename)

	if datatype=="Temperature":
		HV_status, T_profile = extract_measurementmode(filename)
		measurement_info = extract_T_stimulation_params(filename)
		samplename = extract_samplename(filename)
		
		Tdata = loadtxt(filename, skiprows=9)
		
		#previous filter of Tdata
		erase_bools_T = (Tdata[:,1]!=9.9e39)	#overflow on down temperature
		Tdata = Tdata[erase_bools_T]

		filecounter = filecounter + 1
		sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
		sys.stdout.flush()

	elif datatype=="Current":
		Idata = loadtxt(filename)

		#previous filtering of data
		erase_bools_I = (Idata[:,1]!=9.9e39)		#overflow on Keithley amperemeter
		Idata = Idata[erase_bools_I]
		erase_bools_I = (Idata[:,1]!=9.9e37)
		Idata = Idata[erase_bools_I]
		erase_bools_I = (Idata[:,1]!=0.015)			#overflow in measurement program
		Idata = Idata[erase_bools_I]

		filecounter = filecounter + 1
		sys.stdout.write("\rReading: %d/%d completed" % (filecounter,len(filelist)))
		if current_filter_flag == True:
			erase_bools_I = (abs(Idata[:,1])< upper_I_lim)	#user defined low pass filter with upper_I_lim variable
			Idata = Idata[erase_bools_I]
			sys.stdout.write("\rData filter applied")
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
print "\n"+line

#----------------------------------------------------------------------------------------------------------------------------
if filelist == []:
	print "No measurement data files in Folder!"
else:
	#Routines for every measurement_type-------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------------------------------------

	#normal measurement routines without HV (SinWave, LinRamp, ...)
	if measurement_info['hv_mode'] == "Off":
		#Thermostat Method
		#--------------------------------------------------------------------------------------------------------------------
		if measurement_info['waveform'] == "Thermostat":
			print("Mode:\t\t%s"%measurement_info['waveform'])
			print("Temperature:\t%.1fK" % measurement_info['T_Limit_H'])

			#Interpolation and plotting of data ----
			print(line)
			print("...plotting")
			print(line)
			# pre-fit plot
			tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
			bild, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)

			#text box
			box_text = "Temperature: "+str(measurement_info['T_Limit_H']) + "K"
			box = plot_textbox(box_text)
			ax2.add_artist(box)
			show()

			#saving figure
			saving_figure(bild)

		#---------------------------------------------------------------------------------------------------------------------
		#LinearRamp Method
		elif measurement_info['waveform'] == "LinearRamp":
			print("Mode:\t\t%s"%measurement_info['waveform'])
			print("Temperature:\t%.1fK - %.1fK\nSlope:\t%.1fK/h" % (measurement_info['offs'],max(Tdata[:,1]),measurement_info['heat_rate']*3600))

			#Interpolation and plotting of data ----
			print(line)
			print("...plotting")
			print(line)
			# pre-fit plot
			tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
			bild1, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)

			#text box
			box_text = "Temperature: "+str(measurement_info['offs']) +' - ' + str(round(max(Tdata[:,1]),2)) + "K \nSlope: " + str(measurement_info['heat_rate']*3600) + " K/h"
			box = plot_textbox(box_text)
			ax2.add_artist(box)
			show()
			
			# Perform Byer-Roundy Fit and calc p
			#---------------------------------------------------------------------------------------------------------------
			input = raw_input("fit? [y/n]")
			if input == "y":
				
				area, area_error = get_area()
				print(line)
				print("... fitting")
			
				#Byer Roundy evaluation
				head = date+"_"+samplename+"_"+T_profile+'p(T)'
				bild2 = figure(head, figsize=fig_size)
				axp = bild2.add_subplot(111)
				
				#initialize list and dicts for fit
				start = start_index
				end = len(Tnew[:,0])-1
				
				Params = Parameters()
				Params.add('offs', value=measurement_info['offs'], min=273.0)
				Params.add('slope', value=measurement_info['heat_rate'])
				Tresults = minimize(linear, Params, args=(tnew[start:end], Tnew[start:end,0]), method="leastsq")
				ax1.plot(tnew[start:],linear(Params,tnew[start:]),color=temp_color,linestyle='-',label="T-Fit (down)")
				
				pyro_koeff = abs(Inew) / (area * Params['slope'].value)	
				
				axp.plot(Tnew,pyro_koeff*1e6,color=temp_color,marker=".",linestyle="", label='p (BR)')
				axp.set_xlabel('Temperature (K)',size=label_size)
				axp.set_ylabel(u"p (µC/Km²)",color=temp_color,size=label_size)
				axp.grid(b=None, which='major', axis='both', color='grey')
				axp.set_xlim(273,max(Tnew[:,0]))

				bild2.tight_layout()
				
			saving_figure(bild1)
			saving_figure(bild2,pbild=True)
			
		#---------------------------------------------------------------------------------------------------------------------
		#SineWave Method
		elif measurement_info['waveform'] == "SineWave":
			print("Mode:\t\t%s"%measurement_info['waveform'])
			print "Stimulation:\tA=%.1fK\n\t\tf=%.1fmHz\n\t\tO=%.1fK" % (measurement_info['amp'], measurement_info['freq']*1000, measurement_info['offs'])

			#Interpolation and plotting of data ----
			print(line)
			print("...plotting")
			print(line)
			# pre-fit plot
			tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
			bild1, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)

			#---------------------------------------------------------------------------------------------------------------
			input = raw_input("fit? [y/n]")
			if input == "y":
				
				area, area_error = get_area()
				print line
				print "... fitting"
				
				#Fit temperature----------------------------------------------------------------------------------------
				Tresult_down = fit(tnew, Tnew[:,0], start_index, len(Tnew[:,0])-1,1,measurement_info, True, True)
				#correction of phase and amplitudes
				Tparams_down = amp_phase_correction(Tresult_down.params)
				#extract params dict to lists
				Tfit_down, Terror_down = extract_fit_relerr_params(Tparams_down)
				#Plot
				ax1.plot(tnew[start_index:], sinfunc(Tparams_down, tnew[start_index:]), color=temp_color, linestyle='-', label="T-Fit (down)")
				draw()
				#absolute T_high Error
				total_Terror_down = abs(Tparams_down['amp'].stderr/Tparams_down['amp'].value)+abs(Tparams_down['phase'].stderr/Tparams_down['phase'].value)+abs(Tparams_down['freq'].stderr/Tparams_down['freq'].value)+abs(Tparams_down['offs'].stderr/Tparams_down['offs'].value)+abs(Tparams_down['slope'].stderr/Tparams_down['slope'].value)

				#for top temperature
				if temp_filter_flag == False:

					Tresult_high = fit(tnew, Tnew[:,1], start_index, len(Tnew_top)-1,5,measurement_info, True, True)
					#correction of phase and amplitude
					Tparams_high = amp_phase_correction(Tresult_high.params)
					#extract params dict to lists
					Tfit_high, Terror_high = extract_fit_relerr_params(Tparams_high)
					#plot of second fit
					ax1.plot(tnew[start_index:-5], sinfunc(Tparams_high, tnew[start_index:-5]), color=volt_color,linestyle='-', label='T-Fit (top)')
					draw()
					#absolute T_high Error
					total_Terror_high = abs(Tparams_high['amp'].stderr/Tparams_high['amp'].value)+abs(Tparams_high['phase'].stderr/Tparams_high['phase'].value)+abs(Tparams_high['freq'].stderr/Tparams_high['freq'].value)+abs(Tparams_high['offs'].stderr/Tparams_high['offs'].value)+abs(Tparams_high['slope'].stderr/Tparams_high['slope'].value)

				#Fit current ---------------------------------------------------------------------------------------------
				#initialize parameters dict for current fit
				Iparams = Parameters()
				Iparams.add('amp', value=1e-11)
				Iparams.add('freq', value=Tfit_down[1], min=1e-5, max=0.2, vary=False)
				Iparams.add('phase', value=1.0)
				Iparams.add('offs', value=1e-10)
				Iparams.add('slope', value=1e-10)

				#current fit
				Iresult = minimize(sinfunc, Iparams, args=(tnew[start_index:],Inew[start_index:]), method="leastsq")
				#fit correction (amp/phase)
				Iparams = amp_phase_correction(Iresult.params)
				#extract params dict
				Ifit, Ierror = extract_fit_relerr_params(Iparams)
				#plot current fit
				ax2.plot(tnew[start_index:], sinfunc(Iparams, tnew[start_index:]), color=curr_color,linestyle='-', label='I-Fit')
				draw()

				#calculate pyroelectric coefficient------------------------------------------------------------------------
				if single_crystal == False:
					#calculate phase difference
					phi_T = Tfit_down[2]
					phi_I = Ifit[2]
					# if abs(phi_I) > abs(phi_T):
						# phasediff = phase_correction(phi_I-phi_T)
					# else:
						# phasediff = phase_correction(phi_T-phi_I)
					phasediff = phase_correction(phi_T-phi_I)
				else:
					phasediff = -pi/2
				
				# avaraging singnal part to get algebraic sign (oscillation around pos/neg value?)
				meanI = mean(Inew[start_index:])
				if meanI < 0.0:
					polarityI = "neg"
				else:
					polarityI = "pos"

				pyro_koeff = (Ifit[0]*-sin(phasediff))/(area*Tfit_down[0]*2*pi*abs(Tfit_down[1]))		
				perror = pyro_koeff*rel_err(Tfit_down,Terror_down,Ifit,Ierror,area, area_error,phasediff,Xsigma=sigma)

				#Plot Ip and ITSC------------------------------------------------------------------------------------------
				#NonPyroStrom
				#m=magenta (TSC)
				nonpyroparams = Parameters()
				Inp = abs(Ifit[0]*-cos(phasediff))
				# if polarityI == "neg":
					# nonpyroparams.add('amp', value=-1*Inp)
				# else:
				nonpyroparams.add('amp', value=Inp)
				nonpyroparams.add('freq', value=Tfit_down[1])
				nonpyroparams.add('phase', value=Tfit_down[2])
				nonpyroparams.add('offs', value=Ifit[3])
				nonpyroparams.add('slope', value=Ifit[4])
				nonpyroparams = amp_phase_correction(nonpyroparams)
				ax2.plot(tnew[start_index:], sinfunc(nonpyroparams, tnew[start_index:]), color=np_color,linestyle='-',label='non-pyro')
				
				#Calculating Data from Fit - Pyro
				if calculate_data_from_fit_flag == True:
						I_TSC = (array([tnew[start_index:], sinfunc(nonpyroparams, tnew[start_index:])])).T		#transpose!

				#Pyrostrom
				#c=cyan (Pyro)
				pyroparams = Parameters()
				Ip = Ifit[0]*-sin(phasediff)
				pyroparams.add('amp', value=Ip)
				pyroparams.add('freq', value=Tfit_down[1])
				pyroparams.add('phase', value=(Tfit_down[2]+pi/2))
				pyroparams.add('offs', value=Ifit[3])
				pyroparams.add('slope', value=Ifit[4])
				pyroparams = amp_phase_correction(pyroparams)
				ax2.plot(tnew[start_index:], sinfunc(pyroparams, tnew[start_index:]), color=p_color,linestyle='-',label='pyro')
				
				#Calculating Data from Fit - Pyro
				if calculate_data_from_fit_flag == True:
					I_pyro = (array([tnew[start_index:], sinfunc(pyroparams, tnew[start_index:])])).T		#transpose!

				#legend and information box
				box_text = r"$A$:"+"\t     "+format(area,'.3e')+r" $\mathrm{m^2}$"+"\n"+ r"$I_{\mathrm{Amp}}$:"+"\t"+format(Ifit[0],'.3e')+r" A"+"\n"+ r"$T_{\mathrm{Amp}}$:"+"\t"+format(Tfit_down[0],'.3f')+r" K"+"\n"+r"$f$:"+"\t     "+format(Tfit_down[1]*1000,'.3f')+" mHz"+"\n"+r"$\phi$:"+"\t\t"+format(degrees(phasediff),'.3f')+"$^{\circ}$"+"\n"+r"$p$:"+"\t     "+format(pyro_koeff*1e6,'.3f')+r" $\mathrm{\mu C/Km^2}$"
				box = plot_textbox(box_text)
				leg1 = ax1.legend(title="temperatures",loc='upper right')
				ax2.legend(title="currents",loc='lower right')
				# ax2.add_artist(leg1)	#bring legend to foreground
				ax2.add_artist(box)
				draw()

				#console output --------------------------------------------------------------------------------------------
				print 'Area:\t\t', area, 'm2'
				print 'I_pyro:\t\t', fabs(Ip), 'A'
				print 'I_TSC:\t\t', fabs(Inp), 'A'
				print 'phase T-I:\t',(degrees(phasediff))
				print 'p:\t\t', pyro_koeff*1e6,'yC/Km2'
				print '\t\t(+-', perror*1e6,'yC/Km2)'
				print 'B_T:\t\t%f nA/K' % (fabs(Inp/Tfit_down[1])*1e9)
				input = raw_input("Show fits? [y/n]")
				if input == "y":
					consoleprint_fit(Tparams_down, "Temperature (Down)")
					if temp_filter_flag == False:
						consoleprint_fit(Tparams_high, "Temperature (High)")
					consoleprint_fit(Iparams,"Current")
				else:
					pass

				#file output -----------------------------------------------------------------------------------------------
				log = open(date+"_"+samplename+"_"+T_profile+"_T-Fit.txt", 'w+')
				log.write("#Results\n#----------\n")
				fileprint_fit(log,Tparams_down,"Temperature (Down)")
				if temp_filter_flag == False:
					fileprint_fit(log, Tparams_high, "Temperature (High)")
				log.close()
				log = open(date+"_"+samplename+"_"+T_profile+"_I-Fit.txt", 'w+')
				fileprint_fit(log, Iparams, "I-Fit")
				log.close()
				header_string = "#area [m2]\t\t\tI-p [A]\t\t\tI-TSC [A]\t\t\tphasediff [deg]\t\t\tpyroCoeff [yC/Km2]\t\t\tp_error [µC/Km2]\t\t\tB_T [A/K]"
				savetxt(date+"_"+samplename+"_"+T_profile+"_"+"PyroData.txt", [area,Ip,Inp,degrees(phasediff),pyro_koeff,perror, fabs(Inp/Tfit_down[1])], delimiter="\t", header=header_string)
				if calculate_data_from_fit_flag == True:
					header_string = "time [s]\t\t\tI_TSC [A]\t\t\tI_pyro [A]"
					savetxt(date+"_"+samplename+"_"+T_profile+"_"+"DataFromFit.txt", vstack([I_TSC[:,0], I_TSC[:,1], I_pyro[:,1]]).T, delimiter="\t", header=header_string)


			#saving figure----------------------------------------------------------------------------------------------------
			saving_figure(bild1)

		#---------------------------------------------------------------------------------------------------------------------
		#SineWave+LinearRamp Method
		elif measurement_info['waveform'] == "SineWave+LinRamp" or measurement_info['waveform'] == "SineWave+LinearRamp":
			print("Mode:\t\t%s"%measurement_info['waveform'])
			print("Stimulation:\tA=%.1fK\n\t\tf=%.1fmHz\n\t\tO=%.1f-%.1fK\n\t\tb=%.2fK/h" % (measurement_info['amp'], measurement_info['freq']*1000, measurement_info['offs'],measurement_info['T_Limit_H'], measurement_info['heat_rate']*3600))

			#Interpolation and plotting of data ----
			print(line)
			print("...plotting")
			print(line)
			# pre-fit plot
			tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
			bild1, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)

			show()
			
			input = raw_input("fit [y/n]?")
			if input == "y":

				#area for pyroel. coefficent
				area, area_error = get_area()
				print("Area: %e m2" % area)
				print("\n"+line)
				
				usecoolrate_flag = False
				if measurement_info['cool_rate'] != 0:
					choise_rate = raw_input('use "cool" or "heat" rate?:')
					if choise_rate == 'cool':
						usecoolrate_flag = True

				print("...fitting")
				if PartWiseTFit == True:
					print("Partwise Temp.Fit enabled!")
								
				#important calculations for further fit;)
				#check when ramp run into T_Limit_H
				if max(Tnew[:,0]) < measurement_info['T_Limit_H']:
					maxT_ind = Tnew[:,0]>max(Tnew[:,0])-1
				else:
					maxT_ind = Tnew[:,0]>(measurement_info['T_Limit_H']-1)
					
				if usecoolrate_flag == True:
					limit = len(Tnew[:,0])-1
					max_Temp = (tnew[limit]-tnew[0])*-measurement_info['cool_rate']+Tnew[0,0]
					T_perioden = int((tnew[limit]-tnew[0])/(fit_periods/measurement_info['freq']))
					tmax = tnew[limit]-tnew[0]
					satzlaenge = limit/T_perioden
				else:
					number_of_lim = maxT_ind.tolist().count(True)
					limit = len(Tnew[:,0])-number_of_lim-1
					max_Temp = (tnew[limit]+tnew[start_index])*measurement_info['heat_rate']+measurement_info['offs']
					T_perioden = int((tnew[limit]-tnew[start_index])/(fit_periods/measurement_info['freq']))
					tmax = tnew[limit]
				
				satzlaenge = (limit-start_index)/T_perioden
				#print(satzlaenge)
				
				print(line)
				print("temperature fit ...")
				
				# in case of part wise fitting of T -----------------------------------------------------------------------------------------				
				if PartWiseTFit == True:
					
					#initialize fit variables for temperature 
					Tparams_down = Parameters()
					Tparams_down.add('amp', value=measurement_info['amp'],min=0.0)
					Tparams_down.add('freq', value=measurement_info['freq'], min=1e-5, max=0.2)
					Tparams_down.add('phase', value=0.1, vary=True)
					Tparams_down.add('offs', value=measurement_info['offs'], min=0.0)
					Tparams_down.add('slope', value=measurement_info['heat_rate'])
		
					#perform partial fit
					for i in arange(1,T_perioden):
						start = start_index+int((i*satzlaenge)-satzlaenge)
						ende = start_index+int(i*satzlaenge)
					
						#fit of sin
						Tresult_sin = minimize(sinfunc, Tparams_down, args=(tnew[start:ende], Tnew[start:ende,0]), method="leastsq")
					
						#console status
						sys.stdout.write("\rProgress: %d/%d; %.0f %%" % (i,T_perioden-1,100*float(i)/float(T_perioden-1)))
						sys.stdout.flush()
						
						#fit correction (amp/phase)
						Tparams_down = amp_phase_correction(Tresult_sin.params)	
						
						#plot
						ax1.plot(tnew[start:ende], sinfunc(Tparams_down, tnew[start:ende]), color=temp_color, linestyle='-')
						
						#extract params dict to lists
						Tfit_down_temp, Terror_down_temp = extract_fit_relerr_params(Tparams_down)
							
						if i==1:
							Tfit_down = array([Tfit_down_temp])
							Terror_down = array([Terror_down_temp])
							Tresults_down = [Tresult_sin]					#save lmfit minizimer objects for future purpose ... maybe
						else:
							Tfit_down = append(Tfit_down,[array(Tfit_down_temp)],axis=0)
							Terror_down = append(Terror_down,[array(Terror_down_temp)],axis=0)
							Tresults_down.append(Tresult_sin)
														
						Tparams_down['phase'].value=Tfit_down[i-1,2]
						#Tparams_down['phase'].vary=False
					
					header_string = "Amp [I]\t\t\tFreq [Hz]\t\t\tPhase [rad]\t\t\tOffset [A]\t\t\tSlope [A/s]\t\t\tAmp_Err [A]\t\t\tFreq_Err [Hz]\t\t\tPhase_Err [rad]\t\t\tOffs_Err [A]\t\t\tSlope_Err [A/s]"
					savetxt(date+"_"+samplename+"_"+T_profile+"_T-Fit-partwise.txt",hstack([Tfit_down,Terror_down]), delimiter="\t", header=header_string)
				
				# in case of whole range fit of T -----------------------------------------------------------------------------------------				
				else:
					#prepare output log
					log = open(date+"_"+samplename+"_"+T_profile+"_T-Fit.txt", 'w+')
					
					#Temperature Fit -------------------------------------------------------------------------------------
					Tresult_down = fit(tnew, Tnew[:,0],start_index,limit,1,measurement_info, True, True)
					#correction of phase and amplitudes
					Tparams_down = amp_phase_correction(Tresult_down.params)
					#extract params dict to lists
					Tfit_down, Terror_down = extract_fit_relerr_params(Tparams_down)
					#Fit-Plot
					ax1.plot(tnew[start_index:limit], sinfunc(Tparams_down, tnew[start_index:limit]), color=temp_color,linestyle='-', label='T-Fit')
					draw()
					#absolute T_high Error
					total_Terror_down = abs(Tresult_down.params['amp'].stderr/Tresult_down.params['amp'].value)+abs(Tresult_down.params['phase'].stderr/Tresult_down.params['phase'].value)+abs(Tresult_down.params['freq'].stderr/Tresult_down.params['freq'].value)+abs(Tresult_down.params['offs'].stderr/Tresult_down.params['offs'].value)+abs(Tresult_down.params['slope'].stderr/Tresult_down.params['slope'].value)
					#file output
					fileprint_fit(log,Tparams_down,"Temperature (Down)")
					
					#for top temperature-------------------
					if temp_filter_flag == False:
						Tresult_high = fit(tnew[:-5], Tnew[:,1], start_index, limit,1, measurement_info, True, True)
						#correction of phase and amplitude
						Tparams_high = amp_phase_correction(Tresult_high.params)
						#extract params dict to lists
						Tfit_high, Terror_high = extract_fit_relerr_params(Tparams_high)
						#plot of second fit
						ax1.plot(tnew[start_index:-5], sinfunc(Tparams_high, tnew[start_index:-5]), color=volt_color,linestyle='-', label='T-Fit (top)')
						draw()
						#absolute T_high Error
						total_Terror_high = abs(Tparams_high['amp'].stderr/Tparams_high['amp'].value)+abs(Tparams_high['phase'].stderr/Tparams_high['phase'].value)+abs(Tparams_high['freq'].stderr/Tparams_high['freq'].value)+abs(Tparams_high['offs'].stderr/Tparams_high['offs'].value)+abs(Tparams_high['slope'].stderr/Tparams_high['slope'].value)
						#file output
						fileprint_fit(log,Tparams_high,"Temperature (High)")
					
					log.close()
				temperature_custom_legend(ax1)
				draw()

				print('\nTemperature ... done')
				print(line)
				#Current Fit -----------------------------------------------------------------------------------------
				print("current fit ...")
				
				#initialize fit variables
				I_perioden = T_perioden
					
				Iparams = Parameters()
				Iparams.add('amp', value=1e-11)
				if PartWiseTFit == True:
					Iparams.add('freq', value=Tfit_down[0,1], min=1e-5, max=0.2,vary=False)
				else:
					Iparams.add('freq', value=Tfit_down[1], min=1e-5, max=0.2, vary=False)
				Iparams.add('phase', value=1.0)
				Iparams.add('offs', value=1e-10)
				Iparams.add('slope', value=1e-10)
					
				Iparams_lin = Parameters()
				Iparams_lin.add('a', value=1e-10)
				Iparams_lin.add('b', value=0.0)

				#perform partial fits
				for i in arange(1,I_perioden):
					# In case of partwise fit: give the Current Frequency the exact value from the temperature fit ... otherwise the phase will begin to oscillate!!!
					if PartWiseTFit == True:
						Iparams['freq'].value = Tfit_down[i-1,1]
					
					start = start_index+int((i*satzlaenge)-satzlaenge)
					ende = start_index+int(i*satzlaenge)
					
					# avaraging singnal part to get algebraic sign (oscillation around pos/neg value?)
					meanI = mean(Inew[start:ende])
					if meanI < 0.0:
						polarityI = "neg"
					else:
						polarityI = "pos"
					
					#fit of sin and lin func
					Iresult_sin = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					Iresult_lin = minimize(linear, Iparams_lin, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					
					Iparams = Iresult_sin.params

					#Repeat Feature if lin. Fit is better than sine fit
					Ifit_counter = 1
					if Iresult_lin.redchi < 2*Iresult_sin.redchi and Ifit_counter < Ifit_counter_limit:
						
						Iparams['amp'].value = (Ifit_counter)*1e-12

						if PartWiseTFit == True:
							Iparams['phase'].value = Tfit_down[i-1,2]-pi/2
						else:
							Iparams['phase'].value = Tfit_down[2]-pi/2

						Iparams['offs'].value = (Ifit_counter**2)*1e-10
						Iparams['slope'].value = (Ifit_counter**2)*1e-10
						
						Iresult_sin = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
						
						Ifit_counter =  Ifit_counter + 1

					#print i, Ifit_counter
					sys.stdout.write("\rProgress: %d/%d; %.0f %% Rep.: %d" % (i,I_perioden-1,100*float(i)/float(I_perioden-1),Ifit_counter))
					sys.stdout.flush()
					
					#fit correction (amp/phase)
					Iparams = amp_phase_correction(Iparams)	
					
					#plot of sin and line fit
					ax2.plot(tnew[start:ende], sinfunc(Iparams, tnew[start:ende]), 'r-')
				
					#extract params dict to lists
					Ifit_temp, Ierror_temp = extract_fit_relerr_params(Iparams)
					if i==1:
						Ifit = array([Ifit_temp])
						Ierror = array([Ierror_temp])
						Iresults = [Iresult_sin]					#save lmfit minizimer objects for future purpose ... maybe
					else:
						Ifit = append(Ifit,[array(Ifit_temp)],axis=0)
						Ierror = append(Ierror,[array(Ierror_temp)],axis=0)
						Iresults.append(Iresult_sin)

					#calculate phase difference
					if single_crystal==False:
						if PartWiseTFit == True:
							phi_T = Tfit_down[i-1,2]
						else:
							phi_T = Tfit_down[2]
						phi_I = Ifit[i-1,2]
						
						phasediff = phase_correction(phi_T-phi_I)
					else:
						phasediff = -pi/2

					#NonPyroStrom---------------------------------------------------------------------------
					#m=magenta (TSC-Strom)
					
					#Plot
					nonpyroparams = Parameters()
					if polarityI == "neg":
						nonpyroparams.add('amp', value=-1*abs(Ifit[i-1,0]*-cos(phasediff)))
					else:
						nonpyroparams.add('amp', value=abs(Ifit[i-1,0]*-cos(phasediff)))
					nonpyroparams.add('freq', value=Ifit[i-1,1])
					if PartWiseTFit == True:
						nonpyroparams.add('phase', value=Tfit_down[i-1,2])
					else:
						nonpyroparams.add('phase', value=Tfit_down[2])
					nonpyroparams.add('offs', value=Ifit[i-1,3])
					nonpyroparams.add('slope', value=Ifit[i-1,4])
					nonpyroparams = amp_phase_correction(nonpyroparams)
					ax2.plot(tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende]), color=np_color,linestyle='-')
					
					#Calculating Data from Fit - TSC
					if calculate_data_from_fit_flag == True:
						TSC = (array([tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende])])).T		#transpose!
						if i==1:
							I_TSC = TSC
						else:
							I_TSC = append(I_TSC, TSC, axis=0)

					#Pyrostrom + Koeff.---------------------------------------------------------------------
					#c=cyan (Pyrostrom)
					
					#Plot
					pyroparams = Parameters()
					pyroparams.add('amp', value=Ifit[i-1,0]*-sin(phasediff))
					pyroparams.add('freq', value=Ifit[i-1,1])
					if PartWiseTFit == True:
						pyroparams.add('phase', value=(Tfit_down[i-1,2]+pi/2))
					else:
						pyroparams.add('phase', value=(Tfit_down[2]+pi/2))
					pyroparams.add('offs', value=Ifit[i-1,3])
					pyroparams.add('slope', value=Ifit[i-1,4])
					pyroparams = amp_phase_correction(pyroparams)
					ax2.plot(tnew[start:ende], sinfunc(pyroparams, tnew[start:ende]), color=p_color,linestyle='-')
					
					#Calculating Data from Fit - Pyro
					if calculate_data_from_fit_flag == True:
						pyro = (array([tnew[start:ende], sinfunc(pyroparams, tnew[start:ende])])).T		#transpose!
						if i==1:
							I_pyro = pyro
						else:
							I_pyro = append(I_pyro, pyro, axis=0)
					
					#Calc p
					time = mean(tnew[start:ende])
					if PartWiseTFit == True:
						Temp = (tnew[start_index+(i-1)*satzlaenge] + tnew[start_index+(i*satzlaenge)])/2 * Tfit_down[i-1,4] + Tfit_down[i-1,3]
						p_SG = (Ifit[i-1,0]*-sin(phasediff))/(area*Tfit_down[i-1,0]*2*pi*abs(Tfit_down[i-1,1]))						# p (Sharp-Garn) ... with - sin() ! (see manual) ;)
						p_BR = (abs(mean(Idata[start:ende,1]))/(area*Tfit_down[i-1,4]))												# p (Byer-Roundy)
						perror = p_SG * rel_err(Tfit_down[i-1],Terror_down[i-1],Ifit[i-1],Ierror[i-1],area, area_error,phasediff,Xsigma=sigma)
					else:
						Temp = (tnew[start_index+(i-1)*satzlaenge] + tnew[start_index+(i*satzlaenge)])/2 * Tfit_down[4] + Tfit_down[3]
						p_SG = (Ifit[i-1,0]*-sin(phasediff))/(area*Tfit_down[0]*2*pi*abs(Tfit_down[1]))						# p (Sharp-Garn) ... with - sin() ! (see manual) ;)
						p_BR = (abs(mean(Idata[start:ende,1]))/(area*Tfit_down[4]))												# p (Byer-Roundy)
						perror = p_SG * rel_err(Tfit_down,Terror_down,Ifit[i-1],Ierror[i-1],area, area_error,phasediff,Xsigma=sigma)
						
					phasediff = degrees(phasediff)																							# Phasediff. (in deg)
					Ip_TSC_ratio= abs((Ifit[i-1,0]*-sin(radians(phasediff)))/(Ifit[i-1,0]*cos(radians(phasediff))))	# ratio Pyro/TSC
					meanI = mean(Idata[start:ende,1])																					# mean I in Interval
					IAmp_error = Iresults[i-1].chisqr/(Ifit[i-1,0]**2)		# Chi Square / I_amp^2																						# Chi square in Interval

					#wrinting temp list
					p_temp = [time, Temp, p_SG, p_BR, phasediff, Ip_TSC_ratio, meanI, IAmp_error, perror]
					#append list to array 
					if i==1:
						p = array([p_temp])
						p_error = array([perror])
					else:
						p = append(p, [array(p_temp)], axis=0)
						p_error = append(p_error,perror)
				
				current_custom_legend(ax2)
				draw()
				
				header_string = "Amp [I]\t\t\tFreq [Hz]\t\t\tPhase [rad]\t\t\tOffset [A]\t\t\tSlope [A/s]\t\t\tAmp_Err [A]\t\t\tFreq_Err [Hz]\t\t\tPhase_Err [rad]\t\t\tOffs_Err [A]\t\t\tSlope_Err [A/s]"
				savetxt(date+"_"+samplename+"_"+T_profile+"_I-Fit.txt",hstack([Ifit,Ierror]), delimiter="\t", header=header_string)
				
				print('\nCurrent ... done')
				print(line)

				#Plotting p(T)-----------------------------------------------------------------------------------------------------------
				bild2=figure(date+"_"+samplename+"_"+T_profile+'_Pyro', figsize=fig_size)

				#p(T)--------------------------------------------------------------
				ax3=subplot(221)
				ax3.set_autoscale_on(True)
				ax3.set_xlim(p[0,1],p[-1,1])
				ax3.set_xlabel('Temperature (K)',size=label_size)
				ax3.set_ylabel(u"$p$ (µC/Km²)",color=temp_color,size=label_size)

				ax3.grid(b=None, which='major', axis='both', color='grey')
				ax3.errorbar(p[:,1],(p[:,2]*1e6), yerr=p_error[:]*1e6, color=temp_color,marker=".",linestyle="", elinewidth=None, capsize=3, label='$p$ (SG)')
				if BR_flag == True:
					ax3.plot(p[:,1],(p[:,3]*1e6), "r.", label='$p$ (BR)')
					ax3.legend(loc=3)

				#p/TSC ration---------------------------------------------------------
				ax5=subplot(222,sharex=ax3)
				ax5.set_autoscale_on(True)
				ax5.set_xlim(ax3.get_xbound())
				ax5.grid(b=None, which='major', axis='both', color='grey')
				ax5.set_xlabel('Temperature (K)',size=label_size)
				ax5.set_ylabel(r"I$_{p}$/I$_{np}$",color=volt_color,size=label_size)
				ax5.semilogy(p[:,1], p[:,5], color=volt_color,marker=".",linestyle="", label=r"I$_{p}$/I$_{np}$")

				#Chisqr---------------------------------------------------------------
				ax6=subplot(224,sharex=ax3)
				ax6.set_autoscale_on(True)
				ax6.set_xlim(ax3.get_xbound())
				ax6.grid(b=None, which='major', axis='both', color='grey')
				ax6.set_xlabel('Temperature (K)',size=label_size)
				ax6.set_ylabel(r"$X^2 / A_I^2$",color='c',size=label_size)
				ax6.semilogy(p[:,1], p[:,7], color=np_color,marker=".",linestyle="", label=r"$X^2$")

				#Phasediff---------------------------------------------------------------
				ax7=subplot(223,sharex=ax3)
				#ax7.set_autoscale_on(True)
				ax7.set_xlim(ax3.get_xbound())
				ax7.set_ylim(0,360)
				ax7.axhline(180, color='k')
				ax7.axhline(90, color='k',linestyle='--')
				ax7.axhline(270, color='k', linestyle='--')
				ax7.grid(b=None, which='major', axis='both', color='grey')
				ax7.set_xlabel('Temperature (K)',size=label_size)
				ax7.set_ylabel(ur"$\phi$ (°)",color=other,size=label_size)
				ax7.plot(p[:,1],p[:,4],color=other,marker=".",linestyle="", label="Phasediff.")
				
				#CurrAmp---------------------------------------------------------------
				ax8 = ax7.twinx()
				ax8.set_xlim(ax3.get_xbound())
				ax8.plot(p[:,1],Ifit[:,0], color=curr_color,marker=".", linestyle="", label="Amplitude")
				ax8.set_ylabel(r"$I_{\mathrm{Amp}}$ (A)",color=curr_color,size=label_size)

				bild2.tight_layout()
				
				if print_signature == True:
					bild2.subplots_adjust(bottom=0.125)
					signature = operator['name']+' | '+operator['mail']+' | ' + operator['tel'] + ' | ' +operator['company'] + ' | ' + operator['date']
					figtext(0.15,0.02,signature)
				
				show()

				#Calculating p ---------------------------------------------------------------------------------------
				print("spontaneous Polarization ...")
				PS_plot = raw_input("Calculate? (y/n):")
				if PS_plot == "y":
					PS_flag = True
					
					#generate new ax
					axP = ax3.twinx()
					
					number_of_maxima = raw_input("How many max?: ")
					number_of_maxima = int(number_of_maxima)
					print("select TC(s) from the p(T) plot")
					TC = ginput(number_of_maxima)
					
					#get index from array where temp is 300K
					T300 = abs(p[:,1]-300).argmin()
					
					P = []
					TC_index_list = []
					#loop for each selected temperature
					for i in range(number_of_maxima):
						TC_index = abs(p[:,1]-TC[i][0]).argmin()
						TC_index_list.append(TC_index)
						
						#calc PS with partial trapezoidal integration
						for f in range(TC_index):
							PS_interval = trapz(y=p[f:TC_index,2], x=p[f:TC_index,1])
							if f==0:	
								P = [PS_interval]
							else:
								P.append(PS_interval)
						
						#fill rest of array legth with zeros
						for f in range(TC_index,len(p)):
							P.append(0.0)
					
						#make array type 
						Polarization = array(P)

						#append to p array
						p = column_stack((p,Polarization))		#letzte Spalte ist Polarizationsverlauf
					
					
						#user message
						print("%d:\tTC: %.2f K / %.2f C\n\tPS(300K): %.3f mC/km2" % (i+1,TC[i][0],(TC[i][0]-273.15),abs(P[T300])*1e3))

						#Plot
						axP.semilogy(p[:,1],abs(array(P)*1e3), linestylelist[i], color=p_color, label="Polarization")
						cur_ylim = axP.get_ylim()
						axP.set_ylim(1e0,1e3)
						axP.set_xlim(ax3.get_xbound())
						axP.set_ylabel(u'Polarization (mC/m²)',color=p_color,size=label_size)
					
					draw()
				
				#Saving results and figs------------------------------------------------------------------------------
				saving_figure(bild1)
				saving_figure(bild2, pbild=True)

				#writing log files
				print(line)
				print("...writing log files")				
				header_string = "time [s]\t\t\tTemp [K]\t\t\tp_SG [C/Km2]\t\t\tp_BR [C/Km2],\t\t\tPhasediff [deg]\t\t\tp/TSC-ratio\t\t\tMean I [A]\t\t\tRed Chi\t\t\t\tp_err [C/Km2]\t"
				
				if PS_flag == True:
					
					for k in range(number_of_maxima):
						pol_string = "\t\tPS [C/m2] - TC %.2fK" % p[TC_index_list[k],1]
						header_string = header_string + pol_string 
				
				savetxt(date+"_"+samplename+"_"+T_profile+"_"+"PyroData.txt", p, delimiter="\t", header=header_string)
				
				if calculate_data_from_fit_flag == True:
					header_string = "time [s]\t\t\tI_TSC [A]\t\t\tI_pyro [A]"
					savetxt(date+"_"+samplename+"_"+T_profile+"_"+"DataFromFit.txt", vstack([I_TSC[:,0], I_TSC[:,1], I_pyro[:,1]]).T, delimiter="\t", header=header_string)
			
			
			else:
				saving_figure(bild1)

		#---------------------------------------------------------------------------------------------------------------------
		#TriangleHat
		elif measurement_info['waveform'] == "TriangleHat":
			print("Mode:\t\t%s"%measurement_info['waveform'])
			print "Stimulation:\tO1=%.1fK\n\t\tTm=%.1fK\n\t\tO2=%.1fK\n\t\tHR=%.1fK/h\n\t\tCR=%.1fK/h" % (measurement_info['offs'], measurement_info['T_Limit_H'], measurement_info['freq'], measurement_info['heat_rate']*3600, measurement_info['cool_rate']*3600)

			#Interpolation and plotting of data ----
			print(line)
			print("...plotting")
			print(line)
			# pre-fit plot
			tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
			bild1, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)

			show()

			#save figure
			print(line)
			print("...saving figure")
			saving_figure(bild1)

		#---------------------------------------------------------------------------------------------------------------------
		#SineWave+TriangleHat
		elif measurement_info['waveform'] == "SineWave+TriangleHat":
			print("Mode:\t\t%s"%measurement_info['waveform'])
			print "Stimulation:\tO1=%.1fK\n\t\tTm=%.1fK\n\t\tO2=%.1fK\n\t\tHR=%.1fK/h\n\t\tCR=%.1fK/h\n\t\tA=%.1fK\n\t\tf=%.1fmHz" % (measurement_info['offs'], measurement_info['T_Limit_H'], measurement_info['offs'], measurement_info['heat_rate']*3600, measurement_info['cool_rate']*3600, measurement_info['amp'], measurement_info['freq']*1000)

			#Interpolation and plotting of data ----
			print(line)
			print("...plotting")
			print(line)
			# pre-fit plot
			tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
			bild1, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)

			show()

			input = raw_input("fit (y/n)?")
			if input == "y":

				#area for pyroel. coefficent
				area, area_error = get_area()
				print("Area: %e m2" % area)
				
				print(line)
				print("...fitting")
				if PartWiseTFit == True:
					print("Partwise Temp.Fit enabled!")
				print(line)
				
				#prepare output log
				log = open(date+"_"+samplename+"_"+T_profile+"_T-Fits.txt", 'w+')

				#important calculations for further fit;)--------------------------------------------------------------
				#check when ramp runs into T_Limit_H
				turning_point_index = argmax(Tnew[:,0])
				end_point_index = argmin(Tnew[turning_point_index:,0])+turning_point_index		#calculate end_point when measurement time is too long cooling ramp^
				T_perioden_heat = int(tnew[turning_point_index-start_index]/(fit_periods/measurement_info['freq']))
				T_perioden_cool = int(tnew[end_point_index-turning_point_index]/(fit_periods/measurement_info['freq']))
				T_perioden = T_perioden_heat+T_perioden_cool
				satzlaenge = len(tnew[:end_point_index-start_index])/T_perioden

				print("temperature fit ...")
				# in case of part wise fitting of T -----------------------------------------------------------------------------------------				
				if PartWiseTFit == True:
					
					#initialize fit variables for temperature in heating part
					Tparams_down_heat = Parameters()
					Tparams_down_heat.add('amp', value=measurement_info['amp'],min=0.0)
					Tparams_down_heat.add('freq', value=measurement_info['freq'], min=1e-5, max=0.2)
					Tparams_down_heat.add('phase', value=0.1, vary=True)
					Tparams_down_heat.add('offs', value=measurement_info['offs'], min=0.0)
					Tparams_down_heat.add('slope', value=measurement_info['heat_rate'])

					#initialize fit variables for temperature in cooling part
					Tparams_down_cool = Parameters()
					Tparams_down_cool.add('amp', value=measurement_info['amp'],min=0.0)
					Tparams_down_cool.add('freq', value=measurement_info['freq'], min=1e-5, max=0.2)
					Tparams_down_cool.add('phase', value=0.1, vary=True)
					Tparams_down_cool.add('offs', value=measurement_info['offs']+measurement_info['T_Limit_H'], min=0.0)
					Tparams_down_cool.add('slope', value=measurement_info['cool_rate'])
		
					#perform partial fit
					for i in arange(1,T_perioden):

						#calculate start and end index of interval i
						start = start_index+int((i*satzlaenge)-satzlaenge)
						ende = start_index+int(i*satzlaenge)
						
						#fit for heating
						if i < T_perioden_heat:	
							#heating				
							Tresult_sin_heat = minimize(sinfunc, Tparams_down_heat, args=(tnew[start:ende], Tnew[start:ende,0]), method="leastsq")
							Tparams_down_heat = amp_phase_correction(Tresult_sin_heat.params)
							ax1.plot(tnew[start:ende], sinfunc(Tparams_down_heat, tnew[start:ende]), color=temp_color, linestyle='-')
							Tfit_down_temp, Terror_down_temp = extract_fit_relerr_params(Tparams_down_heat)
							if i==1:
								Tfit_down_heat = array([Tfit_down_temp])
								Terror_down_heat = array([Terror_down_temp])
								Tresults_down_heat = [Tresult_sin_heat]					#save lmfit minizimer objects for future purpose ... maybe
							else:
								Tfit_down_heat = append(Tfit_down_heat,[array(Tfit_down_temp)],axis=0)
								Terror_down_heat = append(Terror_down_heat,[array(Terror_down_temp)],axis=0)
								Tresults_down_heat.append(Tresult_sin_heat)
							#paramters for next fit interval								
							Tparams_down_heat['phase'].value=Tfit_down_heat[i-1,2]
							#Tparams_down['phase'].vary=False

						# fit for cooling
						else:
							Tresult_sin_cool = minimize(sinfunc, Tparams_down_cool, args=(tnew[start:ende], Tnew[start:ende,0]), method="leastsq")
							Tparams_down_cool = amp_phase_correction(Tresult_sin_cool.params)
							ax1.plot(tnew[start:ende], sinfunc(Tparams_down_cool, tnew[start:ende]), color=temp_color, linestyle='-')
							Tfit_down_temp, Terror_down_temp = extract_fit_relerr_params(Tparams_down_cool)
							if i==T_perioden_heat:
								Tfit_down_cool = array([Tfit_down_temp])
								Terror_down_cool = array([Terror_down_temp])
								Tresults_down_cool = [Tresult_sin_cool]					#save lmfit minizimer objects for future purpose ... maybe
							else:
								Tfit_down_cool = append(Tfit_down_cool,[array(Tfit_down_temp)],axis=0)
								Terror_down_cool = append(Terror_down_cool,[array(Terror_down_temp)],axis=0)
								Tresults_down_cool.append(Tresult_sin_cool)
							#paramters for next fit interval								
							Tparams_down_cool['phase'].value=Tfit_down_cool[i-1-T_perioden_heat,2]
							#Tparams_down['phase'].vary=False
					
						#console status
						sys.stdout.write("\rProgress: %d/%d; %.0f %%" % (i,T_perioden-1,100*float(i)/float(T_perioden-1)))
						sys.stdout.flush()

					log_output = vstack([hstack([Tfit_down_heat,Terror_down_heat]),hstack([Tfit_down_cool,Terror_down_cool])])
					header_string = "Amp [I]\t\t\tFreq [Hz]\t\t\tPhase [rad]\t\t\tOffset [A]\t\t\tSlope [A/s]\t\t\tAmp_Err [A]\t\t\tFreq_Err [Hz]\t\t\tPhase_Err [rad]\t\t\tOffs_Err [A]\t\t\tSlope_Err [A/s]"
					savetxt(date+"_"+samplename+"_"+T_profile+"_T-Fit-partwise.txt",log_output, delimiter="\t", header=header_string)
				
				# in case of whole range fit of T -----------------------------------------------------------------------------------------				
				else:			
					#Temp fit/plot for heating-----------------------------------------------------------------------------
					Tresult_down_heat = fit(tnew, Tnew[:,0],start_index,turning_point_index,1,measurement_info,True,True)
					#correction of phase and amplitudes
					Tparams_down_heat = amp_phase_correction(Tresult_down_heat.params)
					#extract params dict to lists
					Tfit_down_heat, Terror_down_heat = extract_fit_relerr_params(Tparams_down_heat)
					#Fit-Plot
					ax1.plot(tnew[start_index:turning_point_index], sinfunc(Tparams_down_heat, tnew[start_index:turning_point_index]), color=temp_color, linestyle='-', label="fit")
					draw()
					#absolute T_high Error
					total_Terror_down_heat = abs(Tparams_down_heat['amp'].stderr/Tparams_down_heat['amp'].value)+abs(Tparams_down_heat['phase'].stderr/Tparams_down_heat['phase'].value)+abs(Tparams_down_heat['freq'].stderr/Tparams_down_heat['freq'].value)+abs(Tparams_down_heat['offs'].stderr/Tparams_down_heat['offs'].value)+abs(Tparams_down_heat['slope'].stderr/Tparams_down_heat['slope'].value)
					#file output
					fileprint_fit(log,Tparams_down_heat,"Temperature (Down) - Heating")
					#for top temperature-------------------
					if temp_filter_flag == False:
						Tresult_high_heat = fit(tnew, Tnew[:,1], start_index, turning_point_index,1, measurement_info,True,True)
						#correction of phase and amplitude
						Tparams_high_heat = amp_phase_correction(Tresult_high_heat.params)
						#extract params dict to lists
						Tfit_high_heat, Terror_high_heat = extract_fit_relerr_params(Tparams_high_heat)
						#plot of second fit
						ax1.plot(tnew[start_index:turning_point_index], sinfunc(Tparams_high_heat, tnew[start_index:turning_point_index]), color=volt_color, linestyle='-', label='fit (Top)')
						draw()
						#absolute T_high Error
						total_Terror_high_cool = abs(Tparams_high_heat['amp'].stderr/Tparams_high_heat['amp'].value)+abs(Tparams_high_heat['phase'].stderr/Tparams_high_heat['phase'].value)+abs(Tparams_high_heat['freq'].stderr/Tparams_high_heat['freq'].value)+abs(Tparams_high_heat['offs'].stderr/Tparams_high_heat['offs'].value)+abs(Tparams_high_heat['slope'].stderr/Tparams_high_heat['slope'].value)
						#file output
						fileprint_fit(log,Tparams_high_heat,"Temperature (High) - Heating")
					
					
					#Temp fit/plot for cooling-----------------------------------------------------------------------------
					Tresult_down_cool= fit(tnew, Tnew[:,0],turning_point_index,end_point_index,1,measurement_info, True, heating=False)
					#correction of phase and amplitudes
					Tparams_down_cool = amp_phase_correction(Tresult_down_cool.params)
					#extract params dict to lists
					Tfit_down_cool, Terror_down_cool = extract_fit_relerr_params(Tparams_down_cool)
					#Fit-Plot
					ax1.plot(tnew[turning_point_index:end_point_index], sinfunc(Tparams_down_cool, tnew[turning_point_index:end_point_index]),color=temp_color, linestyle='-')
					draw()
					#absolute T_high Error
					total_Terror_down_heat = abs(Tparams_down_cool['amp'].stderr/Tparams_down_cool['amp'].value)+abs(Tparams_down_cool['phase'].stderr/Tparams_down_cool['phase'].value)+abs(Tparams_down_cool['freq'].stderr/Tparams_down_cool['freq'].value)+abs(Tparams_down_cool['offs'].stderr/Tparams_down_cool['offs'].value)+abs(Tparams_down_cool['slope'].stderr/Tparams_down_cool['slope'].value)
					#file output
					fileprint_fit(log,Tparams_down_cool,"Temperature (Down) - Cooling")  
					if temp_filter_flag == False:
						Tresult_high_cool= fit(tnew, Tnew[:,1], turning_point_index,end_point_index,1, measurement_info, True, heating=False)
						#correction of phase and amplitudes
						Tparams_high_cool = amp_phase_correction(Tresult_high_cool.params)
						#extract params dict to lists
						Tfit_high_cool, Terror_high_cool = extract_fit_relerr_params(Tparams_high_cool)
						#plot of second fit
						ax1.plot(tnew[turning_point_index:end_point_index], sinfunc(Tparams_high_cool, tnew[turning_point_index:end_point_index]), color=volt_color, linestyle='-')
						draw()
						#absolute T_high Error
						total_Terror_high_cool = abs(Tparams_high_cool['amp'].stderr/Tparams_high_cool['amp'].value)+abs(Tparams_high_cool['phase'].stderr/Tparams_high_cool['phase'].value)+abs(Tparams_high_cool['freq'].stderr/Tparams_high_cool['freq'].value)+abs(Tparams_high_cool['offs'].stderr/Tparams_high_cool['offs'].value)+abs(Tparams_high_cool['slope'].stderr/Tparams_high_cool['slope'].value)
						#file output
						fileprint_fit(log,Tparams_high_cool,"Temperature (High) - Cooling")

					log.close()

				temperature_custom_legend(ax1)
				draw()
				print("\nTemperature ... done!")

				#Current Fit -------------------------------------------------------------------------------------
				print(line)
				print("current fit ...")

				I_perioden = int(tnew[end_point_index-start_index]/(fit_periods/measurement_info['freq']))
				satzlaenge = len(tnew[:end_point_index-start_index])/I_perioden
				
				Ifit = zeros((1,5))
				Ierror = zeros((1,5))
				
				Iparams = Parameters()
				Iparams.add('amp', value=1e-11)
				if PartWiseTFit == True:
					Iparams.add('freq', value=Tfit_down_heat[0,1], min=1e-5, max=0.2, vary=False)
				else:
					Iparams.add('freq', value=Tfit_down_heat[1], min=1e-5, max=0.2, vary=False)
				Iparams.add('phase', value=1.0)
				Iparams.add('offs', value=1e-10)
				Iparams.add('slope', value=1e-10)
				
				Iparams_lin = Parameters()
				Iparams_lin.add('a', value=1e-10)
				Iparams_lin.add('b', value=0.0)

				#perform partial fits
				for i in arange(1,I_perioden):
					start = start_index+int((i*satzlaenge)-satzlaenge)
					ende = start_index+int(i*satzlaenge)

					# avaraging singnal part to get algebraic sign (oscillation around pos/neg value?)
					meanI = mean(Inew[start:ende])
					if meanI < 0.0:
						polarityI = "neg"
					else:
						polarityI = "pos"
						
					#fit of sin and lin func
					Iresult_sin = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					Iresult_lin = minimize(linear, Iparams_lin, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					
					Iparams = Iresult_sin.params
					
					#Repeat Feature if lin. Fit is better than sine fit
					Ifit_counter = 1
					if Iresult_lin.redchi < 2*Iresult_sin.redchi and Ifit_counter < Ifit_counter_limit:
						
						Iparams['amp'].value = (Ifit_counter)*1e-12
						if i < T_perioden_heat:
							if PartWiseTFit == True:
								Iparams['phase'].value = Tfit_down_heat[i-1,2]-pi/2
							else:
								Iparams['phase'].value = Tfit_down_heat[2]-pi/2
						else:
							if PartWiseTFit == True:
								Iparams['phase'].value = Tfit_down_cool[i-1-T_perioden_heat,2]-pi/2
							else:
								Iparams['phase'].value = Tfit_down_cool[2]-pi/2
						#Iparams['offs'].value = (Ifit_counter**2)*1e-10
						#Iparams['slope'].value = (Ifit_counter**2)*1e-10
						
						Iresult_sin = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
						
						Ifit_counter =  Ifit_counter + 1

					#print i, Ifit_counter
					sys.stdout.write("\rProgress: %d/%d; %.0f %% Rep.: %d" % (i,I_perioden-1,100*float(i)/float(I_perioden-1),Ifit_counter))
					sys.stdout.flush()
					
					#fit correction (amp/phase)
					Iparams = amp_phase_correction(Iparams)
					
					#plot of sin and line fit
					ax2.plot(tnew[start:ende], sinfunc(Iparams, tnew[start:ende]), 'r-')
					ax2.plot(tnew[start:ende], linear(Iparams_lin, tnew[start:ende]), 'r--')
				
					#extract params dict to lists
					Ifit_temp, Ierror_temp = extract_fit_relerr_params(Iparams)
					if i==1:
						Ifit = array([Ifit_temp])
						Ierror = array([Ierror_temp])
						Iresults = [Iresult_sin]					#save lmfit minizimer objects for future purpose ... maybe
					else:
						Ifit = append(Ifit,[array(Ifit_temp)],axis=0)
						Ierror = append(Ierror,[array(Ierror_temp)],axis=0)
						Iresults.append(Iresult_sin)

					#paramters for next fit interval
					if PartWiseTFit == True:
						if i < T_perioden_heat:
							Iparams['phase'].value=Tfit_down_heat[i-1,2]
							Iparams['freq'].value=Tfit_down_heat[i-1,1]
						else:
							Iparams['phase'].value=Tfit_down_cool[i-1-T_perioden_heat,2]
							Iparams['freq'].value=Tfit_down_cool[i-1-T_perioden_heat,1]

					#calculate phase difference
					if single_crystal==False:
						if PartWiseTFit == True:
							if i < T_perioden_heat:
								phi_T = Tfit_down_heat[i-1,2]
							else:
								phi_T = Tfit_down_cool[i-1-T_perioden_heat,2]
						else:
							if i < T_perioden_heat:
								phi_T = Tfit_down_heat[2]
							else:
								phi_T = Tfit_down_cool[2]

						phi_I = Ifit[i-1,2]
						
						# if abs(phi_I) > abs(phi_T):
							# phasediff = phase_correction(phi_I-phi_T)
						# else:
							# phasediff = phase_correction(phi_T-phi_I)
						
						phasediff = phase_correction(phi_T-phi_I)
					else:
						phasediff = -pi/2

					#NonPyroStrom---------------------------------------------------------------------------
					#m=magenta (TSC-Strom)
					
					#Plot
					nonpyroparams = Parameters()
					if polarityI == "neg":
						nonpyroparams.add('amp', value=-1*abs(Ifit[i-1,0]*-cos(phasediff)))
					else:
						nonpyroparams.add('amp', value=abs(Ifit[i-1,0]*-cos(phasediff)))
					
					if PartWiseTFit == True:		#need different parameters wethere heating or cooling
						if i < T_perioden_heat:
							nonpyroparams.add('freq', value=Tfit_down_heat[i-1,1])
							nonpyroparams.add('phase', value=(Tfit_down_heat[i-1,2]-pi))
						else:
							nonpyroparams.add('freq', value=Tfit_down_cool[i-1-T_perioden_heat,1])
							nonpyroparams.add('phase', value=(Tfit_down_cool[i-1-T_perioden_heat,2]-pi))
					else:
						nonpyroparams.add('freq', value=Tfit_down_heat[1])
						nonpyroparams.add('phase', value=(Tfit_down_heat[2]-pi))

					nonpyroparams.add('offs', value=Ifit[i-1,3])
					nonpyroparams.add('slope', value=Ifit[i-1,4])
					nonpyroparams = amp_phase_correction(nonpyroparams)
					ax2.plot(tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende]), color=np_color,linestyle='-')
					
					#Calculating Data from Fit - TSC
					if calculate_data_from_fit_flag == True:
						TSC = (array([tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende])])).T		#transpose!
						if i==1:
							I_TSC = TSC
						else:
							I_TSC = append(I_TSC, TSC, axis=0)

					#Pyrostrom + Koeff.---------------------------------------------------------------------
					#c=cyan (Pyrostrom)
					
					#Plot
					pyroparams = Parameters()
					pyroparams.add('amp', value=Ifit[i-1,0]*-sin(phasediff))

					if PartWiseTFit == True:		#need different parameters wethere heating or cooling
						if i < T_perioden_heat:
							pyroparams.add('freq', value=Tfit_down_heat[i-1,1])
							pyroparams.add('phase', value=(Tfit_down_heat[i-1,2]+pi/2))
						else:
							pyroparams.add('freq', value=Tfit_down_cool[i-1-T_perioden_heat,1])
							pyroparams.add('phase', value=(Tfit_down_cool[i-1-T_perioden_heat,2]+pi/2))
					else:
						pyroparams.add('freq', value=Tfit_down_heat[1])
						pyroparams.add('phase', value=(Tfit_down_heat[2]+pi/2))
					pyroparams.add('offs', value=Ifit[i-1,3])
					pyroparams.add('slope', value=Ifit[i-1,4])
					pyroparams = amp_phase_correction(pyroparams)
					ax2.plot(tnew[start:ende], sinfunc(pyroparams, tnew[start:ende]), color=p_color,linestyle='-')
					
					#Calculating Data from Fit - Pyro
					if calculate_data_from_fit_flag == True:
						pyro = (array([tnew[start:ende], sinfunc(pyroparams, tnew[start:ende])])).T		#transpose!
						if i==1:
							I_pyro = pyro
						else:
							I_pyro = append(I_pyro, pyro, axis=0)

					#Calc p
					#for heating branche
					if i < T_perioden_heat:
						if PartWiseTFit == True:
							Temp = (tnew[start_index+(i-1)*satzlaenge] + tnew[start_index+(i*satzlaenge)])/2 * Tfit_down_heat[i-1,4] + Tfit_down_heat[i-1,3]
							p_SG = (Ifit[i-1,0]*-sin(phasediff))/(area*Tfit_down_heat[i-1,0]*2*pi*abs(Tfit_down_heat[i-1,1]))
							p_BR = (abs(mean(Idata[start:ende,1]))/(area*Tfit_down_heat[i-1,4]))
							perror = p_SG * rel_err(Tfit_down_heat[i-1],Terror_down_heat[i-1],Ifit[i-1],Ierror[i-1],area, area_error,phasediff,Xsigma=sigma)
						else:
							Temp = (tnew[start_index+(i-1)*satzlaenge] + tnew[start_index+(i*satzlaenge)])/2 * Tfit_down_heat[4] + Tfit_down_heat[3]
							p_SG = (Ifit[i-1,0]*-sin(phasediff))/(area*Tfit_down_heat[0]*2*pi*abs(Tfit_down_heat[1]))
							p_BR = (abs(mean(Idata[start:ende,1]))/(area*Tfit_down_heat[4]))
							perror = p_SG * rel_err(Tfit_down_heat,Terror_down_heat,Ifit[i-1],Ierror[i-1],area, area_error,phasediff,Xsigma=sigma)
						turning_p_index = i
					#for cooling branche
					else:
						if PartWiseTFit == True:
							Temp = (tnew[start_index+(i-1)*satzlaenge] + tnew[start_index+(i*satzlaenge)])/2 * Tfit_down_cool[i-1-T_perioden_heat,4] + Tfit_down_cool[i-1-T_perioden_heat,3]
							p_SG = (Ifit[i-1,0]*-sin(phasediff))/(area*Tfit_down_cool[i-1-T_perioden_heat,0]*2*pi*abs(Tfit_down_cool[i-1-T_perioden_heat,1]))
							p_BR = (abs(mean(Idata[start:ende,1]))/(area*Tfit_down_cool[i-1-T_perioden_heat,4]))
							perror = p_SG * rel_err(Tfit_down_cool[i-1-T_perioden_heat],Terror_down_cool[i-1-T_perioden_heat],Ifit[i-1],Ierror[i-1],area, area_error,phasediff,Xsigma=sigma)			
						else:
							Temp = (tnew[start_index+(i-1)*satzlaenge] + tnew[start_index+(i*satzlaenge)])/2 * Tfit_down_cool[4] + Tfit_down_cool[3]
							p_SG = (Ifit[i-1,0]*-sin(phasediff))/(area*Tfit_down_cool[0]*2*pi*abs(Tfit_down_cool[1]))
							p_BR = (abs(mean(Idata[start:ende,1]))/(area*Tfit_down_cool[4]))
							perror = p_SG * rel_err(Tfit_down_cool,Terror_down_cool,Ifit[i-1],Ierror[i-1],area, area_error,phasediff,Xsigma=sigma)					
					phasediff = degrees(phasediff)					# Phasediff.
					Ip_TSC_ratio= abs((Ifit[i-1,0]*sin(phasediff))/(Ifit[i-1,0]*cos(phasediff)))		# ratio Pyro/TSC
					meanI = mean(Idata[start:ende,1])
					IAmp_error = Iresults[i-1].chisqr/(Ifit[i-1,0]**2)		# Chi Square / I_amp^2
					time = mean(tnew[start:ende])
					
					#wrinting temp list
					p_temp = [time, Temp, p_SG, p_BR, phasediff, Ip_TSC_ratio, meanI, IAmp_error, perror]
					#append list to array 
					if i==1:
						p = array([p_temp])
						p_error = array([perror])
					else:
						p = append(p, [array(p_temp)], axis=0)
						p_error = append(p_error,perror)
					
				current_custom_legend(ax2)
				draw()
				
				header_string = "Amp [I]\t\t\tFreq [Hz]\t\t\tPhase [rad]\t\t\tOffset [A]\t\t\tSlope [A/s]\t\t\tAmp_Err [A]\t\t\tFreq_Err [Hz]\t\t\tPhase_Err [rad]\t\t\tOffs_Err [A]\t\t\tSlope_Err [A/s]"
				savetxt(date+"_"+samplename+"_"+T_profile+"_I-Fit.txt",hstack([Ifit,Ierror]), delimiter="\t", header=header_string)
				print("\nCurrent ... done!")
				print(line)
				
				#Plotting p(T)-----------------------------------------------------------------------------------------------------------
				bild2=figure(date+"_"+samplename+"_"+T_profile+'_Pyro', figsize=fig_size)

				#p(T)--------------------------------------------------------------
				ax3=subplot(221)
				ax3.set_autoscale_on(True)
				ax3.set_xlim(p[0,1],p[turning_p_index,1])
				ax3.set_xlabel('Temperature (K)',size=label_size)
				ax3.set_ylabel(u"$p$ (µC/Km²)",color=temp_color,size=label_size)

				ax3.grid(b=None, which='major', axis='both', color='grey')
				ax3.errorbar(p[:turning_p_index,1],(p[:turning_p_index,2]*1e6), yerr=p_error[:turning_p_index]*1e6, color=temp_color, marker=".",linestyle="", elinewidth=None, capsize=3, label='heating')
				ax3.errorbar(p[turning_p_index:,1],(p[turning_p_index:,2]*1e6), yerr=p_error[turning_p_index:]*1e6, color=temp_color, marker="x",linestyle="", elinewidth=None, capsize=3, label='cooling')
				
				if BR_flag == True:
					ax3.plot(p[:turning_p_index,1],(p[:turning_p_index,3]*1e6), color=volt_color, marker=".",linestyle="", label='$p$ (BR) - heating')
					ax3.plot(p[turning_p_index:,1],(p[turning_p_index:,3]*1e6), color=volt_color, marker="x",linestyle="", label='$p$ (BR) - cooling')
					ax3.legend(loc=3)

				#p/TSC ration---------------------------------------------------------
				ax5=subplot(222,sharex=ax3)
				ax5.set_autoscale_on(True)
				ax5.set_xlim(ax3.get_xbound())
				ax5.grid(b=None, which='major', axis='both', color='grey')
				ax5.set_xlabel('Temperature (K)',size=label_size)
				ax5.set_ylabel(r"I$_p$/I$_{TSC}$",color=volt_color,size=label_size)
				ax5.semilogy(p[:turning_p_index,1], p[:turning_p_index,5], color=volt_color,marker=".",linestyle="", label="heating")
				ax5.semilogy(p[turning_p_index:,1], p[turning_p_index:,5], color=volt_color,marker="x",linestyle="", label="cooling")

				#Chisqr---------------------------------------------------------------
				ax6=subplot(224,sharex=ax3)
				ax6.set_autoscale_on(True)
				ax6.set_xlim(ax3.get_xbound())
				ax6.grid(b=None, which='major', axis='both', color='grey')
				ax6.set_xlabel('Temperature (K)',size=label_size)
				ax6.set_ylabel(r"$X^2 / A_I^2$",color='c',size=label_size)
				ax6.semilogy(p[:turning_p_index,1], p[:turning_p_index,7], color=np_color,marker=".",linestyle="", label="heating")
				ax6.semilogy(p[turning_p_index:,1], p[turning_p_index:,7], color=np_color,marker="x",linestyle="", label="cooling")

				#Phasediff---------------------------------------------------------------
				ax7=subplot(223,sharex=ax3)
				ax7.set_autoscale_on(True)
				ax7.set_xlim(ax3.get_xbound())
				ax7.set_ylim(0,360)
				ax7.axhline(180, color='k')
				ax7.axhline(90, color='k',linestyle='--')
				ax7.axhline(270, color='k', linestyle='--')
				ax7.grid(b=None, which='major', axis='both', color='grey')
				ax7.set_xlabel('Temperature (K)',size=label_size)
				ax7.set_ylabel(ur"$\phi$ (°)",color=other,size=label_size)
				ax7.plot(p[:turning_p_index,1],p[:turning_p_index,4],color=other,marker=".",linestyle="", label="heating")
				ax7.plot(p[turning_p_index:,1],p[turning_p_index:,4],color=other,marker="x",linestyle="", label="cooling")
				
				ax8 = ax7.twinx()
				ax8.set_xlim(ax3.get_xbound())
				ax8.set_ylabel(r"$A_I$ (A)",color='r',size=label_size)
				ax8.plot(p[:turning_p_index,1],Ifit[:turning_p_index,0],color=curr_color,marker=".",linestyle="", label="heating")
				ax8.plot(p[turning_p_index:,1],Ifit[turning_p_index:,0],color=curr_color,marker="x",linestyle="", label="cooling")
				
				bild2.tight_layout()
				show()
				
				#writing log files
				print("...writing log files")				
				header_string = "time [s]\t\t\tTemp [K]\t\t\tp_SG [C/Km2]\t\t\tp_BR [C/Km2],\t\t\tPhasediff [deg]\t\t\tp/TSC-ratio\t\t\tMean I [A]\t\t\tRed Chi\t\t\t\tp_err [C/Km2]\t"
				savetxt(date+"_"+samplename+"_"+T_profile+"_"+"PyroData.txt", p, delimiter="\t", header=header_string)
				
				if calculate_data_from_fit_flag == True:
					header_string = "time [s]\t\t\tI_TSC [A]\t\t\tI_pyro [A]"
					savetxt(date+"_"+samplename+"_"+T_profile+"_"+"DataFromFit.txt", vstack([I_TSC[:,0], I_TSC[:,1], I_pyro[:,1]]).T, delimiter="\t", header=header_string)
					
				#save figure
				saving_figure(bild1)
				saving_figure(bild2, pbild=True)
			
			else:
				saving_figure(bild1)
		
			print (line)
		
		#---------------------------------------------------------------------------------------------------------------------
		#SquareWave
		elif measurement_info['waveform'] == "SquareWave":
			print "Mode:\t\tSquareWave"
			print "Stimulation:\tA=%.1fK\n\t\tf=%.1fmHz\n\t\tO=%.1fK" % (start_parameters[0], start_parameters[1]*1000, start_parameters[2])

			#Interpolation and plotting of data ----
			print(line)
			print("...plotting")
			print(line)
			# pre-fit plot
			tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
			bild1, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)

			show()
			
			
			
			#-----------------------------
			#UNDER CONSTRUCTION
			#-----------------------------
			
			
			
			print "give me fit range ..."
			inputs = ginput(2)
			
			t_idx_min = abs(Idata[:,0]-inputs[0][0]).argmin()
			t_idx_max = abs(Idata[:,0]-inputs[1][0]).argmin()
			
			Params = Parameters()
			Params.add('tau', value=14.0, min=0.1, max=30.0)
			Params.add('offs', value=1e-10)
			Params.add('B',value=1)

			#Params.add('T0', value=1)

			
			Results = minimize(expChy, Params, args=(Idata[t_idx_min:t_idx_max,0],Idata[t_idx_min:t_idx_max,1]), method="leastsq")

			ax2.plot(Idata[t_idx_min:t_idx_max,0], expChy(Params, Idata[t_idx_min:t_idx_max,0]), 'm-', label=r'I$_{Chynoweth}$')

			draw()
			
			B = Params['B'].value						#current faktor in exp. func (in A)
			I_offs = Params['offs'].value				#offset current which has to be subtractet from B
			
			print B
			
			d = 1e-3
			A = pi/4 * (0.015)**2
			nu_Peltier = 0.12							#Efficiency form Quickcool datasheet ... Qmax = 1.5W, Pmax = 1.2A*2.2V = 2.64W; Qmax/Pmax = 0.56
			P_Peltier = 0.35*0.56*nu_Peltier			#example PeltierPower in heating step
			F0 = P_Peltier/A							#heating power per sample area (W/m2)
			c = (0.06*4.1868)/1e-3						#specific heat capacacity (J/K*kg)
			rho = 7.45 * 1000							#density (converted to kg/m3)
			V = A*d										#sample volume (m3)
			m = rho*V									#samplle mass (kg)
			H = c*m										#heat capacity (J/K)
			
			
			
			p = ((B-I_offs)*H)/((A**2) * F0)
			
			
			saving_figure()
		
		#Power SquareWave - Experimental
		elif measurement_info['waveform'] == "PWRSquareWave":
			print "Mode:\t\tPower-SquareWave"
			#note: hier in Zukunft Angaben printen --- d.h. I_set und U_set muessen noch mitgelogt werden, freq. kann auch geschrieben werden!
			
			#plot of data------------------------------------------------------------------------
			print "--------------------------------"
			print "...plotting"
			print "--------------------------------"
			
						# pre-fit plot
			tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
			bild1, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)

			show()
			
			#-----------------------------
			#UNDER CONSTRUCTION
			#-----------------------------
			
			#-----------------------------------------------------------------------------------------------
			#give me some important values 
			#f=0.005	# has to be replaced later with reading from fileheader (measurement_info)
			f = measurement_info["freq"]
			
			#get number of indices in one period
			period_idx_size = int((1/f)*(1/interpolation_step))

			#how many periods are plotted
			perioden = (int(round((max(tinterpol)/(1/f))))-1)
			
			#select start time of fit
			print('select start point of fit:')
			start_time = ginput(1)[0][0]
			
			#start and end index for plotting and so on
			start_index = (abs(tinterpol - start_time)).argmin()
			end_index = start_index + (perioden * period_idx_size)
			
			#mean values after start_index
			T_mean = mean(Tinterpol[start_index:end_index])
			I_mean = mean(Iinterpol[start_index:end_index])
			
			#---------------------------------------------------------------------------------------------
			#modified Chynoweth fitting!
			
			#Fit Dicts for Temp and Current (lmfit package)
			TParams = Parameters()
			TParams.add('A',value=1)
			TParams.add('decay', value=20)
			TParams.add('offs', value=300)
			
			IParams = Parameters()
			IParams.add('A',value=1e-7)
			IParams.add('decay', value=TParams['decay'].value,vary=False)
			IParams.add('offs', value=1e-12)

			
			#getting area
			A = get_area()[0]
			pyro_coeffs = []
			F0s = []
			
			c = 627.9
			m = 8.139e-4
			
			C_LT = c * m
			
			t = arange(0,1/f/2,interpolation_step)
			
			#two fits in very period!
			for i in range(perioden*2):
				#indieces for full half period
				start = start_index + i*(period_idx_size/2)
				end = start_index + (i+1)*(period_idx_size/2)
			
				# getting tau of Temperature decay
								
				TResults = minimize(expdecay, TParams, args=(t,Tinterpol[start:end]), method="leastsq")
				ax1.plot(tinterpol[start:end], expdecay(TParams, t), 'b-', label=r'T$_{exp}$')
				TParams['decay'].vary=False
				
				print "--------------------------------"
				print "Period", i ,"Temp"
				print TParams['A'].value
				print TParams['decay'].value
				print TParams['offs'].value
				
				#consoleprint_fit(TParams,'Temperature - Period '+str(i+1))
				
				#indices for window in half period
				pre = 0.15 #ignoring ...% in front of window
				post = 0.50 #ignoring ...% after window
				
				window = 1 - pre - post
				
				pre_size = int(pre*period_idx_size/2)
				post_size = int(post*period_idx_size/2)
				window_size = int(window*period_idx_size/2)
				
				start = start_index + i*(period_idx_size/2) + (pre_size)
				end = start_index + (i+1)*(period_idx_size/2) - (post_size)

				
				IParams['decay'].value = TParams['decay'].value
				IResults = minimize(expdecay, IParams, args=(t[pre_size:pre_size+window_size],Iinterpol[start:end]), method="leastsq")
				ax2.plot(tinterpol[start:end], expdecay(IParams, t[pre_size:pre_size+window_size]), 'co-', label=r'I$_{Chynoweth}$')
				
				#consoleprint_fit(IParams,'Current - Period '+str(i+1))
				print "--------------------------------"
				print "Period", i ,"Curr"
				print IParams['A'].value
				print IParams['decay'].value
				print IParams['offs'].value
				
				draw()
				
				
			
				#Calcing p
				p = IParams['A'].value * (TParams['decay'].value)/(A*TParams['A'].value) #hier stimmt noch nicht ... eigentlich nicht /2
				pyro_coeffs.append(p)	
				print "--------------------------------"
				print "p:", p*1e6, "µC/Km2"
				
				#calcing F0
				F0 = (IParams['A'].value * C_LT) / (abs(p) * A**2)
				print "--------------------------------"
				print "calc F0:", F0, "W/m2"
				F0 = (TParams['A']*C_LT)/(TParams['decay']*A)
				F0s.append(F0)
				print "--------------------------------"
				print "calc F0:", F0, "W/m2"
			
			print "--------------------------------"
			print "Results"
			print "--------------------------------"
			print "mean p: ", mean(absolute(pyro_coeffs))*1e6, "µC/Km2"
			print "mean F0:", mean(absolute(F0s)), "W/m2"

		
			
			
			
			#I U from log
			strom = 0.02461166
			spannung = 0.03878719
			leistung = strom * spannung

			print "Driv. P :", leistung
			print "theo F0:", leistung/A
		
			
			
			#draw results

			#ax1.legend(title="temperatures", loc='upper left')
			#ax2.legend(title="currents", loc='lower right')
			
			#Calculation of p with original Chynoweth approach - not working!
			#I = 0.5 #A
			#U = 4.5 #V
			#nu = 1 #efficiency
			#area = get_area()
			#area = [(pi/4*0.01**2)]
			#F_0 = ((U*I*nu) / area[0])
			#F_01 = ((U*I) / area[0])
			#c = (0.15*4.1868)/1e-3 #converted from crystec data sheet for LN
			#rho = 4.65 * (1e-3/1e-6) #*1000
			#d = 1e-3 #1mm
			#V = area[0] * d
			#m = rho * V
			#C = c * m
			#Ansatz F0 für tau auszurechen --> auch nicht zufriedenstellend :(
			#F_0 = m * C * TParams['offs'].value / (IParams['decay'].value * area[0])
			#p = (IParams['A'].value * C)/(area[0]**2 * F_0)
			#print (p*1e6)
			
			#Calculation of p with modified Chynoweth Approach (by measuring T and using basic pyroelectric equation)
			#only valid, if tau_th = tau_el. in fit!
			#A = get_area()[0]
			#p = IParams['A'].value * TParams['decay'].value/(A*TParams['A'].value)
			
			#print (p*1e6)
			
			#What is now F0 of the setup?
			#Can someone calc C or other Parameters with original Chynoweth Approach? --> F0 waere zunaechst der interessantere Parameter
			
			
			#saving figure----------------------------------------------------------------------------------------------------
			saving_figure(bild)

	#-----------------------------------------------------------------------------------------------------------------------------
	#AutoPol
	elif measurement_info['hv_mode'] == "Polarize":
		print("Mode:\t\t%s" % measurement_info['hv_mode'])
		print("Temperature:\t%.2f K" % measurement_info['T_Limit_H'])
		if max(HVdata[:,1]) == 0:			# check if poling voltage was postive or negative
			PolVolt = min(HVdata[:,1])
		else:
			PolVolt = max(HVdata[:,1])
		print("max. Voltage:\t%.2f V" % PolVolt)
		print("Compliance:\t%.2e A" % HV_set[1])
		
		#Interpolation and plotting of data ----
		print(line)
		print("...plotting")
		print(line)

		head = date+"_"+samplename + "_AutoPol"
		bild = figure(head,figsize=fig_size)
		ax1 = subplot(111)
		ax2 = ax1.twinx()
		title(samplename+"_AutoPol",size=label_size)

		#Plot Voltage
		ax1.set_xlabel('Time (s)',size=label_size)
		ax1.set_ylabel('Voltage (V)',color=volt_color,size=label_size)
		ax1.grid(b=None, which='major', color='grey', linewidth=1)
		ax1.plot(HVdata[:,0], HVdata[:,1], color=volt_color,marker=".",linestyle="", label='set')
		ax1.plot(HVdata[:,0], HVdata[:,2], color=volt_color,marker="x",linestyle="", label='meas.')
		ax1.set_xlim(HVdata[0,0],HVdata[-1,0])
		ax1.set_ylim(min(HVdata[:,1]),max(HVdata[:,1])+10)
		ax1.legend(loc=4)
		ax1.locator_params(nbins=10)
		
		#Plot Current
		ax2.set_ylabel('Current (A)',color=curr_color,size=label_size)
		ax2.plot(Idata[:,0], Idata[:,1], color=curr_color,marker=".",linestyle="", label='Current')
		ax2.locator_params(nbins=10,axis = 'y')
		ax2.set_xlim(Idata[0,0],Idata[-1,0])

		bild.tight_layout()
		show()
		
		#get switch off time
		sw_off_index = abs(HVdata[:,1]-PolVolt).argmin()
		sw_off_time = HVdata[sw_off_index,0]

		box_string = u"Temperature: %.2f K\nMax. Volt.: %.2f V\nCurr. Compl.: %.3e A\nSwitch off time: %.2f s" % (measurement_info['T_Limit_H'],max(HVdata[:,1]),HV_set[1],sw_off_time)
		
		#Fit exponential decay
		input = raw_input("fit exponential decay? (y/n)")
		if input == "y":
			#console output and graphical input
			print("Select start of fit from graph.")
			input = ginput()
			print("...fitting")

			#getting starting time
			start_time = input[0][0]

			#interpolation
			HVinterpol = interp1d(HVdata[:,0], HVdata[:,1])
			Iinterpol = interp1d(Idata[:,0], Idata[:,1])
			tnew = arange(min(Idata[:,0]),max(HVdata[:,0]), 0.25)
			HVnew = HVinterpol(tnew)
			Inew = Iinterpol(tnew)

			start = abs(tnew-start_time).argmin()
			end = len(tnew)

			#fit
			expparams = Parameters()
			expparams.add('factor', value=1e-9)
			expparams.add('decay', value=2000, min=0.0)
			expparams.add('offs', value=45e-9)#, min=0.0)
			Polresults = minimize(expdecay, expparams, args=(tnew[start:end], Inew[start:end]), method="leastsq")

			#plot
			ax2.plot(tnew[start:end], expdecay(Polresults.params, tnew[start:end]), 'k-')
			
			fit_string = "\nExp. Decay. Fit:\nFactor: %.3e\nDecay time: %.2f s\nCurr Offset: %.3e A" % (Polresults.params['factor'].value, Polresults.params['decay'].value, Polresults.params['offs'].value)
			box_string = box_string + fit_string
			
			#console output
			print("Factor:\t%.2e\nDecay time:\t%.2f s\nCurr Offset:\t%.3e A" % (Polresults.params['factor'].value, Polresults.params['decay'].value, Polresults.params['offs'].value))

		else:
			if HV_set[1]>0:
				maxVolt = str(max(HVdata[:,1]))
			else:
				maxVolt = str(min(HVdata[:,1]))					
		
		box = plot_textbox(box_string)
		ax2.add_artist(box)
		
		draw()
			
		#save figure
		saving_figure(bild,pbild="Polarize")

	#-----------------------------------------------------------------------------------------------------------------------------
	#HighVoltage always on
	elif measurement_info['hv_mode'] == "On":
		#---------------------------------------------------------------------------------------------------------------------
		if T_profile == "Thermostat" or T_profile == "SineWave" or T_profile == "SineWave+LinRamp" or T_profile == "LinearRamp":
			print("Mode:\t\t%s" % measurement_info['waveform'])

			#Interpolation and plotting of data ----
			print(line)
			print("...plotting")
			print(line)

			# pre-fit plot
			tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
			bild1, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)
			
			if Resistance == True:
				for file in filelist:
					if file.endswith('HVsetVoltage-HVmeasVoltage.log'):
						filehandle = open(file)
						fileline = filehandle.readline()
						filehandle.close()
				
				V = float(fileline.split(' ')[-1].strip())
				R = V/abs(Inew)
				
				bild2 = figure('R(T)',figsize=fig_size)
				axR = bild2.add_subplot(111)
				
				axR.plot(Tnew[start_index:],R[start_index:],color=tubafgreen(),marker=".",linestyle="", label='Resistance')
				axR.set_xlim(Tnew[start_index,0],Tnew[-1,0])
				axR.set_yscale('log')
				
				axR.set_xlabel('Temperature (K)',size=label_size)
				axR.set_ylabel('Resistance ($\mathrm{\Omega}$)',size=label_size,color=tubafgreen())
					
				axR.grid()
				bild2.tight_layout()
				show()
				
				saving_figure(bild2,pbild='Resistance')

			if Formation == True:
				print('formation measurement set true!')
				
				#area for pyroel. coefficent
				area, area_error = get_area()
				print("Area: %e m2" % area)
				
				#important calculations for further fit;)---------------------------------------------------------------
				#check when ramp run into T_Limit_H
				if max(Tnew[:,0]) < measurement_info['T_Limit_H']:
					maxT_ind = Tnew[:,0]>max(Tnew[:,0])-1
				else:
					maxT_ind = Tnew[:,0]>(measurement_info['T_Limit_H']-1)
					
				if usecoolrate_flag == True:
					limit = len(Tnew[:,0])-1
					max_Temp = (tnew[limit]-tnew[0])*-measurement_info['cool_rate']+Tnew[0,0]
					T_perioden = int((tnew[limit]-tnew[0])/(fit_periods/measurement_info['freq']))
					tmax = tnew[limit]-tnew[0]
					satzlaenge = limit/T_perioden
				else:
					number_of_lim = maxT_ind.tolist().count(True)
					limit = len(Tnew[:,0])-number_of_lim-1
					max_Temp = (tnew[limit]+tnew[start_index])*measurement_info['heat_rate']+measurement_info['offs']
					T_perioden = int((tnew[limit]-tnew[start_index])/(fit_periods/measurement_info['freq']))
					tmax = tnew[limit]
				
				satzlaenge = (limit-start_index)/T_perioden
				#print(satzlaenge)

				print(line)
				print("...fitting")
				print(line)

				#prepare output log
				log = open(date+"_"+samplename+"_"+T_profile+"_T-Fit.txt", 'w+')
				
				#Temperature Fit -------------------------------------------------------------------------------------
				Tresult_down = fit(tnew, Tnew[:,0], start_index, len(Tnew[:,0])-1,1,measurement_info, True, True)
				#correction of phase and amplitudes
				Tparams_down = amp_phase_correction(Tresult_down.params)
				#extract params dict to lists
				Tfit_down, Terror_down = extract_fit_relerr_params(Tparams_down)
				#Fit-Plot
				ax1.plot(tnew[start_index:limit], sinfunc(Tparams_down, tnew[start_index:limit]), color=temp_color,linestyle='-', label='T-Fit')
				draw()
				#absolute T_high Error
				total_Terror_down = abs(Tparams_down['amp'].stderr/Tparams_down['amp'].value)+abs(Tparams_down['phase'].stderr/Tparams_down['phase'].value)+abs(Tparams_down['freq'].stderr/Tparams_down['freq'].value)+abs(Tparams_down['offs'].stderr/Tparams_down['offs'].value)+abs(Tparams_down['slope'].stderr/Tparams_down['slope'].value)
				#file output
				fileprint_fit(log,Tparams_down,"Temperature (Down)")
		
				#for top temperature-------------------
				if temp_filter_flag == False:
					Tresult_high = fit(tnew[:-5], Tnew[:,1], start_index, limit,1, measurement_info, True, True)
					#correction of phase and amplitude
					Tparams_high = amp_phase_correction(Tparams_high.params)
					#extract params dict to lists
					Tfit_high, Terror_high = extract_fit_relerr_params(Tparams_high)
					#plot of second fit
					ax1.plot(tnew[start_index:-5], sinfunc(Tparams_high, tnew[start_index:-5]), color=volt_color,linestyle='-', label='T-Fit (top)')
					draw()
					#absolute T_high Error
					total_Terror_high = abs(Tparams_high['amp'].stderr/Tparams_high['amp'].value)+abs(Tparams_high['phase'].stderr/Tparams_high['phase'].value)+abs(Tparams_high['freq'].stderr/Tparams_high['freq'].value)+abs(Tparams_high['offs'].stderr/Tparams_high['offs'].value)+abs(Tparams_high['slope'].stderr/Tparams_high['slope'].value)
					#file output
					fileprint_fit(log,Tparams_high,"Temperature (High)")

				leg_T = ax1.legend(loc="upper right",title='Temperatures')
				#ax2.add_artist(leg_T)
				draw()
				
				log.close()
			      
				print("Temperature ... done!")
				print("Current...")

				#Current Fit -----------------------------------------------------------------------------------------
				#initialize fit variables
				I_perioden = int(tnew[limit]/(fit_periods/measurement_info['freq']))
				satzlaenge = limit/I_perioden

				Ifit = zeros((1,5))
				Ierror = zeros((1,5))
				
				Iparams = Parameters()
				Iparams.add('amp', value=1e-11)
				Iparams.add('freq', value=Tfit_down[1], min=1e-5, max=0.2, vary=False)
				Iparams.add('phase', value=1.0)
				Iparams.add('offs', value=1e-10)
				Iparams.add('slope', value=1e-10)
				
				Iparams_lin = Parameters()
				Iparams_lin.add('a', value=1e-10)
				Iparams_lin.add('b', value=0.0)

				#perform partial fits
				for i in arange(1,I_perioden):
					start = start_index+int((i*satzlaenge)-satzlaenge)
					ende = start_index+int(i*satzlaenge)
					
					# avaraging singnal part to get algebraic sign (oscillation around pos/neg value?)
					meanI = mean(Inew[start:ende])
					if meanI < 0.0:
						polarityI = "neg"
					else:
						polarityI = "pos"
					
					#fit of sin and lin func
					Iresult_sin = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					Iresult_lin = minimize(linear, Iparams_lin, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
					
					#Repeat Feature if lin. Feat is better than sine fit
					Ifit_counter = 1
					if Iresult_lin.redchi < 2*Iresult_sin.redchi and Ifit_counter < Ifit_counter_limit:
						
						Iparams['amp'].value = (Ifit_counter)*1e-12
						Iparams['phase'].value = Tfit_down[2]-pi/2
						#Iparams['offs'].value = (Ifit_counter**2)*1e-10
						#Iparams['slope'].value = (Ifit_counter**2)*1e-10
						
						Iresult_sin = minimize(sinfunc, Iparams, args=(tnew[start:ende], Inew[start:ende]), method="leastsq")
						
						Ifit_counter =  Ifit_counter + 1

					#print i, Ifit_counter
					sys.stdout.write("\rProgress: %d/%d; %.0f %% Rep.: %d" % (i,I_perioden-1,100*float(i)/float(I_perioden-1),Ifit_counter))
					sys.stdout.flush()
					
					#fit correction (amp/phase)
					Iparams = amp_phase_correction(Iresult_sin.params)	
					
					#plot of sin and line fit
					ax2.plot(tnew[start:ende], sinfunc(Iparams, tnew[start:ende]), 'r-')
					#ax2.plot(tnew[start:ende], linear(Iparams_lin, tnew[start:ende]), 'r--')
				
					#extract params dict to lists
					Ifit_temp, Ierror_temp = extract_fit_relerr_params(Iparams)
					if i==1:
						Ifit = array([Ifit_temp])
						Ierror = array([Ierror_temp])
						Iresults = [Iresult_sin]					#save lmfit minizimer objects for future purpose ... maybe
					else:
						Ifit = append(Ifit,[array(Ifit_temp)],axis=0)
						Ierror = append(Ierror,[array(Ierror_temp)],axis=0)
						Iresults.append(Iresult_sin)

					#calculate phase difference
					if single_crystal==False:
						phi_T = Tfit_down[2]
						phi_I = Ifit[i-1,2]
						
						# if abs(phi_I) > abs(phi_T):
							# phasediff = phase_correction(phi_I-phi_T)
						# else:
							# phasediff = phase_correction(phi_T-phi_I)
						
						phasediff = phase_correction(phi_T-phi_I)
					else:
						phasediff = -pi/2

					#NonPyroStrom---------------------------------------------------------------------------
					#m=magenta (TSC-Strom)
					
					#Plot
					nonpyroparams = Parameters()
					if polarityI == "neg":
						nonpyroparams.add('amp', value=-1*abs(Ifit[i-1,0]*-cos(phasediff)))
					else:
						nonpyroparams.add('amp', value=abs(Ifit[i-1,0]*-cos(phasediff)))
					nonpyroparams.add('freq', value=Tfit_down[1])
					nonpyroparams.add('phase', value=Tfit_down[2])
					nonpyroparams.add('offs', value=Ifit[i-1,3])
					nonpyroparams.add('slope', value=Ifit[i-1,4])
					nonpyroparams = amp_phase_correction(nonpyroparams)
					ax2.plot(tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende]), color=np_color,linestyle='-')
					
					#Calculating Data from Fit - TSC
					if calculate_data_from_fit_flag == True:
						TSC = (array([tnew[start:ende], sinfunc(nonpyroparams, tnew[start:ende])])).T		#transpose!
						if i==1:
							I_TSC = TSC
						else:
							I_TSC = append(I_TSC, TSC, axis=0)

					#Pyrostrom + Koeff.---------------------------------------------------------------------
					#c=cyan (Pyrostrom)
					
					#Plot
					pyroparams = Parameters()
					pyroparams.add('amp', value=Ifit[i-1,0]*-sin(phasediff))
					pyroparams.add('freq', value=Tfit_down[1])
					pyroparams.add('phase', value=(Tfit_down[2]+pi/2))
					pyroparams.add('offs', value=Ifit[i-1,3])
					pyroparams.add('slope', value=Ifit[i-1,4])
					pyroparams = amp_phase_correction(pyroparams)
					ax2.plot(tnew[start:ende], sinfunc(pyroparams, tnew[start:ende]), color=p_color,linestyle='-')
					
					#Calculating Data from Fit - Pyro
					if calculate_data_from_fit_flag == True:
						pyro = (array([tnew[start:ende], sinfunc(pyroparams, tnew[start:ende])])).T		#transpose!
						if i==1:
							I_pyro = pyro
						else:
							I_pyro = append(I_pyro, pyro, axis=0)
					
					#Calc p
					time = mean(tnew[start:ende])
					Temp = (tnew[start_index+(i-1)*satzlaenge] + tnew[start_index+(i*satzlaenge)])/2 * Tfit_down[4] + Tfit_down[3]					
					p_SG = (Ifit[i-1,0]*-sin(phasediff))/(area*Tfit_down[0]*2*pi*abs(Tfit_down[1]))						# p (Sharp-Garn) ... with - sin() ! (see manual) ;)
					p_BR = (abs(mean(Idata[start:ende,1]))/(area*Tfit_down[4]))												# p (Byer-Roundy)
					perror = p_SG * rel_err(Tfit_down,Terror_down,Ifit[i-1],Ierror[i-1],area, area_error,phasediff,Xsigma=sigma)
					phasediff = degrees(phasediff)																							# Phasediff. (in deg)
					Ip_TSC_ratio= abs((Ifit[i-1,0]*-sin(radians(phasediff)))/(Ifit[i-1,0]*cos(radians(phasediff))))	# ratio Pyro/TSC
					meanI = mean(Idata[start:ende,1])																					# mean I in Interval
					Chisqr = Iresults[i-1].chisqr																								# Chi square in Interval

					#wrinting temp list
					p_temp = [time, Temp, p_SG, p_BR, phasediff, Ip_TSC_ratio, meanI, Chisqr, perror]
					#append list to array 
					if i==1:
						p = array([p_temp])
						p_error = array([perror])
					else:
						p = append(p, [array(p_temp)], axis=0)
						p_error = append(p_error,perror)

				current_custom_legend(ax2)
				draw()
				
				header_string = "Amp [I]\t\t\tFreq [Hz]\t\t\tPhase [rad]\t\t\tOffset [A]\t\t\tSlope [A/s]\t\t\tAmp_Err [A]\t\t\tFreq_Err [Hz]\t\t\tPhase_Err [rad]\t\t\tOffs_Err [A]\t\t\tSlope_Err [A/s]"
				savetxt(date+"_"+samplename+"_"+T_profile+"_I-Fit.txt",hstack([Ifit,Ierror]), delimiter="\t", header=header_string)
				print "\nCurrent ... done!"
				print line
				
				#Plotting p(T)-----------------------------------------------------------------------------------------------------------
				bild2=figure(date+"_"+samplename+"_"+T_profile+'_Pyro', figsize=fig_size)

				#p(T)--------------------------------------------------------------
				ax3=subplot(221)
				ax3.set_autoscale_on(True)
				ax3.set_xlim(p[0,0],p[-1,0])
				#ax3.set_ylim(min(p[:,2])*1e6-50, max(p[:,2])*1e6+50)
				ax3.set_xlabel('Time (s)',size=label_size)
				ax3.set_ylabel(u"$p$ (µC/Km²)",color=temp_color,size=label_size)

				ax3.grid(b=None, which='major', axis='both', color='grey')
				ax3.errorbar(p[:,0],(p[:,2]*1e6), yerr=p_error[:]*1e6, color=temp_color,marker=".",linestyle="", elinewidth=None, capsize=3, label='p (SG)')
				
				#p/TSC ration---------------------------------------------------------
				ax5=subplot(222,sharex=ax3)
				ax5.set_autoscale_on(True)
				ax5.set_xlim(ax3.get_xbound())
				ax5.grid(b=None, which='major', axis='both', color='grey')
				ax5.set_xlabel('Time (s)',size=label_size)
				ax5.set_ylabel(r"I$_{p}$/I$_{np}$",color=volt_color,size=label_size)
				ax5.semilogy(p[:,0], p[:,5], color=volt_color,marker=".",linestyle="", label=r"I$_{p}$/I$_{np}$")

				#Chisqr---------------------------------------------------------------
				ax6=subplot(224,sharex=ax3)
				ax6.set_autoscale_on(True)
				ax6.set_xlim(ax3.get_xbound())
				ax6.grid(b=None, which='major', axis='both', color='grey')
				ax6.set_xlabel('Time (s)',size=label_size)
				ax6.set_ylabel(r"$X^2$",color=np_color,size=label_size)
				ax6.semilogy(p[:,0], p[:,7], color=np_color,marker=".",linestyle="", label=r"$X^2$")

				#Phasediff---------------------------------------------------------------
				ax7=subplot(223,sharex=ax3)
				#ax7.set_autoscale_on(True)
				ax7.set_xlim(ax3.get_xbound())
				ax7.set_ylim(0,360)
				ax7.axhline(180, color='k')
				ax7.axhline(90, color='k',linestyle='--')
				ax7.axhline(270, color='k', linestyle='--')
				ax7.grid(b=None, which='major', axis='both', color='grey')
				ax7.set_xlabel('Time (s)',size=label_size)
				ax7.set_ylabel(ur"$\phi$ (°)",color=other,size=label_size)
				ax7.plot(p[:,0],p[:,4],color=other,marker=".",linestyle="", label="Phasediff.")
				
				#CurrAmp---------------------------------------------------------------
				ax8 = ax7.twinx()
				ax8.set_xlim(ax3.get_xbound())
				ax8.plot(p[:,0],Ifit[:,0], color=curr_color,marker=".", linestyle="", label="Amplitude")
				ax8.set_ylabel(r"$I_{\mathrm{Amp}}$ (A)",color=curr_color,size=label_size)

				bild2.tight_layout()
				show()
				
				#Saving results and figs------------------------------------------------------------------------------
				saving_figure(bild1)
				saving_figure(bild2, pbild=True)

				#writing log files
				print line
				print "...writing log files"				
				header_string = "time [s]\t\t\tTemp [K]\t\t\tp_SG [C/Km2]\t\t\tp_BR [C/Km2],\t\t\tPhasediff [deg]\t\t\tp/TSC-ratio\t\t\tMean I [A]\t\t\tRed Chi\t\t\t\tp_err [C/Km2]\t"			
				savetxt(date+"_"+samplename+"_"+T_profile+"_"+"PyroData.txt", p, delimiter="\t", header=header_string)
				
				if calculate_data_from_fit_flag == True:
					header_string = "time [s]\t\t\tI_TSC [A]\t\t\tI_pyro [A]"
					savetxt(date+"_"+samplename+"_"+T_profile+"_"+"DataFromFit.txt", vstack([I_TSC[:,0], I_TSC[:,1], I_pyro[:,1]]).T, delimiter="\t", header=header_string)

			else:
				saving_figure(bild1)

		elif T_profile == "TriangleHat+SineWave" or T_profile == "TriangleHat":
			print("Mode:\t\t"+measurement_info['waveform'])
			
			if Resistance == True:

				#Interpolation and plotting of data ----
				print(line)
				print("...plotting")
				print(line)
	
				# pre-fit plot
				tnew, Tnew, Inew = interpolate_data(Tdata, Idata, interpolation_step, temp_filter_flag)
				bild1, ax1, ax2 = plot_graph(tnew, Tnew, Inew, T_profile)
				
				if Resistance == True:
					for file in filelist:
						if file.endswith('HVsetVoltage-HVmeasVoltage.log'):
							filehandle = open(file)
							fileline = filehandle.readline()
							filehandle.close()
					
					V = float(fileline.split(' ')[-1].strip())
					R = V/abs(Inew)
					
					turning_point_index = argmax(Tnew[:,0])
					
					bild2 = figure('R(T)',figsize=fig_size)
					axR = bild2.add_subplot(111)
					
					axR.plot(Tnew[start_index:turning_point_index],R[start_index:turning_point_index],color=tubafgreen(),marker=".",linestyle="", label='heating')
					axR.plot(Tnew[turning_point_index:],R[turning_point_index:],color=tubafcyan(),marker=".",linestyle="", label='cooling')

					axR.set_xlim(Tnew[start_index,0],Tnew[turning_point_index,0])
					axR.set_yscale('log')
					axR.legend(loc=1)
				
					axR.set_xlabel('Temperature (K)',size=label_size)
					axR.set_ylabel('Resistance ($\mathrm{\Omega}$)',size=label_size,color=tubafgreen())
					
					axR.grid()
					bild2.tight_layout()
					show()
					
					saving_figure(bild1)
					saving_figure(bild2,pbild='Resistance')
					
			else:
				print('Just resistance calculation implemented in these waveforms implemented yet!')

		else:
			print("Mode not implemented yet ...")

	#-----------------------------------------------------------------------------------------------------------------------------
	#for every other
	else:
		pass
ioff()