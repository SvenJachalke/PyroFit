# -*- coding: utf-8 -*-
# Plotting of all p(T) in a given folder - console Version
# date:	2014-07-29
# author:	Sven Jachalke

#import of modules
import pylab as pl
import glob
import os

#General Settings ---------------------------------------------------------------
pl.ion()

#Welcome Message ----------------------------------------------------------------
print "-------PolyPyroPlot----------------------------------------"


#Creating Plot ------------------------------------------------------------------
#Looking for SinLinRamp Folders
foldercontent = os.listdir('.') #'.' gives current relative path
sinewavefolders = []
for i in range(len(foldercontent)):
    if 'SinLinRamp' in foldercontent[i]:
        sinewavefolders.append(foldercontent[i])
sinewavefolders.sort() #ascending sorting of the folders

#loading pyro data and plot
linedict = {}
labellist = []
skip = 1 							#how many points have to be skipped?
bild = pl.figure('p(T)')
ax = pl.subplot(111)
colorlist = pl.array([(0,0,1), (0,0.5,0), (1,0,0), (0,0.75,0.75), (0.75,0,0.75), (0.75,0.75,0), (0,0,0), (0.0,1.0,0.5), (0.5,1.0,0.0), (1.0,0.5,0.0)])

for i in range(len(sinewavefolders)):
	#print("%d : %s" % (i+1,sinewavefolders[i]))
	#getting data
	os.chdir(sinewavefolders[i])
	filelist = glob.glob('*.txt')
	#print i, sinewavefolders[i], os.path.abspath('.')
	for f in range(len(filelist)):
		if 'p-Fits.txt' in filelist[f]:
			p = pl.loadtxt(filelist[f],skiprows = 6)
			coloritem = pl.mod(i,len(colorlist))
			line = pl.plot(p[::skip,0],p[::skip,1]*1e6,label=sinewavefolders[i],color=colorlist[coloritem])
			try:
				linedict.update({i+1:line})
				labellist.append(sinewavefolders[i])
			except NameError:
				pass

	os.chdir('..')

ax.set_xlabel(r'Temperature [K]')
ax.set_ylabel(r'pyroel. Coefficient [$\mu$C/Km$^2$]')
figure_legend = ax.legend(loc=3)
ax.grid()
pl.show()

#Menue -----------------------------------------------------------------
if len(sinewavefolders) == len(labellist):
	while True:
		print "-----------------------------------------------------------"
		print "Plotted Lines:"
		for i in range(len(linedict)):
			if linedict[i+1][0].get_visible() == True:
				print("%d : %s" % (i+1,labellist[i]))
		print "-----------------------------------------------------------"
		print "What to do?"
		print "1 - delete line(s)"
		print "2 - add line(s)"
		print "3 - rename line label(s)"
		print "4 - save figure"
		print "0 - exit"
		top_menu_nr = raw_input()

		# deleting lines in plot
		if top_menu_nr == "1":
			erase_line_nrs = raw_input("Line Nr(s):")
			plot_numbers = [int(s) for s in erase_line_nrs.split(',')]
			try:
				for i in range(len(plot_numbers)):
					linedict[plot_numbers[i]][0].set_visible(False)
					ax.lines.remove(linedict[plot_numbers[i-1]][0])
					figure_legend.set_visible(False)
					figure_legend = ax.legend()
					#ax.autoscale(True)
					pl.draw()
			except ValueError:
				print "invalid number!"
			except KeyError:
				print "invalid number!"
			
				#linedict[plot_numbers[i-1]][0].set_visible(False)
			
			bild.canvas.draw()

		# adding lines in plot
		elif top_menu_nr == "2":
			#print "not implemented yet!"
			print "not plotted lines:"
			for i in range(len(linedict)):
				if linedict[i+1][0].get_visible() == False:
					print("%d : %s" % (i+1,labellist[i]))
			
			add_line_nrs = raw_input("Line Nr(s):")
			plot_numbers = [int(s) for s in add_line_nrs.split(',')]
			for i in range(len(plot_numbers)):
				if linedict[plot_numbers[i]][0].get_visible() == True:
					print("Line %d already plotted"% plot_numbers[i])
				else:
					linedict[plot_numbers[i]][0].set_visible(True)
					ax.lines.append(linedict[plot_numbers[i]][0])
					#ax.lines.sort()
					figure_legend.set_visible(False)
					figure_legend = ax.legend()
					pl.draw()

		# costom legend labels
		elif top_menu_nr == "3":
			rename_label_nrs = raw_input("Line Nr(s):")
			if rename_label_nrs == 'all':
				label_numbers = pl.arange(1,len(linedict)+1)
			else:
				label_numbers = [int(s) for s in rename_label_nrs.split(',')]
			
			print "Enter new label name:"
			try:
				for i in range(len(label_numbers)):
					new_label = raw_input("Line %d:"%label_numbers[i])
					linedict[label_numbers[i]][0].set_label(new_label)
					labellist[label_numbers[i]-1] = new_label
					figure_legend.set_visible(False)
					figure_legend = ax.legend()
					pl.draw()
			except KeyError:
				print "invalid number!"

		elif top_menu_nr == "4":
			figure_name = raw_input("Enter figure name: ")
			if figure_name == "":
				figure_name = "p(T)-Zsf"
			bild.savefig(figure_name+".png", dpi=300, transparent=False)
			print "... figure saved"
		# exit
		else:
			print "Bye!"
			break
else:
	print "No PyroFit-Data in one or more folders!\nPlease run PyroFit in each folder first and try again!"
