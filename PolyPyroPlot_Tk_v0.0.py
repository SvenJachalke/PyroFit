# -*- coding: utf-8 -*-
# Plotting of all p(T) in a given folder
# date:	2014-07-29
# author:	Sven Jachalke

#import of modules
from pylab import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
import glob
import os
import sys
#version check of Tkinter
if sys.version_info[0] < 3:
  import Tkinter as Tk
else:
  import tkinter as Tk

root = Tk.Tk()
root.title('PolyPyroPlot')

#Creating Plot ---------------------------------------------------
#Looking for SinLinRamp Folders
foldercontent = os.listdir('.') #'.' gives current relative path
sinewavefolders = []
for i in range(len(foldercontent)):
    if 'SinLinRamp' in foldercontent[i]:
        sinewavefolders.append(foldercontent[i])
sinewavefolders.sort() #ascending sorting of the folders

#loading pyro data and plot
linedict = {}
skip = 1 							#how many points have to be skipped?
bild = figure('p(T)')

for i in range(len(sinewavefolders)):
    #getting data
    os.chdir(sinewavefolders[i])
    filelist = glob.glob('*.txt')
    #print i, sinewavefolders[i], os.path.abspath('.')
    for f in range(len(filelist)):
        if 'p-Fits.txt' in filelist[f]:
            p = loadtxt(filelist[f],skiprows = 6)
            line = plot(p[::skip,0],p[::skip,1]*1e6,label=sinewavefolders[i])
            #line = plot(p[::skip,0],p[::skip,1]*1e6,label=labellist[i])
            linedict.update({f+1:line})
    
    #linelist.append(line)
    os.chdir('..')

#deactivate uninteressting lines
#plot_numbers = [1]
#for i in range(len(plot_numbers)):
#    linedict[labellist[plot_numbers[i]-1]][0].set_visible(False)   
    
#cosmetics
xlabel(r'Temperature [K]')
ylabel(r'pyroel. Coefficient [$\mu$C/Km$^2$]')
legend(loc=3)
grid()


# a tk.DrawingArea ------------------------------------
canvas = FigureCanvasTkAgg(bild, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg( canvas, root )
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

def on_key_event(event):
    print('you pressed %s'%event.key)
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect('key_press_event', on_key_event)

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)

Tk.mainloop()








#creating GUI with standart Tkinter
#class PolyPyroPlot_App(Tkinter.Tk):
 # def __init__(self,parent):
  #  Tkinter.Tk.__init__(self,parent)
   # self.parent = parent
    #self.initialize()
  #def initialize(self):
   # pass

#if __name__ == "__main__":
 # app = PolyPyroPlot_App(None)
  #app.title('PolyPyroPlot')
  #app.mainloop()



#labellist = ['05 - 1.Seite - ungepolt (initial)', #05
 #            '06 - 2.Seite - ungepolt', #06
  #           '07 - 1.Seite - ungepolt', #07
   #          '09 - 1.Seite - +1000V', #09
    #         '11 - 1.Seite - -1000V', #11
     #        '13 - 2.Seite - +1000V', #13
      #       '15 - 2.Seite - -1000V', #15
       #      '16 - 2.Seite - ungepolt',
        #     '17 - 2.Seite - ungepolt (fein)'
         #    ] 


#show()

