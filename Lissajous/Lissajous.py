#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:59:41 2018

@author: svenjachalke
"""

from pylab import *
from tubafcdpy import *
colors = [tubafblue(),tubafcyan(),tubafgreen(),tubaforange(),tubafred()]
style.use('science')


f = figure(figsize=(6,6))
ax = f.add_subplot(111)

t = arange(0,101,1)

A1 = 1
f1 = 0.01
p1 = 0

A2 = 1
f2 = 0.01
p2 = linspace(pi/2,0,5) 


func1 = A1 * sin(2*pi*f1 * t + p1)

for i in arange(len(p2)):
	ang = p2[i]*180/pi
	func2 = A2 * sin(2*pi*f2 * t + p2[i])
	ax.plot(func1, func2, label=r'%.1f\,$^{\circ}$'%ang,linewidth=1.2,color=colors[i])

ax.set_xticks([])
ax.set_yticks([])

ax.legend(loc=2)

f.tight_layout()

f.savefig('Lissajous.png',dpi=200)
f.savefig('Lissajous.eps')
f.savefig('Lissajous.pdf')


#ax.plot(t,f1)
#ax.plot(t,f2)

