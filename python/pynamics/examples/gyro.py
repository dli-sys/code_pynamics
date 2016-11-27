# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
#pynamics.script_mode = True
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body,Body2
from pynamics.dyadic import Dyadic
from pynamics.output import Output
from pynamics.particle import Particle

#import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()


m = Constant('m',.1,system)
r = Constant('r',.2,system)
L = Constant('L',.2,system)

g = Constant('g',9.81,system)

tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]
w = 300*2*pi/60

qA,qA_d,qA_dd = Differentiable(system,'qA')
qB,qB_d,qB_dd = Differentiable(system,'qB')
wC,wC_d,qC_dd = Differentiable(system,'wC')

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[qB]=20*pi/180
initialvalues[qB_d]=0
initialvalues[wC]=0*pi/180
#initialvalues[qC_d]=300*2*pi/60

statevariables = [qA,qA_d,qB,qB_d,wC]
statevariables = statevariables[:-1]
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')
B = Frame('B')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],-qA,system)
B.rotate_fixed_axis_directed(A,[1,0,0],-qB,system)

Origin=0*N.x
pCcm=Origin + L*B.z
vCcm = pCcm.diff_in_parts(N,system)

II = 1/4*m*r**2

IC = Dyadic.build(B,II,II,2*II)

#BodyC = Body2('BodyC',B,pCcm,vCcm,wC*B.z,m,IC,system)

#system.addforcegravity(-g*N.y)
#
#KE = system.KE
#PE = system.getPEGravity(Origin) - system.getPESprings()
#    
#pynamics.tic()
#print('solving dynamics...')
#f,ma = system.getdynamics()
#print('creating second order function...')
#func1 = system.state_space_post_invert(f,ma)
#print('integrating...')
#states=scipy.integrate.odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14)
#pynamics.toc()
#print('calculating outputs..')
#output = Output([KE-PE,qA,qB],system)
#y = output.calc(states)
#pynamics.toc()
#
#plt.figure()
#plt.hold(True)
#plt.plot(t,y[:,0])
#plt.show()
#
#plt.figure()
#plt.hold(True)
#plt.plot(t,y[:,1])
#plt.show()
#
#plt.figure()
#plt.hold(True)
#plt.plot(t,y[:,2]*180/pi)
#plt.show()
#
