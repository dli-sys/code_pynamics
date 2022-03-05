# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
import pynamics.integration

#import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)

lA = Constant(1,'lA',system)
lB = Constant(1,'lB',system)
lC = Constant(1,'lC',system)

mA = Constant(1,'mA',system)
mB = Constant(1,'mB',system)
mC = Constant(1,'mC',system)

g = Constant(9.81,'g',system)
b = Constant(1e2,'b',system)
k = Constant(1e2,'k',system)

preload1 = Constant(0*pi/180,'preload1',system)
preload2 = Constant(0*pi/180,'preload2',system)
preload3 = Constant(0*pi/180,'preload3',system)

Ixx_A = Constant(1,'Ixx_A',system)
Iyy_A = Constant(1,'Iyy_A',system)
Izz_A = Constant(1,'Izz_A',system)
Ixx_B = Constant(1,'Ixx_B',system)
Iyy_B = Constant(1,'Iyy_B',system)
Izz_B = Constant(1,'Izz_B',system)
Ixx_C = Constant(1,'Ixx_C',system)
Iyy_C = Constant(1,'Iyy_C',system)
Izz_C = Constant(1,'Izz_C',system)

qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[qB]=0*pi/180
initialvalues[qB_d]=0*pi/180
initialvalues[qC]=0*pi/180
initialvalues[qC_d]=0*pi/180

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(N,[0,0,1],qB,system)
C.rotate_fixed_axis(N,[0,0,1],qC,system)

pNA=0*N.x
pAB=pNA+lA*A.x
pBC = pAB + lB*B.x
pCtip = pBC + lC*C.x

pAcm=pNA+lA/2*A.x
pBcm=pAB+lB/2*B.x
pCcm=pBC+lC/2*C.x

wNA = N.get_w_to(A)
wAB = A.get_w_to(B)
wBC = B.get_w_to(C)

IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

BodyA = Body('BodyA',A,pAcm,mA,IA,system)
BodyB = Body('BodyB',B,pBcm,mB,IB,system)
#BodyC = Body('BodyC',C,pCcm,mC,IC,system)
BodyC = Particle(pCcm,mC,'ParticleC',system)

vSoil = -1*N.y

vAcm = pAcm.time_derivative()
vBcm = pBcm.time_derivative()
vCcm = pCcm.time_derivative()

k_perp = 1
k_par = 1/3

faperp = k_perp*vSoil.dot(A.y)*A.y
fapar= k_par*vSoil.dot(A.x)*A.x
fbperp = k_perp*vSoil.dot(B.y)*B.y
fbpar= k_par*vSoil.dot(B.x)*B.x
fcperp = k_perp*vSoil.dot(C.y)*C.y
fcpar= k_par*vSoil.dot(C.x)*C.x

system.addforce((faperp+fapar),vAcm)
system.addforce((fbperp+fbpar),vBcm)
system.addforce((fcperp+fcpar),vCcm)
system.addforce(-b*wNA,wNA)
system.addforce(-b*wAB,wAB)
system.addforce(-b*wBC,wBC)

system.add_spring_force1(k,(qA-preload1)*N.z,wNA) 
system.add_spring_force1(k,(qB-qA-preload2)*N.z,wAB)
system.add_spring_force1(k,(qC-qB-preload3)*N.z,wBC)

# system.addforcegravity(-g*N.y)

f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma)

tol = 1e-5
tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

def myfunc(x):
    b1,k1 = x
    constants = system.constant_values
    constants[b] = b1
    constants[k] = k1
    states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=tol,atol=tol, args=({'constants':constants},))
    return states, constants

points = [pNA,pAB,pBC,pCtip]

states0,constants0 = myfunc([1e1,1e2])
points_output = PointsOutput(points,system,constant_values = constants0)
y = points_output.calc(states0,t)
points_output.plot_time()
#points_output.animate(fps = fps,movie_name = 'triple_pendulum_rft.mp4',lw=2)

