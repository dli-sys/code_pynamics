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
import scipy.interpolate

import sympy
import numpy
import matplotlib.pyplot as plt
# plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)

global_q = True
tol = 1e-7
lA = Constant(1,'lA',system)
lB = Constant(1,'lB',system)
lC = Constant(1,'lC',system)

mA = Constant(1,'mA',system)
mB = Constant(1,'mB',system)
mC = Constant(1,'mC',system)

g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e1,'k',system)

tinitial = 0
tfinal = 5
tstep = 1/1000
t = numpy.r_[tinitial:tfinal:tstep]


qA_exp = t**3/100
qB_exp = 2*t**3/100
qC_exp = 3*t**3/100

qA_d_exp = (qA_exp[2:]-qA_exp[:-2])/(2*tstep)
qB_d_exp = (qB_exp[2:]-qB_exp[:-2])/(2*tstep)
qC_d_exp = (qC_exp[2:]-qC_exp[:-2])/(2*tstep)

qA_dd_exp = (qA_d_exp[2:]-qA_d_exp[:-2])/(2*tstep)
qB_dd_exp = (qB_d_exp[2:]-qB_d_exp[:-2])/(2*tstep)
qC_dd_exp = (qC_d_exp[2:]-qC_d_exp[:-2])/(2*tstep)

qA_d_exp = qA_d_exp[1:-1]
qB_d_exp = qB_d_exp[1:-1]
qC_d_exp = qC_d_exp[1:-1]

qA_exp = qA_exp[2:-2]
qB_exp = qB_exp[2:-2]
qC_exp = qC_exp[2:-2]

t = t[2:-2]
states_exp = numpy.c_[qA_exp,qB_exp,qC_exp,qA_d_exp,qB_d_exp,qC_d_exp,qA_dd_exp,qB_dd_exp,qC_dd_exp]

q_exp = numpy.array([qA_exp,qB_exp,qC_exp]).T
plt.figure()
plt.plot(t,q_exp)
plt.title('q_exp')


q_dd_exp = numpy.array([qA_dd_exp,qB_dd_exp,qC_dd_exp]).T
plt.figure()
plt.plot(t,q_dd_exp)
plt.title('q_dd_exp')



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

t1 = sympy.Symbol('t1')
t2 = sympy.Symbol('t2')
t3 = sympy.Symbol('t3')

initialvalues = {}
initialvalues[qA]=qA_exp[0]
initialvalues[qB]=qB_exp[0]
initialvalues[qC]=qC_exp[0]
initialvalues[qA_d]=qA_d_exp[0]
initialvalues[qB_d]=qB_d_exp[0]
initialvalues[qC_d]=qC_d_exp[0]

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)

system.set_newtonian(N)
if not global_q:
    A.rotate_fixed_axis(N,[0,0,1],qA,system)
    B.rotate_fixed_axis(A,[0,0,1],qB,system)
    C.rotate_fixed_axis(B,[0,0,1],qC,system)
else:
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

# system.addforce(-b*wNA,wNA)
# system.addforce(-b*wAB,wAB)
# system.addforce(-b*wBC,wBC)


system.addforce(t1*N.z,wNA)
system.addforce(t2*N.z,wAB)
system.addforce(t3*N.z,wBC)


# if not global_q:
#     system.add_spring_force1(k,(qA-preload1)*N.z,wNA) 
#     system.add_spring_force1(k,(qB-preload2)*A.z,wAB)
#     system.add_spring_force1(k,(qC-preload3)*B.z,wBC)
# else:
#     system.add_spring_force1(k,(qA-preload1)*N.z,wNA) 
#     system.add_spring_force1(k,(qB-qA-preload2)*N.z,wAB)
#     system.add_spring_force1(k,(qC-qB-preload3)*N.z,wBC)

system.addforcegravity(-g*N.y)

vCtip = pCtip.time_derivative(N,system)

eq = []
# eq.append(pCtip.dot(N.y))
eq_d=[(system.derivative(item)) for item in eq]
eq_dd=[(system.derivative(item)) for item in eq_d]

f,ma = system.getdynamics()

f = sympy.Matrix(f)
f = f.subs(system.constant_values)

ma = sympy.Matrix(ma)
ma = ma.subs(system.constant_values)

zero = f-ma
torques = (t1,t2,t3)
sol = sympy.solve(zero,torques)
sol2 = [sol[item] for item in torques]
# sol2 = sympy.Matrix([sol[item] for item in torques])
f_torques = sympy.lambdify(system.get_q(0)+system.get_q(1)+system.get_q(2),sol2)
res = numpy.array(f_torques(*(states_exp.T))).T

ft1 = scipy.interpolate.interp1d(t,res[:,0],fill_value = 'extrapolate', kind='quadratic')
ft2 = scipy.interpolate.interp1d(t,res[:,1],fill_value = 'extrapolate', kind='quadratic')
ft3 = scipy.interpolate.interp1d(t,res[:,2],fill_value = 'extrapolate', kind='quadratic')

plt.figure()
plt.plot(t,numpy.array([ft1(t),ft2(t),ft3(t)]).T,'-o')
plt.title('joint torques')

variable_functions = {t1:ft1,t2:ft2,t3:ft3}

func1 = system.state_space_post_invert(f,ma,variable_functions=variable_functions)
# # func1,lambda1 = system.state_space_post_invert(f,ma,eq_dd,return_lambda = True)
states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=tol,atol=tol,args=({'constants':system.constant_values},) )

# lambda2 = numpy.array([lambda1(item1,item2,system.constant_values) for item1,item2 in zip(t,states)])

KE = system.get_KE()
PE = system.getPEGravity(pNA) - system.getPESprings()

points = [pNA,pAB,pBC,pCtip]
#points = [item for item2 in points for item in [item2.dot(system.newtonian.x),item2.dot(system.newtonian.y)]]
points_output = PointsOutput(points,system)
y = points_output.calc(states_exp[:,:6],t)
# points_output.plot_time()
plt.figure()
plt.plot(*(y[::int(len(y)/20)].T))
plt.axis('equal')
plt.title('y_exp')


y = points_output.calc(states,t)
#y.resize(y.shape[0],int(y.shape[1]/2),2)

plt.figure()
plt.plot(t,states[:,:3])
plt.title('q')

plt.figure()
plt.plot(*(y[::int(len(y)/20)].T))
plt.axis('equal')
plt.title('y')

energy_output = Output([KE-PE],system)
energy_output.calc(states,t)

# plt.figure()
# plt.plot(energy_output.y)
# plt.title('energy')
plt.show()


# points_output.animate(fps = 100,movie_name = 'render.mp4',lw=2,marker='o',color=(1,0,0,1),linestyle='-')
#a()
