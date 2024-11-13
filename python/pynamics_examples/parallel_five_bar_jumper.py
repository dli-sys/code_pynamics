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
import sympy
import numpy
import matplotlib.pyplot as plt
from pynamics.constraint import KinematicConstraint,AccelerationConstraint

plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)
tol=1e-5

lO = Constant(name='lO',system=system)
lA = Constant(name='lA',system=system)
lB = Constant(name='lB',system=system)
lC = Constant(name='lC',system=system)
lD = Constant(name='lD',system=system)

mO = Constant(name='mO',system=system)
mA = Constant(name='mA',system=system)
mB = Constant(name='mB',system=system)
mC = Constant(name='mC',system=system)
mD = Constant(name='mD',system=system)

I_main = Constant(name='I_main',system=system)

g = Constant(name='g',system=system)
b = Constant(name='b',system=system)
k = Constant(name='k',system=system)
stall_torque = Constant(name='stall_torque',system=system)

k_constraint = Constant(name='k_constraint',system=system)
b_constraint = Constant(name='b_constraint',system=system)

tinitial = 0
tfinal = 10
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant(name='preload1',system=system)
preload2 = Constant(name='preload2',system=system)
preload3 = Constant(name='preload3',system=system)
preload4 = Constant(name='preload4',system=system)
preload5 = Constant(name='preload5',system=system)

constants = {}
constants[lO]=.5
constants[lA] = .75
constants[lB] = 1
constants[lC] = .75
constants[lD] = 1
constants[mO] = 3
constants[mA] = .1
constants[mB] = .1
constants[mC] = .1
constants[mD] = .1
constants[I_main] = 1
constants[g] = 9.81
constants[b] = 1e0
constants[k] = 1e2
constants[stall_torque] = 1e2
constants[k_constraint] = 1e5
constants[b_constraint] = 1e3
constants[preload1] = 0*pi/180
constants[preload2] = 0*pi/180
constants[preload3] = -180*pi/180
constants[preload4] = 0*pi/180
constants[preload5] = 180*pi/180


x,x_d,x_dd = Differentiable(name='x',system=system)
y,y_d,y_dd = Differentiable(name='y',system=system)
qO,qO_d,qO_dd = Differentiable(name='qO',system=system)
qA,qA_d,qA_dd = Differentiable(name='qA',system=system)
qB,qB_d,qB_dd = Differentiable(name='qB',system=system)
qC,qC_d,qC_dd = Differentiable(name='qC',system=system)
qD,qD_d,qD_dd = Differentiable(name='qD',system=system)

initialvalues={
        x: 0,
        x_d: 0,
        y: 1.25,
        y_d: 0,
        qO: 0,
        qO_d: 0,
        qA: -0.89,
        qA_d: 0,
        qB: -2.64,
        qB_d: 0,
        qC: -pi+0.89,
        qC_d: 0,
        qD: -pi+2.64,
        qD_d: 0}


statevariables = system.get_state_variables()
ini0 = [initialvalues[item] for item in statevariables]

N = Frame('N',system)
O = Frame('O',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)
D = Frame('D',system)

system.set_newtonian(N)
O.rotate_fixed_axis(N,[0,0,1],qO,system)
A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(N,[0,0,1],qB,system)
C.rotate_fixed_axis(N,[0,0,1],qC,system)
D.rotate_fixed_axis(N,[0,0,1],qD,system)

pOrigin = 0*N.x+0*N.y
pOcm=x*N.x+y*N.y
pOA = pOcm+lO/2*O.x
pOC = pOcm-lO/2*O.x
pAB = pOA+lA*A.x
pBtip = pAB + lB*B.x
vBtip = pBtip.time_derivative(N,system)

pCD = pOC + lC*C.x
pDtip = pCD + lD*D.x
vDtip = pDtip.time_derivative(N,system)

points = [pDtip,pCD,pOC,pOA,pAB,pBtip]

eqs = []
eqs.append((pBtip-pDtip).dot(N.x))
eqs.append((pBtip-pDtip).dot(N.y))


constraint_system=KinematicConstraint(eqs)

variables = [qO, qA, qB, qC, qD]
guess = [initialvalues[item] for item in variables]
result = constraint_system.solve_numeric(variables,guess,constants)

ini = []
for item in system.get_state_variables():
    if item in variables:
        ini.append(result[item])
    else:
        ini.append(initialvalues[item])

points = PointsOutput(points, constant_values=constants)
points.calc(numpy.array([ini0,ini]),[0,1])
points.plot_time()
    
pAcm=pOA+lA/2*A.x
pBcm=pAB+lB/2*B.x
pCcm=pOC+lC/2*C.x
pDcm=pCD+lD/2*D.x

wOA = O.get_w_to(A)
wAB = A.get_w_to(B)
wOC = O.get_w_to(C)
wCD = C.get_w_to(D)
wBD = B.get_w_to(D)

BodyO = Body('BodyO',O,pOcm,mO,Dyadic.build(O,I_main,I_main,I_main),system)
#BodyA = Body('BodyA',A,pAcm,mA,Dyadic.build(A,I_leg,I_leg,I_leg),system)
#BodyB = Body('BodyB',B,pBcm,mB,Dyadic.build(B,I_leg,I_leg,I_leg),system)
#BodyC = Body('BodyC',C,pCcm,mC,Dyadic.build(C,I_leg,I_leg,I_leg),system)
#BodyD = Body('BodyD',D,pDcm,mD,Dyadic.build(D,I_leg,I_leg,I_leg),system)

ParticleA = Particle(pAcm,mA,'ParticleA')
ParticleB = Particle(pBcm,mB,'ParticleB')
ParticleC = Particle(pCcm,mC,'ParticleC')
ParticleD = Particle(pDcm,mD,'ParticleD')

system.addforce(-b*wOA,wOA)
system.addforce(-b*wAB,wAB)
system.addforce(-b*wOC,wOC)
system.addforce(-b*wCD,wCD)
system.addforce(-b*wBD,wBD)
#
stretch = -pBtip.dot(N.y)
stretch_s = (stretch+abs(stretch))
on = stretch_s/(2*stretch+1e-10)
system.add_spring_force1(k_constraint,-stretch_s*N.y,vBtip)
system.addforce(-b_constraint*vBtip*on,vBtip)

system.add_spring_force1(k,(qA-qO-preload1)*N.z,wOA)
system.add_spring_force1(k,(qB-qA-preload2)*N.z,wAB)
system.add_spring_force1(k,(qC-qO-preload3)*N.z,wOC)
system.add_spring_force1(k,(qD-qC-preload4)*N.z,wCD)
system.add_spring_force1(k,(qD-qB-preload5)*N.z,wBD)

system.addforcegravity(-g*N.y)

import pynamics.time_series
x = [0,2,2,5,5,6,6,10]
y = [0,0,1,1,-1,-1,0,0]
my_signal, ft2 = pynamics.time_series.build_smoothed_time_signal(x,y,t,'my_signal',window_time_width = .1)

torque = my_signal*stall_torque
system.addforce(torque*O.z,wOA)
system.addforce(-torque*O.z,wOC)

#
eq = []
eq.append(pBtip-pDtip)
eq.append(O.y)
eq_d= [item.time_derivative() for item in eq]
eq_dd= [item.time_derivative() for item in eq_d]

eq_dd_scalar = []
eq_dd_scalar.append(eq_dd[0].dot(N.x))
eq_dd_scalar.append(eq_dd[0].dot(N.y))
eq_dd_scalar.append(eq_dd[1].dot(N.y))

c = AccelerationConstraint(eq_dd_scalar)
# c.linearize(0)
system.add_constraint(c)

#
f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma,constants = constants,variable_functions = {my_signal:ft2})
states=pynamics.integration.integrate(func1,ini,t,rtol=tol,atol=tol)

KE = system.get_KE()
PE = system.getPEGravity(0*N.x) - system.getPESprings()
energy = Output([KE-PE], constant_values=constants)
energy.calc(states,t)
energy.plot_time()

#torque_plot = Output([torque])
#torque_plot.calc(states,t)
#torque_plot.plot_time()

points = [pDtip,pCD,pOC,pOA,pAB,pBtip]
points = PointsOutput(points, constant_values=constants)
y = points.calc(states,t)
y = y.reshape((-1,6,2))
plt.figure()
for item in y[::30]:
    plt.plot(*(item.T))

points.animate(fps = 30, movie_name='parallel_five_bar_jumper.gif',lw=2)
