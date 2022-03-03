# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
from pynamics.frame import Frame
import idealab_tools.units
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
import pynamics.integration
from pynamics.constraint import KinematicConstraint,AccelerationConstraint

import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)

idealab_tools.units.Unit.set_scaling(meter=100)

lA = Constant(1.1*idealab_tools.units.length,'lA',system)
lB = Constant(1*idealab_tools.units.length,'lB',system)
lC = Constant(.9*idealab_tools.units.length,'lC',system)
lD = Constant(1.01*idealab_tools.units.length,'lD',system)

m = Constant(1*idealab_tools.units.mass,'m',system)

g = Constant(9.81*idealab_tools.units.acceleration,'g',system)

Ixx_A = Constant(1*idealab_tools.units.inertia,'Ixx_A',system)
Iyy_A = Constant(1*idealab_tools.units.inertia,'Iyy_A',system)
Izz_A = Constant(1*idealab_tools.units.inertia,'Izz_A',system)
Ixx_B = Constant(1*idealab_tools.units.inertia,'Ixx_B',system)
Iyy_B = Constant(1*idealab_tools.units.inertia,'Iyy_B',system)
Izz_B = Constant(1*idealab_tools.units.inertia,'Izz_B',system)
Ixx_C = Constant(1*idealab_tools.units.inertia,'Ixx_C',system)
Iyy_C = Constant(1*idealab_tools.units.inertia,'Iyy_C',system)
Izz_C = Constant(1*idealab_tools.units.inertia,'Izz_C',system)
Ixx_D = Constant(1*idealab_tools.units.inertia,'Ixx_D',system)
Iyy_D = Constant(1*idealab_tools.units.inertia,'Iyy_D',system)
Izz_D = Constant(1*idealab_tools.units.inertia,'Izz_D',system)

qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)
qD,qD_d,qD_dd = Differentiable('qD',system)

initialvalues = {}
initialvalues[qA]=20*pi/180*idealab_tools.units.radian
initialvalues[qA_d]=0*pi/180*idealab_tools.units.rotational_speed
initialvalues[qB]=90*pi/180*idealab_tools.units.radian
initialvalues[qB_d]=0*pi/180*idealab_tools.units.rotational_speed
initialvalues[qC]=140*pi/180*idealab_tools.units.radian
initialvalues[qC_d]=0*pi/180*idealab_tools.units.rotational_speed
initialvalues[qD]=-90*pi/180*idealab_tools.units.radian
initialvalues[qD_d]=0*pi/180*idealab_tools.units.rotational_speed

statevariables = system.get_state_variables()

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)
D = Frame('D',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(A,[0,0,1],qB,system)
C.rotate_fixed_axis(N,[0,0,1],qC,system)
D.rotate_fixed_axis(C,[0,0,1],qD,system)

pNA = 0*N.x
pAB = pNA + lA*A.x
pBD = pAB + lB*B.x
pCD = pNA + lC*C.x
pDB = pCD + lD*D.x

points = [pBD,pAB,pNA,pCD,pDB]

statevariables = system.get_state_variables()
ini0 = [initialvalues[item] for item in statevariables]


eq = []
eq.append(pBD-pDB)

eq_scalar = []
eq_scalar.append(eq[0].dot(N.x))
eq_scalar.append(eq[0].dot(N.y))

c=KinematicConstraint(eq_scalar)
variables = [qB,qD]
constant_states = list(set(system.get_q(0))-set(variables))

constants = system.constant_values.copy()
for key in constant_states:
    constants[key] = initialvalues[key] 
guess = [initialvalues[item] for item in variables]
result = c.solve_numeric(variables,guess,constants)


ini = []
for item in system.get_state_variables():
    if item in variables:
        ini.append(result[item])
    else:
        ini.append(initialvalues[item])
        
points = PointsOutput(points, constant_values=system.constant_values)
points.calc(numpy.array([ini0,ini]),[0,1])
points.plot_time()

pAcm=pNA+lA/2*A.x
pBcm=pAB+lB/2*B.x
pCcm=pNA+lC/2*C.x
pDcm=pCD+lD/2*D.x

wND = N.get_w_to(D)

IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)
ID = Dyadic.build(D,Ixx_D,Iyy_D,Izz_D)

BodyA = Body('BodyA',A,pAcm,m,IA,system)
BodyB = Body('BodyB',B,pBcm,m,IB,system)
BodyC = Body('BodyC',C,pCcm,m,IC,system)
BodyC = Body('BodyC',D,pDcm,m,ID,system)
# BodyA = Particle(pAcm,m,'ParticleA',system)
# BodyB = Particle(pBcm,m,'ParticleB',system)
# BodyC = Particle(pCcm,m,'ParticleC',system)
# BodyD = Particle(pDcm,m,'ParticleD',system)

system.addforcegravity(-g*N.y)

eq_d = [item.time_derivative() for item in eq]
eq_dd = [item.time_derivative() for item in eq_d]
eq_dd_scalar = []
eq_dd_scalar.append(eq_dd[0].dot(N.x))
eq_dd_scalar.append(eq_dd[0].dot(N.y))
system.add_constraint(AccelerationConstraint(eq_dd_scalar))


f,ma = system.getdynamics()

func1 = system.state_space_post_invert(f,ma)

fps = 30/(1*idealab_tools.units.time)
tinitial = 0*idealab_tools.units.time
tfinal = 5*idealab_tools.units.time
tstep = 1/fps
t = numpy.r_[tinitial:tfinal:tstep]
tolerance = 1e-12

states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=tolerance,atol=tolerance,args=({'constants':system.constant_values},))

KE = system.get_KE()
PE = system.getPEGravity(pNA) - system.getPESprings()

points.calc(states,t)
points.plot_time()
# points.animate(fps = fps,movie_name = 'four_bar.mp4',lw=2)
