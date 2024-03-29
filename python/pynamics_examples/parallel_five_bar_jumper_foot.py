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
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)
tol=1e-7


lO = Constant(.5,'lO',system)
lA = Constant(.75,'lA',system)
lB = Constant(1,'lB',system)
lC = Constant(.75,'lC',system)
lD = Constant(1,'lD',system)
lE = Constant(1,'lE',system)

mO = Constant(2,'mO',system)
mA = Constant(.1,'mA',system)
mB = Constant(.1,'mB',system)
mC = Constant(.1,'mC',system)
mD = Constant(.1,'mD',system)
mE = Constant(.1,'mE',system)

I_main = Constant(1,'I_main',system)
I_leg = Constant(.1,'I_leg',system)

g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e2,'k',system)
k_ankle = Constant(1e3,'k_ankle',system)
b_ankle = Constant(1e1,'b_ankle',system)
stall_torque = Constant(2e2,'stall_torque',system)

k_constraint = Constant(1e4,'k_constraint',system)
b_constraint = Constant(1e2,'b_constraint',system)

tinitial = 0
tfinal = 10
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant(0*pi/180,'preload1',system)
preload2 = Constant(0*pi/180,'preload2',system)
preload3 = Constant(-180*pi/180,'preload3',system)
preload4 = Constant(0*pi/180,'preload4',system)
preload5 = Constant(180*pi/180,'preload5',system)
preload6 = Constant(0*pi/180,'preload6',system)

x,x_d,x_dd = Differentiable('x',system)
y,y_d,y_dd = Differentiable('y',system)
qO,qO_d,qO_dd = Differentiable('qO',system)
qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)
qD,qD_d,qD_dd = Differentiable('qD',system)
qE,qE_d,qE_dd = Differentiable('qE',system)

initialvalues={
        x: 0,
        x_d: .5,
        y: 2,
        y_d: 0,
        qO: 5*pi/180,
        qO_d: 0,
        qA: -0.89,
        qA_d: 0,
        qB: -2.64,
        qB_d: 0,
        qC: -pi+0.89,
        qC_d: 0,
        qD: -pi+2.64,
        qD_d: 0,
        qE: 0,
        qE_d: 0,
        }

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
O = Frame('O')
A = Frame('A')
B = Frame('B')
C = Frame('C')
D = Frame('D')
E = Frame('E')


system.set_newtonian(N)
O.rotate_fixed_axis_directed(N,[0,0,1],qO,system)
A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)
B.rotate_fixed_axis_directed(N,[0,0,1],qB,system)
C.rotate_fixed_axis_directed(N,[0,0,1],qC,system)
D.rotate_fixed_axis_directed(N,[0,0,1],qD,system)
E.rotate_fixed_axis_directed(N,[0,0,1],qE,system)

pOrigin = 0*N.x+0*N.y
pOcm=x*N.x+y*N.y
pOA = pOcm+lO/2*O.x
pOC = pOcm-lO/2*O.x
pAB = pOA+lA*A.x
pBtip = pAB + lB*B.x
#vBtip = pBtip.time_derivative(N,system)

pCD = pOC + lC*C.x
pDtip = pCD + lD*D.x
#vDtip = pDtip.time_derivative(N,system)

#pE1 = pBtip+lE/2*E.x
#vE1 = pE1.time_derivative(N,system)
#
#pE2 = pBtip-lE/2*E.x
#vE2 = pE2.time_derivative(N,system)

def gen_init():
    eqs = []
    eqs.append(pBtip-pDtip)
#    eqs.append(pBtip-pOrigin)
    a=[(item).express(N) for item in eqs]
    b=[item.subs(system.constant_values) for item in a]
    c = numpy.array([vec.dot(item) for vec in b for item in list(N.principal_axes)])
    d = (c**2).sum()
    e = system.get_state_variables()
    #e = sorted(list(d.atoms(Differentiable)),key=lambda x:str(x))
    f = sympy.lambdify(e,d)
    g = lambda args:f(*args)
    return g
fun = gen_init()

import scipy.optimize
result = scipy.optimize.minimize(fun,ini)

if result.fun<1e-7:
    points = [pDtip,pCD,pOC,pOA,pAB,pBtip]
    points = PointsOutput(points)
    state = numpy.array([ini,result.x])
    ini1 = list(result.x)
    y = points.calc(state)
    y = y.reshape((-1,6,2))
    plt.figure()
    for item in y:
        plt.plot(*(item.T))
#    for item,value in zip(system.get_state_variables(),result.x):
#        initialvalues[item]=value
    
pAcm=pOA+lA/2*A.x
pBcm=pAB+lB/2*B.x
pCcm=pOC+lC/2*C.x
pDcm=pCD+lD/2*D.x
pEcm=pBtip -.1*E.y

pE1 = pEcm+lE/2*E.x
vE1 = pE1.time_derivative(N,system)

pE2 = pEcm-lE/2*E.x
vE2 = pE2.time_derivative(N,system)

wOA = O.getw_(A)
wAB = A.getw_(B)
wOC = O.getw_(C)
wCD = C.getw_(D)
wBD = B.getw_(D)
wOE = O.getw_(E)

BodyO = Body('BodyO',O,pOcm,mO,Dyadic.build(O,I_main,I_main,I_main),system)
#BodyA = Body('BodyA',A,pAcm,mA,Dyadic.build(A,I_leg,I_leg,I_leg),system)
#BodyB = Body('BodyB',B,pBcm,mB,Dyadic.build(B,I_leg,I_leg,I_leg),system)
#BodyC = Body('BodyC',C,pCcm,mC,Dyadic.build(C,I_leg,I_leg,I_leg),system)
#BodyD = Body('BodyD',D,pDcm,mD,Dyadic.build(D,I_leg,I_leg,I_leg),system)
BodyE = Body('BodyE',E,pEcm,mE,Dyadic.build(D,I_leg,I_leg,I_leg),system)

ParticleA = Particle(pAcm,mA,'ParticleA')
ParticleB = Particle(pBcm,mB,'ParticleB')
ParticleC = Particle(pCcm,mC,'ParticleC')
ParticleD = Particle(pDcm,mD,'ParticleD')
#ParticleE = Particle(pEcm,mE,'ParticleE')

system.addforce(-b*wOA,wOA)
system.addforce(-b*wAB,wAB)
system.addforce(-b*wOC,wOC)
system.addforce(-b*wCD,wCD)
system.addforce(-b_ankle*wOE,wOE)
#
stretch1 = -pE1.dot(N.y)
stretch1_s = (stretch1+abs(stretch1))
on = stretch1_s/(2*stretch1+1e-10)
system.add_spring_force1(k_constraint,-stretch1_s*N.y,vE1)
system.addforce(-b_constraint*vE1*on,vE1)

toeforce = k_constraint*-stretch1_s

stretch2 = -pE2.dot(N.y)
stretch2_s = (stretch2+abs(stretch2))
on = stretch2_s/(2*stretch2+1e-10)
system.add_spring_force1(k_constraint,-stretch2_s*N.y,vE2)
system.addforce(-b_constraint*vE2*on,vE2)

heelforce = k_constraint*-stretch2_s

system.add_spring_force1(k,(qA-qO-preload1)*N.z,wOA)
system.add_spring_force1(k,(qB-qA-preload2)*N.z,wAB)
system.add_spring_force1(k,(qC-qO-preload3)*N.z,wOC)
system.add_spring_force1(k,(qD-qC-preload4)*N.z,wCD)
system.add_spring_force1(k,(qD-qB-preload5)*N.z,wBD)
system.add_spring_force1(k_ankle,(qE-qO-preload6)*N.z,wOE)

system.addforcegravity(-g*N.y)

import pynamics.time_series
x = [0,5,5,7,7,9,9,10]
y = [0,0,1,1,-1,-1,0,0]
my_signal, ft2 = pynamics.time_series.build_smoothed_time_signal(x,y,t,'my_signal',window_time_width = .1)

torque = my_signal*stall_torque
system.addforce(torque*O.z,wOA)
system.addforce(-torque*O.z,wOC)

#
eq = []
eq.append((pBtip-pDtip).dot(N.x))
eq.append((pBtip-pDtip).dot(N.y))
#eq.append((O.y.dot(N.y)))
eq_d= [system.derivative(item) for item in eq]
eq_dd= [system.derivative(item) for item in eq_d]
#
f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma,eq_dd,constants = system.constant_values,variable_functions = {my_signal:ft2})
states=pynamics.integration.integrate_odeint(func1,ini1,t,rtol=tol,atol=tol)

KE = system.get_KE()
PE = system.getPEGravity(0*N.x) - system.getPESprings()
energy = Output([KE-PE,toeforce,heelforce])
energy.calc(states)
energy.plot_time()

#torque_plot = Output([torque])
#torque_plot.calc(states)
#torque_plot.plot_time()

points = [pDtip,pCD,pOC,pOA,pAB,pBtip,pE1,pE2,pBtip]
points = PointsOutput(points)
y = points.calc(states)
y = y.reshape((-1,9,2))
plt.figure()
for item in y[::30]:
    plt.plot(*(item.T))

points.animate(fps = 30, movie_name='parallel_five_bar_jumper_foot.mp4',lw=2)
