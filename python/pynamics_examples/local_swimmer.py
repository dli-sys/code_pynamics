import sys
import os
# # Add the path to the pynamics module
# module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pynamics'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import pynamics
import pynamics.frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
import pynamics.integration
import pynamics.tanh as tanh

from numpy import pi
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import sympy
# import sympy
import numpy
from math import pi
import scipy
import scipy.optimize
from sympy import sin, cos

import cma
import time


plt.close('all')
plt.ion()



def Cal_robot(direction, given_l, angular_vel,start_angle, end_angle, ini_states, name1, system, video_on, x1,force_coeff):

  [body_force,arm_force_prep,arm_force_par] = force_coeff

  time_a = time.time()
  pynamics.set_system(__name__, system)
  given_k, given_b = x1
  global_q = True

  damping_r = 0
  tinitial = 0
  tfinal = abs((start_angle - end_angle) / angular_vel)
  tstep = 1 / 30
  t = numpy.r_[tinitial:tfinal:tstep]

  tol_1 = 1e-6
  tol_2 = 1e-6
  lO = Constant(20 / 1000, 'lO', system)
  lR = Constant(40 / 1000, 'lR', system)
  # lA = Constant(given_l / 1000, 'lA', system)
  # lB = Constant(given_l / 1000, 'lB', system)
  # lC = Constant(given_l / 1000, 'lC', system)

  mO = Constant(60 / 1000, 'mO', system)
  mR = Constant(10 / 1000, 'mR', system)
  k = Constant(given_k, 'k', system)

  friction_arm_perp = Constant(arm_force_prep, 'fr_perp', system)
  friction_arm_par = Constant(arm_force_par, 'fr_par', system)
  # b_damping = Constant(given_b, 'b_damping', system)


  Ixx_O = Constant(1, 'Ixx_O', system)
  Iyy_O = Constant(1, 'Iyy_O', system)
  Izz_O = Constant(1, 'Izz_O', system)
  Ixx_R = Constant(1, 'Ixx_R', system)
  Iyy_R = Constant(1, 'Iyy_R', system)
  Izz_R = Constant(1, 'Izz_R', system)

  y, y_d, y_dd = Differentiable('y', system)
  qO, qO_d, qO_dd = Differentiable('qO', system)
  qR, qR_d, qR_dd = Differentiable('qR', system)

  initialvalues = {}
  initialvalues[y] = ini_states[0] + tol_1
  initialvalues[qO] = ini_states[1] + tol_1
  initialvalues[qR] = ini_states[2] + tol_1

  initialvalues[y_d] = ini_states[3] + tol_1
  initialvalues[qO_d] = ini_states[4] + tol_1
  initialvalues[qR_d] = ini_states[5] + tol_1


  statevariables = system.get_state_variables()
  ini = [initialvalues[item] for item in statevariables]
  N = Frame('N')
  O = Frame('O')
  R = Frame('R')


  system.set_newtonian(N)
  if not global_q:
    O.rotate_fixed_axis_directed(N, [0, 0, 1], qO, system)
    R.rotate_fixed_axis_directed(O, [0, 0, 1], qR, system)

  else:
    O.rotate_fixed_axis_directed(N, [0, 0, 1], qO, system)
    R.rotate_fixed_axis_directed(N, [0, 0, 1], qR, system)


  pNO = 0 * N.x + y * N.y
  pOR = pNO + lO * N.x
  pRA = pOR + lR * R.x


  pOcm = pNO + lO / 2 * N.x
  pRcm = pOR + lR / 2 * R.x


  wNO = N.getw_(O)
  wOR = N.getw_(R)


  IO = Dyadic.build(O, Ixx_O, Iyy_O, Izz_O)
  IR = Dyadic.build(R, Ixx_R, Iyy_R, Izz_R)

  BodyO = Body('BodyO', O, pOcm, mO, IO, system)
  BodyR = Body('BodyR', R, pRcm, mR, IR, system)

  j_tol = 3 * pi / 180
  inv_k = 10


  vOcm = y_d * N.y
  vRcm = pRcm.time_derivative()


  nvRcm = 1 / (vRcm.length() + tol_1) * vRcm

  vSoil = -direction * 1 * N.y
  nSoil = 1 / vSoil.length() * vSoil
  foperp = body_force * nSoil
  system.addforce(-foperp, vOcm)

  frperp = friction_arm_perp * nvRcm.dot(R.y) * R.y
  frpar = friction_arm_par * nvRcm.dot(R.x) * R.x
  system.addforce(-(frperp + frpar), vRcm)


  eq = []
  eq_d = [(system.derivative(item)) for item in eq]
  eq_d.append(qR_d - angular_vel)
  eq_dd = [(system.derivative(item)) for item in eq_d]

  f, ma = system.getdynamics()
  func1 = system.state_space_post_invert(f, ma, eq_dd)
  points = [pNO, pOR,pRA]
  # points = [pNO, pOR, pRA, pAB, pBC, pCtip]

  constants = system.constant_values
  states = pynamics.integration.integrate_odeint(func1, ini, t, args=({'constants': constants},))
  final = numpy.asarray(states[-1, :])


  logger1 = logging.getLogger('pynamics.system')
  logger2 = logging.getLogger('pynamics.integration')
  logger3 = logging.getLogger('pynamics.output')
  logger1.disabled = True
  logger2.disabled = True
  logger3.disabled = True

  # Here is how to use points to calculatethe video
  points_output = PointsOutput(points, system, constant_values=constants)
  y1 = points_output.calc(states)
  points_output.animate(fps=1 / tstep, movie_name=name1, lw=2, marker='o', color=(1, 0, 0, 1), linestyle='-')
  plt.xlim(0,0.06)
  plt.ylim(-0.5,0.5)
  # plt.axis("equal")
  # plt.ion()
  plt.show()
  # plt.plot(states[:,0:3])

  if video_on == 1:
    plt.figure()
    plt.plot(*(y1[::int(len(y1) / 20)].T) * 1000)
    # plt.axis('equal')
    # plt.axis('equal')
    plt.title("Plate Configuration vs Distance")
    # plt.xlabel("Configuration")
    plt.ylabel("Distance (mm)")

    # plt.figure()
    # plt.plot(t, numpy.rad2deg(states[:, 2]))
    # # plt.plot(t, numpy.rad2deg(states[:, 8]))
    # plt.legend(["qR", "qR_d"])
    # plt.hlines(numpy.rad2deg(start_angle), tinitial, tfinal)
    # plt.hlines(numpy.rad2deg(end_angle), tinitial, tfinal)
    # plt.title("Robot Arm angle and velocitues (qR and qR_d) over Time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Angles,Velocities (deg, deg/s)")

    # plt.figure()
    # q_states = numpy.c_[(states[:, 2], states[:, 3], states[:, 4], states[:, 5])]
    # plt.plot(t, numpy.rad2deg(q_states))
    # plt.title("Joint Angule over Time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Joint Angles (deg)")
    # plt.legend(["Arm", "Joint 1", "Joint 2", "Joint 3"])

    # plt.figure()
    # # qd_states = numpy.c_[(states[:, 8], states[:, 9], states[:, 10], states[:, 11])]
    # plt.plot(t, numpy.rad2deg(qd_states))
    # plt.legend(["qR_d", "qA_d", "qB_d", "qC_d"])
    # plt.title("Joint Angular Velocities over Time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Joint Angular Velocities (deg/s)")
    # plt.legend(["Arm", "Joint 1", "Joint 2", "Joint 3"])

    plt.figure()
    plt.plot(t, states[:, 0], '--')
    # plt.plot(t, states[:, -1])
    plt.title("Robot Distance and Velocity over time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Distance (mm)")
    # plt.legend(["Distance", "Velocity of the robot"])


  else:
    pass
  return final, states, y1, points


def cal_eff(video_flag):

  # if (0 <= start_angle) & (start_angle <= 90):
  #   error1 = 0
  # else:
  #   print("Theta_1 should <=90!!! and >=0!!!")
  #   error1 = 1e5
  # if (-50 <= end_angle) & (end_angle <= 0):
  #   error2 = 0
  # else:
  #   error2 = 1e5
  #   print("Theta_2 should <=0!!! and >=-50!!!")
  if True:

  # if error1 + error2 == 0:
    logger1 = logging.getLogger('pynamics.system')
    logger2 = logging.getLogger('pynamics.integration')
    logger3 = logging.getLogger('pynamics.output')

    logger1.disabled = True
    logger2.disabled = True
    logger3.disabled = True
    direction = 1

    # Looks like a2 is the initial angle of the arm, vir_vel is the initial velocity, omega is some angular velocity or angle
    # DEfination
    # [y,qO,qR,y_d,qO_d,qR_d]
    # End def
    servo_speed = pi*10/180
    ini_angle = -pi/3
    ini_states = numpy.array([0, 0, ini_angle, 0, 0, servo_speed])
    stroke_angle = 2*pi/3
    system1 = System()
    # In here we start calculate the front proposion
    fin_drag_reduction_coef = 1
    body_fin_drag_reduction_coef = 1
    fin_perp = 4
    fin_par = -0.1
    body_drag = 2
    video_on = video_flag
    given_l = 20
    start_angle, end_angle = [ini_angle, ini_angle+stroke_angle]

    final1, states1, y1,forward_points = Cal_robot(direction, given_l, servo_speed, start_angle, end_angle, ini_states,'robot_p1.gif', system1, 1, [0, 0],[body_drag,fin_perp*fin_drag_reduction_coef,fin_par*fin_drag_reduction_coef])

    # plt.plot(numpy.rad2deg(states1[:,2]))
    # plt.show()
    final = final1
    final[3::] = 0
    final[-1] = -servo_speed

    system2 = System()
    final2, states2, y2,recovery_points = Cal_robot(-direction, given_l, servo_speed, start_angle, end_angle, final,'robot_p2.gif', system2, 1, [0, 0],[body_drag*body_fin_drag_reduction_coef,fin_perp,fin_par])


    full_stroke_points = forward_points
    # points_output = PointsOutput(full_stroke_points, system1, constant_values=constants)
    # points_output.animate(fps=1 / tstep, movie_name=name1, lw=2, marker='o', color=(1, 0, 0, 1), linestyle='-')

    dis1 = states1[:, 0]
    dis2 = states2[:, 0]
    dis = numpy.append(dis1, dis2)
    real_dis = abs(dis[0] - dis[-1])
    forward_dis = abs(dis1[0] - dis1[-1])
    backward_dis = abs(dis2[0] - dis2[-1])
    ieta = 1 - real_dis / abs(dis2[0] - dis2[-1])
  else:
    ieta = 1
  if video_on == 1:
    plt.figure()
    plt.plot(dis * 1000)
    plt.title("Robot distance over time")
    plt.ylabel("Distance (mm)")
    plt.show(block=True)
  else:
    pass
  total_eta = ieta + error1 + error2
  return total_eta, forward_dis, backward_dis


if __name__ == "__main__":
  total_eta, forward_dis, backward_dis = cal_eff( 1)
