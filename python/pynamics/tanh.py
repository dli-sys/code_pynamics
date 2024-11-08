# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:00:27 2020

@author: danaukes
"""

import matplotlib.pyplot as plt
import sympy
import numpy
from math import pi

def gen_spring_force(s, scaling, x0n, x1n, k1,k2,k3,plot=False):

    x = sympy.Symbol('x')
    x0 = sympy.Symbol('x0')
    x1 = sympy.Symbol('x1')
    
    
#    scaling = 10
    
#    x0n = 0
#    x1n = -4
#    x1n = -pi/4
#    x1n = -10*pi/180
    
    f1 = (sympy.tanh(scaling*(x-x0))+1)/2
    f2 = (sympy.tanh(-scaling*(x-x0))+1)/2
    f3 = (sympy.tanh(scaling*(x-x1))+1)/2
    f4 = (sympy.tanh(-scaling*(x-x1))+1)/2
    
    if plot:
        xn = numpy.r_[-.5:.5:100j]
        
        f1n = numpy.array([f1.evalf(subs={x:item,x0:x0n,x1:x1n}) for item in xn])
        f2n = numpy.array([f2.evalf(subs={x:item,x0:x0n,x1:x1n}) for item in xn])
        f3n = numpy.array([f3.evalf(subs={x:item,x0:x0n,x1:x1n}) for item in xn])
        f4n = numpy.array([f4.evalf(subs={x:item,x0:x0n,x1:x1n}) for item in xn])
        
        plt.plot(xn,f1n)
        plt.plot(xn,f2n)
        
        plt.figure()
        
        plt.plot(xn,f3n)
        plt.plot(xn,f4n)
#    
#    k1 = 2e1
#    k2 = 1e1
#    k3 = 0e1

    y0 = (k1*f1+k2*f2)*(x)
    y2 = k3*x

    y0n = y0.subs({x:x1n,x0:x0n,x1:x1n}).evalf()
    y2n = y2.subs({x:x1n,x0:x0n,x1:x1n}).evalf()
    
    y = (k1*f1+k2*f2*f3+k3*f4)*(x) + f4*y0n - f4*y2n

    if plot:
        yn = numpy.array([y.evalf(subs={x:item,x0:x0n,x1:x1n}) for item in xn])
        plt.figure()
        plt.plot(xn,yn)
    y2 = y.subs({x:s,x1:x1n,x0:x0n})
    return y2


def gen_spring_force1(s, scaling, x0n, x1n, k1, k2, k3, plot=False):
    x = sympy.Symbol('x')
    x0 = sympy.Symbol('x0')
    x1 = sympy.Symbol('x1')

    #    scaling = 10

    # x0n = 0
    #    x1n = -4
    #    x1n = -pi/4
    #    x1n = -10*pi/180

    f1 = (sympy.tanh(scaling * (x - x0)) + 1) / 2
    # f2 = (sympy.tanh(-scaling * (x - x0)) + 1) / 2
    #
    # f3 = (sympy.tanh(scaling * (x - x1)) + 1) / 2
    # f4 = (sympy.tanh(-scaling * (x - x1)) + 1) / 2

    # f12 = f1+f2

    # if plot:
    #     xn = numpy.r_[-.5:.5:100j]
    #
    #     f1n = numpy.array([f1.evalf(subs={x: item, x0: x0n, x1: x1n}) for item in xn])
        # f2n = numpy.array([f2.evalf(subs={x: item, x0: x0n, x1: x1n}) for item in xn])
        # f3n = numpy.array([f3.evalf(subs={x: item, x0: x0n, x1: x1n}) for item in xn])
        # f4n = numpy.array([f4.evalf(subs={x: item, x0: x0n, x1: x1n}) for item in xn])
        #
        # f12n = numpy.array([f12.evalf(subs={x: item, x0: x0n, x1: x1n}) for item in xn])

        # plt.figure()
        # plt.plot(xn,f1n)


        # plt.title("f1 and f2")
        # plt.plot(xn, f1n)
    #     plt.plot(xn, f2n)
    #
    #     plt.figure()
    #     plt.title("f3 and f4")
    #     plt.plot(xn, f3n)
    #     plt.plot(xn, f4n)
    # #
    # #    k1 = 2e1
    # #    k2 = 1e1
    # #    k3 = 0e1
    #
    # y0 = (k1 * f1 + k2 * f2) * (x)
    # y2 = k3 * x
    #
    # y0n = y0.subs({x: x1n, x0: x0n, x1: x1n}).evalf()
    # y2n = y2.subs({x: x1n, x0: x0n, x1: x1n}).evalf()
    #
    # y = (k1 * f1 + k2 * f2 * f3 + k3 * f4) * (x) + f4 * y0n - f4 * y2n
    #
    # if plot:
    #     yn = numpy.array([y.evalf(subs={x: item, x0: x0n, x1: x1n}) for item in xn])
    #     plt.figure()
    #     plt.plot(xn, yn)
    # y2 = y.subs({x: s, x1: x1n, x0: x0n})
    return f1

if __name__=='__main__':
    # the purpose of tanh: since symbplic can not solve abs value. Dan introduce a tanh function to minmic
    s = sympy.Symbol('s')
    #f = gen_spring_force(s,100, 0, -10*pi/180, 32.3783, 7.6504, 0e1)
#    f2 = gen_spring_force(s,1000, 0, -0.00866, 32.3783, 1*6.68, 0e1,plot=True)
#    f2 = gen_spring_force(s,1000, 0, 0*pi/180, 1e0, 1e0, 0e1,plot=True)
    f2 = gen_spring_force1(s,1000, 0, 0, 1e3, 0e1, 0e1,plot=True)
    
    
    plt.xlabel('Theta')
    plt.ylabel('Torque')
    plt.title('Stiffness (Torque vs. Theta)')