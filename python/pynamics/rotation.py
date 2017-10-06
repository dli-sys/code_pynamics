# -*- coding: utf-8 -*-

"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import sympy
from sympy import cos, sin
from pynamics.vector import Vector
import pynamics

def build_fixed_axis(axis,q,frame,sys = None):
    sys = sys or pynamics.get_system()

    axis = sympy.Matrix(axis)
    l_2 = axis.dot(axis)
    axis = axis/(l_2**.5)
    
    e0 = axis[0]        
    e1 = axis[1]        
    e2 = axis[2]        
    
    c = axis * axis.T
    a = (sympy.Matrix.eye(3) - c) * cos(q)
    b = sympy.Matrix([[0, -e2, e1], [e2, 0, -e0],[-e1, e0, 0]]) * sin(q)
    R = a + b + c 
    R = R.T

    w = sys.derivative(q)
    w_ = w*Vector({frame:axis})
    return R,w_,axis


class Rotation(object):
    def __init__(self,f1,f2,R,w_):
        self.f1 = f1
        self.f2 = f2
        f1.add_rotation(self)
        f2.add_rotation(self)
        self._R = R
        self.w_ = w_
    def to_other(self,f):
        if f==self.f1:
            return self._R
        elif f==self.f2:
            return self._R.T
        else:
            raise(Exception('frame not in this rotation'))
    def w__from(self,f):
        if f==self.f1:
            return self.w_
        elif f==self.f2:
            return -self.w_
        else:
            raise(Exception('frame not in this rotation'))
    def other(self,f):
        if f==self.f1:
            return self.f2
        elif f==self.f2:
            return self.f1
        else:
            raise(Exception('frame not in this rotation'))
            
    @classmethod
    def build_fixed_axis(cls,f1,f2,axis,q,sys = None):
#        self.f1 = f1
#        self.f2 = f2
        R,w,fixedaxis = build_fixed_axis(axis,q,f1,sys)
        return cls(f1,f2,R,w)
        
        
    @classmethod
    def build_xyz(cls,f1,f2,q1,q2,q3,sys = None):
        r1,w1,a1 = build_fixed_axis([1,0,0],q1,f1,sys)
        r2,w2,a2 = build_fixed_axis([0,1,0],q2,f1,sys)
        r3,w3,a3 = build_fixed_axis([0,0,1],q3,f1,sys)
        r = r1*r2*r3
        w = w1,w2,w3
        return cls(f1,f2,r,w)
            