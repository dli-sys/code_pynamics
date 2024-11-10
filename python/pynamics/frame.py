# -*- coding: utf-8 -*-

"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
from pynamics.tree_node import TreeNode
from pynamics.vector import Vector
from pynamics.rotation import Rotation, RotationalVelocity
from pynamics.name_generator import NameGenerator
from pynamics.quaternion import Quaternion

import sympy


class Frame(NameGenerator):
    def __init__(self,name,system):
        super(Frame,self).__init__()
        
        self.connections={}
        self.connections['R'] = {}
        self.connections['w'] = {}

        self.precomputed={}
        self.precomputed['R'] = {}
        self.precomputed['w'] = {}

        self.tree={}
        self.tree['R'] = TreeNode(self)
        self.tree['w'] = TreeNode(self)

        self.reps = {}

        self.name = name
        
        self.x = Vector()
        self.y = Vector()
        self.z = Vector()

        self.x_sym = sympy.Symbol(name+'.x')
        self.y_sym = sympy.Symbol(name+'.y')
        self.z_sym = sympy.Symbol(name+'.z')
        self.syms = sympy.Matrix([self.x_sym,self.y_sym,self.z_sym])
        
        self.x.add_component(self,[1,0,0])
        self.y.add_component(self,[0,1,0])
        self.z.add_component(self,[0,0,1])
        
        r = Rotation(self,self,sympy.Matrix.eye(3),Quaternion(0,0,0,0))
        w = RotationalVelocity(self,self,sympy.Number(0)*self.x,Quaternion(0,0,0,0))

        self.add_generic(r,'R')
        self.add_generic(w,'w')
        self.system = system

        self.system.add_frame(self)

    def add_generic(self,rotation,my_type):
        self.connections[my_type][rotation.other(self)] = rotation
        
    def add_precomputed_generic(self,rotation,my_type):
        self.precomputed[my_type][rotation.other(self)] = rotation

    @property
    def principal_axes(self):
        return [self.x,self.y,self.z]
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)

    def get_generic(self,other,my_type):
        if other in self.connections[my_type]:
            return self.connections[my_type][other]
        elif other in self.precomputed[my_type]:
            return self.precomputed[my_type][other]
        else: 
            path = self.tree['R'].path_to(other.tree['R'])
            path = [item.myclass for item in path]
            from_frames = path[:-1]
            to_frames = path[1:]
            if my_type=='R':
                items = [from_frame.connections[my_type][to_frame].get_r_to(to_frame) for from_frame,to_frame in zip(from_frames,to_frames)]
                q_items = [from_frame.connections[my_type][to_frame].get_rq_to(to_frame) for from_frame,to_frame in zip(from_frames,to_frames)]
            elif my_type=='w':
                items = [from_frame.connections[my_type][to_frame].get_w_to(to_frame) for from_frame,to_frame in zip(from_frames,to_frames)]                
            item_final= items.pop(0)      
            if my_type=='R':
                q_item_final= q_items.pop(0)      
                for item,to_frame in zip(items,to_frames[1:]):
                    item_final = item*item_final
                for q_item,to_frame in zip(q_items,to_frames[1:]):
                    q_item_final = q_item*q_item_final
                result = Rotation(self,to_frame,item_final,q_item_final)
            elif my_type=='w':
                for item,to_frame in zip(items,to_frames[1:]):
                    item_final += item
                    result = RotationalVelocity(self,to_frame,item_final,Quaternion(0,0,0,0))
                self.add_precomputed_generic(result,my_type)
                to_frame.add_precomputed_generic(result,my_type)
            return result

    def get_r_to(self,other):
        return self.get_generic(other,'R').get_r_to(other)

    def get_r_from(self,other):
        return self.get_generic(other,'R').get_r_from(other)

    def get_rq_to(self,other):
        return self.get_generic(other,'R').get_rq_to(other)

    def get_rq_from(self,other):
        return self.get_generic(other,'R').get_rq_from(other)

    def get_w_from(self, other: object) -> object:
        return self.get_generic(other,'w').get_w_from(other)

    def get_w_to(self,other):
        return self.get_generic(other,'w').get_w_to(other)

    def set_generic(self,other,item,my_type):
        if my_type=='R':
            result = Rotation(self, other, item,Quaternion(0,0,0,0))
        elif my_type=='w':
            result = RotationalVelocity(self, other, item,Quaternion(0,0,0,0))
        self.add_generic(result,my_type)
        other.add_generic(result,my_type)
        
    def set_parent_generic(self,parent,item,my_type):
        self.set_generic(parent,item,my_type)
        parent.tree[my_type].add_branch(self.tree[my_type])        

    def set_child_generic(self,child,item,my_type):
        self.set_generic(child,item,my_type)
        self.tree[my_type].add_branch(child.tree[my_type])        

    def set_w(self,other,w):
        self.set_child_generic(other,w,'w')

    def rotate_fixed_axis(self,fromframe,axis,q,system):
        import pynamics.misc_tools
        if not all([pynamics.misc_tools.is_literal(item) for item in axis]):
            raise(Exception('not all axis variables are constant'))

        rotation = Rotation.build_fixed_axis(fromframe,self,axis,q,system)
        rotational_velocity = RotationalVelocity.build_fixed_axis(fromframe,self,axis,q,system)
        self.set_parent_generic(fromframe,rotation,'R')
        self.set_parent_generic(fromframe,rotational_velocity,'w')
        self.add_generic(rotation,'R')
        self.add_generic(rotational_velocity,'w')
        fromframe.add_generic(rotation,'R')
        fromframe.add_generic(rotational_velocity,'w')
        
        fromframe.tree['R'].add_branch(self.tree['R'])        
        fromframe.tree['w'].add_branch(self.tree['w'])        

