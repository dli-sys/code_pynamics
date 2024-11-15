# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:32:41 2020

@author: danaukes
"""
import numpy
import pynamics
from pynamics.output_generic import Output


class PointsOutput(Output):
    def __init__(self,y_exp, system=None, constant_values = None,state_variables = None,dot = None):
        system = system or pynamics.get_system()
        dot = dot or [system.newtonian.x,system.newtonian.y]
        y_exp = [item for item2 in y_exp for item in [item2.dot(dotitem) for dotitem in dot]]
        Output.__init__(self,y_exp, system, constant_values,state_variables)

    def calc(self,x,t):
        Output.calc(self,x,t)
        self.y.resize(self.y.shape[0],int(self.y.shape[1]/2),2)
        return self.y

    def animate(self,fps = 30,stepsize=1,scale=1,movie_name = None,*args,**kwargs):
        # import numpy as np
        import matplotlib.pyplot as plt
        
        from matplotlib import animation, rc

        y = self.y * scale

        f = plt.figure()
        ax = f.add_subplot(1,1,1,aspect = 'equal',autoscale_on=False)
#        ax.axis('equal')
        limits   = [y[:,:,0].min(),y[:,:,0].max(),y[:,:,1].min(),y[:,:,1].max()]
        ax.axis(limits)

#        y = self.y[::stepsize]
        
        line, = ax.plot([], [], *args,**kwargs)
        
        def init():
            line.set_data([], [])
            return (line,)
        
        def run(item):
            line.set_data(*(item.T))
#            ax.axis('equal')
#            ax.axis(limits)
            return (line,)

        self.anim = animation.FuncAnimation(f, run, init_func=init,frames=y[::stepsize], interval=1/fps*1000, blit=True,repeat = True,repeat_delay=3000)        
        if movie_name is not None:
            self.anim.save(movie_name, fps=fps,writer='ffmpeg')

        # return ax
            
    def plot_time(self,stepsize=1,fig = None,linestyle='b-'):
        import matplotlib.pyplot as plt
        fig = fig or plt.figure()
        ax = fig.add_subplot()
        try:
            self.y
        except AttributeError:
            self.calc()

        ax.plot(self.y[::stepsize,:,0].T,self.y[::stepsize,:,1].T, linestyle = 'solid')
        ax.axis('equal')     
        return ax
    
    def plot_time_c(self,stepsize=1,fig = None,linestyle='solid',color='',displacement=[0,0],amplify=1,ax1=[]):
        import matplotlib.pyplot as plt
        fig = fig or plt.figure()
        ax = fig.add_subplot()
        try:
            self.y
        except AttributeError:
            self.calc()
        if color == '':
            ax.plot(self.y[::stepsize,:,0].T*amplify+displacement[0],self.y[::stepsize,:,1].T*amplify+displacement[1],linestyle=linestyle)
            ax.axis('equal')
        else:
            ax.plot(self.y[::stepsize,:,0].T*amplify+displacement[0],self.y[::stepsize,:,1].T*amplify+displacement[1],linestyle=linestyle,color=color)
            ax.axis('equal')
        return ax

    def point_anim(y1, fps=30, stepsize=1, scale=1e3, movie_name=None,title="None", *args, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib import animation, rc
        f = plt.figure()
        y = y1 * scale
        ax = f.add_subplot(1, 1, 1, aspect='equal', autoscale_on=False)
        limits = [y[:, :, 0].min(), y[:, :, 0].max(), y[:, :, 1].min(), y[:, :, 1].max()]
        ax.axis(limits)
        plt.title(title)
        line, = ax.plot([], [], *args, **kwargs)
        def init():
            line.set_data([], [])
            return (line,)

        def run(item):
            line.set_data(*(item.T))
            #            ax.axis('equal')
            #            ax.axis(limits)
            return (line,)

        anim = animation.FuncAnimation(f, run, init_func=init, frames=y[::stepsize], interval=1 / fps * 1000, blit=True,
                                       repeat=True, repeat_delay=3000)
        if movie_name is not None:
            anim.save(movie_name, fps=fps, writer='ffmpeg')