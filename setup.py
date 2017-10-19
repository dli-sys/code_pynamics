# -*- coding: utf-8 -*-
'''
Written by Daniel M. Aukes and CONTRIBUTORS
Email: danaukes<at>asu.edu.
Please see LICENSE for full license.
'''

from setuptools import setup
import sys
import shutil

shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)
shutil.rmtree('pynamics.egg-info', ignore_errors=True)


packages = [ 'pynamics']

package_data = {}
package_data['pynamics'] = []

setup_kwargs = {}
setup_kwargs['name']='pynamics'
setup_kwargs['version']='0.0.1'
setup_kwargs['classifiers']=['Programming Language :: Python','Programming Language :: Python :: 3']   
setup_kwargs['description']='Foldable robotics is a package for designing and analyzing foldable laminate robots'
setup_kwargs['author']='Dan Aukes'
setup_kwargs['author_email']='danaukes@danaukes.com'
setup_kwargs['url']='https://github.com/idealabasu/code_pynamics'
setup_kwargs['license']='MIT'
setup_kwargs['packages']=packages
setup_kwargs['package_dir']={'pynamics' : 'python/pynamics'}
setup_kwargs['package_data'] = package_data

  
setup(**setup_kwargs)