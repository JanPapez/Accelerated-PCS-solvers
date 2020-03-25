#!/usr/bin/env python

"""
script for the experiments from the paper Papez, Grigori, Stompor 2020: "Accelerating
    linear system solvers for time domain component separation of cosmic microwave
    background data", submitted to Astronomy & Astrophysics

this script runs standard PCG for a sequence of PCS systems
"""

import numpy as np
import os
import time
import sys

import param
from PCS_functions import *
from PCS_solvers import *

__author__ = "Jan Papez"
__license__ = "GPL"
__version__ = "1.0.0 of March 25, 2020"
__maintainer__ = "Jan Papez"
__email__ = "jan@papez.org"
__status__ = "Code for the experiments for the paper"


#### setting the default parameters
params = ['case','niter','scalen','Betas_index','ncomp','TOL']
param.case = "case0"  # test case
param.scalen = 100   # noise scaling; it seems reasonable to set scalen in [1000,10000]
param.niter = 200                  #
TOL_log = 8; param.TOL = 10**(-TOL_log)
param.Betas_index = 6
param.ncomp = 2

Beta_init = 0;

spec_params_def = { 'nu_ref':150.0, 'Bd':1.59, 'Td':19.6, 'h_over_k':0.0479924,\
    'drun':0.0, 'Bs':-3.1, 'srun':0.0, 'cst':56.8, 'nu_pivot_sync_curv':23.0 }

Betas_orig = [spec_params_def['Bd'], spec_params_def['Bs']]

frequencies = [30,40,90,150,220,270]


if __name__== "__main__":
    my_parser()

print(" ")
print("START AT "+time.ctime())
print(" ")

print('************ parameters ***************')
for i in np.arange(len(params)):
    print(str(params[i]+" = "+str(eval('param.'+params[i]))))
print('***************************************')


print("*** reading the input files")
[s, data, A, N, Betas] = read_data(param.case, len(frequencies)) # noise -> data

# scaling the noise
if param.scalen > 0:
    data *= param.scalen
    for i in np.arange(len(N)):
        N[i].invspectrum *= 1./(param.scalen**2)
    #end for
#endif

# set the input signal to zero if not loaded
if len(s) == 0:
    s = np.zeros((param.Np,param.Nsigcomp))
# endif

#### assembling the original (target) matrix and rhs using the known Betas

print("*** building the mixing-pointing matrix with the target value of parameters")
origA = full_problem_matrix(A,N,frequencies,spectral_parameters=spec_params_def)

print("*** constructing the rhs = observed data")
if param.scalen > 0:
    data += origA.full_pointing(s)
else:
    data = origA.full_pointing(s)
#end if

#### iterating over the sampled Betas

for Beta_i in np.arange(Beta_init, param.Betas_index):
    #
    print("**** iteration: {}, Beta = [{}, {}]".format(Beta_i, Betas[Beta_i, 0], Betas[Beta_i, 1]))
    #
    spec_params_def['Bd'] = Betas[Beta_i, 0]
    spec_params_def['Bs'] = Betas[Beta_i, 1]
    #
    print("* building the full matrix with sampled parameters")
    fullA = full_problem_matrix(A,N,frequencies,spectral_parameters=spec_params_def)
    #
    print("* running the PCG with zero initial guess")
    xPCG, res_normPCG, normb = PCG_BDprec(fullA, data, [], param.niter,
        stopcrit=(lambda a,b: stopcrit_relres(a,b, param.TOL)))
    print(res_normPCG)
    rel_resnorm = res_normPCG/normb
    #
#endfor


print(" ")
print("END AT "+time.ctime())
print(" ")
