#!/usr/bin/env python

"""
script for the experiments from the paper Papez, Grigori, Stompor 2020: "Accelerating
    linear system solvers for time domain component separation of cosmic microwave
    background data", submitted to Astronomy & Astrophysics

this script runs deflated PCG with subspace recycling (the final algorithm proposed)
    in the paper) for a sequence of PCS problems
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
param.scalen = 100.0   # noise scaling; set 0 for no noise
param.niter = 200                  #
TOL_log = 8; param.TOL = 10**(-TOL_log)
param.Betas_index = 3
param.ncomp = 2

Beta_init = 0;      # the index of sampled parameters where to start the solver

param.dimP = 100 # number of vectors kept for approximating the eigenvectors
param.k = 5    # number of eigenvectors to approximate

# chosing the variant how the initial guess is chosen
#variant = "cont"   # initial guess as in Section 3.4.1
variant = "adapt"   # initial guess as in Section 3.4.2
#varaint = ""       # start always with zero initial guess

varpam = variant + "_dimP_"+str(param.dimP)+"_k_"+str(param.k)

spec_params_def = { 'nu_ref':150.0, 'Bd':1.59, 'Td':19.6, 'h_over_k':0.0479924,\
    'drun':0.0, 'Bs':-3.1, 'srun':0.0, 'cst':56.8, 'nu_pivot_sync_curv':23.0 }

Betas_orig = [spec_params_def['Bd'], spec_params_def['Bs']]

frequencies = [30,40,90,150,220,270]
param.nfrequencies = len(frequencies)


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
data += origA.full_pointing(s)

rel_resnorm_history = []    # relative residual appended for all systems
xPCG = []       # current approximation
U = []          # deflation vectors

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
    print("* running the PCG with deflation")
    if variant == "cont":
        xPCG, res_normPCG, normb, U = PCG_BDplusprojectionprec(fullA, data, xPCG, U, param.niter,
            stopcrit=(lambda a,b: stopcrit_relresb(a,b, param.TOL)), dimP=param.dimP, k=param.k)
        #
    elif variant == "adapt":
        if Beta_i > Beta_init:
            print("* modifying the computed solution for the new system")
            xPCG = apply_pseudoinverseM(fullA.weights, xPCG)
        #endif
        xPCG, res_normPCG, normb, U = PCG_BDplusprojectionprec(fullA, data, xPCG, U, param.niter,
            stopcrit=(lambda a,b: stopcrit_relresb(a,b, param.TOL)), dimP=param.dimP, k=param.k)
    else:
        xPCG, res_normPCG, normb, U = PCG_BDplusprojectionprec(fullA, data, [], U, param.niter,
            stopcrit=(lambda a,b: stopcrit_relresb(a,b, param.TOL)), dimP=param.dimP, k=param.k)
        #
    #endif
    #
    print(res_normPCG)
    rel_resnorm_history = np.append(rel_resnorm_history, res_normPCG/normb)
    #
    if variant == "adapt":
        xPCG = apply_mixing(fullA.weights, xPCG)
    #endif
#endfor

print(" ")
print("END AT "+time.ctime())
print(" ")
