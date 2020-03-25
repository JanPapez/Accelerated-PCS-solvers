"""
script for the experiments from the paper Papez, Grigori, Stompor 2020: "Accelerating
    linear system solvers for time domain component separation of cosmic microwave
    background data", submitted to Astronomy & Astrophysics

this script defines the solvers for the experiments: standard PCG and PCG with
    deflation and subspace recycling
"""

import numpy as np
import param
import scipy.linalg as spla
import time

from PCS_functions import *

__author__ = "Jan Papez"
__license__ = "GPL"
__version__ = "1.0.0 of March 25, 2020"
__maintainer__ = "Jan Papez"
__email__ = "jan@papez.org"
__status__ = "Code for the experiments for the paper"


def stopcrit_maxiter(relres,i):
    # the default stopping criterion requiring performing maxiter iterations
    return False
#end stopcrit_maxiter

def stopcrit_relres(relres,i, TOL):
    # the stopping criterion on relative residual norm(r_i)/norm(r_0)
    out = False
    if relres[i+1]/relres[0] < TOL:
        out = True
    return out
#end stopcrit_relres

def stopcrit_relresb(relres,i, TOL):
    # the stopping criterion on relative residual norm(r_i)/norm(rhs)
    out = False
    if relres[i+1]/relres[-1] < TOL:
        out = True
    return out
#end stopcrit_relres

def stopcrit_absres(relres,i, TOL):
    # the stopping criterion on (absolute) residual
    out = False
    if relres[i+1] < TOL:
        out = True
    return out
#end stopcrit_absres

# PCG method with block diagonal preconditioner (At diag(invN) A)^-1
def PCG_BDprec(fullA, d, x0, maxiter, stopcrit=stopcrit_maxiter):
    """preconditioned conjugate gradient method with block-diagonal
    preconditioner for map-making problem"""
    #
    # evaluate the corresponding right-hand side
    r = fullA.full_depointing( fullA.full_noise_weightening( d ))
    normb = inner_signals(r, r)**0.5
    # build the block diagonal preconditioner
    BD_prec = BD_preconditioner(fullA)
    # the initial approximation and the residual
    if len(x0) == 0: #x0 is empty
        x = np.zeros((param.Np, param.Nsigcomp))
        #r = r - 0
    elif (len(x0) == 6) and (x0 == "binned"): #x0 is the binned map x_0 = (A^t diag(invN) A)^-1 A^t diag(invN) d
        x = BD_prec.apply_inverse( fullA.full_depointing( fullA.N.return_diag_invN() * d) )
        r -= fullA.apply_full_matrix(x)
    else:
        x = x0.copy()
        r -= fullA.apply_full_matrix(x)
    # endifs
    res_normPCG = np.zeros(maxiter + 1)
    res_normPCG[0] = inner_signals(r, r)**0.5
    res_normPCG[-1] = normb
    # start PCG algorithm
    z_Ap = BD_prec.apply_inverse(r) # in the script we use z_Ap for z and for A*p vectors
    p = z_Ap.copy()
    zr_old = inner_signals(z_Ap, r)
    #
    for i in np.arange(maxiter):
        z_Ap = fullA.apply_full_matrix(p)
        alpha = zr_old/inner_signals(p, z_Ap)
        alphazr = alpha*zr_old
        x += alpha * p
        r -= alpha * z_Ap
        z_Ap = BD_prec.apply_inverse(r)
        zr_new = inner_signals(z_Ap, r)
        beta = zr_new/zr_old
        zr_old = zr_new
        p = z_Ap + beta*p
        #
        res_normPCG[i+1] = inner_signals(r, r)**0.5
        #
        if stopcrit(res_normPCG, i):
            res_normPCG = res_normPCG[0:i+2]
            print("* stop.crit. satisfied in iteration {}".format(i+1))
            break
    # endfor
    return x, res_normPCG, normb
# end PCG_BDprec



# PCG method with block diagonal preconditioner (At diag(invN) A)^-1 and projection preconditioner
def PCG_BDplusprojectionprec(fullA, d, x0, Z, maxiter, stopcrit=stopcrit_maxiter, dimP=0, k=0, approximation="harmonic", deflation="def1", AZknown=[]):
    """preconditioned conjugate gradient method with block-diagonal
    preconditioner and the deflation for the vectors Z in map-making problem"""
    # using the notation of [Tang, Nabben, Vuik, Erlanga, 2009]
    #
    deflation = deflation.lower()
    approximation = approximation.lower()
    #
    # evaluate the corresponding right-hand side
    r = fullA.full_depointing( fullA.full_noise_weightening( d ))
    normb = inner_signals(r, r)**0.5
    rhs = r.copy()
    #
    P = []
    AP = []
    AZ = list(Z) # copy the list to get list of the same size
    #
    """building the deflation operators"""
    #
    # build the "coarse" matrix inv(Z^t A Z)
    coarse_matrix = np.zeros((len(Z),len(Z)))
    for i in np.arange(len(Z)):
        if (len(AZknown) > 0):
            AZ[i] = AZknown[i]
        else:
            AZ[i] = fullA.apply_full_matrix(np.array(Z[i]))
        #endif
        for j in np.arange(len(Z)):
            coarse_matrix[i,j] = inner_signals(AZ[i], np.array(Z[j]))
        # endfor
    #endfor
    if (not np.allclose(coarse_matrix, coarse_matrix.T, rtol=1e-15)):
        print("*** warning: coarse matrix is not symmetric !")
    # (V E^-1 W^T) v where E = Z^T A Z = coarse matrix
    def apply_VcoarsesolveWt(v, V, W):
        out = np.zeros_like(v)
        Wtv = np.zeros(len(W))
        for i in np.arange(len(W)):
            Wtv[i] = inner_signals(np.array(W[i]), v)
        # endfor
        temp = np.linalg.solve(coarse_matrix, Wtv)
        for i in np.arange(len(V)):
            out += np.array(V[i])*temp[i]
        # endfor
        return out
    # then we have
    #   Q*v  := apply_VcoarsesolveWt(v, Z, Z)
    #   QA*v := apply_VcoarsesolveWt(v, Z, AZ)
    #   AQ*v := apply_VcoarsesolveWt(v, AZ, Z)
    #
    # define the projector P
    def apply_P(v):
        return (v - apply_VcoarsesolveWt(v, AZ, Z))
    # define the transpose projector P^T
    def apply_Pt(v):
        return (v - apply_VcoarsesolveWt(v, Z, AZ))
    # define the modification of the initial guess or the computed solution as
    #   x := Q*b + P^t*x
    def modify_x(v):
        return apply_VcoarsesolveWt(rhs, Z, Z) + apply_Pt(v)
    #
    """ the residual correction suggested by [Saad, Yeung, Erhel, Guyomarc'h 2000] """
    # build the matrix used for the residual correction
    ZtZ = np.zeros((len(Z),len(Z)))
    for i in np.arange(len(Z)):
        for j in np.arange(len(Z)):
            ZtZ[i,j] = inner_signals(np.array(Z[i]), np.array(Z[j]))
        # endfor
    #endfor
    def correct_residuals( r ):
        out = np.zeros_like(r)
        Ztr = np.zeros(len(Z))
        for i in np.arange(len(Z)):
            Ztr[i] = inner_signals(np.array(Z[i]), r)
        # endfor
        temp = np.linalg.solve(ZtZ, Ztr)
        for i in np.arange(len(Z)):
            out += np.array(Z[i])*temp[i]
        # endfor
        return r - out
    #
    """ building the preconditioners for various deflations techniques"""
    # build the block diagonal preconditioner
    M = BD_preconditioner(fullA)
    #
    def M1(v):
        if (deflation == "def1"):
            out = M.apply_inverse(v)
        elif (deflation == "def2"):
            out = M.apply_inverse(v)
        elif (deflation == "a-def1"):
            out = M.apply_inverse( apply_P(v) ) + apply_VcoarsesolveWt(v, Z, Z)
        return out
    #end M1
    #
    def M2(v):
        if (deflation == "def1"):
            out = v.copy()
        elif (deflation == "def2"):
            out = apply_Pt(v)
        elif (deflation == "a-def1"):
            out = v.copy()
        return out
    #end M2
    #
    def M3(v):
        if (deflation == "def1"):
            out = apply_P(v)
        elif (deflation == "def2"):
            out = v.copy()
        elif (deflation == "a-def1"):
            out = v.copy()
        return out
    #end M3
    #
    def init_vec(v):
        if (deflation == "def1"):
            out = v.copy()
        elif (deflation == "def2"):
            out = modify_x(v)
        elif (deflation == "a-def1"):
            out = v.copy()
        return out
    #end init_vec
    def output_vec(v):
        if (deflation == "def1"):
            out = modify_x(v)
        elif (deflation == "def2"):
            out = v.copy()
        elif (deflation == "a-def1"):
            out = v.copy()
        return out
    #end output_vec
    """ running deflated PCG """
    # the initial approximation and the residual
    if len(x0) == 0: #x0 is empty
        x = np.zeros((param.Np, param.Nsigcomp))
    elif (len(x0) == 6) and (x0 == "binned"): #x0 is the binned map x_0 = (A^t diag(invN) A)^-1 A^t diag(invN) d
        x = M.apply_inverse( fullA.full_depointing( fullA.N.return_diag_invN() * d) )
    else:
        x = x0.copy()
    # endif
    x = init_vec(x)
    r = M3(r - fullA.apply_full_matrix(x)) #in def1, the residual must be projected !!!
    #
    res_normPCG = np.zeros(maxiter + 1)
    res_normPCG[0] = inner_signals(r, r)**0.5
    res_normPCG[-1] = normb
    # start PCG algorithm
    y = M1(r)
    p = M2(y)
    #
    yr = inner_signals(y,r)
    for i in np.arange(maxiter):
        w = M3(fullA.apply_full_matrix(p))
        if i < dimP:
            if (deflation == "def1"):
                P.append(apply_Pt(p))
            else:
                P.append(p)
            #endif
            AP.append(w)
        #endif
        alpha = yr/inner_signals(p, w)
        alphazr = alpha*yr
        x += alpha * p
        r -= alpha * w
        #r = correct_residuals( r ) # use if necessary; see [Saad, Yeung, Erhel, Guyomarc'h 2000]
        y = M1(r)
        beta = 1.0 / yr
        yr = inner_signals(y,r)
        beta *= yr
        p = M2(y) + beta*p
        #
        res_normPCG[i+1] = inner_signals(r, r)**0.5
        #
        if stopcrit(res_normPCG, i):
            res_normPCG = res_normPCG[0:i+2]
            print("* stop.crit. satisfied in iteration {}".format(i+1))
            break
        #endif
    # endfor
    if len(res_normPCG) == maxiter+1:
        print("* maximum number of iterations reached")
    #
    x = output_vec(x)
    #
    """(harmonic) Ritz approximation and checking the results"""
    #
    if approximation != "none":
        # concatenate Z and P
        if param.multiple_eigvals:
            Z = Z[::2]
            AZ = AZ[::2]
        #endif
        U = Z+P
        AU = AZ+AP
        F = np.zeros((len(U),len(U)))
        G = F.copy()
        if approximation == "harmonic":
            for i in np.arange(len(U)):
                for j in np.arange(len(U)):
                    F[i,j] = inner_signals(AU[i], U[j])
                    G[i,j] = inner_signals(AU[i], M.apply_inverse(AU[j]))
                # endfor
            #endfor
        elif approximation == "ritz":
            for i in np.arange(len(U)):
                for j in np.arange(len(U)):
                    F[i,j] = inner_signals(U[i], M.apply(U[j]))
                    G[i,j] = inner_signals(AU[i], U[j])
        #endif
        if len(U) == 1:
            tempk = 0
            Ritz_values = G[0][0]/F[0][0]
            Ritz_vectors = np.ones((1,1))
        else:
            tempk = min(k, len(U)-1)
            Ritz_values, Ritz_vectors = spla.eigh(G, F, eigvals=(0,tempk))
        #endif
        Unew = []
        for i in np.arange(tempk+1):
            tmp = np.zeros_like(U[0])
            for j in np.arange(len(U)):
                tmp += (U[j]*Ritz_vectors[j,i])
            #endfor
            Unew.append(tmp)
        #endfor
        # compute error
        print("* number of approximated eigenvalues: {}".format(tempk))
        URv = np.zeros_like(Unew)
        for i in np.arange(tempk+1):
            URv[i] = Unew[i]*Ritz_values[i]
        # endfor
        AUnew = []
        for i in np.arange(tempk+1):
            tmp = np.zeros_like(AU[0])
            for j in np.arange(len(AU)):
                tmp += (AU[j]*Ritz_vectors[j,i])
            #endfor
            AUnew.append(tmp)
        #endfor
        for i in np.arange(len(AUnew)):
            AUnew[i] = M.apply_inverse(AUnew[i])
        error = spla.norm(AUnew - URv)
        print("* error of approximated eigenvalues: {}".format(error))
        # generate a "copy" of the deflation vectors in U if the eigenvalues are multiple
        if param.multiple_eigvals:
            UU = []
            for i in np.arange(len(Unew)):
                UU.append(Unew[i])
                UU.append(np.array([ -1*Unew[i][:,1], Unew[i][:,0], -1*Unew[i][:,3], Unew[i][:,2], -1*Unew[i][:,5], Unew[i][:,4]]).T)
            #endfor
            Unew = UU
        #endif
    else:
        Unew = []
    #endif approximation of eigenvalues
    return x, res_normPCG, normb, Unew
# end PCG_BDplusprojectionprec
