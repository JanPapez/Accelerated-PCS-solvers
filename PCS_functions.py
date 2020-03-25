"""
script for the experiments from the paper Papez, Grigori, Stompor 2020: "Accelerating
    linear system solvers for time domain component separation of cosmic microwave
    background data", submitted to Astronomy & Astrophysics

this script defines the auxiliary functions for the experiments
"""

import numpy as np
import os
import param
import copy
import sys
import argparse

import matrix_operations_wrapper

units = 'CMB-CMB'
whatfft = 'fftw_inC' #variant of fft; other options: "numpy", "scipy" and "scipyr"
#import scipy.fftpack as scfft

__author__ = "Jan Papez"
__license__ = "GPL"
__version__ = "1.0.0 of March 25, 2020"
__maintainer__ = "Jan Papez"
__email__ = "jan@papez.org"
__status__ = "Code for the experiments for the paper"


def BB_factor_computation(nu):
	# computing the CMB->RJ conversion factor
    BB_factor = 1.0
    cst = 56.8
    if units == 'CMB-CMB':
        BB_factor = (nu/cst)**2*np.exp(nu/cst)/(np.exp(nu/cst)-1)**2
	return BB_factor
#enddef


def read_data(case, nfrequencies):
    """ function for reading the data"""
    #
    def my_loadtxt(file_):
        return np.load(file_)
    # end my_loadtxt
    #
    param.pathtopointing = param.pathtopointing+case+"/"
    #
    # loading the processed signals
    cmb = np.loadtxt(param.pathtosignal+"cmb_observed.txt")/BB_factor_computation(150.0)
    dust = np.loadtxt(param.pathtosignal+"dust_observed.txt")/BB_factor_computation(150.0)
    sync = np.loadtxt(param.pathtosignal+"sync_observed.txt")/BB_factor_computation(150.0)
    s = np.hstack((cmb[:,3-param.ncomp:3], dust[:,3-param.ncomp:3], sync[:,3-param.ncomp:3]))
    #
    param.Nsigcomp = param.ncomp * 3
    #
    """the scanning strategy"""
    #
    param.ndetectors = 2
    #
    """read the pointing matrix A """
    #
    p = [my_loadtxt(param.pathtopointing+"hscan_"+case+"_ns1024_processed.npy"),
        my_loadtxt(param.pathtopointing+"vscan_"+case+"_ns1024_processed.npy")]
    #
    param.Nt = p[0].size
    param.Np = p[0].max() + 1
    #
    twght = []
    #
    if os.path.isfile(param.pathtopointing+"qwght_hscan_"+case+"_short.npy"):
        qwght_hscan = my_loadtxt(param.pathtopointing+"qwght_hscan_"+case+"_short.npy")
    else:
        qwght_hscan = my_loadtxt(param.pathtopointing+"qwght_hscan_"+case+".npy")
    #endif
    if os.path.isfile(param.pathtopointing+"qwght_vscan_"+case+"_short.npy"):
        qwght_vscan = my_loadtxt(param.pathtopointing+"qwght_vscan_"+case+"_short.npy")
    else:
        qwght_vscan = my_loadtxt(param.pathtopointing+"qwght_vscan_"+case+".npy")
    #endif
    qwght = [qwght_hscan, qwght_vscan]
    del qwght_hscan, qwght_vscan
    #
    if os.path.isfile(param.pathtopointing+"uwght_hscan_"+case+"_short.npy"):
        uwght_hscan = my_loadtxt(param.pathtopointing+"uwght_hscan_"+case+"_short.npy")
    else:
        uwght_hscan = my_loadtxt(param.pathtopointing+"uwght_hscan_"+case+".npy")
    #endif
    if os.path.isfile(param.pathtopointing+"uwght_vscan_"+case+"_short.npy"):
        uwght_vscan = my_loadtxt(param.pathtopointing+"uwght_vscan_"+case+"_short.npy")
    else:
        uwght_vscan = my_loadtxt(param.pathtopointing+"uwght_vscan_"+case+".npy")
    #endif
    uwght = [uwght_hscan, uwght_vscan]
    del uwght_hscan, uwght_vscan
    #
    A = pointing_matrix(p, twght, qwght, uwght)
    #
    """defining the fft operations"""
    my_fft = my_fft_operations(whatfft)
    #
    """read noise covariance matrix N and the noise"""
    N = []
    n = np.zeros(param.Nt*param.ndetectors*nfrequencies)
    for i in np.arange(nfrequencies):
        invspectrum = my_loadtxt(param.pathtonoise+"inverse_pf"+str(i)+".npy")/(param.Nt)
        Tisizes = np.array([invspectrum.size, invspectrum.size])
        if (whatfft == "numpy"):
            invspectrum = invspectrum[0:param.Nt/2+1]
        elif (whatfft == "scipyr"):
            tmp = np.arange(param.Nt/2+1)
            tmp = np.array([tmp,tmp])
            invspectrum = invspectrum[tmp.T.flatten()[1:-1]]
        #endif
        N.append(noise_covariance_matrix(np.ascontiguousarray(invspectrum), Tisizes, my_fft))
        #
        n_hscan = my_loadtxt(param.pathtonoise+"noisestream_hscan_sim"+str(i)+".npy")
        n_vscan = my_loadtxt(param.pathtonoise+"noisestream_vscan_sim"+str(i)+".npy")
        n[i*param.Nt*param.ndetectors:(i+1)*param.Nt*param.ndetectors] = np.append(n_hscan, n_vscan)
    del n_hscan, n_vscan
    #
    """ read sampled Betas """
    Betas = np.load(param.pathtosignal+"sampled_betas.npy")
    #
    return [s, n, A, N, Betas]
#end read_data

#### defining the operations
class my_fft_operations:
    """ the class for fft and ifft"""
    def __init__(self, type):
        """ intialize the class """
        self.type = type
        #
    #end
    def ifft_w_fft(self, v, spectrum):
        # apply fft, weight by the noise covariance, and apply ifft
        # various FFTs require various size/shapes of the vector "spectrum",
        #   which has been modified right after loading
        out = []
        if self.type == 'scipy':
            temp = scfft.fft(v)
            temp *= spectrum
            out = scfft.ifft(temp)
        elif self.type == 'scipyr':
            temp = scfft.rfft(v)
            temp *= spectrum
            out = scfft.irfft(temp)
        elif self.type == 'numpy':
            temp = np.fft.rfft(v)
            temp *= spectrum
            out = np.fft.irfft(temp)
        elif self.type == 'fftw_inC':
            out = np.zeros_like(v)
            matrix_operations_wrapper.apply_invN(v, out, spectrum)
            out /= param.Nt #this scaling is necessary due to using FFTW3
        #endif
        return out.real
    #end ifft_w_fft
#end class my_fft_operations


#### defining the operations
class pointing_matrix:
    """ the pointing matrix and the operations with"""
    def __init__(self, p, twght, qwght, uwght):
        """ intialize the pointing matrix """
        self.p = [np.ascontiguousarray(p[0]).astype(np.double), np.ascontiguousarray(p[1]).astype(np.double)]
        self.twght = []
        self.qwght = [np.ascontiguousarray(qwght[0]).astype(np.double), np.ascontiguousarray(qwght[1]).astype(np.double)]
        self.uwght = [np.ascontiguousarray(uwght[0]).astype(np.double), np.ascontiguousarray(uwght[1]).astype(np.double)]
    # end init
    #
    def apply_A(self, v, weights):
        """ apply the pointing matrix to a vector v ("pointing") """
        out = np.empty(param.Nt*2).astype(np.double) #for 2 detectors
        matrix_operations_wrapper.apply_A_twodetectors(self.p[0], self.p[1], self.qwght[0], self.qwght[1], self.uwght[0], self.uwght[1], v.flatten(), out[0:param.Nt], out[param.Nt:], weights)
        return out
    # end apply_A
    #
    def apply_At(self, w, weights):
        """ apply the transpose of the pointing matrix to a vector w ("depointing") """
        out = np.zeros(param.Np * param.Nsigcomp).astype(np.double)
        matrix_operations_wrapper.apply_At_twodetectors(self.p[0], self.p[1], self.qwght[0], self.qwght[1], self.uwght[0], self.uwght[1], w[0:param.Nt].flatten(), w[param.Nt:].flatten(), out, weights)
        return out.reshape(param.Np, param.Nsigcomp)
    # end apply_At
    #
# end class pointing_matrix

class noise_covariance_matrix:
    """ the noise covariance matrix and the operations with """
    def __init__(self, invspectrum, Tisizes, my_fft):
        """intialize the spectrum and the sizes of its Toeplitz blocks"""
        self.invspectrum = np.ascontiguousarray(invspectrum)
        self.nTi = Tisizes.size
        self.Tisizes_accum = np.zeros(self.nTi + 1).astype('int')
        for i in np.arange(self.nTi):
            self.Tisizes_accum[i+1] = self.Tisizes_accum[i] + Tisizes[i]
        # endfor
        if (self.Tisizes_accum[-1] != param.ndetectors * param.Nt):
            raise ValueError("wrong Tisizes")
        # endif
        self.my_fft = my_fft
    # end init
    #
    def apply_inverse(self, v):
        """ apply the inverse of the noise covariance matrix ("noise-weighting")"""
        out = np.zeros_like(v)
        #
        for i in np.arange(self.nTi):
            begin = int(self.Tisizes_accum[i])
            end = int(self.Tisizes_accum[i+1])
            out[begin:end] = self.my_fft.ifft_w_fft(v[begin:end], self.invspectrum)
        #endfor
        return out
    # end apply_inverse
    #
    def return_diag_invN(self):
        """return the diagonal of the noise covariance matrix"""
        # diagonal entry = e_1 ^T invN e_1
        #
        e1 = np.zeros(param.Nt)
        e1[0] = 1.
        out = self.my_fft.ifft_w_fft(e1, self.invspectrum)
        #
        return out[0]
    # end return_diag_invN
# end class noise_covariance_matrix

class white_noise_covariance_matrix:
    """ the noise covariance matrix and the operations with """
    def __init__(self, sigma, Tisizes):
        self.sigma = sigma
        self.nTi = Tisizes.size
        self.Tisizes_accum = np.zeros(self.nTi + 1).astype('int')
        for i in np.arange(self.nTi):
            self.Tisizes_accum[i+1] = self.Tisizes_accum[i] + Tisizes[i]
        # endfor
        if (self.Tisizes_accum[-1] != param.ndetectors * param.Nt):
            raise ValueError("wrong Tisizes")
        # endif
    # end init
    #
    def apply_inverse(self, v):
        """ apply the inverse of the noise covariance matrix ("noise-weighting")"""
        out = np.zeros_like(v)
        #
        for i in np.arange(self.nTi):
            begin = int(self.Tisizes_accum[i])
            end = int(self.Tisizes_accum[i+1])
            out[begin:end] = v[begin:end]/(self.sigma**2)
        #endfor
        return out
    # end apply_inverse
    #
    def return_diag_invN(self):
        """return the diagonal of the noise covariance matrix"""
        #
        return 1./(self.sigma**2)
    #
    def return_noise_realization(self):
        return self.sigma * np.random.randn(param.Nt)
    # end return_noise_realization
# end class white_noise_covariance_matrix


class full_problem_matrix:
    """the matrix of the problem that involves all the observed frequencies"""
    def __init__(self, A, N, frequencies, spectral_parameters):
        #
        self.frequencies = frequencies
        self.N_freq = len(frequencies)
        self.A = A
        self.N = N
        #
        weights = np.zeros((self.N_freq, param.Nsigcomp)) #mixing weights depending on the frequency
        #
        for f_index in np.arange(self.N_freq):
            spectral_parameters['nu'] = frequencies[f_index]*1.0
            nu = spectral_parameters['nu']*1.0
            cst = spectral_parameters['cst']*1.0
            nu_ref = spectral_parameters['nu_ref']*1.0
            Bd = spectral_parameters['Bd']*1.0
            Td = spectral_parameters['Td']*1.0
            h_over_k = spectral_parameters['h_over_k']*1.0
            drun = spectral_parameters['drun']*1.0
            Bs = spectral_parameters['Bs']*1.0
            srun = spectral_parameters['srun']*1.0
            nu_pivot_sync_curv = spectral_parameters['nu_pivot_sync_curv']*1.0
            #
            if param.ncomp == 3:
                weights[f_index,1] = weights[f_index,2] = \
                    (nu / cst) ** 2 * ( np.exp ( nu / cst ) ) / ( ( np.exp ( nu / cst ) - 1 ) ** 2 ) \
                    * BB_factor_computation(150.0)/BB_factor_computation(nu)
                weights[f_index,4] = weights[f_index,5] = \
                    ( np.exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( np.exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * np.log( nu/nu_ref ) ) \
                    * BB_factor_computation(150.0)/BB_factor_computation(nu)
                weights[f_index,7] = weights[f_index,8] = \
                    ( nu / nu_ref ) ** (Bs + srun * np.log( nu / nu_pivot_sync_curv ) )\
                    * BB_factor_computation(150.0)/BB_factor_computation(nu)
                weights[f_index,0] = weights[f_index,1]
                weights[f_index,3] = weights[f_index,4]
                weights[f_index,6] = weights[f_index,7]
            else: #param.ncomp == 2
                weights[f_index,0] = weights[f_index,1] = \
                    (nu / cst) ** 2 * ( np.exp ( nu / cst ) ) / ( ( np.exp ( nu / cst ) - 1 ) ** 2 ) \
                    * BB_factor_computation(150.0)/BB_factor_computation(nu)
                weights[f_index,2] = weights[f_index,3] = \
                    ( np.exp( nu_ref / (Td / h_over_k ) ) - 1 ) / ( np.exp( nu / ( Td / h_over_k ) ) - 1 ) * ( nu / nu_ref ) ** ( 1 + Bd + drun * np.log( nu/nu_ref ) ) \
                    * BB_factor_computation(150.0)/BB_factor_computation(nu)
                weights[f_index,4] = weights[f_index,5] = \
                    ( nu / nu_ref ) ** (Bs + srun * np.log( nu / nu_pivot_sync_curv ) ) \
                    * BB_factor_computation(150.0)/BB_factor_computation(nu)
            # endif
        #endfor
        self.weights = weights
    #end __init__
    #
    def apply_full_matrix(self, v):
        #
        v_in = np.ascontiguousarray(v.ravel())
        out = np.ascontiguousarray(np.zeros_like(v_in), dtype=np.double)
        for f_index in np.arange(self.N_freq):
            matrix_operations_wrapper.apply_A_invN_At_twodetectors(self.A.p[0], self.A.p[1], self.A.qwght[0], self.A.qwght[1],
                self.A.uwght[0], self.A.uwght[1], v_in, self.N[f_index].invspectrum, out, self.weights[f_index,:])
        #endfor
        out /= param.Nt
        out = out.reshape(param.Np, param.Nsigcomp)
        return out
    #end apply_full_matrix
    #
    def full_pointing(self, v):
        #
        f_datasize = param.Nt*param.ndetectors
        out = np.zeros(f_datasize*self.N_freq).astype(np.double)
        for f_index in np.arange(self.N_freq):
            out[ (f_datasize*f_index):(f_datasize*(f_index+1)) ]\
                = self.A.apply_A( v,  self.weights[f_index,:] )
        #endfor
        return out
    #end full_pointing
    #
    def full_depointing(self, w):
        #
        out = np.zeros((param.Np, param.Nsigcomp)).astype(np.double)
        f_datasize = param.Nt*param.ndetectors
        for f_index in np.arange(self.N_freq):
            out += self.A.apply_At( w[(f_datasize*f_index):(f_datasize*(f_index+1))], self.weights[f_index,:] )
        #endfor
        return out
    #end full_depointing
    #
    def full_noise_weightening(self, w):
        #
        out = np.zeros_like(w)
        f_datasize = param.Nt*param.ndetectors
        for f_index in np.arange(self.N_freq):
            out[ (f_datasize*f_index):(f_datasize*(f_index+1)) ]\
             = self.N[f_index].apply_inverse( w[(f_datasize*f_index):(f_datasize*(f_index+1))] )
        #endfor
        return out
    #end full_noise_weightening
    #
# end class full_problem_matrix

class BD_preconditioner:
    """block-diagonal preconditioner for the map-making problem,
    we assume that the noise covariance is the same for each detector and frequency
    """
    def __init__(self, fullA):
        """intialize the preconditioner"""
        nblockentries = param.Nsigcomp * param.Nsigcomp #number of entries in the block
        self.prec_diag = np.ascontiguousarray(np.zeros(param.Np * nblockentries))
        self.ifdiagonal = False
        #endif
        #
        for f_index in np.arange(fullA.N_freq):
            invN_diag = np.array(fullA.N[f_index].return_diag_invN())
            for detector in np.arange(param.ndetectors):
                matrix_operations_wrapper.build_Prec_BD(fullA.A.p[detector], fullA.A.qwght[detector], fullA.A.uwght[detector], invN_diag, self.prec_diag, fullA.weights[f_index,:])
            #endfor
        #endfor
        # inverting the diagonal blocks
        for pixel in np.arange(param.Np):
            tmp = self.prec_diag[pixel*nblockentries:(pixel+1)*nblockentries]
            out = np.linalg.inv(tmp.reshape((param.Nsigcomp,param.Nsigcomp)))
            self.prec_diag[pixel*nblockentries:(pixel+1)*nblockentries] = out.flatten()
        #endfor
        self.prec_diag = self.prec_diag.reshape((param.Np, nblockentries))
    # end init
    #
    def apply_inverse(self, v):
        """evaluate inv(At* diag(invN) * A) * v"""
        out = np.empty_like(v)
        if self.ifdiagonal: # the preconditioner is diagonal
            for j in np.arange(param.Nsigcomp):
                out[:, j] = self.prec_diag[:, j] * v[:, j]
            #endfor
        else:
        # if the preconditioner is BLOCK diagonal
            for pixel in np.arange(param.Np):
                tmp = self.prec_diag[pixel, :].reshape((param.Nsigcomp,param.Nsigcomp))
                out[pixel, :] = np.dot(tmp, v[pixel, :])
            # endfor
        #endif
        return out
    # end apply_inverse
    #
    def apply(self, v):
        """evaluate (At* diag(invN) * A) * v"""
        out = np.empty_like(v)
        if self.ifdiagonal: # the preconditioner is diagonal
            for j in np.arange(param.Nsigcomp):
                out[:, j] = v[:, j] / self.prec_diag[:, j]
            #endfor
        else:
        # if the preconditioner is BLOCK diagonal
            for pixel in np.arange(param.Np):
                tmp = self.prec_diag[pixel, :].reshape((param.Nsigcomp,param.Nsigcomp))
                out[pixel, :] = np.linalg.solve(tmp, v[pixel, :])
            # endfor
        #endif
        return out
    # end apply
# end class BD_preconditioner


def inner_signals(s1, s2):
    """evaluate the inner product of two signals, s = [sT, sQ, sU]"""
    out = 0.0
    for i in np.arange(s1.shape[1]):
        out += np.dot(s1[:, i], s2[:, i])
    #endfor
    return out
# end inner_signals


def apply_mixing(weights, v):
    """apply mixing matrix to a signal"""
    out = np.zeros((param.nfrequencies*param.Np,2))
    for i in np.arange(param.nfrequencies):
        out[i*param.Np:(i+1)*param.Np,0] = weights[i,0]*v[:,0] + weights[i,2]*v[:,2] + weights[i,4]*v[:,4]
        out[i*param.Np:(i+1)*param.Np,1] = weights[i,1]*v[:,1] + weights[i,3]*v[:,3] + weights[i,5]*v[:,5]
    #endif
    return out
#end apply_mixing

def apply_mixingT(weights, w):
    """apply transpose of the mixing matrix to a signal"""
    out = np.zeros((param.Np,param.Nsigcomp))
    for j in np.arange(param.nfrequencies):
        for i in np.arange(3):
            out[:,2*i]   += w[j*param.Np:(j+1)*param.Np,0] * weights[j,2*i]
            out[:,2*i+1] += w[j*param.Np:(j+1)*param.Np,1] * weights[j,2*i+1]
        #endif
    #endif
    return out
#end apply_mixingT

def apply_pseudoinverseM(weights, w):
    """apply the pseudoinverse (M^t M)^-1 M^t, where M is the mixing matrix"""
    Mtemp = weights.copy()
    Mtemp = Mtemp[:,0::2]
    new_weights = np.linalg.solve(np.dot(Mtemp.T,Mtemp), Mtemp.T).T
    new_weights = new_weights[:,[0,0,1,1,2,2]]
    out = apply_mixingT(new_weights, w)
    return out
#end apply_pseudoinverseM


# parser for input parameters
def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--case', type=str, help='number of the test case')
    parser.add_argument('-i','--niter', type=int, help='maximal number of iterations')
    parser.add_argument('-n','--scalen', type=np.double, help='scaling of the noise (0 = no noise)')
    parser.add_argument('-s','--ncomp', type=int, help='number of the signal components')
    parser.add_argument('-b','--beta', type=int, help='index of the problem to be solved (-1 = true Betas)')
    parser.add_argument('-t','--TOL', type=np.double, help='tolerance')
    parser.add_argument('-k','--ndefvecs', type=int, help='number of deflated vectors')
    parser.add_argument('-P','--dimP', type=int, help='dimension of the space to compute the deflation vectors')
    args = parser.parse_args()
    #
    if args.case:
        if len(args.case) == 1:
            args.case = "case"+str(args.case)
        param.case = args.case
    if args.niter is not None: param.niter = args.niter
    if args.scalen is not None: param.scalen = args.scalen
    if args.ncomp is not None: param.ncomp = args.ncomp
    if args.beta is not None: param.Betas_index = args.beta
    if args.TOL is not None: param.TOL = args.TOL
    if args.ndefvecs is not None: param.k = args.ndefvecs
    if args.dimP is not None: param.dimP = args.dimP
    #
    return
#end my_parser
