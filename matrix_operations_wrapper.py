"""
script for the experiments from the paper Papez, Grigori, Stompor 2020: "Accelerating
    linear system solvers for time domain component separation of cosmic microwave
    background data", submitted to Astronomy & Astrophysics

this script is a wrapper for matrix operation functions implemented in C
"""

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import param

__author__ = "Jan Papez"
__license__ = "GPL"
__version__ = "1.0.0 of March 25, 2020"
__maintainer__ = "Jan Papez"
__email__ = "jan@papez.org"
__status__ = "Code for the experiments for the paper"

# define the pointers
array_1d_int = npct.ndpointer(dtype=np.int, ndim=1, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# load the C library
matops = npct.load_library("libmatrix_operations.so", "PATH_TO_WORKDIR")

# setup the return types and argument types

# pointing
matops.apply_A.restype = None
matops.apply_A.argtypes = [
    c_int,
    array_1d_double, c_int,
    array_1d_double, c_int,
    array_1d_double, c_int,
    array_1d_double,
    array_1d_double,
    array_1d_double]

def apply_A(p, q, u, v, out, weights):
    return matops.apply_A(
        param.ncomp,
        p, param.Nt,
        q, int(q.size),
        u, int(u.size),
        np.ascontiguousarray(v),
        np.ascontiguousarray(weights),
        out)

# pointing for two detectors
matops.apply_A_twodetectors.restype = None
matops.apply_A_twodetectors.argtypes = [
    c_int,
    array_1d_double, array_1d_double, c_int,
    array_1d_double, c_int, array_1d_double, c_int,
    array_1d_double, c_int, array_1d_double, c_int,
    array_1d_double,
    array_1d_double,
    array_1d_double, array_1d_double]

def apply_A_twodetectors(p1, p2, q1, q2, u1, u2, v, out1, out2, weights):
    return matops.apply_A_twodetectors(
        param.ncomp,
        p1, p2, param.Nt,
        q1, int(q1.size), u1, int(u1.size),
        q2, int(q2.size), u2, int(u2.size),
        np.ascontiguousarray(v),
        np.ascontiguousarray(weights),
        out1, out2)

# depointing
matops.apply_At.restype = None
matops.apply_At.argtypes = [
    c_int,
    array_1d_double, c_int,
    array_1d_double, c_int,
    array_1d_double, c_int,
    array_1d_double,
    array_1d_double,
    array_1d_double]

def apply_At(p, q, u, w, out, weights):
    return matops.apply_At(
        param.ncomp,
        p, param.Nt,
        q, int(q.size),
        u, int(u.size),
        np.ascontiguousarray(w),
        np.ascontiguousarray(weights),
        out)

# depointing for two detectors
matops.apply_At_twodetectors.restype = None
matops.apply_At_twodetectors.argtypes = [
    c_int,
    array_1d_double, array_1d_double, c_int,
    array_1d_double, c_int, array_1d_double, c_int,
    array_1d_double, c_int, array_1d_double, c_int,
    array_1d_double, array_1d_double,
    array_1d_double,
    array_1d_double]

def apply_At_twodetectors(p1, p2, q1, q2, u1, u2, w1, w2, out, weights):
    return matops.apply_At_twodetectors(
        param.ncomp,
        p1, p2, param.Nt,
        q1, int(q1.size), u1, int(u1.size),
        q2, int(q2.size), u2, int(u2.size),
        np.ascontiguousarray(w1), np.ascontiguousarray(w2),
        np.ascontiguousarray(weights),
        out)

# apply the problem matrix using fft in C
matops.apply_A_invN_At.restype = None
matops.apply_A_invN_At.argtypes = [
    c_int,
    array_1d_double, c_int,
    array_1d_double, c_int, array_1d_double, c_int,
    array_1d_double,
    array_1d_double,
    array_1d_double,
    array_1d_double]

def apply_A_invN_At(p, q, u, signal, invspectrum, out_signal, weights):
    return matops.apply_A_invN_At(
        param.ncomp,
        p, param.Nt,
        q, int(q.size), u, int(u.size),
        np.ascontiguousarray(invspectrum),
        np.ascontiguousarray(signal),
        np.ascontiguousarray(weights),
        out_signal)

# apply the problem matrix using fft in C
matops.apply_A_invN_At_twodetectors.restype = None
matops.apply_A_invN_At_twodetectors.argtypes = [
    c_int,
    array_1d_double, array_1d_double, c_int,
    array_1d_double, c_int, array_1d_double, c_int, array_1d_double, c_int, array_1d_double, c_int,
    array_1d_double,
    array_1d_double,
    array_1d_double,
    array_1d_double]

def apply_A_invN_At_twodetectors(p1, p2, q1, q2, u1, u2, signal, invspectrum, out_signal, weights):
    return matops.apply_A_invN_At_twodetectors(
        param.ncomp,
        p1, p2, param.Nt,
        q1, int(q1.size), u1, int(u1.size), q2, int(q2.size), u2, int(u2.size),
        np.ascontiguousarray(invspectrum),
        np.ascontiguousarray(signal),
        np.ascontiguousarray(weights),
        out_signal)

# apply the noise weightening using fft in C
matops.apply_invN.restype = None
matops.apply_invN.argtypes = [c_int, array_1d_double, array_1d_double, array_1d_double]

def apply_invN(indata, outdata, invspectrum):
    return matops.apply_invN(
        param.Nt, np.ascontiguousarray(indata), np.ascontiguousarray(outdata), np.ascontiguousarray(invspectrum))

matops.build_Prec_BD.restype = None
matops.build_Prec_BD.argtypes = [
    c_int,
    c_int, c_int,
    array_1d_double,
    array_1d_double, c_int,
    array_1d_double, c_int,
    array_1d_double, c_int,
    array_1d_double,
    array_1d_double
    ]


def build_Prec_BD(p, q, u, invNdiag, prec_diag, weights):
    return matops.build_Prec_BD(
        param.ncomp,
        param.Nt, param.Np,
        p,
        q, int(q.size),
        u, int(u.size),
        np.ascontiguousarray(invNdiag), int(invNdiag.size),
        np.ascontiguousarray(weights),
        np.ascontiguousarray(prec_diag))
