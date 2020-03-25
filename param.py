"""
script for the experiments from the paper Papez, Grigori, Stompor 2020: "Accelerating
    linear system solvers for time domain component separation of cosmic microwave
    background data", submitted to Astronomy & Astrophysics

(global) parameters and paths are defined here
"""

ndetectors = []     #number of detectors, we use 2 (horizontal+vertical scans)
Np = []     # number of pixels
Nt = []     # number of observations, size of each scan
ncomp = []  # number of components of signal [TQU], we use 2
Nsigcomp = []  # ncomp x number of different signals, we use 2x3
nfrequencies = [] # number of frequencies, we use 6

multiple_eigvals = True # see the comment in Section 4.2

case = 'case0'      # test case to be used
niter = 200         # maximal number of iterations
scalen = 1          # scaling of the loaded noise
Betas_index = -1    # index of the first parameters to start with
TOL = 1e-8          # tolerance for stopping criterion
k = 0               # number of eigenvalues to deflate recycle
dimP = 0            # size of the subspace used for recycling

# the directories for data
pathtosignal = "/global/cscratch1/sd/USER/data/signal/"
pathtonoise = "/global/cscratch1/sd/USER/data/noise/"
pathtopointing = "/global/cscratch1/sd/USER/data/"
#    pathtopointing+case is added in PCS_functions.read_data
