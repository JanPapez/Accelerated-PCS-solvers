# Accelarated PCS solvers

Scripts (in python2.7 and C) for the experiments from the paper Papez, Grigori, Stompor 2020: "Accelerating linear system solvers for time domain component separation of cosmic microwave background data", submitted to Astronomy & Astrophysics. The paper is available from https://hal.inria.fr/hal-02470964.

The scripts are prepared to be run from NERSC's Cori machine; but any other machine can be used after properly updating the paths.

The experiments require simulated data, which are not freely available due to an extensive memory they require. Nevertheless, they can be demanded on "jan@papez.org". Any user can use the codes for any other data, which satisfy the requirements described below.

After copying the scripts, the C-code matrix_operations.c must be compiled to generate a shared library. The compile commands are for Mac OS and Cori machine given in the very bottom of matrix_operations.c.

# Data description

signal
*  values of signal components in observed pixels (3 components: cmb, dust, sync): 3 matrices of size (N_pix,3). FILES: cmb_observed.txt, dust_observed.txt, sync_observed.txt; 
*  a priori generated sequence of mixing parameters. FILES: sampled_betas.npy

noise
*  for each frequency: inverse spectrum and 2 noise realizations (for horizontal and vertical scans), all vectors of length N_t. FILES: inverse_pfI.npy, noisestream_hscan_simI.npy, noisestream_vscan_simI.npy where "I" is replaced by 0, 1, ..., N_freq

pointing
*  index of the observed pixel in time t, indices should be in [0, N_pix-1] for horizontal and vertical scans; 2 vectors of length N_t. FILES: hscan_CASE_ns1024_processed.npy, vscan_CASE_ns1024_processed.npy where "CASE" is replaced by the test case, in our case, e.g., "case0"
*  Q and U weights for horizontal and vertical scans; 2x2 vectors of length N_t, can be shortened if periodic. FILES: qwght_hscan_CASE.npy, qwght_vscan_CASE.npy, uwght_hscan_CASE.npy, uwght_vscan_CASE.npy (or for example qwght_hscan_CASE_short.npy if the file is shortened thanks to periodicity)
