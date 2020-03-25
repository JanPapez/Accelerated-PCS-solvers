/*
script for the experiments from the paper Papez, Grigori, Stompor 2020: "Accelerating
    linear system solvers for time domain component separation of cosmic microwave
    background data", submitted to Astronomy & Astrophysics

this script defined several matrix operations that would be slow in Python
*/

#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h> //package for FFT

/*
author = "Jan Papez"
license = "GPL"
version = "1.0.0 of March 25, 2020"
maintainer = "Jan Papez"
email = "jan@papez.org"
status = "Code for the experiments for the paper"
*/

/* application of the pointing matrix */
void apply_A(int ncomp, double * p, int Nt, double * qwght, int qwght_size, double * uwght, int uwght_size, double * v, double * weights, double * out) {
    int t;
    int index;
    double qvalue, uvalue;

    if (ncomp == 3) {
      for (t = 0; t < Nt; t++) {
        index = (int) p[t];

        /* we assume that tvalue = 1 and it is therefore ommited from the formula */
        qvalue = qwght[t % qwght_size];
        uvalue = uwght[t % uwght_size];

        out[t] = weights[0]* v[9*index+0] + weights[1]* v[9*index+1] * qvalue + weights[2]* v[9*index+2] * uvalue +
                 weights[3]* v[9*index+3] + weights[4]* v[9*index+4] * qvalue + weights[5]* v[9*index+5] * uvalue +
                 weights[6]* v[9*index+6] + weights[7]* v[9*index+7] * qvalue + weights[8]* v[9*index+8] * uvalue ;
      }
    }
    else { // ncomp == 2
      for (t = 0; t < Nt; t++) {
        index = (int) p[t];

        qvalue = qwght[t % qwght_size];
        uvalue = uwght[t % uwght_size];

        out[t] = weights[0]* v[6*index+0] * qvalue + weights[1]* v[6*index+1] * uvalue +
                 weights[2]* v[6*index+2] * qvalue + weights[3]* v[6*index+3] * uvalue +
                 weights[4]* v[6*index+4] * qvalue + weights[5]* v[6*index+5] * uvalue ;
      }
    }
    return ;
}

/* application of the pointing matrix for two detectors*/
void apply_A_twodetectors(int ncomp, double * p1, double * p2, int Nt, double * qwght1, int qwght1_size, double * uwght1, int uwght1_size, double * qwght2, int qwght2_size, double * uwght2, int uwght2_size, double * v, double * weights, double * out1, double * out2) {
    int t;
    int index1, index2;
    double qvalue1, uvalue1, qvalue2, uvalue2;

    if (ncomp == 3) {
      for (t = 0; t < Nt; t++) {
        index1 = (int) p1[t];
        index2 = (int) p2[t];

        /* we assume that tvalue = 1 and it is therefore ommited from the formula */
        qvalue1 = qwght1[t % qwght1_size];
        uvalue1 = uwght1[t % uwght1_size];
        qvalue2 = qwght2[t % qwght2_size];
        uvalue2 = uwght2[t % uwght2_size];

        out1[t] = weights[0]* v[9*index1+0] + weights[1]* v[9*index1+1] * qvalue1 + weights[2]* v[9*index1+2] * uvalue1 +
                  weights[3]* v[9*index1+3] + weights[4]* v[9*index1+4] * qvalue1 + weights[5]* v[9*index1+5] * uvalue1 +
                  weights[6]* v[9*index1+6] + weights[7]* v[9*index1+7] * qvalue1 + weights[8]* v[9*index1+8] * uvalue1 ;
        out2[t] = weights[0]* v[9*index2+0] + weights[1]* v[9*index2+1] * qvalue2 + weights[2]* v[9*index2+2] * uvalue2 +
                  weights[3]* v[9*index2+3] + weights[4]* v[9*index2+4] * qvalue2 + weights[5]* v[9*index2+5] * uvalue2 +
                  weights[6]* v[9*index2+6] + weights[7]* v[9*index2+7] * qvalue2 + weights[8]* v[9*index2+8] * uvalue2 ;
      }
    }
    else { // ncomp == 2
      for (t = 0; t < Nt; t++) {
        index1 = (int) p1[t];
        index2 = (int) p2[t];

        qvalue1 = qwght1[t % qwght1_size];
        uvalue1 = uwght1[t % uwght1_size];
        qvalue2 = qwght2[t % qwght2_size];
        uvalue2 = uwght2[t % uwght2_size];

        out1[t] = weights[0]* v[6*index1+0] * qvalue1 + weights[1]* v[6*index1+1] * uvalue1 +
                 weights[2]* v[6*index1+2] * qvalue1 + weights[3]* v[6*index1+3] * uvalue1 +
                 weights[4]* v[6*index1+4] * qvalue1 + weights[5]* v[6*index1+5] * uvalue1 ;
        out2[t] = weights[0]* v[6*index2+0] * qvalue2 + weights[1]* v[6*index2+1] * uvalue2 +
                 weights[2]* v[6*index2+2] * qvalue2 + weights[3]* v[6*index2+3] * uvalue2 +
                 weights[4]* v[6*index2+4] * qvalue2 + weights[5]* v[6*index2+5] * uvalue2 ;
      }
    }
    return ;
}

/* application of the depointing matrix */
void apply_At(int ncomp, double * p, int Nt, double * qwght, int qwght_size, double * uwght, int uwght_size, double * w, double * weights, double * out) {
    int t;
    int index;
    double qvalue, uvalue, wvalue;

    if (ncomp == 3) {
      for (t = 0; t < Nt; t++) {
        index = (int) p[t];

        /* we assume that tvalue = 1 and it is therefore ommited from the formula */
        qvalue = qwght[t % qwght_size];
        uvalue = uwght[t % uwght_size];
        wvalue = w[t];

        out[9*index + 0] += weights[0]* wvalue;
        out[9*index + 1] += weights[1]* qvalue * wvalue;
        out[9*index + 2] += weights[2]* uvalue * wvalue;
        out[9*index + 3] += weights[3]* wvalue;
        out[9*index + 4] += weights[4]* qvalue * wvalue;
        out[9*index + 5] += weights[5]* uvalue * wvalue;
        out[9*index + 6] += weights[6]* wvalue;
        out[9*index + 7] += weights[7]* qvalue * wvalue;
        out[9*index + 8] += weights[8]* uvalue * wvalue;
      }
    }
    else { // ncomp == 2
      for (t = 0; t < Nt; t++) {
        index = (int) p[t];

        qvalue = qwght[t % qwght_size];
        uvalue = uwght[t % uwght_size];
        wvalue = w[t];

        out[6*index + 0] += weights[0]* qvalue * wvalue;
        out[6*index + 1] += weights[1]* uvalue * wvalue;
        out[6*index + 2] += weights[2]* qvalue * wvalue;
        out[6*index + 3] += weights[3]* uvalue * wvalue;
        out[6*index + 4] += weights[4]* qvalue * wvalue;
        out[6*index + 5] += weights[5]* uvalue * wvalue;
      }
    }
    return ;
}

/* application of the depointing matrix for two detectors*/
void apply_At_twodetectors(int ncomp, double * p1, double * p2, int Nt, double * qwght1, int qwght1_size, double * uwght1, int uwght1_size, double * qwght2, int qwght2_size, double * uwght2, int uwght2_size, double * w1, double * w2, double * weights, double * out) {
    int t;
    int index1, index2;
    double qvalue1, uvalue1, wvalue1, qvalue2, uvalue2, wvalue2;

    if (ncomp == 3) {
      for (t = 0; t < Nt; t++) {
        index1 = (int) p1[t];
        index2 = (int) p2[t];

        /* we assume that tvalue = 1 and it is therefore ommited from the formula */
        qvalue1 = qwght1[t % qwght1_size];
        uvalue1 = uwght1[t % uwght1_size];
        wvalue1 = w1[t];
        qvalue2 = qwght2[t % qwght2_size];
        uvalue2 = uwght2[t % uwght2_size];
        wvalue2 = w2[t];

        out[9*index1 + 0] += weights[0]* wvalue1;
        out[9*index1 + 1] += weights[1]* qvalue1 * wvalue1;
        out[9*index1 + 2] += weights[2]* uvalue1 * wvalue1;
        out[9*index1 + 3] += weights[3]* wvalue1;
        out[9*index1 + 4] += weights[4]* qvalue1 * wvalue1;
        out[9*index1 + 5] += weights[5]* uvalue1 * wvalue1;
        out[9*index1 + 6] += weights[6]* wvalue1;
        out[9*index1 + 7] += weights[7]* qvalue1 * wvalue1;
        out[9*index1 + 8] += weights[8]* uvalue1 * wvalue1;

        out[9*index2 + 0] += weights[0]* wvalue2;
        out[9*index2 + 1] += weights[1]* qvalue2 * wvalue2;
        out[9*index2 + 2] += weights[2]* uvalue2 * wvalue2;
        out[9*index2 + 3] += weights[3]* wvalue2;
        out[9*index2 + 4] += weights[4]* qvalue2 * wvalue2;
        out[9*index2 + 5] += weights[5]* uvalue2 * wvalue2;
        out[9*index2 + 6] += weights[6]* wvalue2;
        out[9*index2 + 7] += weights[7]* qvalue2 * wvalue2;
        out[9*index2 + 8] += weights[8]* uvalue2 * wvalue2;
      }
    }
    else { // ncomp == 2
      for (t = 0; t < Nt; t++) {
        index1 = (int) p1[t];
        index2 = (int) p2[t];

        /* we assume that tvalue = 1 and it is therefore ommited from the formula */
        qvalue1 = qwght1[t % qwght1_size];
        uvalue1 = uwght1[t % uwght1_size];
        wvalue1 = w1[t];
        qvalue2 = qwght2[t % qwght2_size];
        uvalue2 = uwght2[t % uwght2_size];
        wvalue2 = w2[t];

        out[6*index1 + 0] += weights[0]* qvalue1 * wvalue1;
        out[6*index1 + 1] += weights[1]* uvalue1 * wvalue1;
        out[6*index1 + 2] += weights[2]* qvalue1 * wvalue1;
        out[6*index1 + 3] += weights[3]* uvalue1 * wvalue1;
        out[6*index1 + 4] += weights[4]* qvalue1 * wvalue1;
        out[6*index1 + 5] += weights[5]* uvalue1 * wvalue1;

        out[6*index2 + 0] += weights[0]* qvalue2 * wvalue2;
        out[6*index2 + 1] += weights[1]* uvalue2 * wvalue2;
        out[6*index2 + 2] += weights[2]* qvalue2 * wvalue2;
        out[6*index2 + 3] += weights[3]* uvalue2 * wvalue2;
        out[6*index2 + 4] += weights[4]* qvalue2 * wvalue2;
        out[6*index2 + 5] += weights[5]* uvalue2 * wvalue2;
      }
    }
    return ;
}


/* application of the pointing matrix, weightening by noise and depointing */
void apply_A_invN_At(int ncomp, double * p, int Nt, double * qwght, int qwght_size, double * uwght, int uwght_size, double * invspectrum, double * v, double * weights, double * out) {
  int t;
  int index;
  double qvalue, uvalue, wvalue;
  /* variables for fft */
  fftw_complex *fft_in;
  fftw_plan fft, ifft;

  fftw_import_system_wisdom();

  fft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nt);
  fft = fftw_plan_dft_1d(Nt, fft_in, fft_in, FFTW_FORWARD, FFTW_PATIENT);
  ifft = fftw_plan_dft_1d(Nt, fft_in, fft_in, FFTW_BACKWARD, FFTW_PATIENT);

  /* pointing */
  if (ncomp == 3) {
    for (t = 0; t < Nt; t++) {
      index = (int) p[t];

      /* we assume that tvalue = 1 and it is therefore ommited from the formula */
      qvalue = qwght[t % qwght_size];
      uvalue = uwght[t % uwght_size];

      fft_in[t][0] = weights[0]* v[9*index+0] + weights[1]* v[9*index+1] * qvalue + weights[2]* v[9*index+2] * uvalue +
                     weights[3]* v[9*index+3] + weights[4]* v[9*index+4] * qvalue + weights[5]* v[9*index+5] * uvalue +
                     weights[6]* v[9*index+6] + weights[7]* v[9*index+7] * qvalue + weights[8]* v[9*index+8] * uvalue ;
      fft_in[t][1] = 0;
    }
  }
  else { // ncomp == 2
    for (t = 0; t < Nt; t++) {
      index = (int) p[t];

      qvalue = qwght[t % qwght_size];
      uvalue = uwght[t % uwght_size];

      fft_in[t][0] = weights[0]* v[6*index+0] * qvalue + weights[1]* v[6*index+1] * uvalue +
                     weights[2]* v[6*index+2] * qvalue + weights[3]* v[6*index+3] * uvalue +
                     weights[4]* v[6*index+4] * qvalue + weights[5]* v[6*index+5] * uvalue ;
      fft_in[t][1] = 0;
    }
  }

  /* noise weightening */

  fftw_execute(fft);

  for (t = 0; t < Nt; t++) {
    fft_in[t][0] *= invspectrum[t];
    fft_in[t][1] *= invspectrum[t];
  }

  fftw_execute(ifft);

  /* depointing */
  if (ncomp == 3) {
    for (t = 0; t < Nt; t++) {
      index = (int) p[t];

      /* we assume that tvalue = 1 and it is therefore ommited from the formula */
      qvalue = qwght[t % qwght_size];
      uvalue = uwght[t % uwght_size];
      wvalue = fft_in[t][0];

      out[9*index + 0] += weights[0]* wvalue;
      out[9*index + 1] += weights[1]* qvalue * wvalue;
      out[9*index + 2] += weights[2]* uvalue * wvalue;
      out[9*index + 3] += weights[3]* wvalue;
      out[9*index + 4] += weights[4]* qvalue * wvalue;
      out[9*index + 5] += weights[5]* uvalue * wvalue;
      out[9*index + 6] += weights[6]* wvalue;
      out[9*index + 7] += weights[7]* qvalue * wvalue;
      out[9*index + 8] += weights[8]* uvalue * wvalue;
    }
  }
  else { // ncomp == 2
    for (t = 0; t < Nt; t++) {
      index = (int) p[t];

      qvalue = qwght[t % qwght_size];
      uvalue = uwght[t % uwght_size];
      wvalue = fft_in[t][0];

      out[6*index + 0] += weights[0]* qvalue * wvalue;
      out[6*index + 1] += weights[1]* uvalue * wvalue;
      out[6*index + 2] += weights[2]* qvalue * wvalue;
      out[6*index + 3] += weights[3]* uvalue * wvalue;
      out[6*index + 4] += weights[4]* qvalue * wvalue;
      out[6*index + 5] += weights[5]* uvalue * wvalue;
    }
  }

  fftw_destroy_plan(fft);
  fftw_destroy_plan(ifft);
  fftw_free(fft_in);

  return ;
}


/* application of the pointing matrix, weightening by noise and depointing for two detectors */
void apply_A_invN_At_twodetectors(int ncomp, double * p1, double * p2, int Nt, double * qwght1, int qwght1_size, double * uwght1, int uwght1_size, double * qwght2, int qwght2_size, double * uwght2, int uwght2_size, double * invspectrum, double * v, double * weights, double * out) {
  int t;
  int index;
  double qvalue, uvalue, wvalue;
  /* variables for fft */
  fftw_complex *fft_in;
  fftw_plan fft, ifft;

  fftw_import_system_wisdom();

  fft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nt);
  fft = fftw_plan_dft_1d(Nt, fft_in, fft_in, FFTW_FORWARD, FFTW_MEASURE);
  ifft = fftw_plan_dft_1d(Nt, fft_in, fft_in, FFTW_BACKWARD, FFTW_MEASURE);

  // FIRST DETECTOR
  /* pointing */
  if (ncomp == 3) {
    for (t = 0; t < Nt; t++) {
      index = (int) p1[t];

      /* we assume that tvalue = 1 and it is therefore ommited from the formula */
      qvalue = qwght1[t % qwght1_size];
      uvalue = uwght1[t % uwght1_size];

      fft_in[t][0] = weights[0]* v[9*index+0] + weights[1]* v[9*index+1] * qvalue + weights[2]* v[9*index+2] * uvalue +
                     weights[3]* v[9*index+3] + weights[4]* v[9*index+4] * qvalue + weights[5]* v[9*index+5] * uvalue +
                     weights[6]* v[9*index+6] + weights[7]* v[9*index+7] * qvalue + weights[8]* v[9*index+8] * uvalue ;
      fft_in[t][1] = 0;
    }
  }
  else { // ncomp == 2
    for (t = 0; t < Nt; t++) {
      index = (int) p1[t];

      qvalue = qwght1[t % qwght1_size];
      uvalue = uwght1[t % uwght1_size];

      fft_in[t][0] = weights[0]* v[6*index+0] * qvalue + weights[1]* v[6*index+1] * uvalue +
                     weights[2]* v[6*index+2] * qvalue + weights[3]* v[6*index+3] * uvalue +
                     weights[4]* v[6*index+4] * qvalue + weights[5]* v[6*index+5] * uvalue ;
      fft_in[t][1] = 0;
    }
  }

  /* noise weightening */

  fftw_execute(fft);

  for (t = 0; t < Nt; t++) {
    fft_in[t][0] *= invspectrum[t];
    fft_in[t][1] *= invspectrum[t];
  }

  fftw_execute(ifft);

  /* depointing */
  if (ncomp == 3) {
    for (t = 0; t < Nt; t++) {
      index = (int) p1[t];

      /* we assume that tvalue = 1 and it is therefore ommited from the formula */
      qvalue = qwght1[t % qwght1_size];
      uvalue = uwght1[t % uwght1_size];
      wvalue = fft_in[t][0];

      out[9*index + 0] += weights[0]* wvalue;
      out[9*index + 1] += weights[1]* qvalue * wvalue;
      out[9*index + 2] += weights[2]* uvalue * wvalue;
      out[9*index + 3] += weights[3]* wvalue;
      out[9*index + 4] += weights[4]* qvalue * wvalue;
      out[9*index + 5] += weights[5]* uvalue * wvalue;
      out[9*index + 6] += weights[6]* wvalue;
      out[9*index + 7] += weights[7]* qvalue * wvalue;
      out[9*index + 8] += weights[8]* uvalue * wvalue;
    }
  }
  else { // ncomp == 2
    for (t = 0; t < Nt; t++) {
      index = (int) p1[t];

      qvalue = qwght1[t % qwght1_size];
      uvalue = uwght1[t % uwght1_size];
      wvalue = fft_in[t][0];

      out[6*index + 0] += weights[0]* qvalue * wvalue;
      out[6*index + 1] += weights[1]* uvalue * wvalue;
      out[6*index + 2] += weights[2]* qvalue * wvalue;
      out[6*index + 3] += weights[3]* uvalue * wvalue;
      out[6*index + 4] += weights[4]* qvalue * wvalue;
      out[6*index + 5] += weights[5]* uvalue * wvalue;
    }
  }

  // SECOND DETECTOR
  /* pointing */
  if (ncomp == 3) {
    for (t = 0; t < Nt; t++) {
      index = (int) p2[t];

      /* we assume that tvalue = 1 and it is therefore ommited from the formula */
      qvalue = qwght2[t % qwght2_size];
      uvalue = uwght2[t % uwght2_size];

      fft_in[t][0] = weights[0]* v[9*index+0] + weights[1]* v[9*index+1] * qvalue + weights[2]* v[9*index+2] * uvalue +
                     weights[3]* v[9*index+3] + weights[4]* v[9*index+4] * qvalue + weights[5]* v[9*index+5] * uvalue +
                     weights[6]* v[9*index+6] + weights[7]* v[9*index+7] * qvalue + weights[8]* v[9*index+8] * uvalue ;
      fft_in[t][1] = 0;
    }
  }
  else { // ncomp == 2
    for (t = 0; t < Nt; t++) {
      index = (int) p2[t];

      qvalue = qwght2[t % qwght2_size];
      uvalue = uwght2[t % uwght2_size];

      fft_in[t][0] = weights[0]* v[6*index+0] * qvalue + weights[1]* v[6*index+1] * uvalue +
                     weights[2]* v[6*index+2] * qvalue + weights[3]* v[6*index+3] * uvalue +
                     weights[4]* v[6*index+4] * qvalue + weights[5]* v[6*index+5] * uvalue ;
      fft_in[t][1] = 0;
    }
  }

  /* noise weightening */

  fftw_execute(fft);

  for (t = 0; t < Nt; t++) {
    fft_in[t][0] *= invspectrum[t];
    fft_in[t][1] *= invspectrum[t];
  }

  fftw_execute(ifft);

  /* depointing */
  if (ncomp == 3) {
    for (t = 0; t < Nt; t++) {
      index = (int) p2[t];

      /* we assume that tvalue = 1 and it is therefore ommited from the formula */
      qvalue = qwght2[t % qwght2_size];
      uvalue = uwght2[t % uwght2_size];
      wvalue = fft_in[t][0];

      out[9*index + 0] += weights[0]* wvalue;
      out[9*index + 1] += weights[1]* qvalue * wvalue;
      out[9*index + 2] += weights[2]* uvalue * wvalue;
      out[9*index + 3] += weights[3]* wvalue;
      out[9*index + 4] += weights[4]* qvalue * wvalue;
      out[9*index + 5] += weights[5]* uvalue * wvalue;
      out[9*index + 6] += weights[6]* wvalue;
      out[9*index + 7] += weights[7]* qvalue * wvalue;
      out[9*index + 8] += weights[8]* uvalue * wvalue;
    }
  }
  else { // ncomp == 2
    for (t = 0; t < Nt; t++) {
      index = (int) p2[t];

      qvalue = qwght2[t % qwght2_size];
      uvalue = uwght2[t % uwght2_size];
      wvalue = fft_in[t][0];

      out[6*index + 0] += weights[0]* qvalue * wvalue;
      out[6*index + 1] += weights[1]* uvalue * wvalue;
      out[6*index + 2] += weights[2]* qvalue * wvalue;
      out[6*index + 3] += weights[3]* uvalue * wvalue;
      out[6*index + 4] += weights[4]* qvalue * wvalue;
      out[6*index + 5] += weights[5]* uvalue * wvalue;
    }
  }

  fftw_destroy_plan(fft);
  fftw_destroy_plan(ifft);
  fftw_free(fft_in);

  return ;
}


/* application of the pointing matrix, weightening by noise and depointing */
void apply_invN(int Nt, double * in, double * out, double * invspectrum) {
  int t;
  /* variables for fft */
  fftw_complex *fft_in;
  fftw_plan fft, ifft;

  fftw_import_system_wisdom();

  fft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nt);
  fft = fftw_plan_dft_1d(Nt, fft_in, fft_in, FFTW_FORWARD, FFTW_PATIENT);
  ifft = fftw_plan_dft_1d(Nt, fft_in, fft_in, FFTW_BACKWARD, FFTW_PATIENT);

  for (t = 0; t < Nt; t++) {
      fft_in[t][0] = in[t];
      fft_in[t][1] = 0;
  }


  fftw_execute(fft);

  for (t = 0; t < Nt; t++) {
    fft_in[t][0] *= invspectrum[t];
    fft_in[t][1] *= invspectrum[t];
  }

  fftw_execute(ifft);

  for (t = 0; t < Nt; t++) {
      out[t] = fft_in[t][0];
  }

  fftw_destroy_plan(fft);
  fftw_destroy_plan(ifft);
  fftw_free(fft_in);

  return ;
}


/* construction of the preconditioner diagonal blocks */
void build_Prec_BD(int ncomp, int Nt, int Np, double * p, double * qwght, int qwght_size, double * uwght, int uwght_size, double * invN_diag, int invN_size, double * weights, double * out) {
  int t, i, j, locindex, index;
  double inv_diag_value, qvalue, uvalue;
  double v[9];

  /* build the local matrix */
  if (ncomp == 3) {
    for (t = 0; t < Nt; t++) {
      index = (int) p[t];
      inv_diag_value = invN_diag[t % invN_size];
      qvalue = qwght[t % qwght_size];
      uvalue = uwght[t % uwght_size];

      /* we assume that tvalue = 1 and it is therefore ommited from the formula */
      v[0] = weights[0]* 1;
      v[1] = weights[1]* qvalue;
      v[2] = weights[2]* uvalue;
      v[3] = weights[3]* 1;
      v[4] = weights[4]* qvalue;
      v[5] = weights[5]* uvalue;
      v[6] = weights[6]* 1;
      v[7] = weights[7]* qvalue;
      v[8] = weights[8]* uvalue;

      locindex = 0;
      for (j = 0; j < 9; j++) {
        for (i = 0; i < 9; i++) {
          out[81*index + locindex] += inv_diag_value * v[j] * v[i];
          locindex++;
        }
      }
    }
  }
  else { // ncomp == 2
    for (t = 0; t < Nt; t++) {
      index = (int) p[t];
      inv_diag_value = invN_diag[t % invN_size];
      qvalue = qwght[t % qwght_size];
      uvalue = uwght[t % uwght_size];

      v[0] = weights[0]* qvalue;
      v[1] = weights[1]* uvalue;
      v[2] = weights[2]* qvalue;
      v[3] = weights[3]* uvalue;
      v[4] = weights[4]* qvalue;
      v[5] = weights[5]* uvalue;

      locindex = 0;
      for (j = 0; j < 6; j++) {
        for (i = 0; i < 6; i++) {
          out[36*index + locindex] += inv_diag_value * v[j] * v[i];
          locindex++;
        }
      }
    }
  }
  return ;
}

/* to compile */
/*  COMPILATION ON MAC
gcc -c -lfftw -lm -Wall -Werror -fpic matrix_operations.c
gcc -shared -o -lfftw3 -lm libmatrix_operations.so matrix_operations.o
*/
/*  COMPILATION ON CORI
module load gcc
module load cray-fftw
gcc -O3 -I$FFTW_INC -L$FFTW_DIR -lfftw3 -c -Wall -Werror -fpic matrix_operations.c
gcc -I$FFTW_INC -L$FFTW_DIR -lfftw3 -shared -o libmatrix_operations.so matrix_operations.o
*/
