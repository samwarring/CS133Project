#ifndef DENOISE_H
#define DENOISE_H

/* Method Parameters */
#define DT          5.0
#define EPSILON     1.0E-20
#define MAXITER     500

#define sigma		0.05
#define lambda		0.065
#define Tol			2.e-3

int denoise(float *u, const float *f, int M, int N);

#endif

