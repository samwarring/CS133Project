#ifndef MRI_H
#define MRI_H

#define SQR(a)		(a)*(a)
#define max(a,b)	((a)>(b))?(a):(b)

#define dx(i,j)		dx[(i)*N2+(j)]
#define dy(i,j)		dy[(i)*N2+(j)]

#define mask(i,j)	mask[(i)*N2+(j)]
#define diff(i,j)	diff[(i)*N2+(j)]

#define dx_new(i,j) dx_new[(i)*N2+(j)]
#define dy_new(i,j) dy_new[(i)*N2+(j)]

#define u(i,j)		u[(i)*N2+(j)]
#define f(i,j)		f[(i)*N2+(j)]
#define f0(i,j)		f0[(i)*N2+(j)]
#define img(i,j)	img[(i)*N2+(j)]

#define fftmul(i,j)		fftmul[(i)*N2+(j)]
#define Lap(i,j)		Lap[(i)*N2+(j)]
#define u_fft2(i,j)		u_fft2[(i)*N2+(j)]
#define dtildex(i,j)	dtildex[(i)*N2+(j)]
#define dtildey(i,j)	dtildey[(i)*N2+(j)]

#define MaxIter		30
#define MaxOutIter  5
#define Gamma1  	1
#define Gamma2		0.01

#define PI			3.1415926

int mri(float *, float complex*, float*, float, int, int);

#endif

