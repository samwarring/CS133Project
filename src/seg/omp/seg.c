#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "seg.h"

int seg(float* contour, const float* u, int m, int n, int MaxIter) {

	float* curv	= (float*)calloc(m*n,sizeof(float));
	float* phi	= (float*)calloc(m*n,sizeof(float));

	float c1,c2;

	int iter;
	int i, j, k;	

	float mu = 0.18*255*255;
	float dt = 0.225/mu;

	float xcent = (m-1) / 2.0;
	float ycent = (n-1) / 2.0;
	float r = fmin(m,n) / 2.0;

#pragma omp parallel for private(j)
	for(i=0; i < m; i++) {
		for(j=0; j < n; j++) {

			float xx = i;
			float yy = j;
			phi(i, j)	= sqrtf(SQR(xx-xcent) + SQR(yy-ycent)) - r;
			curv(i, j)	= 0;
		}
	}
	for(iter=0; iter<MaxIter; iter++) {

		float num1 = 0;
		float num2 = 0;
		int   den1 = 0;
		int   den2 = 0;

#pragma omp parallel for private(j) reduction(+:num1, num2, den1, den2)
		for(i=0; i<m; i++) {
			for(j=0; j < n; j++) {
				if(phi(i,j) < 0) {
					num1 += 256*u(i,j);
					den1 +=  1;
				}
				else if(phi(i,j) > 0) {
					num2  += 256*u(i,j);
					den2  += 1;
				}
			}
		}

		c1 = num1/den1;
		c2 = num2/den2;

#pragma omp parallel for private(j)
		for(i=1;i<m-1;i++) {
			for(j=1; j < n-1; j++) {
				float Dx_p = phi(i+1,j) - phi(i,j);
				float Dx_m = phi(i,j) - phi(i-1,j);
				float Dy_p = phi(i,j+1) - phi(i,j);
				float Dy_m = phi(i,j) - phi(i,j-1);

				float Dx_0 = (phi(i+1,j) - phi(i-1,j))/2;
				float Dy_0 = (phi(i,j+1) - phi(i,j-1))/2;

				float Dxx = Dx_p - Dx_m ;
				float Dyy = Dy_p - Dy_m ;

				float Dxy = (phi(i+1,j+1) - phi(i+1,j-1) - phi(i-1,j+1) + phi(i-1,j-1)) / 4;

				float Grad	= sqrtf(Dx_0*Dx_0 + Dy_0*Dy_0);
				float K		= (Dx_0*Dx_0*Dyy - 2*Dx_0*Dy_0*Dxy + Dy_0*Dy_0*Dxx) / (CUB(Grad) + epsilon);

				curv(i, j) = Grad*(mu*K + SQR(256*u(i,j)-c1) - SQR(256*u(i,j)-c2));
			}
		}
#pragma omp parallel for
		for(j=0; j < n; j++) {
			curv( 0, j) = curv( 1, j);
			curv(m-1,j) = curv(m-2,j);
		}
#pragma omp parallel for
		for(i=0; i < m; i++) {
			curv(i, 0 ) = curv(i, 1 );
			curv(i,n-1) = curv(i,n-2);
		}
#pragma omp parallel for private(j)
		for(i=0; i<m; i++) {
			for (j=0; j<n; j++) {
				phi(i, j) += curv(i, j) * dt;
			}
		}
	}
#pragma omp parallel for private(j)
	for(i=1; i<m; i++) {
		for (j=1; j<n; j++) {
			if (phi(i, j)*phi(i-1, j)<0 || phi(i, j)*phi(i, j-1)<0) 
				contour[i*n+j] = 0.99;
			else 
				contour[i*n+j] = 0;
		}
	}

	free(phi);
	free(curv);
	return 0;
}
