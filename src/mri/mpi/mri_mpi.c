#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <string.h>

#include "mri.h"
#include "util_mpi.h"

int mpi_nprocs;
int mpi_rank;
MPI_Datatype FLOAT_COMPLEX;
MPI_Op FC_SUM;

char* str(float complex z) {
	
	char* s = malloc(64);
	sprintf(s, "%f+%fi", creal(z), cimag(z));
	return s;
}

int check_matrices(float complex* m1, float complex* m2, int mrows, int mcols) {

	for (int r=0; r<mrows; r++) {
		for (int c=0; c<mcols; c++) {
			float complex x1 = m1[r*mcols+c];
			float complex x2 = m2[r*mcols+c];
			if (x1 != x2) {
				fprintf(stderr, "check_matrices mismatch: row=%d col=%d (%s vs %s)\n",
					r, c, str(x1), str(x2));
				while(1);
				return 0; // Mismatch!
			}
		}
	}
	fprintf(stderr, "check_matrices success\n");
	while(1);
	return 1; // Equal matrices
}

int check_transpose(float complex* matrix, float complex* matrixT, int mrows, int mcols) {
	
	for (int r=0; r<mrows; r++) {
		for (int c=0; c<mcols; c++) {
			float complex m  = matrix[r*mcols+c];
			float complex mT = matrixT[c*mrows+r];
			if (m != mT) {
				fprintf(stderr, "check_transpose mismatch: row=%d col=%d\n", r, c);
				while(1);
				return 0; // Mismatch!
			}
		}
	}
	fprintf(stderr, "check_transpose success\n");
	while(1);
	return 1; // Valid transpose
}

int dft_init(float complex **w1, float complex **w2, float complex **tmp, int N, int M) {
	
	*w1 = (float complex*)malloc(((M-1)*(M-1)+1)*sizeof(float complex));
	*w2 = (float complex*)malloc(((N-1)*(N-1)+1)*sizeof(float complex));

	*tmp = (float complex*)malloc(M*N*sizeof(float complex));

	int i, j;
	(*w1)[0] = 1;
	(*w2)[0] = 1;

	for (i=1; i<M; i++) {
		for (j=i; j<M; j++) {
			(*w1)[i*j] = cexp(-2.0*PI*I*i*j/M);
		}
	}
	for (i=1; i<N; i++) {
		for (j=i; j<N; j++) {
			(*w2)[i*j] = cexp(-2.0*PI*I*i*j/N);
		}
	}
	return 0;
}




int dft_seq(float complex *dst, float complex* src, float complex* w1, float complex* w2, float complex* tmp, int N, int M) {
	
	int k, l, m, n;
	for (l=0; l<M; l++) {
		for (n = 0; n<N; n++) {
			float complex l_dst = 0.0;
			for (m = 0; m<M; m++) {
				l_dst += src[n*M+m]*w1[l*m];
			}
			tmp[n*M+l] = l_dst;
		}
	}
	for (k=0; k<N; k++) {
		for (l=0; l<M; l++) {
			float complex l_dst = 0.0;
			for (n = 0; n<N; n++) {
				l_dst += tmp[n*M+l]*w2[n*k];
			}
			dst[k*M+l] = l_dst;
		}
	}
	return 0;
}

int idft_seq(float complex *dst, float complex* src, float complex* w1, float complex* w2, float complex* tmp, int N, int M) {

	int k, l, m, n;
	for (l=0; l<M; l++) {
		for (n = 0; n<N; n++) {
			float complex l_dst = 0.0;
			for (m = 0; m<M; m++) {
				l_dst += src[n*M+m]/w1[l*m];
			}
			tmp[n*M+l] = l_dst/M;
		}
	}
	for (k=0; k<N; k++) {
		for (l=0; l<M; l++) {
			float complex l_dst = 0.0;
			for (n = 0; n<N; n++) {
				l_dst += tmp[n*M+l]/w2[n*k];
			}
			dst[k*M+l] = l_dst/N;
		}
	}
	return 0;
}


int dft(float complex *dst, float complex* src, float complex* w1, float complex* w2, float complex* tmp, int N, int M) {
	
	// Each process stores to a horizontal slice of the transposed tmp array.
	int start, count, share, extra, end;
	start = mpi_rank * (M/mpi_nprocs);
	share = M / mpi_nprocs;
	end   = start + share;
	extra = M % mpi_nprocs;
	if (mpi_rank == mpi_nprocs-1)
		end += extra;

	int k, l, m, n;
	for (l=start; l<end; l++) {
		for (n=0; n<N; n++) {
			float complex l_dst = 0.0;
			for (m=0; m<M; m++) {
				l_dst += src[n*M+m]*w1[l*m];
			}
			tmp[l*N+n] = l_dst; // Store into transpose(tmp) for better hit rate.
		}
	}

	for (k=0; k<N; k++) {
		for (l=start; l<end; l++) {
			float complex l_dst = 0.0;
			for (n=0; n<N; n++) {
				l_dst += tmp[l*N+n]*w2[n*k];
			}
			dst[l*N+k] = l_dst; // Store into transpose(dst) for better hit rate.
		}
	}

	// Gather the transposed dst matrix.
	MPI_Allgather(&dst[start*N], share*N, FLOAT_COMPLEX,
		dst, share*N, FLOAT_COMPLEX, MPI_COMM_WORLD);
	MPI_Bcast(&dst[mpi_nprocs*share*N], extra*N, FLOAT_COMPLEX,
		mpi_nprocs-1, MPI_COMM_WORLD);
	
	// Transpose the final matrix.
	for (m=0; m<M; m++) {
		for (n=0; n<m; n++) {
			float complex t = dst[m*N+n];
			dst[m*N+n] = dst[n*M+m];
			dst[n*M+m] = t;
		}
	}
}


int idft(float complex *dst, float complex* src, float complex* w1, float complex* w2, float complex* tmp, int N, int M) {
	
	// Each process stores to a horizontal slice of the transposed tmp array.
	int start, count, share, extra, end;
	start = mpi_rank * (M/mpi_nprocs);
	share = M / mpi_nprocs;
	end   = start + share;
	extra = M % mpi_nprocs;
	if (mpi_rank == mpi_nprocs-1)
		end += extra;

	int k, l, m, n;
	for (l=start; l<end; l++) {
		for (n=0; n<N; n++) {
			float complex l_dst = 0.0;
			for (m=0; m<M; m++) {
				l_dst += src[n*M+m]/w1[l*m];
			}
			tmp[l*N+n] = l_dst/M; // Store into transpose(tmp) for better hit rate.
		}
	}

	for (k=0; k<N; k++) {
		for (l=start; l<end; l++) {
			float complex l_dst = 0.0;
			for (n=0; n<N; n++) {
				l_dst += tmp[l*N+n]/w2[n*k];
			}
			dst[l*N+k] = l_dst/N; // Store into transpose(dst) for better hit rate.
		}
	}

	// Gather the transposed dst matrix.
	MPI_Allgather(&dst[start*N], share*N, FLOAT_COMPLEX,
		dst, share*N, FLOAT_COMPLEX, MPI_COMM_WORLD);
	MPI_Bcast(&dst[mpi_nprocs*share*N], extra*N, FLOAT_COMPLEX,
		mpi_nprocs-1, MPI_COMM_WORLD);
	
	// Transpose the final matrix.
	for (m=0; m<M; m++) {
		for (n=0; n<m; n++) {
			float complex t = dst[m*N+n];
			dst[m*N+n] = dst[n*M+m];
			dst[n*M+m] = t;
		}
	}
}



int mri(
		float* img, 
		float complex* f, 
		float* mask, 
		float lambda,
		int N1,
		int N2)
{
	int i, j;

	float complex* f0	    = (float complex*) calloc(N1*N2,sizeof(float complex));
	float complex* dx	    = (float complex*) calloc(N1*N2,sizeof(float complex));
	float complex* dy	    = (float complex*) calloc(N1*N2,sizeof(float complex));

	float complex* dx_new   = (float complex*) calloc(N1*N2,sizeof(float complex));
	float complex* dy_new   = (float complex*) calloc(N1*N2,sizeof(float complex));

	float complex* dtildex	= (float complex*) calloc(N1*N2,sizeof(float complex));
	float complex* dtildey	= (float complex*) calloc(N1*N2,sizeof(float complex));
	float complex* u_fft2	= (float complex*) calloc(N1*N2,sizeof(float complex));
	float complex* u		= (float complex*) calloc(N1*N2,sizeof(float complex));

	float complex* fftmul	= (float complex*) calloc(N1*N2,sizeof(float complex));
	float complex* Lap		= (float complex*) calloc(N1*N2,sizeof(float complex));
	float complex* diff		= (float complex*) calloc(N1*N2,sizeof(float complex));

	float sum = 0;

	for(i=0; i<N1; i++)
		for(j=0; j<N2; j++)
			sum += (SQR(crealf(f(i,j))/N1) + SQR(cimagf(f(i,j))/N1));

	float normFactor = 1.f/sqrtf(sum);
	float scale		 = sqrtf(N1*N2);

	for(i=0; i<N1; i++) {
		for(j=0; j<N2; j++) {
			f(i, j)  = f(i, j)*normFactor;
			f0(i, j) = f(i, j);
		}
	}
	Lap(N1-1, N2-1)	= 0.f;
	Lap(N1-1, 0)	= 1.f; 
	Lap(N1-1, 1)	= 0.f;
	Lap(0, N2-1)	= 1.f;
	Lap(0, 0)		= -4.f; 
	Lap(0, 1)		= 1.f;
	Lap(1, N2-1)	= 0.f;
	Lap(1, 0)		= 1.f; 
	Lap(1, 1)		= 0.f;

	float complex *w1;
	float complex *w2;
	float complex *buff;

	dft_init(&w1, &w2, &buff, N1, N2);
	dft(Lap, Lap, w1, w2, buff, N1, N2);

	for(i=0;i<N1;i++)
		for(j=0;j<N2;j++)					
			fftmul(i,j) = 1.0/((lambda/Gamma1)*mask(i,j) - Lap(i,j) + Gamma2);

	int OuterIter,iter;
	for(OuterIter= 0; OuterIter<MaxOutIter; OuterIter++) {
		for(iter = 0; iter<MaxIter; iter++) {

			for(i=0;i<N1;i++)	
				for(j=0;j<N2;j++)
					diff(i,j)  = dtildex(i,j)-dtildex(i,(j-1)>=0?(j-1):0) + dtildey(i,j)- dtildey((i-1)>=0?(i-1):0,j) ;

			dft(diff, diff, w1, w2, buff, N1, N2);

			for(i=0;i<N1;i++)
				for(j=0;j<N2;j++)
					u_fft2(i,j) = fftmul(i,j)*(f(i,j)*lambda/Gamma1*scale-diff(i,j)+Gamma2*u_fft2(i,j)) ;

			idft(u, u_fft2, w1, w2, buff, N1, N2);

			for(i=0;i<N1;i++) {
				for(j=0;j<N2;j++) {
					float tmp;
					float Thresh=1.0/Gamma1;

					dx(i,j)     = u(i,j<(N2-1)?(j+1):j)-u(i,j)+dx(i,j)-dtildex(i,j) ;
					dy(i,j)     = u(i<(N1-1)?(i+1):i,j)-u(i,j)+dy(i,j)-dtildey(i,j) ;

					tmp = sqrtf(SQR(crealf(dx(i,j)))+SQR(cimagf(dx(i,j))) + SQR(crealf(dy(i,j)))+SQR(cimagf(dy(i,j))));
					tmp = max(0,tmp-Thresh)/(tmp+(tmp<Thresh));
					dx_new(i,j) =dx(i,j)*tmp;
					dy_new(i,j) =dy(i,j)*tmp;
					dtildex(i,j) = 2*dx_new(i,j) - dx(i,j);
					dtildey(i,j) = 2*dy_new(i,j) - dy(i,j);
					dx(i,j)      = dx_new(i,j);
					dy(i,j)      = dy_new(i,j);
				}
			}
		}
		for(i=0;i<N1;i++) {
			for(j=0;j<N2;j++) {
				f(i,j) += f0(i,j) - mask(i,j)*u_fft2(i,j)/scale;  
			}
		}
	}

	for(i=0; i<N1; i++) {
		for(j=0; j<N2; j++) {
			img(i, j) = sqrt(SQR(crealf(u(i, j))) + SQR(cimagf(u(i, j))));
		}
	}

	free(w1);
	free(w2);
	free(buff);
	return 0;
}
