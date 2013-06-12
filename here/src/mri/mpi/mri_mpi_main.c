#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <complex.h>
#include <sys/time.h>

#include "mri.h"
#include "util_mpi.h"

MPI_Datatype FLOAT_COMPLEX;
MPI_Op FC_SUM;

void mpi_sum_complexf(float complex* in, float complex* inout, int* len, MPI_Datatype *dptr) {
	
	int i;
	float complex c;
	for (i=0; i< *len; i++) {
		c = *in + *inout;
		*inout = c;
		in++;
		inout++;
	}
}

int main(int argc, char *argv[]) {
	
	char* fname;
	char fname_out[50] = "recon.bmp";
	FILE* fin;
	int N1;
	int N2;
	int i, j;
	float lambda;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	// Include float complex as an MPI datatype and create the
	// float complex sum operator.
	MPI_Type_contiguous(2, MPI_FLOAT, &FLOAT_COMPLEX);
	MPI_Type_commit(&FLOAT_COMPLEX);
	MPI_Op_create((void*) &mpi_sum_complexf, 1, &FC_SUM);

	if(argc<2) {
		MPI_Finalize();
		return -1;
	}

	fname = argv[1];

	int exit = 0;
	if (mpi_rank == 0 && argc>2) {
		strcpy(fname_out, argv[2]);
		fin =	fopen(fname,"rb");
		if (!fin) {
			printf("no such file: %s\n", fname);
			exit = 1;
		}
	}
	MPI_Bcast(&exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (exit == 1) {
		MPI_Finalize();
		return -1;
	}

	if (mpi_rank == 0) {
		fread(&N1, sizeof(int), 1, fin);
		fread(&N2, sizeof(int), 1, fin);
		fread(&lambda, sizeof(float), 1, fin);
	}

	MPI_Bcast(&N1, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N2, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lambda, 1, MPI_INT, 0, MPI_COMM_WORLD);

	float complex *f = (float complex*) calloc(N1*N2,sizeof(float complex));
	float *img		 = (float*) calloc(N1*N2,sizeof(float));
	float *mask		 = (float*) calloc(N1*N2,sizeof(float));
	float *f_r		 = (float*) calloc(N1*N2,sizeof(float));
	float *f_i	 	 = (float*) calloc(N1*N2,sizeof(float));

	// Read the data from files
	if (mpi_rank == 0) {
		fread(f_r, sizeof(float), N1*N2, fin);
		fread(f_i, sizeof(float), N1*N2, fin);
		fread(mask, sizeof(float), N1*N2, fin);
		fclose(fin);

		for(i=0; i<N1; i++) {
			for(j=0; j<N2; j++) {
				f[i*N2+j] = f_r[i*N2+j] + f_i[i*N2+j]*I;
			}
		}
	}
	free(f_r);
	free(f_i);
	MPI_Bcast(f, N1*N2, FLOAT_COMPLEX, 0, MPI_COMM_WORLD);
	MPI_Bcast(mask, N1*N2, MPI_FLOAT, 0, MPI_COMM_WORLD);

	mri(img, f, mask, lambda, N1, N2);

	if (mpi_rank == 0) {
		imnorm(img, N1, N2);
		imwrite(img, N1, N2, fname_out);
	}
	
	free(f);
	free(img);
	free(mask);

	MPI_Finalize();

	return 0;
}
