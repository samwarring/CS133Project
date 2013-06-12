#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <complex.h>
#include <sys/time.h>

#include "mri.h"
#include "util.h"

int main(int argc, char *argv[]) {

	if(argc<2) 
		return -1;

	char* fname = argv[1];
	char fname_out[50] = "recon.bmp";

	if (argc>2)
		strcpy(fname_out, argv[2]);

	FILE* fin =	fopen(fname,"rb");

	if (!fin) {
		printf("no such file: %s\n", fname);
		return -1;
	}

	int N1;
	int	N2;
	int i, j;
	float lambda;

	fread(&N1, sizeof(int), 1, fin);
	fread(&N2, sizeof(int), 1, fin);
	fread(&lambda, sizeof(float), 1, fin);

	float complex *f = (float complex*) calloc(N1*N2,sizeof(float complex));
	float *img		 = (float*) calloc(N1*N2,sizeof(float));
	float *mask		 = (float*) calloc(N1*N2,sizeof(float));
	float *f_r		 = (float*) calloc(N1*N2,sizeof(float));
	float *f_i	 	 = (float*) calloc(N1*N2,sizeof(float));

	// Read the data from files
	fread(f_r, sizeof(float), N1*N2, fin);
	fread(f_i, sizeof(float), N1*N2, fin);
	fread(mask, sizeof(float), N1*N2, fin);
	fclose(fin);

	for(i=0; i<N1; i++) {
		for(j=0; j<N2; j++) {
			f[i*N2+j] = f_r[i*N2+j] + f_i[i*N2+j]*I;
		}
	}
	free(f_r);
	free(f_i);

	mri(img, f, mask, lambda, N1, N2);

	imnorm(img, N1, N2);
	imwrite(img, N1, N2, fname_out);
	
	free(f);
	free(img);
	free(mask);

	return 0;
}
