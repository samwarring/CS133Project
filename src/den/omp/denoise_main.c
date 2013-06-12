#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <sys/time.h>

#include "denoise.h"
#include "util.h"

int main(int argc, char *argv[]) {

	if(argc<2) 
		return -1;

	char* fname = argv[1];
	char fname_out[50] = "denoise.bmp";

	if (argc>2)
		strcpy(fname_out, argv[2]);

	int N1;
	int	N2;
	int i, j;

	float *img;
	float *img_denoised;
	
	int err = imread(&img, &N1, &N2, fname);
	if (err!=0) return err;

	img_denoised = (float*)calloc(N1*N2, sizeof(float));
	denoise(img_denoised, img, N1, N2);
	imwrite(img_denoised, N1, N2, fname_out);

	return 0;
}
