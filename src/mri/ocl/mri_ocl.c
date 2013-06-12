#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include "mri.h"
#include <CL/cl.h>

#define print(x) fprintf(stderr,x)

// OpenCL variables:
const char*      kernelFile = "kernels.cl";
char*            oclCode;
int              oclCodeLen;
size_t           matrixSize;
size_t           w1Size;
size_t           w2Size;
cl_int           status;
cl_uint          numPlatforms;
cl_platform_id*  platforms;
cl_uint          numDevs;
cl_device_id*    devs;
cl_context       context;
cl_command_queue cmdQueue;
cl_mem           bufSrc, bufTmp, bufDst, bufW1, bufW2, bufDims;
cl_program       program;
cl_kernel        kernDft1, kernDft2, kernIdft1, kernIdft2;
size_t           globalWorkSize[2];
cl_event         prevEvent = 0;

const char* errorDescription(cl_int status) {

	switch(status) {
		case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
		case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
		default: {
			fprintf(stderr,"(((%d)))", status);
			return "";
		}
	}

}

char* toStr(float complex fc) {
	
	char* s = malloc(sizeof(char)*32);
	sprintf(s, "(%f,%f)", creal(fc), cimag(fc));
	return s;
}

void check(cl_int status, const char* errmsg) {
	
	if (status != CL_SUCCESS) {
		fprintf(stderr, "%s: %s\n", errmsg, errorDescription(status));
		exit(status);
	}
}

void loadFile(const char* filename, char** text, int* size) {
	
	FILE* fp = fopen(filename, "r");
	fseek(fp, 0, SEEK_END);
	*size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	*text = (char*) malloc(*size * sizeof(char));
	fread(*text, 1, *size, fp);
	fclose(fp);
}

void init_everything(int N1, int N2) {
	
	matrixSize = N1*N2*sizeof(float complex);
	w1Size = N2*N2*sizeof(float complex);
	w2Size = N1*N1*sizeof(float complex);

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	check(status, "Error getting number of platforms");
	platforms = (cl_platform_id*) malloc(numPlatforms*sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	check(status, "Error filling in platforms");

	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevs);
	check(status, "Error getting number of devices");
	devs = (cl_device_id*) malloc(numDevs*sizeof(cl_device_id));
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevs, devs, NULL);
	check(status, "Error filling in devices");

	context = clCreateContext(NULL, numDevs, devs, NULL, NULL, &status);
	check(status, "Error creating context");

	cmdQueue = clCreateCommandQueue(context, devs[0], 0, &status);
	check(status, "Error creating command queue");

	bufSrc = clCreateBuffer(context, CL_MEM_READ_ONLY, matrixSize, NULL, &status);
	check(status, "Error creating src buffer");
	bufTmp = clCreateBuffer(context, CL_MEM_READ_WRITE, matrixSize, NULL, &status);
	check(status, "Error creating tmp buffer");
	bufDst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrixSize, NULL, &status);
	check(status, "Error creating dst buffer");
	bufW1 = clCreateBuffer(context, CL_MEM_READ_ONLY, w1Size, NULL, &status);
	check(status, "Error creating w1 buffer");
	bufW2 = clCreateBuffer(context, CL_MEM_READ_ONLY, w2Size, NULL, &status);
	check(status, "Error creating w2 buffer");
	bufDims = clCreateBuffer(context, CL_MEM_READ_ONLY, 2*sizeof(int), NULL, &status);
	check(status, "Error creating dims buffer");

	// Read in the kernels file
	loadFile(kernelFile, &oclCode, &oclCodeLen);
	program = clCreateProgramWithSource(context, 1, (const char**)&oclCode, NULL, &status);
	check(status, "Error creating program");
	status = clBuildProgram(program, numDevs, devs, NULL, NULL, NULL);

	kernDft1 = clCreateKernel(program, "dft1", &status);
	check(status, "Error creating kernel dft1");
	kernDft2 = clCreateKernel(program, "dft2", &status);
	check(status, "Error creating kernel dft2");
	kernIdft1 = clCreateKernel(program, "idft1", &status);
	check(status, "Error creating kernel idft1");
	kernIdft2 = clCreateKernel(program, "idft2", &status);
	check(status, "Error creating kernel idft2");
}

void kernel_dft1(float complex* src, int N, int M) {
	
	// Send the src matrix to the dft1 kernel.
	status = clEnqueueWriteBuffer
		(cmdQueue, bufSrc, CL_TRUE, 0, matrixSize, src, 0, NULL, NULL);
	check(status, "Error enqueueing write to src buffer");

	// Execute the dft1 kernel.
	globalWorkSize[0] = M;
	globalWorkSize[1] = N;
	status = clEnqueueNDRangeKernel
		(cmdQueue, kernDft1, 2, NULL, globalWorkSize, NULL, 0, NULL, &prevEvent);
	check(status, "Error enqueueing kernDft1");
	clWaitForEvents(1, &prevEvent);
}

void kernel_dft2(float complex* dst, int N, int M) {
	
	// Execute the dft2 kernel.
	globalWorkSize[0] = N;
	globalWorkSize[1] = M;
	status = clEnqueueNDRangeKernel
		(cmdQueue, kernDft2, 2, NULL, globalWorkSize, NULL, 0, NULL, &prevEvent);
	check(status, "Error enqueueing kernDft2");
	clWaitForEvents(1, &prevEvent);

	// Read from the dst buffer to the dst array.
	status = clEnqueueReadBuffer
		(cmdQueue, bufDst, CL_TRUE, 0, matrixSize, dst, 0, NULL, &prevEvent);
	check(status, "Error enqueueing read from dst buffer");
}

void kernel_idft1(float complex* src, int N, int M, float complex* w1, float complex* tmp) {

	// Send the src matrix to the idft1 kernel.
	status = clEnqueueWriteBuffer
		(cmdQueue, bufSrc, CL_TRUE, 0, matrixSize, src, 0, NULL, NULL);
	check(status, "Error enqueueing write to src buffer");
	clWaitForEvents(1, &prevEvent);

	// Execute the idft1 kernel.
	globalWorkSize[0] = M;
	globalWorkSize[1] = N;
	status = clEnqueueNDRangeKernel
		(cmdQueue, kernIdft1, 2, NULL, globalWorkSize, NULL, 0, NULL, &prevEvent);
	check(status, "Error enqueueing kernIdft1");
	clWaitForEvents(1, &prevEvent);
}

void kernel_idft2(float complex* dst, int N, int M) {
	
	// Execute the idft2 kernel.
	globalWorkSize[0] = N;
	globalWorkSize[1] = M;
	status = clEnqueueNDRangeKernel
		(cmdQueue, kernIdft2, 2, NULL, globalWorkSize, NULL, 0, NULL, &prevEvent);
	check(status, "Error enqueueing kernIdft2");
	clWaitForEvents(1, &prevEvent);

	// Read from the dst buffer to the dst array.
	status = clEnqueueReadBuffer
		(cmdQueue, bufDst, CL_TRUE, 0, matrixSize, dst, 0, NULL, &prevEvent);
	check(status, "Error enqueueing read from dst buffer");
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

int dft(float complex *dst, float complex* src, float complex* w1, float complex* w2, float complex* tmp, int N, int M) {
	
	kernel_dft1(src, N, M);
	kernel_dft2(src, N, M);

	return 0;
}

int idft(float complex *dst, float complex* src, float complex* w1, float complex* w2, float complex* tmp, int N, int M) {

	kernel_idft1(src, N, M, w1, tmp);
	kernel_idft2(dst, N, M);

	return 0;
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

	init_everything(N1, N2);

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

	// OpenCL can transfer w1, w2, N, and M to the device.
	int Ns[] = {N1, N2};
	status = clEnqueueWriteBuffer(cmdQueue, bufW1, CL_FALSE, 0, w1Size, w1, 0, NULL, NULL);
	check(status, "Error enqueueing write to w1 buffer");
	status = clEnqueueWriteBuffer(cmdQueue, bufW2, CL_FALSE, 0, w2Size, w2, 0, NULL, NULL);
	check(status, "Error enqueueing write to w2 buffer");
	status = clEnqueueWriteBuffer(cmdQueue, bufDims, CL_FALSE, 0, 2*sizeof(int), Ns, 0, NULL, &prevEvent);
	check(status, "Error enqueueing write to dims buffer");

	// Associate the arguments for the dft1 kernel.
	status = clSetKernelArg(kernDft1, 0, sizeof(cl_mem), &bufSrc);
	check(status, "Error setting src argument for kernel dft1");
	status = clSetKernelArg(kernDft1, 1, sizeof(cl_mem), &bufTmp);
	check(status, "Error setting tmp argument for kernel dft1");
	status = clSetKernelArg(kernDft1, 2, sizeof(cl_mem), &bufW1);
	check(status, "Error setting w1 argument for kernel dft1");
	status = clSetKernelArg(kernDft1, 3, sizeof(cl_mem), &bufDims);
	check(status, "Error setting dims argument for kernel dft1");

	// Associate the arguments for the dft2 kernel.
	status = clSetKernelArg(kernDft2, 0, sizeof(cl_mem), &bufTmp);
	check(status, "Error setting tmp argument for kernel dft2");
	status = clSetKernelArg(kernDft2, 1, sizeof(cl_mem), &bufDst);
	check(status, "Error setting dst argument for kernel dft2");
	status = clSetKernelArg(kernDft2, 2, sizeof(cl_mem), &bufW2);
	check(status, "Error setting w2 argument for kernel dft2");
	status = clSetKernelArg(kernDft2, 3, sizeof(cl_mem), &bufDims);
	check(status, "Error setting dims argument for kernel dft2");

	// Associate the arguments for the idft1 kernel.
	status = clSetKernelArg(kernIdft1, 0, sizeof(cl_mem), &bufSrc);
	check(status, "Error setting src argument for kernel idft1");
	status = clSetKernelArg(kernIdft1, 1, sizeof(cl_mem), &bufTmp);
	check(status, "Error setting tmp argument for kernel idft1");
	status = clSetKernelArg(kernIdft1, 2, sizeof(cl_mem), &bufW1);
	check(status, "Error setting w1 argument for kernel idft1");
	status = clSetKernelArg(kernIdft1, 3, sizeof(cl_mem), &bufDims);
	check(status, "Error setting dims argument for kernel idft1");
	
	// Associate the arguments for the idft2 kernel.
	status = clSetKernelArg(kernIdft2, 0, sizeof(cl_mem), &bufTmp);
	check(status, "Error setting tmp argument for kernel idft2");
	status = clSetKernelArg(kernIdft2, 1, sizeof(cl_mem), &bufDst);
	check(status, "Error setting dst argument for kernel idft2");
	status = clSetKernelArg(kernIdft2, 2, sizeof(cl_mem), &bufW2);
	check(status, "Error setting w2 argument for kernel idft2");
	status = clSetKernelArg(kernIdft2, 3, sizeof(cl_mem), &bufDims);
	check(status, "Error setting dims argument for kernel idft2");

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

	// free opencl data
	clReleaseKernel(kernDft1);
	clReleaseKernel(kernDft2);
	clReleaseKernel(kernIdft1);
	clReleaseKernel(kernIdft2);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufSrc);
	clReleaseMemObject(bufTmp);
	clReleaseMemObject(bufDst);
	clReleaseMemObject(bufW1);
	clReleaseMemObject(bufW2);
	clReleaseMemObject(bufDims);
	clReleaseContext(context);

	free(w1);
	free(w2);
	free(buff);
	return 0;
}
