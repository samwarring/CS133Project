#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#include <CL/cl.h>

#include "seg.h"

// TODO: opencl kernel code
const char* programSource =
"#define SQR(x) (x)*(x)\n"
"#define CUB(x) (x)*(x)*(x)\n"
"#define phi(i,j)	phi[(i)*m+(j)]\n"
"#define u(i,j)		u[(i)*m+(j)]\n"
"#define curv(i,j)	curv[(i)*m+(j)]\n"
"#define mu 0.18*255*255\n"
"#define epsilon 5e-5f\n"
"__kernel\n"
"void dostuff(__constant float *phi, __constant float *u, __global float *curv, __constant float *c1, __constant float *c2, __constant int *sizes) {\n"
  "    int m = sizes[0];\n"
  "    int n = sizes[1];\n"
  "    int idx = get_global_id(0);\n"
  "    int i = idx / 1024;\n"
  "    int j = idx % 1024;\n"
  "    if (i==0 || j==0 || i == m-1 || j == n-1) return; \n"
"float Dx_p = phi(i+1,j) - phi(i,j);\n"
"float Dy_p = phi(i,j+1) - phi(i,j);\n"

  "    float Dx_m = phi(i,j) - phi(i-1,j);\n"
  "    float Dy_m = phi(i,j) - phi(i,j-1);\n"
  "    float Dx_0 = (phi(i+1,j) - phi(i-1,j))/2;\n"
  "    float Dy_0 = (phi(i,j+1) - phi(i,j-1))/2;\n"
  "    float Dxx = Dx_p - Dx_m ;\n"
  "    float Dyy = Dy_p - Dy_m ;\n"
  "    float Dxy = (phi(i+1,j+1) - phi(i+1,j-1) - phi(i-1,j+1) + phi(i-1,j-1)) / 4;\n"
  "    float Grad	= sqrt(Dx_0*Dx_0 + Dy_0*Dy_0);\n"
  "    float K		= (Dx_0*Dx_0*Dyy - 2*Dx_0*Dy_0*Dxy + Dy_0*Dy_0*Dxx) / (CUB(Grad) + epsilon);\n"
  "    curv(i, j) = Grad*(mu*K + SQR(256*u(i,j)-*c1) - SQR(256*u(i,j)-*c2));\n"
"}"
;

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

  // -------------------------------------------------------------------------
  // BEGIN OPENCL
  // -------------------------------------------------------------------------

  size_t datasize = sizeof(float) * n*m;

  // Use this to check the output of each API call
  cl_int status;  
   
  // Retrieve the number of platforms
  cl_uint numPlatforms = 0;
  status = clGetPlatformIDs(0, NULL, &numPlatforms);

  // Allocate enough space for each platform
  cl_platform_id *platforms = NULL;
  platforms = (cl_platform_id*)malloc(
      numPlatforms*sizeof(cl_platform_id));

  // Fill in the platforms
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);

  // Retrieve the number of devices
  cl_uint numDevices = 0;
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, 
      NULL, &numDevices);

  // Allocate enough space for each device
  cl_device_id *devices;
  devices = (cl_device_id*)malloc(
      numDevices*sizeof(cl_device_id));

  // Fill in the devices 
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,        
      numDevices, devices, NULL);

  // Create a context and associate it with the devices
  cl_context context;
  context = clCreateContext(NULL, numDevices, devices, NULL, 
      NULL, &status);

  // Create a command queue and associate it with the device 
  cl_command_queue cmdQueue;
  cmdQueue = clCreateCommandQueue(context, devices[0], 0, 
      &status);

  // Create a buffer object that will contain the data 
  // from the host array A
  cl_mem bufPhi;
  bufPhi = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize,                       
     NULL, &status);

  // Create a buffer object that will contain the data 
  // from the host array B
  cl_mem bufU;
  bufU = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize,                        
      NULL, &status);

  // Create a buffer object that will hold the output data
  cl_mem bufCurv;
  bufCurv = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize,
      NULL, &status); 

  cl_mem bufC1;
  bufC1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);

  cl_mem bufC2;
  bufC2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), NULL, &status);

  cl_mem bufSize;
  bufSize = clCreateBuffer(context, CL_MEM_READ_ONLY, 2*sizeof(int), NULL, &status);

  // Write input array B to the device buffer bufferB
  // U is read only and not changed ever????
  status = clEnqueueWriteBuffer(cmdQueue, bufU, CL_FALSE, 
      0, datasize, u, 0, NULL, NULL);

  // Create a program with source code
  cl_program program = clCreateProgramWithSource(context, 1, 
      (const char**)&programSource, NULL, &status);

  // Build (compile) the program for the device
  status = clBuildProgram(program, numDevices, devices, 
      NULL, NULL, NULL);
  
  if (status != CL_SUCCESS) {
    printf("build error\n");
    // Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    // Allocate memory for the log
    char *log = (char *) malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    // Print the log
    printf("%s\n", log);

    free(log);
  }

  // Create the vector addition kernel
  cl_kernel kernel;
  kernel = clCreateKernel(program, "dostuff", &status);

  // Associate the input and output buffers with the kernel 
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufPhi);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufU);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufCurv);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufC1);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufC2);
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufSize);

  // Define an index space (global work size) of work 
  // items for execution. A workgroup size (local work size) 
  // is not required, but can be used.
  size_t globalWorkSize[1];   

  // There are n*m work-items 
  globalWorkSize[0] = n*m;

  // Write input array A to the device buffer bufferA
  //
  int sizes[2] = {m, n};
  status = clEnqueueWriteBuffer(cmdQueue, bufSize, CL_FALSE, 
      0, 2*sizeof(int), sizes, 0, NULL, NULL);


  // -------------------------------------------------------------------------
  // END OPENCL INIT
  // -------------------------------------------------------------------------


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

    // -----------------------------------------------------------------------
    // START OPENCL DO
    // -----------------------------------------------------------------------
  
    // Write input array A to the device buffer bufferA
    status = clEnqueueWriteBuffer(cmdQueue, bufPhi, CL_FALSE, 
        0, datasize, phi, 0, NULL, NULL);

    // Write input array A to the device buffer bufferA
    status = clEnqueueWriteBuffer(cmdQueue, bufC1, CL_FALSE, 
        0, sizeof(float), &c1, 0, NULL, NULL);

    // Write input array A to the device buffer bufferA
    status = clEnqueueWriteBuffer(cmdQueue, bufC2, CL_FALSE, 
        0, sizeof(float), &c2, 0, NULL, NULL);

    // Execute the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, 
        globalWorkSize, NULL, 0, NULL, NULL);

    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufCurv, CL_TRUE, 0, 
        datasize, curv, 0, NULL, NULL);
    

    // -----------------------------------------------------------------------
    // END OPENCL DO
    // -----------------------------------------------------------------------

		for(j=0; j < n; j++) {
			curv( 0, j) = curv( 1, j);
			curv(m-1,j) = curv(m-2,j);
		}

		for(i=0; i < m; i++) {
			curv(i, 0 ) = curv(i, 1 );
			curv(i,n-1) = curv(i,n-2);
		}

		for(i=0; i<m; i++) {
			for (j=0; j<n; j++) {
				phi(i, j) += curv(i, j) * dt;
			}
		}
	}

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
