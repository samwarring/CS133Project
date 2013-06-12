/*========================================================================
 *
 * RICIANDENOISE.C  Total variation minimization for Rician denoising
 *
 * int riciandenoisemx(u, f, sigma,lambda,Tol) performs denoising on image f
 * with Rician noise with parameter sigma.  The denoised image u is found
 * as the minimizer of 
 *
 *         /                      / [ u^2 + f^2            u f    ]
 *    min  | |grad u| dx + lambda | [ --------- - log I0( ----- ) ] dx.
 *     u   /                      / [ 2 sigma^2          sigma^2  ]
 *
 * Parameter lambda >= 0 determines the strength of the denoising: smaller
 * lambda implies stronger denoising.  Tol specifies the stopping 
 * tolerance, the method stops when ||u^Iter - u^Iter-1||_inf < Tol.
 *
 *======================================================================*/

//THIS IS THE OPENCL-PARALLELIZED VERSION OF THE CODE. (MILESTONE 6)
#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <string.h>
//added
#include <CL/cl.h>
#include <stdlib.h>

#include "denoise.h"

/* Macro functions */
#define SQR(x)	 ((x)*(x))

/* Macros for referring to pixel neighbors */
#define CENTER   (m+n*M)
#define RIGHT    (m+n*M+M)
#define LEFT     (m+n*M-M)
#define DOWN     (m+n*M+1)
#define UP       (m+n*M-1)        

int denoise(
		float *u, 
		const float *f, 
		int M, 
		int N)
{
	float *g;
    float *ulast; 
	float sigma2, gamma, r;
	int Converged;
    int m, n;
    int iter;
    FILE *fp = fopen("g_kernel.cl", "r");
    if (!fp)
        printf("Error opening the file\n");
    char g_kernel_source[10000];

    int i = 0; 
    while (feof(fp))
    {
        g_kernel_source[i++] = fgetc(fp);
    }
    g_kernel_source[i] = '\0';
    close(fp);
    
    
    fp = fopen("u_kernel.cl", "r");
    if (!fp)
        printf("Error opening the file\n");
    char u_kernel_source[10000];

    i = 0; 
    while (feof(fp))
    {
        u_kernel_source[i++] = fgetc(fp);
    }
    u_kernel_source[i] = '\0';
    close(fp);
    
    
    /* Initializations */
    sigma2 = SQR(sigma);
    gamma = lambda/sigma2;
    memcpy(u, f, sizeof(float)*M*N);
    g = (float*) calloc(M*N, sizeof(float));
    
        
    

    //copy over current value of u to test later

    cl_int status;
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    //allocate enough space for each platfrm
    cl_platform_id *platforms = NULL;
    platforms = (cl_platform_id*)malloc(
        numPlatforms*sizeof(cl_platform_id));
    //fill in the platforms
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    //retrieve the number of devices
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0,
        NULL, &numDevices);
    //allocate space for each device
    cl_device_id *devices;
    devices = (cl_device_id*)malloc(
        numDevices*sizeof(cl_device_id));

    //fill in the devices
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,
        numDevices, devices, NULL);

    //create a context and associate with the devices
    cl_context context;
    context = clCreateContext(NULL, numDevices, devices, NULL,
        NULL, &status);

    cl_command_queue cmdQueue;
    cmdQueue = clCreateCommandQueue(context, devices[0], 0,
        &status);
 
 
        //have to copy u to test later
        /*for (n = 1; n < N-1; n++)
            for (m = 1; m < M-1; m++)
                ulast[CENTER] = u[CENTER]; */
        
        /* Approximate g = 1/|grad u| */
        /*for(n = 1; n < N-1; n++)
            for(m = 1; m < M-1; m++)
                g[CENTER] = 1.0/sqrt( EPSILON
                   + SQR(u[CENTER] - u[RIGHT])
                   + SQR(u[CENTER] - u[LEFT])
                   + SQR(u[CENTER] - u[DOWN])
                   + SQR(u[CENTER] - u[UP]) );  */

        cl_mem Gammabuf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                    sizeof(float), NULL, &status);
                
        //write input array to device buffer
        float *gamma_ptr = malloc(sizeof(float));
        *gamma_ptr = gamma;
        status = clEnqueueWriteBuffer(cmdQueue, Gammabuf, CL_FALSE,
                    0, sizeof(float), gamma_ptr, 0, NULL, NULL);


        cl_mem Gbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                    sizeof(g), NULL, &status);
                
        //write input array to device buffer
        status = clEnqueueWriteBuffer(cmdQueue, Gbuf, CL_FALSE,
                    0, sizeof(g), g, 0, NULL, NULL);


        cl_mem Fbuf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                    sizeof(f), NULL, &status);
                
        //write input array to device buffer
        status = clEnqueueWriteBuffer(cmdQueue, Fbuf, CL_FALSE,
                   0, sizeof(f), f, 0, NULL, NULL);


        
        cl_mem Sigma2buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                    sizeof(float), NULL, &status);
        

        float *sigma2_ptr = malloc(sizeof(float));
        *sigma2_ptr = sigma2;
        //write input array to device buffer
        status = clEnqueueWriteBuffer(cmdQueue, Sigma2buf, CL_FALSE,
                    0, sizeof(float), sigma2_ptr, 0, NULL, NULL);


        cl_mem Ubuf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                    sizeof(u), NULL, &status);
                
        //write input array to device buffer
        status = clEnqueueWriteBuffer(cmdQueue, Ubuf, CL_FALSE,
                    0, sizeof(u), u, 0, NULL, NULL);
        
        cl_mem Nbuf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                    sizeof(int), NULL, &status);

        int *N_ptr = malloc(sizeof(int));
        *N_ptr = N;
                
        //write input array to device buffer
        status = clEnqueueWriteBuffer(cmdQueue, Nbuf, CL_FALSE,
                    0, sizeof(int), N_ptr, 0, NULL, NULL);

        cl_mem Mbuf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                    sizeof(int), NULL, &status);
                
        int *M_ptr = malloc(sizeof(int));
        *M_ptr = M;

        //write input array to device buffer
        status = clEnqueueWriteBuffer(cmdQueue, Mbuf, CL_FALSE,
                    0, sizeof(int), M_ptr, 0, NULL, NULL);


        //copy program and build it
        cl_program g_program = clCreateProgramWithSource(context, 1,
                    (const char**)&g_kernel_source, NULL, &status);
        status = clBuildProgram(g_program, numDevices, devices,
                    NULL, NULL, NULL);

        if (status == CL_BUILD_PROGRAM_FAILURE) {
              size_t log_size;
              clGetProgramBuildInfo(g_program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
              char *log = (char *) malloc(log_size);
              clGetProgramBuildInfo(g_program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
              printf("%s\n", log);
        }
        
        cl_program u_program = clCreateProgramWithSource(context, 1,
                    (const char**)&u_kernel_source, NULL, &status);
        status = clBuildProgram(u_program, numDevices, devices,
                    NULL, NULL, NULL);
        if (status == CL_BUILD_PROGRAM_FAILURE) {
              size_t log_size;
              clGetProgramBuildInfo(u_program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
              char *log = (char *) malloc(log_size);
              clGetProgramBuildInfo(u_program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
              printf("%s\n", log);
        }


        cl_kernel g_kernel;
        g_kernel = clCreateKernel(g_program, "g_kernel", &status);
        status = clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &Gbuf);
        status = clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &Ubuf);
        status = clSetKernelArg(g_kernel, 2, sizeof(cl_mem), &Nbuf);
        status = clSetKernelArg(g_kernel, 3, sizeof(cl_mem), &Mbuf);


        cl_kernel u_kernel;
        u_kernel = clCreateKernel(u_program, "u_kernel", &status);
        //parameters
        status = clSetKernelArg(u_kernel, 0, sizeof(cl_mem), &Gammabuf);
        status = clSetKernelArg(u_kernel, 1, sizeof(cl_mem), &Gbuf);
        status = clSetKernelArg(u_kernel, 2, sizeof(cl_mem), &Fbuf);
        status = clSetKernelArg(u_kernel, 3, sizeof(cl_mem), &Sigma2buf);
        status = clSetKernelArg(u_kernel, 4, sizeof(cl_mem), &Ubuf);
        status = clSetKernelArg(u_kernel, 5, sizeof(cl_mem), &Nbuf);
        status = clSetKernelArg(u_kernel, 6, sizeof(cl_mem), &Mbuf);


        size_t globalWorkSize[2];

        globalWorkSize[0] = N;

        globalWorkSize[1] = M;

        //set the kernel for execution
    for(iter = 1; iter <= MAXITER; iter++) 
    {
        status = clEnqueueNDRangeKernel(cmdQueue, g_kernel, 1, NULL,
                    globalWorkSize, NULL, 0, NULL, NULL);
        status = clEnqueueNDRangeKernel(cmdQueue, u_kernel, 1, NULL,
                    globalWorkSize, NULL, 0, NULL, NULL);
        status = clEnqueueReadBuffer(cmdQueue, Ubuf, CL_TRUE, 0, 
                    sizeof(u), u, 0, NULL, NULL);
    }
        /*
        for (n = 1; n < N-1; n++) 
        {
            for (m = 1; m < M-1; m++) 
            {
                if (fabs(ulast[CENTER] - u[CENTER]) > Tol)
                {
                    Converged = 0;
                    break;
                }
            }
        }
        if(Converged==1)
            break;  */
    free(g);  /* Free temporary array */    

    return 0;
}

