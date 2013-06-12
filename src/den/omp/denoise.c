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

//THIS IS THE OPENMP-PARALLELIZED VERSION OF THE CODE. (MILESTONE 3)
#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <string.h>
//added
#include <omp.h>

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
    float *unew;
	float sigma2, gamma, r;
    int m, n;
    int iter;

	int* Converged = malloc(sizeof(int));
    
    /* Initializations */
    sigma2 = SQR(sigma);
    gamma = lambda/sigma2;
    memcpy(u, f, sizeof(float)*M*N);
    g = (float*) calloc(M*N, sizeof(float));
    
    /*** Main gradient descent loop ***/
    for(iter = 1; iter <= MAXITER; iter++) {
       
        /* Approximate g = 1/|grad u| */
        for(n = 1; n < N-1; n++)
            for(m = 1; m < M-1; m++)
                g[CENTER] = 1.0/sqrt( EPSILON
                   + SQR(u[CENTER] - u[RIGHT])
                   + SQR(u[CENTER] - u[LEFT])
                   + SQR(u[CENTER] - u[DOWN])
                   + SQR(u[CENTER] - u[UP]) );        
        
        /* Update u by a sem-implict step */
        *Converged = 1;

        #pragma omp parallel for default(shared) private(r)
        for(n = 1; n < N-1; n++)
        {
            for(m = 1; m < M-1; m++)
            {
                /* Evaluate r = I1(u*f/sigma^2) / I0(u*f/sigma^2) with
                 a cubic rational approximation. */
                r = u[CENTER]*f[CENTER]/sigma2;
                r = ( r*(2.38944 + r*(0.950037 + r)) )
                   / ( 4.65314 + r*(2.57541 + r*(1.48937 + r)) );
                /* Update u */           
                float utemp;
                utemp = u[CENTER];
                u[CENTER] = ( u[CENTER] + DT*(u[RIGHT]*g[RIGHT]
                   + u[LEFT]*g[LEFT] + u[DOWN]*g[DOWN] + u[UP]*g[UP] 
                   + gamma*f[CENTER]*r) ) /
                   (1.0 + DT*(g[RIGHT] + g[LEFT] + g[DOWN] + g[UP] + gamma));
                if (fabs(utemp - u[CENTER]) > Tol)
                    *Converged = 0;
            }
        }

                /* Test for convergence, sequentially */
        /*
        for (n = 1; n < N-1; n++) 
        {
            for (m = 1; m < M-1; m++) 
            {
                if (fabs(ulast[CENTER] - unew[CENTER] > Tol))
                {
                    Converged = 0;
                    break;
                }
            }
        } */

        if(*Converged==1)
            break;
    }
       
    free(g);  /* Free temporary array */    

    return 0;
}

