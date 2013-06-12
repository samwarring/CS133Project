#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "seg.h"

#define u_sub(i,j)		u_sub[(i)*m+(j)]
#define phi_sub(i,j)		phi_sub[(i+1)*m+(j)]
#define curv_sub(i,j)		curv_sub[(i)*m+(j)]

int seg(float* contour, const float* u, int m, int n, int MaxIter) {

	//float* curv	= (float*)calloc(m*n,sizeof(float));

	float c1,c2;

	int iter;
	int i, j, k;	

	float mu = 0.18*255*255;
	float dt = 0.225/mu;

	float xcent = (m-1) / 2.0;
	float ycent = (n-1) / 2.0;
	float r = fmin(m,n) / 2.0;


  // MPI stuff
  int npes, myrank;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  // Figure out offsets
  int m_per_proc = m/npes;
  int my_m_start = myrank * m_per_proc;
  int my_m_end = my_m_start + m_per_proc;
  int sliceArea = m_per_proc * n;
  int phiArea = (m_per_proc + 2) * n;
  //printf("%i, %i, %i\n", sliceArea, phiArea, 2*n);

  // New sub-buffers
  float *curv_sub = calloc(sliceArea, sizeof(float));
  float *u_sub = calloc(sliceArea, sizeof(float));
  float *phi_sub = calloc(phiArea, sizeof(float));

  //printf("%d: %d to %d\n", myrank, my_m_start, my_m_end);

  for (k = 0; k < m_per_proc * n; k++)
    curv_sub[k] = 0;

  MPI_Scatter((void*)u, n*m_per_proc, MPI_FLOAT,
              (void*)u_sub, n*m_per_proc, MPI_FLOAT,
              0, MPI_COMM_WORLD);

  // Send out all parts of phi to all others
  /*
  if (myrank == 0) {
    int r;
    printf("%i copying phi slice...\n", myrank);
    memcpy((void*)(phi_sub + n), (const void*)phi, (size_t)(phiArea - n));
    printf("%i sending...\n", myrank);
    for (r = 1; r < npes; r++) {
      printf("%i sending to %i...\n", myrank, r);
      MPI_Send(phi, phiArea, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
    }
    printf("%i all sent!\n", myrank);
  }
  else {
    printf("%i receiving...\n", myrank);
    MPI_Status status;
    MPI_Recv(phi_sub, phiArea, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("%i received!\n", myrank);
  }
  */

  // Initialize phi
  //printf("%i\n", m/npes);
  for(i=0; i < m_per_proc; i++) {
    for(j=0; j < n; j++) {
      float xx = i + myrank*m_per_proc;
      float yy = j;
      phi_sub(i, j)	= sqrtf(SQR(xx-xcent) + SQR(yy-ycent)) - r;
    }
  }

	for(iter=0; iter<MaxIter; iter++) {
    // Send forward phi
    //if (myrank % 2 == 1) {
      //MPI_Send(phi+n+sliceArea, n, MPI_FLOAT, 
    MPI_Status status;
    if (myrank != 0) {
      //printf("%i waiting to recv...\n", myrank);
      MPI_Recv(phi_sub, m, MPI_FLOAT, myrank-1, iter, MPI_COMM_WORLD, &status);
      //printf("%i got!\n", myrank);
    }
    if (myrank != npes-1) {
      //printf("%i waiting to send...\n", myrank);
      MPI_Send(phi_sub+sliceArea, m, MPI_FLOAT, myrank+1, iter, MPI_COMM_WORLD);
      //printf("%i sent!\n", myrank);
    }

    // Send backward phi
    if (myrank != npes-1) {
      //printf("%i waiting to recv...\n", myrank);
      MPI_Recv(phi_sub+m+sliceArea, m, MPI_FLOAT, myrank+1, iter, MPI_COMM_WORLD, &status);
      //printf("%i got!\n", myrank);
    }
    if (myrank != 0) {
      //printf("%i waiting to send...\n", myrank);
      MPI_Send(phi_sub+m, m, MPI_FLOAT, myrank-1, iter, MPI_COMM_WORLD);
      //printf("%i sent!\n", myrank);
    }

		float num1 = 0;
		float num2 = 0;
		int   den1 = 0;
		int   den2 = 0;

		for(i=0; i<m_per_proc; i++) {
			for(j=0; j < n; j++) {
				if(phi_sub(i,j) < 0) {
					num1 += 256*u_sub(i,j);
					den1 +=  1;
				}
				else if(phi_sub(i,j) > 0) {
					num2  += 256*u_sub(i,j);
					den2  += 1;
				}
			}
		}

		float xnum1 = 0;
		float xnum2 = 0;
		int   xden1 = 0;
		int   xden2 = 0;

    MPI_Allreduce((void*)&num1, (void*)&xnum1, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce((void*)&num2, (void*)&xnum2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce((void*)&den1, (void*)&xden1, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce((void*)&den2, (void*)&xden2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		c1 = xnum1/xden1;
		c2 = xnum2/xden2;

    k = 0;
    for(i=0;i<m_per_proc;i++) {
      for(j=0; j < n; j++) {
        if ((i == 0 && myrank == 0) ||
            (i == m_per_proc-1 && myrank == npes-1) ||
            j == 0 ||
            j == n-1) {
          k++;
          continue;
        }

        float Dx_p = phi_sub(i+1,j) - phi_sub(i,j);
        float Dx_m = phi_sub(i,j) - phi_sub(i-1,j);
        float Dy_p = phi_sub(i,j+1) - phi_sub(i,j);
        float Dy_m = phi_sub(i,j) - phi_sub(i,j-1);

        float Dx_0 = (phi_sub(i+1,j) - phi_sub(i-1,j))/2;
        float Dy_0 = (phi_sub(i,j+1) - phi_sub(i,j-1))/2;

        float Dxx = Dx_p - Dx_m ;
        float Dyy = Dy_p - Dy_m ;

        float Dxy = (phi_sub(i+1,j+1) - phi_sub(i+1,j-1) - phi_sub(i-1,j+1) + phi_sub(i-1,j-1)) / 4;

        float Grad      = sqrtf(Dx_0*Dx_0 + Dy_0*Dy_0);
        float K         = (Dx_0*Dx_0*Dyy - 2*Dx_0*Dy_0*Dxy + Dy_0*Dy_0*Dxx) / (CUB(Grad) + epsilon);

        curv_sub(i,j) = Grad*(mu*K + SQR(256*u_sub(i,j)-c1) - SQR(256*u_sub(i,j)-c2));

        if (i == 1 && myrank == 0)
          curv_sub(i-1,j) = curv_sub(i,j);
        else if (i == m_per_proc-2 && myrank == npes-1)
          curv_sub(i+1,j) = curv_sub(i,j);
        else if (j == 1)
          curv_sub(i,j-1) = curv_sub(i,j);
        else if (j == n-2)
          curv_sub(i,j+1) = curv_sub(i,j);

        k++;
      }
    }

    for(i=0; i<m_per_proc; i++) {
      for (j=0; j<n; j++) {
        phi_sub(i, j) += curv_sub(i,j) * dt;
      }
    }


    MPI_Barrier(MPI_COMM_WORLD);

    // Gather up everyone else's phi
    /*
    MPI_Allgather(phi + my_m_start*n,
                  m_per_proc*n,
                  MPI_FLOAT,
                  phi,
                  m_per_proc*n,
                  MPI_FLOAT,
                  MPI_COMM_WORLD);
    */

	}


  float* phi = NULL;
  if (myrank == 0) {
    phi = (float*)calloc(m*n,sizeof(float));
  }

  MPI_Gather(phi_sub+n,
             m_per_proc*n,
             MPI_FLOAT,
             phi,
             m_per_proc*n,
             MPI_FLOAT,
             0,
             MPI_COMM_WORLD);

  free(curv_sub);
  free(phi_sub);
  free(u_sub);

  MPI_Finalize();
  if (myrank == 0) {
    for(i=1; i<m; i++) {
      for (j=1; j<n; j++) {
        if (phi(i, j)*phi(i-1, j)<0 || phi(i, j)*phi(i, j-1)<0) 
          contour[i*n+j] = 0.99;
        else 
          contour[i*n+j] = 0;
      }
    }
  }

	free(phi);
	//free(curv);
	return 0;
}
