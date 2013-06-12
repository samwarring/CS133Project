#define CENTER (m+n*M) 
#define RIGHT (m+n*M+M) 
#define LEFT (m+n*M-M) 
#define DOWN (m+n*M+1) 
#define UP  (m+n*M-1)  
#define SQR(x)  ((x)*(x))
#define DT 5.0 
 
__kernel void g_kernel(__global float* g, __global float* u, __global int* N_ptr, __global int* M_ptr) 
{
 
    int n = get_global_id(0);                                            
    int m = get_global_id(1);                                    
    int N = *N_ptr;
    int M = *M_ptr;
 
    /* if on the edges, don't do anything */
    
    if (n == 0 || n == N-1 || m == 0 || m == M-1)
        return;

    /* Update g */
    g[CENTER] = 1.0 / sqrt ( DBL_EPSILON
       + SQR(u[CENTER] - u[RIGHT])
       + SQR(u[CENTER] - u[LEFT])
       + SQR(u[CENTER] - u[DOWN])
       + SQR(u[CENTER] - u[UP]) );
}
