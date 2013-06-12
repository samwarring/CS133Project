#define CENTER (m+n*M) 
#define RIGHT (m+n*M+M) 
#define LEFT (m+n*M-M) 
#define DOWN (m+n*M+1) 
#define UP  (m+n*M-1)  
#define SQR(x)  ((x)*(x))
#define DT 5.0 
 
__kernel void u_kernel( __global int* gamma_ptr, __global float* g, __global int* f, __global float* sigma2_ptr,
      __global float* u, __global int* N_ptr, __global int* M_ptr) 
{
 
    float gamma = *gamma_ptr;                                            
    int n = get_global_id(0);                                            
    int m = get_global_id(1);                                    
    int N = *N_ptr;
    int M = *M_ptr;
    float sigma2  = *sigma2_ptr;                                            
    float r = u[CENTER]*f[CENTER]/sigma2;                                
    r = ( r*(2.38944 + r*(0.950037 + r)) )                         
        / ( 4.65314 + r*(2.57541 + r*(1.48937 + r)) );              
 
    /* if on the edges, don't do anything */
    
    if (n == 0 || n == N-1 || m == 0 || m == M-1)
        return;

    /* Update u */                
    u[CENTER] = ( u[CENTER] + DT*(u[RIGHT]*g[RIGHT]              
        + u[LEFT]*g[LEFT] + u[DOWN]*g[DOWN] + u[UP]*g[UP]          
        + gamma*f[CENTER]*r) ) /                                   
        (1.0 + DT*(g[RIGHT] + g[LEFT] + g[DOWN] + g[UP] + gamma)); 
}
 

