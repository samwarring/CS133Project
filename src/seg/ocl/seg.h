#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#define SQR(x) (x)*(x)
#define CUB(x) (x)*(x)*(x)

#define phi(i,j)	phi[(i)*m+(j)]
#define u(i,j)		u[(i)*m+(j)]
#define curv(i,j)	curv[(i)*m+(j)]

#define epsilon 5e-5f

int seg(float* phi, const float* u, int m, int n, int MaxIter);

#endif
