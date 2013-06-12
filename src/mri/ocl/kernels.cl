// Complex number multiplication.

inline float2 cmul(float2 a, float2 b) {

	float2 r;
	r.x = (a.x * b.x) - (a.y * b.y);
	r.y = (a.x * b.y) + (a.y * b.x);
	return r;
}

// Complex number division.

inline float2 cdiv(float2 a, float2 b) {
	
	float2 r;
	float scale = b.x*b.x + b.y*b.y;
	b.y = -b.y;
	r = cmul(a,b);
	r.x = r.x / scale;
	r.y = r.y / scale;
	return r;
}

// The dft and idft functions are each split into two halves: the part storing
// into tmp, and the part storing into dst. The functions for the first part are
// dft1 and idft1. The functions for the second part ar dft2, and idft2.

__kernel void dft1(__global float2* src,
						 __global float2* tmp,
						 __global float2* w1,
						 __global int* dims) {

	int l = get_global_id(0); // col [0,M)
	int n = get_global_id(1); // row [0,N)
	int M = dims[0];
	
	float2 l_dst;
	l_dst.x = l_dst.y = 0.0;
	for (int m=0; m<M; m++) {
		l_dst += cmul(src[n*M+m], w1[l*m]);
	}	
	tmp[n*M+l] = l_dst;
}

__kernel void dft2(__global float2* tmp,
						 __global float2* dst,
						 __global float2* w2,
						 __global int* dims) {
	
	int k = get_global_id(0); // row [0,N)
	int l = get_global_id(1); // col [0,M)
	int M = dims[0];
	int N = dims[1];

	float2 l_dst;
	l_dst.x = l_dst.y = 0.0;
	for (int n=0; n<N; n++) {
		l_dst += cmul(tmp[n*M+l], w2[n*k]);
	}
	dst[k*M+l] = l_dst;
}

__kernel void idft1(__global float2* src,
						  __global float2* tmp,
						  __global float2* w1,
						  __global int* dims) {
	
	int l = get_global_id(0); // col [0,M)
	int n = get_global_id(1); // row [0,N)
	int M = dims[0];
	
	float2 l_dst;
	l_dst.x = l_dst.y = 0.0;
	for (int m=0; m<M; m++) {
		l_dst += cdiv(src[n*M+m], w1[l*m]);
	}	
	tmp[n*M+l] = l_dst/M;
}

__kernel void idft2(__global float2* tmp,
						  __global float2* dst,
						  __global float2* w2,
						  __global int* dims) {

	int k = get_global_id(0); // row [0,N)
	int l = get_global_id(1); // col [0,M)
	int M = dims[0];
	int N = dims[1];

	float2 l_dst;
	l_dst.x = l_dst.y = 0.0;
	for (int n=0; n<N; n++) {
		l_dst += cdiv(tmp[n*M+l], w2[n*k]);
	}
	dst[k*M+l] = l_dst/N;
}
