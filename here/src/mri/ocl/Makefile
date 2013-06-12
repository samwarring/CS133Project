BIN_DIR      = ../../../bin

MRI_OCL_BIN  = $(BIN_DIR)/mri_ocl

MRI_OCL_SRCS = mri_main.c \
               mri_ocl.c  \
					util.c

all:
	ioc64 -cmd=build -input=kernels.cl
	gcc -O3 -std=c99 -o $(MRI_OCL_BIN) $(MRI_OCL_SRCS) -lm -lOpenCL
