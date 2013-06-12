BIN_DIR      = ../../../bin

OCL_BIN  = $(BIN_DIR)/seg_ocl

OCL_SRCS = seg_main.c \
               seg.c  \
					util.c

all:
	ioc64 -cmd=build -input=kernels.cl
	cp kernels.cl $(BIN_DIR)/
	gcc -O3 -std=c99 -o $(OCL_BIN) $(OCL_SRCS) -lm -lOpenCL
