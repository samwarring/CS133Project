BIN_DIR      = ../../../bin

MRI_OMP_SRCS = mri_main.c \
               mri_omp.c  \
					util.c
					
MRI_OMP_BIN  = $(BIN_DIR)/mri_omp 

all:
	gcc -O3 -std=c99 -o $(MRI_OMP_BIN) $(MRI_OMP_SRCS) -lm -fopenmp
