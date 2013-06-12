BIN_DIR      = ../../../bin

OMP_SRCS = seg_main.c \
               seg.c  \
					util.c
					
OMP_BIN  = $(BIN_DIR)/seg_omp

all:
	gcc -O3 -std=c99 -o $(OMP_BIN) $(OMP_SRCS) -lm -fopenmp
