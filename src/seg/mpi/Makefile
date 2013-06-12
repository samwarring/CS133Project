BIN_DIR      = ../../../bin

MPI_BIN  = $(BIN_DIR)/seg_mpi

MPI_SRCS = seg_main.c \
               seg.c  \
					util.c

all:
	mpicc -O3 -std=c99 -o $(MPI_BIN) $(MPI_SRCS) -lm

test:
	clear
	mpicc -O3 -std=c99 -o $(MPI_BIN) $(MPI_SRCS) -lm
	time mpirun -n 4 $(MPI_BIN) ../../data/coronal.dat ../../parallel.bmp
	eog ../../parallel.bmp
