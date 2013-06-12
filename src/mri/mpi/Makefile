BIN_DIR      = ../../../bin

MRI_MPI_BIN  = $(BIN_DIR)/mri_mpi

MRI_MPI_SRCS = mri_mpi_main.c \
               mri_mpi.c  \
					util_mpi.c

all:
	mpicc -O3 -std=c99 -o $(MRI_MPI_BIN) $(MRI_MPI_SRCS) -lm

test:
	clear
	mpicc -O3 -std=c99 -o $(MRI_MPI_BIN) $(MRI_MPI_SRCS) -lm
	time mpirun -n 4 $(MRI_MPI_BIN) ../../data/coronal.dat ../../parallel.bmp
	eog ../../parallel.bmp
