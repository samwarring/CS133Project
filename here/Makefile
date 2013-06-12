TARBALL_FILES = Makefile src/
TARBALL_NAME = 18.tgz

all: omp mpi ocl

omp:
	make -C src/mri/omp/

mpi:
	make -C src/mri/mpi/

ocl:
	make -C src/mri/ocl/

clean:
	rm bin/*

tgz:
	tar -cvzf $(TARBALL_NAME) . --exclude .git
