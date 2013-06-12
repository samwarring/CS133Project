TARBALL_FILES = Makefile readme.txt src/ bin/
TARBALL_NAME = 18.tgz

all:
	make -C src/mri/omp/
	make -C src/mri/mpi/
	make -C src/mri/ocl/
	make -C src/seg/omp/
	make -C src/seg/mpi/
	make -C src/seg/ocl/

clean:
	rm bin/*

tgz:
	tar -cvzf $(TARBALL_NAME) . --exclude .git
