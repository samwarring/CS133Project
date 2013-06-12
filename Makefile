TARBALL_FILES = Makefile readme.txt src/
TARBALL_NAME = 18.tgz

all:
	make -C src/mri/omp/
	make -C src/mri/mpi/
	make -C src/mri/ocl/

clean:
	rm bin/*

tgz:
	tar -cvzf $(TARBALL_NAME) . --exclude .git
