src/
 mri/             all mri code
	omp/           mri_omp sources
 	 mri.omp.txt	mri_omp project write-up
	 ...
	mpi/           mri_mpi sources
	 mri.mpi.txt	mri_mpi project write-up
  	 ...
	ocl/           mri_ocl sources
	 mri.ocl.txt	mri_ocl project write-up
	 ...
 denoise/         all denoise code
		...
 seg/             all segmentation code
		...

To make all binaries, invoke 'make.'
To make all mri binaries, invoke 'make' in the src/mri/ directory.

To clear the bin/ directory, invoke 'make clean'
To create the submission tarball, invoke 'make tgz'
