src/
	mri/        all mri code
		omp/     mri_omp sources
		mpi/     mri_mpi sources
		ocl/     mri_ocl sources
	denoise/    all denoise code
		...
	seg/        all segmentation code
		...

To make all binaries, invoke 'make.'
To make all mri binaries, invoke 'make' in the src/mri/ directory.

To clear the bin/ directory, invoke 'make clean'
To create the submission tarball, invoke 'make tgz'
