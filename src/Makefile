CC=mpicc

FFTWDIR	= ../fftw
OBJDIR	= ../obj
EXEDIR	= ../bin

MRI_SRC  = 	mri_main.c \
	     	mri.c
DEN_SRC  = 	denoise_main.c \
	     	denoise.c
SEG_SRC  = 	seg_main.c \
	   		segmentation.c 
COMM_SRC =	util.c

MRI_EXEC = mri
DEN_EXEC = denoise
SEG_EXEC = seg

INCLUDES	= -I$(FFTWDIR)/include
LIBRARIES	= -L$(FFTWDIR)/lib -lfftw3f -lm
CFLAGS		= -O3 -g -pg 

MRI_DEPS = $(MRI_SRC:%.c=%.o)
DEN_DEPS = $(DEN_SRC:%.c=%.o)
SEG_DEPS = $(SEG_SRC:%.c=%.o)
COMM_DEPS = $(COMM_SRC:%.c=%.o)

MRI_OBJ = $(addprefix $(OBJDIR)/, $(MRI_DEPS) $(COMM_DEPS)) 
DEN_OBJ = $(addprefix $(OBJDIR)/, $(DEN_DEPS) $(COMM_DEPS)) 
SEG_OBJ = $(addprefix $(OBJDIR)/, $(SEG_DEPS) $(COMM_DEPS)) 

COMPILE	= $(CFLAGS) -c $(INCLUDES)
LINK	= $(CFLAGS) $(LIBRARIES) -I$(OPENCL_ROOT)/include -L$(OPENCL_ROOT)/lib64 -lOpenCL

all: $(EXEDIR)/$(MRI_EXEC) $(EXEDIR)/$(DEN_EXEC) $(EXEDIR)/$(SEG_EXEC)

$(EXEDIR)/$(MRI_EXEC): $(MRI_OBJ)
	$(CC) -o $(EXEDIR)/$(MRI_EXEC) $(MRI_OBJ) $(LINK)

$(EXEDIR)/$(DEN_EXEC): $(DEN_OBJ)
	$(CC) -o $(EXEDIR)/$(DEN_EXEC) $(DEN_OBJ) $(LINK)

$(EXEDIR)/$(SEG_EXEC): $(SEG_OBJ) 
	$(CC) -o $(EXEDIR)/$(SEG_EXEC) $(SEG_OBJ) $(LINK)

$(OBJDIR)/%.o: %.c %.h
	$(CC) $(COMPILE) $< -o $@  

clean:
	rm -rf $(EXEDIR)/$(MRI_EXEC) $(EXEDIR)/$(DEN_EXEC) $(EXEDIR)/$(SEG_EXEC)
	rm -rf $(OBJDIR)/*.o
