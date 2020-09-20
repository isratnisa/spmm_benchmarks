
CXX=g++
# CXX=icc
NVCC=nvcc

CXXFLAGS=-O3 -std=c++11 -g -fopenmp $(BOOSTFLAG)

NVCCFLAGS += -O3 -w -gencode arch=compute_70,code=sm_70 -lcublas -lcusparse -rdc=true --use_fast_math -Xptxas "-v -dlcm=ca" -Xcompiler -fopenmp --std=c++11 -m64 -lineinfo 
NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart
# nvcc -ccbin=/cm/shared/apps/intel/compilers_and_libraries_2016.3.210/linux/bin/intel64/icc -std=c++11 -o t912 t912.cu

all: spmm spmm_2D spmm_1p5 spmm_1D

spmm: cuSPARSE_newSpmm.cu
	${NVCC} ${NVCCFLAGS} -o spmm cuSPARSE_newSpmm.cu $(NVCCLINKFLAGS)  

spmm_2D: spmm_2D_blocked.cu
	${NVCC} ${NVCCFLAGS} -o spmm_2D spmm_2D_blocked.cu $(NVCCLINKFLAGS)  

spmm_1D: spmm_1D_blocked.cu
	${NVCC} ${NVCCFLAGS} -o spmm_1D spmm_1D_blocked.cu $(NVCCLINKFLAGS)  

spmm_1p5: spmm_1p5_blocked.cu
	${NVCC} ${NVCCFLAGS} -o spmm_1p5 spmm_1p5_blocked.cu $(NVCCLINKFLAGS)  

clean:
	rm -rf spmm spmm_2D spmm_1p5 spmm_1D *.o f

# nvcc -O3 -gencode arch=compute_70,code=sm_70 -lcublas -lcusparse cuSPARSE_newspmm_blocked.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o cuSPARSE_newsp_bl
# srun -n 1 ./cuSPARSE_newsp_bl $mat 32 4






