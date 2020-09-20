
CXX=g++
# CXX=icc
NVCC=nvcc

CXXFLAGS=-O3 -std=c++11 -g -fopenmp $(BOOSTFLAG)

NVCCFLAGS += -O3 -w -gencode arch=compute_70,code=sm_70 -lcublas -lcusparse -rdc=true --use_fast_math -Xptxas "-v -dlcm=ca" -Xcompiler -fopenmp --std=c++11 -m64 -lineinfo 
NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart
SPUTNIKFLAGS=-I/global/homes/i/inisa/spmm/sputnik/ -I/global/homes/i/inisa/spmm/sputnik/sputnik/spmm -L/global/homes/i/inisa/spmm/sputnik/build/sputnik/ -lsputnik
# nvcc -ccbin=/cm/shared/apps/intel/compilers_and_libraries_2016.3.210/linux/bin/intel64/icc -std=c++11 -o t912 t912.cu

all: spmm sptnk_spmm_2D sptnk_spmm_1p5 sptnk_spmm_1D

spmm: spmm.cu
	${NVCC} ${NVCCFLAGS} -o spmm spmm.cu $(SPUTNIKFLAGS) $(NVCCLINKFLAGS)  

sptnk_spmm_2D: sptnk_spmm_2Dblocked.cu
	${NVCC} ${NVCCFLAGS} -o sptnk_spmm_2D sptnk_spmm_2Dblocked.cu $(SPUTNIKFLAGS) $(NVCCLINKFLAGS)  

sptnk_spmm_1D: sptnk_spmm_1Dblocked.cu
	${NVCC} ${NVCCFLAGS} -o sptnk_spmm_1D sptnk_spmm_1Dblocked.cu $(SPUTNIKFLAGS) $(NVCCLINKFLAGS)  

sptnk_spmm_1p5: sptnk_spmm_1p5blocked.cu
	${NVCC} ${NVCCFLAGS} -o sptnk_spmm_1p5 sptnk_spmm_1p5blocked.cu $(SPUTNIKFLAGS) $(NVCCLINKFLAGS)  

clean:
	rm -rf spmm sptnk_spmm_2D sptnk_spmm_1p5 sptnk_spmm_1D *.o f

#nvcc -O3 -gencode arch=compute_70,code=sm_70 -I/global/homes/i/inisa/spmm/sputnik/ -I/global/homes/i/inisa/spmm/sputnik/sputnik/spmm -L/global/homes/i/inisa/spmm/sputnik/build/sputnik/ -lsputnik  -lcusparse -lcudart  spmm.cu -o out
#srun -n 1 ./spmm_2D ~/can_62.mtx 32 4

