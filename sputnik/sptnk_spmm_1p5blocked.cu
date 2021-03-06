/* Blocking: Figure 2: https://dl.acm.org/doi/pdf/10.1145/3016078.2851152/*/

#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_spmm.h"
#include <cuda_runtime.h>
#include "cusparse.h"
#define FTYPE float

using namespace sputnik;

#define CLEANUP(s)                                   \
do {                                                 \
    printf ("%s\n", s);                              \
    if (yHostPtr)           free(yHostPtr);          \
    if (zHostPtr)           free(zHostPtr);          \
    if (xIndHostPtr)        free(xIndHostPtr);       \
    if (xValHostPtr)        free(xValHostPtr);       \
    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);\
    if (cooColIndexHostPtr) free(cooColIndexHostPtr);\
    if (cooValHostPtr)      free(cooValHostPtr);     \
    if (y)                  cudaFree(y);             \
    if (z)                  cudaFree(z);             \
    if (xInd)               cudaFree(xInd);          \
    if (xVal)               cudaFree(xVal);          \
    if (csrRowPtr)          cudaFree(csrRowPtr);     \
    if (cooRowIndex)        cudaFree(cooRowIndex);   \
    if (cooColIndex)        cudaFree(cooColIndex);   \
    if (cooVal)             cudaFree(cooVal);        \
    if (handle)             cusparseDestroy(handle); \
    fflush (stdout);                                 \
} while (0)

struct v_struct {
        int row, col;
        FTYPE val;
};

double rtclock(void)
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

int compare1(const void *a, const void *b)
{
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
        return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}


int main(int argc, char **argv)
{

    if(argc < 4){
        printf("Wrong arg list. Try with: ./exec matrix rhs nBlocks.\n"); 
        printf("E.g., ./spmm_blocked tmp.mtx 32 1 \n"); 
        exit(0);
    }

	FILE *fp;
	FILE *fpo = fopen("SpMM_GPU_SP_spmm.out", "a");
	srand(time(NULL));

    cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4,cudaStat5,cudaStat6;
    cusparseStatus_t status;
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descra=0;

    int n, nr, nc, nnz, nflag, nnz_vector, i, j;

	struct v_struct *temp_v;

	char buf[300];
	int sflag;
	int dummy, pre_count=0, tmp_ne;
    int sc = atoi(argv[2]);   
    int nBlock = atoi(argv[3]);
	fp = fopen(argv[1], "r");

	fgets(buf, 300, fp);

    if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
    else sflag = 0;
    if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
    else if(strstr(buf, "complex") != NULL) nflag = -1;
    else nflag = 1;

    while(1) {
            pre_count++;
            fgets(buf, 300, fp);
            if(strstr(buf, "%") == NULL) break;
    }
    fclose(fp);

    fp = fopen(argv[1], "r");
    for(i=0;i<pre_count;i++)
            fgets(buf, 300, fp);

    fscanf(fp, "%d %d %d", &nr, &nc, &nnz);
    nnz *= (sflag+1);

    temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(nnz+1));

    /*------------------------
      Read input matrix
    ------------------------*/
    for(i=0;i<nnz;i++) {
        fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
        temp_v[i].row--; temp_v[i].col--;

        if(temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc) {
                fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
                exit(0);
        }
        if(nflag == 0) temp_v[i].val = (FTYPE)(rand()%1048576)/1048576;
        else if(nflag == 1) {
                FTYPE ftemp;
                fscanf(fp, " %f ", &ftemp);
                temp_v[i].val = ftemp;
        } else { // complex
                FTYPE ftemp1, ftemp2;
                fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
                temp_v[i].val = ftemp1;
        }
        if(sflag == 1) {
                i++;
                temp_v[i].row = temp_v[i-1].col;
                temp_v[i].col = temp_v[i-1].row;
                temp_v[i].val = temp_v[i-1].val;
        }
    }
    qsort(temp_v, nnz, sizeof(struct v_struct), compare1);

    /*------------------------
      Estimate block nnz
    ------------------------*/  
    int p = nBlock;
    int rootp = sqrt(nBlock);

    int nRowsBlock = (nr + rootp - 1) / rootp;
    int nColsBlock = (nc + rootp - 1) / rootp; 
    // sc = (sc + rootp -1)/rootp; 
    nr = nRowsBlock;
    nc = nColsBlock;
    int *nnzBlock = (int*)malloc(nBlock * sizeof(int));    
    memset(nnzBlock, 0, nBlock * sizeof(int));
    int bi, bj, bid;

    for(i = 0; i < nnz; i++){
        bi = temp_v[i].row/nRowsBlock;
        bj = temp_v[i].col/nColsBlock;
        bid = bi * rootp + bj;
        nnzBlock[bid]++;  
    }
    int tot_nnz = 0;
    
    /*------------------------
      Created blocked COO/CSR
    ------------------------*/  
    int **b_rowPtr = (int **)malloc(nBlock * sizeof (int *) );
    int **b_rowInd = (int **)malloc(nBlock * sizeof (int *) );
    int **b_colInd = (int **)malloc(nBlock * sizeof (int *) );
    FTYPE **b_val = (FTYPE **)malloc(nBlock * sizeof (FTYPE *) );

    for (int b = 0; b < nBlock; b++) {
        b_rowPtr[b] = (int *) malloc ((nRowsBlock+1) * sizeof (int)) ;
        memset(&b_rowPtr[b][0], 0, (nRowsBlock+1) * sizeof(int));
        b_rowInd[b] = (int *) malloc (nRowsBlock * sizeof (int)) ;
        b_colInd[b] = (int *) malloc ((nnzBlock[b]) * sizeof (int)) ;
        b_val[b] = (FTYPE  *) malloc ((nnzBlock[b]) * sizeof (FTYPE   )) ;
    }
    memset( nnzBlock, 0, (nBlock) * sizeof(int) );

    for(i = 0; i < nnz; i++){
        int br = temp_v[i].row/nRowsBlock;
        int bc = temp_v[i].col/nColsBlock;
        int bId = br * rootp + bc;
        
        int local_rowInd = temp_v[i].row % nRowsBlock;
        b_rowPtr[bId][1+local_rowInd] = nnzBlock[bId]+1;
        b_rowInd[bId][nnzBlock[bId]] = temp_v[i].row % nRowsBlock;
        b_colInd[bId][nnzBlock[bId]] = temp_v[i].col % nColsBlock;
        b_val[bId][nnzBlock[bId]] = temp_v[i].val;
        nnzBlock[bId]++;
    }
    for (int b = 0; b < rootp; ++b){
        for(int r = 0; r < nRowsBlock; r++) {
            b_rowInd[b][i] = r;//
            if(b_rowPtr[b][r] == 0)
                b_rowPtr[b][r] = b_rowPtr[b][r-1];
        }   
    }

    /* 1.5D Blocked SpMM */
    
    float tot_time = 0;
    // loop through blocks of each benchmark
    for (int br = 0; br < rootp; ++br) {//loop over row blocks of A  
         
        for (int bk = 0; bk < rootp; ++bk) {// loop over col blocks of A   
          
            int b = br * rootp + bk;
          
            if(!nnzBlock[b]) continue;  

            nnz = nnzBlock[b];
            nr = nRowsBlock;
            nc = nColsBlock;
            
            int *d_rowPtr, *d_rowInd, *d_colInd; FTYPE *d_vals;

            cudaMalloc((void **) &d_rowPtr, sizeof(int)*(nr+1));
            cudaMalloc((void **) &d_rowInd, sizeof(int)*(nr));
            cudaMalloc((void **) &d_colInd, sizeof(int)*nnz);
            cudaMalloc((void **) &d_vals, sizeof(FTYPE)*nnz);
            cudaMemcpy(d_rowPtr, &(b_rowPtr[b][0]), sizeof(int)*(nr+1), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rowInd, &(b_rowInd[b][0]), sizeof(int)*(nr), cudaMemcpyHostToDevice);
            cudaMemcpy(d_colInd, &(b_colInd[b][0]), sizeof(int)*(nnz), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vals, &(b_val[b][0]), sizeof(FTYPE)*(nnz), cudaMemcpyHostToDevice);
       

        	cudaError_t err = cudaSuccess;
        	FTYPE *y_in, *cy_in, *y_out, *cy_out; 
        	y_in = (FTYPE *)malloc(sizeof(FTYPE)*nc*sc);
        	y_out = (FTYPE *)malloc(sizeof(FTYPE)*(nr)*sc);
        	
            for(int i=0;i<nc*sc;i++)
        		y_in[i] = ((FTYPE)1);//(rand()%1048576))/1048576;

        	err = cudaMalloc((void **) &cy_in, sizeof(FTYPE)*nc*sc);
                if(err != cudaSuccess)  {fprintf(stdout, "\n"); exit(0); }
        	err = cudaMalloc((void **) &cy_out, sizeof(FTYPE)*(nr)*sc);
                if(err != cudaSuccess)  {fprintf(stdout, "\n"); exit(0); }
        	cudaMemcpy(cy_in, y_in, sizeof(FTYPE)*nc*sc, cudaMemcpyHostToDevice);
            	cudaMemset((void *)cy_out, 0, sc*(nr)*sizeof(FTYPE));    

        	float tot_ms;
            cudaEvent_t event1, event2;
            cudaEventCreate(&event1);
            cudaEventCreate(&event2);

        	const FTYPE alpha=1.0f, beta=0.0f;

            /*new SpMM*/

            cudaDeviceSynchronize();
            cudaEventRecord(event1,0);
            #define ITER (10)
            for(int ik=0;ik<ITER;ik++) {
                CudaSpmm(nr, nc, sc, nnz,
                         d_rowInd,
                         d_vals,
                         d_rowPtr,
                         d_colInd,
                         cy_in,
                         cy_out, nullptr);
            }

            cudaEventRecord(event2,0);
            cudaEventSynchronize(event1);
            cudaEventSynchronize(event2);
            cudaEventElapsedTime(&tot_ms, event1, event2);
            cudaDeviceSynchronize();

            if (status != CUSPARSE_STATUS_SUCCESS) return EXIT_FAILURE;
    	    cudaMemcpy(y_out, cy_out, sizeof(FTYPE)*(nr)*sc, cudaMemcpyDeviceToHost);

        	cudaFree(cy_out); cudaFree(cy_in); free(y_out); free(y_in);
            cudaFree(d_rowPtr); cudaFree(d_rowInd); cudaFree(d_colInd); cudaFree(d_vals);
            // free(csr_v), free(csr_colIdx); free(csr_vals);
            tot_time += tot_ms;
            fprintf(stdout, "Block: %d, nnz: %d, tot_ms: %f s\n", b, nnzBlock[b], tot_ms/ITER);
   
        
        }
    }
    for (int b = 0; b < rootp; ++b){
        free(b_rowPtr[b]); free(b_colInd[b]); free(b_val[b]);
    }
    fprintf(stdout, "1.5D Blocking, K=%d : nBlocks: %d, nnz: %d, tot_ms: %f ms, GFLOPS: %f \n", sc, nBlock, nnz, tot_time, (double)ITER*(double)nnz*2*sc/tot_time/1000000);
	fclose(fpo);     
}


