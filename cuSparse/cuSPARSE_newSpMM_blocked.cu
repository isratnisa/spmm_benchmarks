#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#define FTYPE float

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

int *csr_v, *csr_colIdx;
FTYPE *csr_vals;

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
    sc = (sc + rootp -1)/rootp;
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
    //correctness check of blocks
    // for (int b = 0; b < nBlock; ++b) {tot_nnz += nnzBlock[b];
    // printf("nnzBlock %d %d\n", nnzBlock[b], tot_nnz);}


    // loop through blocks of each benchmark
    for (int br = 0; br < rootp; ++br) {//loop over row blocks of C 
        for (int bc = 0; bc < rootp; ++bc) {//loop over col blocks C          
            for (int bk = 0; bk < rootp; ++bk) {// loop over blocks of A & B   
                int b = br * rootp + bk;
                int bb = bk * rootp + bc; //doesnt matter here..all sublcoks are same 
                // printf("Processing block %d %d %d - nnz:  %d\n", br, bc, bk, nnzBlock[b] );
                if(!nnzBlock[b]) continue;  

                nnz = nnzBlock[b];
                nr = nRowsBlock;
                nc = nColsBlock;

                int nrows_ = nr;
                int nvals_ = nnz;
                int *h_csrRowPtr_ = (int *)malloc(sizeof(int)*(nr+1));
                // csr_v = (int *)malloc(sizeof(int)*(nr+1));
                csr_colIdx = (int *)malloc(sizeof(int)*nnz);
                csr_vals = (FTYPE *)malloc(sizeof(FTYPE)*nnz);

                // Convert to CSR/CSC
                int temp, row, col, dest, cumsum=0;

                // Set all rowPtr to 0
                for( int i=0; i<=nrows_; i++ )
                  h_csrRowPtr_[i] = 0;
                // Go through all elements to see how many fall in each row
                for( int i=0; i<nvals_; i++ ) {
                  row = temp_v[i].row % nRowsBlock;
                  // if( row>=nrows_ ) return GrB_INDEX_OUT_OF_BOUNDS;
                  h_csrRowPtr_[ row ]++;
                }
                // Cumulative sum to obtain rowPtr
                for( int i=0; i<nrows_; i++ ) {
                  temp = h_csrRowPtr_[i];
                  h_csrRowPtr_[i] = cumsum;
                  cumsum += temp;
                }
                h_csrRowPtr_[nrows_] = nvals_;

                // Store colInd and val
                for( int i=0; i<nvals_; i++ ) {
                  row = temp_v[i].row % nRowsBlock;
                  dest= h_csrRowPtr_[row];
                  col = temp_v[i].col % nColsBlock;
                  // if( col>=ncols_ ) return GrB_int_OUT_OF_BOUNDS;
                  csr_colIdx[dest] = col;
                  csr_vals[dest]    = temp_v[i].val;
                  h_csrRowPtr_[row]++;
                }
                cumsum = 0;
                
                // Undo damage done to rowPtr
                for( int i=0; i<=nrows_; i++ ) {
                  temp = h_csrRowPtr_[i];
                  h_csrRowPtr_[i] = cumsum;
                  cumsum = temp;
                }
            	
                int *ccsr_v, *ccsr_e; FTYPE *ccsr_ev;

            	cudaMalloc((void **) &ccsr_v, sizeof(int)*(nr+1));
            	cudaMalloc((void **) &ccsr_e, sizeof(int)*nnz);
            	cudaMalloc((void **) &ccsr_ev, sizeof(FTYPE)*nnz);
            	cudaMemcpy(ccsr_v, h_csrRowPtr_, sizeof(int)*(nr+1), cudaMemcpyHostToDevice);
            	cudaMemcpy(ccsr_e, csr_colIdx, sizeof(int)*(nnz), cudaMemcpyHostToDevice);
            	cudaMemcpy(ccsr_ev, csr_vals, sizeof(FTYPE)*(nnz), cudaMemcpyHostToDevice);
            	

                /* initialize cusparse library */
                status= cusparseCreate(&handle);
                if (status != CUSPARSE_STATUS_SUCCESS) {
                    return EXIT_FAILURE;
                }
                /* create and setup matrix descriptor */ 
                status= cusparseCreateMatDescr(&descra); 
                if (status != CUSPARSE_STATUS_SUCCESS) {
                    return EXIT_FAILURE;
                }       
                cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
                cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);  

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

                cusparseSpMatDescr_t a_cusparse;
                status = cusparseCreateCsr(&a_cusparse, nr, nc, nnz,
                                ccsr_v, ccsr_e, ccsr_ev, 
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
           
                cusparseDnMatDescr_t b_cusparse;
                status = cusparseCreateDnMat(&b_cusparse, nc, sc, sc,
                                  cy_in, CUDA_R_32F, CUSPARSE_ORDER_ROW);

                cusparseDnMatDescr_t c_cusparse;
                status = cusparseCreateDnMat(&c_cusparse, nr, sc, sc,
                                      cy_out, CUDA_R_32F, CUSPARSE_ORDER_ROW);
                
                size_t bufferSize = 0;
          
                status = cusparseSpMM_bufferSize(handle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      a_cusparse,
                                      b_cusparse,
                                      &beta,
                                      c_cusparse,
                                      CUDA_R_32F,
                                      CUSPARSE_SPMM_CSR_ALG2,
                                      &bufferSize);

                if (status != CUSPARSE_STATUS_SUCCESS) return EXIT_FAILURE;
                
                char* externalBuffer = NULL;
                cudaMalloc(&externalBuffer, bufferSize);
                
                cudaDeviceSynchronize();
                cudaEventRecord(event1,0);
            #define ITER (1)
                for(int ik=0;ik<ITER;ik++) {
                    status = cusparseSpMM(handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha,
                               a_cusparse,
                               b_cusparse,
                               &beta,
                               c_cusparse,
                               CUDA_R_32F,
                               CUSPARSE_SPMM_CSR_ALG2,
                               externalBuffer);
                }

                cudaEventRecord(event2,0);
                cudaEventSynchronize(event1);
                cudaEventSynchronize(event2);
                cudaEventElapsedTime(&tot_ms, event1, event2);
                cudaDeviceSynchronize();

                if (status != CUSPARSE_STATUS_SUCCESS) return EXIT_FAILURE;
        	    cudaMemcpy(y_out, cy_out, sizeof(FTYPE)*(nr)*sc, cudaMemcpyDeviceToHost);

            	cudaFree(cy_out); cudaFree(cy_in); free(y_out); free(y_in);
                cudaFree(externalBuffer);
                // free(loc);
            	fprintf(stdout, "K=%d : nBlocks: %d, nnz: %d, tot_ms: %f ms, GFLOPS: %f \n", sc, nBlock, nnz, tot_ms, (double)ITER*(double)nnz*2*sc/tot_ms/1000000);
            	fprintf(fpo, "%f,", (double)ITER*(double)nnz*2*sc/tot_ms/1000000);
                
                cudaFree(ccsr_v), cudaFree(ccsr_e); cudaFree(ccsr_ev);
                free(csr_v), free(csr_colIdx); free(csr_vals), free(h_csrRowPtr_);
            }
        }
    }
	fclose(fpo);     
}


