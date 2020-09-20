#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cuda_spmm.h"
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

int *csr_v, *csr_e, *row_indices;
FTYPE *csr_ev;

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
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) 
            return 1;
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) 
            return -1; //The element pointed to by p1 goes before the element pointed to by p2
        return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}


int main(int argc, char **argv)
{
    if(argc < 3){
        printf("Wrong arg list. Try with: ./exec matrix rhs.\n");
        printf("E.g., ./spmm tmp.mtx 32 \n"); 
        exit(0);
    }

	FILE *fp;
	FILE *fpo = fopen("SpMM_GPU_SP_spmm.out", "a");
	srand(time(NULL));

  // cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4,cudaStat5,cudaStat6;
  cusparseStatus_t status;
  // cusparseHandle_t handle=0;
  // cusparseMatDescr_t descra=0;

  int n, nr, nc, nnz, nflag, nnz_vector, i, j;

	struct v_struct *temp_v;

	char buf[300];
	int sflag;
	int dummy, pre_count=0, tmp_ne;

	fp = fopen(argv[1], "r");
	//sc = atoi(argv[2]);
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
    int sc = atoi(argv[2]); 
    fp = fopen(argv[1], "r");
    for(i=0;i<pre_count;i++)
            fgets(buf, 300, fp);

    fscanf(fp, "%d %d %d", &nr, &nc, &nnz);
    nnz *= (sflag+1);

    temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(nnz+1));

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

        // Convert to CSR/CSC
    int temp, row, col, dest, cumsum=0;
    int nrows_ = nr;
    int nvals_ = nnz;
    int *h_csrRowPtr_ = (int *)malloc(sizeof(int)*(nr+1));

    csr_v = (int *)malloc(sizeof(int)*(nr+1));
    row_indices = (int *)malloc(sizeof(int)*(nr));
    csr_e = (int *)malloc(sizeof(int)*nnz);
    csr_ev = (FTYPE *)malloc(sizeof(FTYPE)*nnz);
    
    // Set all rowPtr to 0
    for( int i=0; i<=nrows_; i++ )
      h_csrRowPtr_[i] = 0;
    // Go through all elements to see how many fall in each row
    for( int i=0; i<nvals_; i++ ) {
      row = temp_v[i].row;
      // if( row>=nrows_ ) return GrB_int_OUT_OF_BOUNDS;
      h_csrRowPtr_[ row ]++;
    }
    // Cumulative sum to obtain rowPtr
    for( int i=0; i<nrows_; i++ ) {
      temp = h_csrRowPtr_[i];
      h_csrRowPtr_[i] = cumsum;
      cumsum += temp;
    }
    h_csrRowPtr_[nrows_] = nnz;

    // Store colInd and val
    for( int i=0; i<nvals_; i++ ) {
      row = temp_v[i].row;
      dest= h_csrRowPtr_[row];
      col = temp_v[i].col;
      // if( col>=ncols_ ) return GrB_int_OUT_OF_BOUNDS;
      csr_e[dest] = col;
      csr_ev[dest]    = temp_v[i].val;
      h_csrRowPtr_[row]++;
    }
    cumsum = 0;
    
    // Undo damage done to rowPtr
    for( int i=0; i<=nrows_; i++ ) {
      temp = h_csrRowPtr_[i];
      h_csrRowPtr_[i] = cumsum;
      cumsum = temp;
    }

	csr_v = h_csrRowPtr_;
  for (int i = 0; i < nr; ++i)
    row_indices[i] = i; //from sputnik: should be random for ld dist. 
  

  int *row_offset, *col_indices, *d_row_indices; 
  FTYPE *values;

	cudaMalloc((void **) &row_offset, sizeof(int)*(nr+1));
  cudaMalloc((void **) &d_row_indices, sizeof(int)*(nr));
	cudaMalloc((void **) &col_indices, sizeof(int)*nnz);
	cudaMalloc((void **) &values, sizeof(FTYPE)*nnz);
	cudaMemcpy(row_offset, csr_v, sizeof(int)*(nr+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_indices, row_indices, sizeof(int)*nr, cudaMemcpyHostToDevice);
	cudaMemcpy(col_indices, csr_e, sizeof(int)*(nnz), cudaMemcpyHostToDevice);
	cudaMemcpy(values, csr_ev, sizeof(FTYPE)*(nnz), cudaMemcpyHostToDevice);
	
	cudaError_t err = cudaSuccess;
	FTYPE *y_in, *cy_in, *y_out, *cy_out; 
	y_in = (FTYPE *)malloc(sizeof(FTYPE)*nc*sc);
	y_out = (FTYPE *)malloc(sizeof(FTYPE)*(nr)*sc);
	
    for(i=0;i<nc*sc;i++)
		y_in[i] = ((FTYPE)(1));;//rand()%1048576))/1048576;

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
    
    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);
#define ITER (1)
    for(int ik=0;ik<ITER;ik++) {
    
      // status = CudaSpmm(int m, int k, int n, int nnz,
      //                const int* __restrict__ row_indices,
      //                const float* __restrict__ values,
      //                const int* __restrict__ row_offsets,
      //                const int* __restrict__ column_indices,
      //                const float* __restrict__ cy_in,
      //                float* __restrict__ cy_out, nullptr);

            CudaSpmm(nr, nc, sc, nnz,
                     d_row_indices,
                     values,
                     row_offset,
                     col_indices,
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

    /*Correctness check*/

    // #define VALID
    #ifdef VALID
	FTYPE *gold_out = (FTYPE *)malloc(sizeof(FTYPE)*(nr)*sc);
    FTYPE *y_out_col = (FTYPE *)malloc(sizeof(FTYPE)*(nr)*sc);
	memset(gold_out, 0, sizeof(FTYPE)*nr*sc);
	for(i=0;i<nnz;i++) {
		for(j=0;j<sc;j++) {
			//gold_out[temp_v[i].row + (nr+1)*j] += y_in[temp_v[i].col + nc*j] * temp_v[i].val;
			gold_out[temp_v[i].row + nr*j] += y_in[sc*temp_v[i].col + j] * temp_v[i].val;
        }
	}
    //covert to col major
    for(i=0;i<nr;i++) {
        for(j=0;j<sc;j++) {
            //gold_out[temp_v[i].row + (nr+1)*j] += y_in[temp_v[i].col + nc*j] * temp_v[i].val;
            y_out_col[j * sc + i] += y_out[i * nr + j];
        }
    }

	long num_diff=0;
    // fprintf(stdout, "beg diff : %f\t%f\n", y_out[0], gold_out[0]);

	for(int r=0; r < nr; r++) {
        for(int c=0; c < sc ; c++) 
        //  if(i < 200 )
        fprintf(stdout, "i: %d GPU: %f\t CPU %f\n", i, y_out_col[r * nr + c], gold_out[r * nr +c]);
	 //    if(abs(y_out[i] -gold_out[i])/max(abs(y_out[i]), abs(gold_out[i])) > 0.01) {
		// 	num_diff++;
		// 	// if(num_diff < 3) {
		// 	// 	fprintf(stdout, "diff : %f\t%f\n", y_out[i], gold_out[i]);
		// 	// }
		// }
	}

	fprintf(stdout, "(%ld),",num_diff);	
	// fprintf(stdout, "time(ms) : %f\tGFlops : %f\t%ld\n", (end-start)*1000, (double)nnz*2*sc/(end-start)/1000000000, num_diff);
#endif
	cudaFree(cy_out); cudaFree(cy_in); free(y_out); free(y_in);
 
	fprintf(stdout, "K=%d : tot_ms: %f ms, GFLOPS: %f\n", sc, tot_ms, (double)ITER*(double)nnz*2*sc/tot_ms/1000000);
	fprintf(fpo, "%f,", (double)ITER*(double)nnz*2*sc/tot_ms/1000000);

  cudaFree(row_offset), cudaFree(col_indices); cudaFree(values); cudaFree(d_row_indices);
	fclose(fpo);     
}


