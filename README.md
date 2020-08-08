# spmm_benchmarks

### Build
```
cd cuSparse
make
```

### Run
```
Run cuSparse spmm as follows:
./exec /path/to/mat.mtx right_hand_side
E.g., ./spmm tmp.mtx 32 

Run the blocked version of cuSparse spmm as follows:
./exec /path/to/mat.mtx right_hand_side #blocks
E.g., ./spmm_blocked tmp.mtx 32 1

On coriGPU, run as follows:
srun -n 1 ./spmm_blocked tmp.mtx 32 1

```
