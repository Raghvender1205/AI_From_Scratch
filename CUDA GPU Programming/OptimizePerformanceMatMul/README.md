## Optimize CUDA MatMul Kernel Performance 
In this, we implement different kernel which would iteratively improve the performance of the MatMul Kernel

### 1. Naive Implementation
For this, we use `grid`, `block` and `thread` hierarchy to assign each thread a unique entry in the result matrix `C`. That thread would compute the `dot` product. As each location of `C` is written only by one thread, there is no `synchronization`.

## Refernces
- https://siboehm.com/articles/22/CUDA-MMM
- https://github.com/siboehm/SGEMM_CUDA/tree/master