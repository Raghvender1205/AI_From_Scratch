#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

constexpr size_t div_up(size_t a, size_t b) { return (a + b - 1) / b; }

template <typename T>
__global__ void transpose_read_coalesced(T* output_matrix,
                                         T const* input_matrix, size_t M,
                                         size_t N)
{
    size_t const j{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const i{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const from_idx{i * N + j};
    if ((i < M) && (j < N))
    {
        size_t const to_idx{j * M + i};
        output_matrix[to_idx] = input_matrix[from_idx];
    }
}

template <typename T>
__global__ void transpose_write_coalesced(T* output_matrix,
                                          T const* input_matrix, size_t M,
                                          size_t N)
{
    size_t const j{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const i{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const to_idx{i * M + j};
    if ((i < N) && (j < M))
    {
        size_t const from_idx{j * N + i};
        output_matrix[to_idx] = input_matrix[from_idx];
    }
}

template <typename T>
void launch_transpose_read_coalesced(T* output_matrix, T const* input_matrix,
                                     size_t M, size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(N, warp_size)),
                               static_cast<unsigned int>(div_up(M, warp_size))};
    transpose_read_coalesced<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output_matrix, input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
void launch_transpose_write_coalesced(T* output_matrix, T const* input_matrix,
                                      size_t M, size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(M, warp_size)),
                               static_cast<unsigned int>(div_up(N, warp_size))};
    transpose_write_coalesced<<<blocks_per_grid, threads_per_block, 0,
                                stream>>>(output_matrix, input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T, size_t BLOCK_SIZE = 32>
__global__ void transpose_read_write_coalesced(T* output_matrix,
                                               T const* input_matrix, size_t M,
                                               size_t N)
{
    // BLOCK_SIZE + 1 for avoiding the shared memory bank conflicts.
    // https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/
    // Try setting it to BLOCK_SIZE instead of BLOCK_SIZE + 1 to see the
    // performance drop.
    __shared__ T buffer[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Make sure blockDim.x == blockDim.y == BLOCK_SIZE

    size_t const matrix_j{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const matrix_i{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const matrix_from_idx{matrix_i * N + matrix_j};

    if ((matrix_i < M) && (matrix_j < N))
    {
        buffer[threadIdx.x][threadIdx.y] = input_matrix[matrix_from_idx];
    }

    // Make sure the buffer in a block is filled.
    __syncthreads();

    size_t const matrix_transposed_j{threadIdx.x + blockIdx.y * blockDim.y};
    size_t const matrix_transposed_i{threadIdx.y + blockIdx.x * blockDim.x};

    if ((matrix_transposed_i < N) && (matrix_transposed_j < M))
    {
        size_t const to_idx{matrix_transposed_i * M + matrix_transposed_j};
        output_matrix[to_idx] = buffer[threadIdx.y][threadIdx.x];
    }
}

template <typename T>
void launch_transpose_read_write_coalesced(T* output_matrix,
                                           T const* input_matrix, size_t M,
                                           size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(N, warp_size)),
                               static_cast<unsigned int>(div_up(M, warp_size))};
    transpose_read_write_coalesced<T, warp_size>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(output_matrix,
                                                            input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
bool is_equal(T const* data_1, T const* data_2, size_t size)
{
    for (size_t i{0}; i < size; ++i)
    {
        if (data_1[i] != data_2[i])
        {
            return false;
        }
    }
    return true;
}

template <typename T>
bool verify_transpose_implementation(
    std::function<void(T*, T const*, size_t, size_t, cudaStream_t)>
        transpose_function,
    size_t M, size_t N)
{
    // Fixed random seed for reproducibility
    std::mt19937 gen{0};
    cudaStream_t stream;
    size_t const matrix_size{M * N};
    std::vector<T> matrix(matrix_size, 0.0f);
    std::vector<T> matrix_transposed(matrix_size, 1.0f);
    std::vector<T> matrix_transposed_reference(matrix_size, 2.0f);
    std::uniform_real_distribution<T> uniform_dist(-256, 256);
    for (size_t i{0}; i < matrix_size; ++i)
    {
        matrix[i] = uniform_dist(gen);
    }
    // Create the reference transposed matrix using CPU.
    for (size_t i{0}; i < M; ++i)
    {
        for (size_t j{0}; j < N; ++j)
        {
            size_t const from_idx{i * N + j};
            size_t const to_idx{j * M + i};
            matrix_transposed_reference[to_idx] = matrix[from_idx];
        }
    }
    T* d_matrix;
    T* d_matrix_transposed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, matrix.data(),
                                matrix_size * sizeof(T),
                                cudaMemcpyHostToDevice));
    transpose_function(d_matrix_transposed, d_matrix, M, N, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(matrix_transposed.data(), d_matrix_transposed,
                                matrix_size * sizeof(T),
                                cudaMemcpyDeviceToHost));
    bool const correctness{is_equal(matrix_transposed.data(),
                                    matrix_transposed_reference.data(),
                                    matrix_size)};
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return correctness;
}

template <typename T>
void profile_transpose_implementation(
    std::function<void(T*, T const*, size_t, size_t, cudaStream_t)>
        transpose_function,
    size_t M, size_t N)
{
    constexpr int const num_repeats{100};
    constexpr int const num_warmups{10};
    cudaStream_t stream;
    size_t const matrix_size{M * N};
    T* d_matrix;
    T* d_matrix_transposed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> const transpose_function_wrapped{
        std::bind(transpose_function, d_matrix_transposed, d_matrix, M, N,
                  std::placeholders::_1)};
    float const transpose_function_latency{measure_performance(
        transpose_function_wrapped, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
              << "Latency: " << transpose_function_latency << " ms"
              << std::endl;
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main()
{
    // Unit tests.
    for (size_t m{1}; m <= 64; ++m)
    {
        for (size_t n{1}; n <= 64; ++n)
        {
            assert(verify_transpose_implementation<float>(
                &launch_transpose_write_coalesced<float>, m, n));
            assert(verify_transpose_implementation<float>(
                &launch_transpose_read_coalesced<float>, m, n));
            assert(verify_transpose_implementation<float>(
                &launch_transpose_read_write_coalesced<float>, m, n));
        }
    }

    // M: Number of rows.
    size_t const M{12800};
    // N: Number of columns.
    size_t const N{12800};
    std::cout << M << " x " << N << " Matrix" << std::endl;
    std::cout << "Transpose Write Coalesced" << std::endl;
    profile_transpose_implementation<float>(
        &launch_transpose_write_coalesced<float>, M, N);
    std::cout << "Transpose Read Coalesced" << std::endl;
    profile_transpose_implementation<float>(
        &launch_transpose_read_coalesced<float>, M, N);
    std::cout << "Transpose Read and Write Coalesced" << std::endl;
    profile_transpose_implementation<float>(
        &launch_transpose_read_write_coalesced<float>, M, N);
}