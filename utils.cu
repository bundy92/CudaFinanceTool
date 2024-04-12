#ifndef UTILS_CU
#define UTILS_CU

#include "utils.cuh"

// Utility function for error checking
#define CUDA_CHECK_ERROR(err) \
    do { \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Utility function for generating random numbers using cuRAND
__global__ void setup_rng(curandState* state, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void generate_random_numbers(curandState* state, float* random_numbers, int num_elements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_elements) {
        random_numbers[tid] = curand_uniform(&state[tid]);
    }
}

// Other utility functions for input/output, error handling, etc.

// Function to calculate the mean of an array
__host__ float mean_of_array(float* array, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum / size;
}

#endif //UTILS_CU
