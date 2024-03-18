#ifndef UTILS_CU
#define UTILS_CU

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118
#endif

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

__global__ void generate_random_numbers(curandState* state, double* random_numbers, int num_elements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_elements) {
        random_numbers[tid] = curand_uniform(&state[tid]);
    }
}

// Other utility functions for input/output, error handling, etc.

// Function to calculate the mean of an array
__host__ double mean_of_array(double* array, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum / size;
}

#endif //UTILS_CU