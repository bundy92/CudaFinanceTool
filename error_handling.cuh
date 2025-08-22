#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"

// Error handling macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_CHECK_ASYNC(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while(0)

#define CURAND_CHECK(call) \
    do { \
        curandStatus_t err = call; \
        if (err != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "CURAND error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Memory allocation with error checking
#define CUDA_MALLOC(ptr, size) \
    do { \
        cudaError_t err = cudaMalloc((void**)&ptr, size); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA malloc failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_MALLOC_HOST(ptr, size) \
    do { \
        cudaError_t err = cudaMallocHost((void**)&ptr, size); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA mallocHost failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Memory copy with error checking
#define CUDA_MEMCPY(dst, src, size, kind) \
    do { \
        cudaError_t err = cudaMemcpy(dst, src, size, kind); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA memcpy failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel launch with error checking
#define CUDA_LAUNCH(kernel, grid, block, shared, stream, ...) \
    do { \
        kernel<<<grid, block, shared, stream>>>(__VA_ARGS__); \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device synchronization with error checking
#define CUDA_SYNC() \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA sync failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Financial parameter validation
#define VALIDATE_STOCK_PRICE(price) \
    do { \
        if (price < MIN_STOCK_PRICE || price > MAX_STOCK_PRICE) { \
            fprintf(stderr, "Invalid stock price at %s:%d: %f (valid range: %f-%f)\n", __FILE__, __LINE__, price, MIN_STOCK_PRICE, MAX_STOCK_PRICE); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define VALIDATE_STRIKE_PRICE(strike) \
    do { \
        if (strike < MIN_STRIKE_PRICE || strike > MAX_STRIKE_PRICE) { \
            fprintf(stderr, "Invalid strike price at %s:%d: %f (valid range: %f-%f)\n", __FILE__, __LINE__, strike, MIN_STRIKE_PRICE, MAX_STRIKE_PRICE); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define VALIDATE_VOLATILITY(vol) \
    do { \
        if (vol < MIN_VOLATILITY || vol > MAX_VOLATILITY) { \
            fprintf(stderr, "Invalid volatility at %s:%d: %f (valid range: %f-%f)\n", __FILE__, __LINE__, vol, MIN_VOLATILITY, MAX_VOLATILITY); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define VALIDATE_TIME_TO_MATURITY(time) \
    do { \
        if (time < MIN_TIME_TO_MATURITY || time > MAX_TIME_TO_MATURITY) { \
            fprintf(stderr, "Invalid time to maturity at %s:%d: %f (valid range: %f-%f)\n", __FILE__, __LINE__, time, MIN_TIME_TO_MATURITY, MAX_TIME_TO_MATURITY); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define VALIDATE_RISK_FREE_RATE(rate) \
    do { \
        if (rate < MIN_RISK_FREE_RATE || rate > MAX_RISK_FREE_RATE) { \
            fprintf(stderr, "Invalid risk-free rate at %s:%d: %f (valid range: %f-%f)\n", __FILE__, __LINE__, rate, MIN_RISK_FREE_RATE, MAX_RISK_FREE_RATE); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define VALIDATE_CONFIDENCE_LEVEL(level) \
    do { \
        if (level <= MIN_CONFIDENCE_LEVEL || level >= MAX_CONFIDENCE_LEVEL) { \
            fprintf(stderr, "Invalid confidence level at %s:%d: %f (valid range: %f-%f)\n", __FILE__, __LINE__, level, MIN_CONFIDENCE_LEVEL, MAX_CONFIDENCE_LEVEL); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Error handling functions
__host__ void check_cuda_error(const char* file, int line, const char* func);
__host__ void validate_financial_parameters(float stock_price, float strike_price, 
                                          float volatility, float time_to_maturity, 
                                          float risk_free_rate);
__host__ void print_device_info();
__host__ void check_device_memory(size_t required_bytes);

// Error codes
typedef enum {
    CUDA_FINANCE_SUCCESS = 0,
    CUDA_FINANCE_INVALID_PARAMETERS = 1,
    CUDA_FINANCE_MEMORY_ERROR = 2,
    CUDA_FINANCE_KERNEL_ERROR = 3,
    CUDA_FINANCE_DEVICE_ERROR = 4,
    CUDA_FINANCE_NUMERICAL_ERROR = 5
} cuda_finance_error_t;

// Error context structure
typedef struct {
    cuda_finance_error_t error_code;
    const char* error_message;
    const char* file;
    int line;
    const char* function;
} cuda_finance_error_context_t;

// Error handling function declarations
__host__ cuda_finance_error_t get_last_error();
__host__ const char* get_error_string(cuda_finance_error_t error);
__host__ void set_error_context(cuda_finance_error_t error, const char* message, 
                               const char* file, int line, const char* function);
__host__ void clear_error_context();

#endif // ERROR_HANDLING_H 