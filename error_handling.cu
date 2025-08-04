#include "error_handling.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global error context
static cuda_finance_error_context_t g_error_context = {CUDA_FINANCE_SUCCESS, NULL, NULL, 0, NULL};

// Error handling function implementations
__host__ void check_cuda_error(const char* file, int line, const char* func) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", 
                func, file, line, cudaGetErrorString(error));
        set_error_context(CUDA_FINANCE_KERNEL_ERROR, cudaGetErrorString(error), 
                         file, line, func);
    }
}

__host__ void validate_financial_parameters(float stock_price, float strike_price, 
                                          float volatility, float time_to_maturity, 
                                          float risk_free_rate) {
    VALIDATE_STOCK_PRICE(stock_price);
    VALIDATE_STRIKE_PRICE(strike_price);
    VALIDATE_VOLATILITY(volatility);
    VALIDATE_TIME_TO_MATURITY(time_to_maturity);
    VALIDATE_RISK_FREE_RATE(risk_free_rate);
}

__host__ void print_device_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    printf("Found %d CUDA device(s):\n", device_count);
    
    for (int device = 0; device < device_count; device++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        
        printf("Device %d: %s\n", device, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Max Grid Size: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max Block Size: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock Rate: %d MHz\n", prop.memoryClockRate / 1000);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n", 
               (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6));
        printf("\n");
    }
}

__host__ void check_device_memory(size_t required_bytes) {
    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    
    if (free_memory < required_bytes) {
        fprintf(stderr, "Insufficient GPU memory. Required: %zu bytes, Available: %zu bytes\n", 
                required_bytes, free_memory);
        set_error_context(CUDA_FINANCE_MEMORY_ERROR, "Insufficient GPU memory", 
                         __FILE__, __LINE__, __FUNCTION__);
        exit(EXIT_FAILURE);
    }
    
    printf("GPU memory check passed. Required: %zu bytes, Available: %zu bytes\n", 
           required_bytes, free_memory);
}

__host__ cuda_finance_error_t get_last_error() {
    return g_error_context.error_code;
}

__host__ const char* get_error_string(cuda_finance_error_t error) {
    switch (error) {
        case CUDA_FINANCE_SUCCESS:
            return "Success";
        case CUDA_FINANCE_INVALID_PARAMETERS:
            return "Invalid financial parameters";
        case CUDA_FINANCE_MEMORY_ERROR:
            return "CUDA memory allocation error";
        case CUDA_FINANCE_KERNEL_ERROR:
            return "CUDA kernel execution error";
        case CUDA_FINANCE_DEVICE_ERROR:
            return "CUDA device error";
        case CUDA_FINANCE_NUMERICAL_ERROR:
            return "Numerical computation error";
        default:
            return "Unknown error";
    }
}

__host__ void set_error_context(cuda_finance_error_t error, const char* message, 
                               const char* file, int line, const char* function) {
    g_error_context.error_code = error;
    g_error_context.error_message = message;
    g_error_context.file = file;
    g_error_context.line = line;
    g_error_context.function = function;
}

__host__ void clear_error_context() {
    g_error_context.error_code = CUDA_FINANCE_SUCCESS;
    g_error_context.error_message = NULL;
    g_error_context.file = NULL;
    g_error_context.line = 0;
    g_error_context.function = NULL;
}

// Additional utility functions for error handling
__host__ void print_error_context() {
    if (g_error_context.error_code != CUDA_FINANCE_SUCCESS) {
        fprintf(stderr, "Error Context:\n");
        fprintf(stderr, "  Code: %d (%s)\n", g_error_context.error_code, 
                get_error_string(g_error_context.error_code));
        if (g_error_context.error_message) {
            fprintf(stderr, "  Message: %s\n", g_error_context.error_message);
        }
        if (g_error_context.file) {
            fprintf(stderr, "  Location: %s:%d in %s\n", 
                    g_error_context.file, g_error_context.line, 
                    g_error_context.function ? g_error_context.function : "unknown");
        }
    }
}

__host__ void reset_device() {
    CUDA_CHECK(cudaDeviceReset());
    clear_error_context();
    printf("CUDA device reset completed.\n");
}

__host__ void check_device_status() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Device status check failed: %s\n", cudaGetErrorString(error));
        set_error_context(CUDA_FINANCE_DEVICE_ERROR, cudaGetErrorString(error), 
                         __FILE__, __LINE__, __FUNCTION__);
    } else {
        printf("Device status: OK\n");
    }
}

// Memory management with error checking
__host__ void* safe_cuda_malloc(size_t size) {
    void* ptr;
    CUDA_MALLOC(ptr, size);
    return ptr;
}

__host__ void safe_cuda_free(void* ptr) {
    if (ptr != NULL) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

__host__ void safe_cuda_memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    CUDA_MEMCPY(dst, src, size, kind);
}

// Kernel launch wrapper with error checking
__host__ void safe_kernel_launch(dim3 grid, dim3 block, size_t shared_mem, cudaStream_t stream) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Pre-kernel launch error: %s\n", cudaGetErrorString(error));
        set_error_context(CUDA_FINANCE_KERNEL_ERROR, cudaGetErrorString(error), 
                         __FILE__, __LINE__, __FUNCTION__);
        return;
    }
    
    // Note: This is a placeholder. Actual kernel launch would be done by the calling function
    // with the CUDA_LAUNCH macro
}

// Performance monitoring
__host__ void start_timer(cudaEvent_t* start, cudaEvent_t* stop) {
    CUDA_CHECK(cudaEventCreate(start));
    CUDA_CHECK(cudaEventCreate(stop));
    CUDA_CHECK(cudaEventRecord(*start));
}

__host__ float stop_timer(cudaEvent_t start, cudaEvent_t stop) {
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds;
} 