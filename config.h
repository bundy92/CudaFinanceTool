#ifndef CONFIG_H
#define CONFIG_H

// Build configuration
#define PROJECT_VERSION "1.0.0"
#define PROJECT_NAME "CUDA Finance Tool"

// Default parameters
#define DEFAULT_NUM_OPTIONS 8196
#define DEFAULT_NUM_SIMULATIONS 100000
#define DEFAULT_BLOCK_SIZE 512
#define DEFAULT_CONFIDENCE_LEVEL 0.95f

// Memory configuration
#define MAX_NUM_OPTIONS 1000000
#define MAX_NUM_SIMULATIONS 1000000
#define MAX_BLOCK_SIZE 1024

// Financial parameter bounds
#define MIN_STOCK_PRICE 0.01f
#define MAX_STOCK_PRICE 10000.0f
#define MIN_STRIKE_PRICE 0.01f
#define MAX_STRIKE_PRICE 10000.0f
#define MIN_VOLATILITY 0.001f
#define MAX_VOLATILITY 5.0f
#define MIN_TIME_TO_MATURITY 0.001f
#define MAX_TIME_TO_MATURITY 50.0f
#define MIN_RISK_FREE_RATE -0.1f
#define MAX_RISK_FREE_RATE 1.0f
#define MIN_CONFIDENCE_LEVEL 0.5f
#define MAX_CONFIDENCE_LEVEL 0.999f

// Performance tuning
#define SHARED_MEMORY_SIZE 16384  // 16KB
#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK 32

// Error handling
#define ERROR_TOLERANCE 1e-6f
#define MAX_ERROR_MESSAGES 100

// Testing configuration
#define TEST_NUM_OPTIONS 1024
#define TEST_NUM_SIMULATIONS 10000
#define TEST_BLOCK_SIZE 256
#define TEST_TOLERANCE 1e-4f

// Output formatting
#define MAX_DISPLAY_OPTIONS 10
#define PRECISION_DISPLAY 4

// Feature flags
#define ENABLE_DEBUG_OUTPUT 1
#define ENABLE_PERFORMANCE_MONITORING 1
#define ENABLE_PARAMETER_VALIDATION 1
#define ENABLE_ERROR_HANDLING 1

// Platform-specific settings
#ifdef _WIN32
    #define PATH_SEPARATOR "\\"
#else
    #define PATH_SEPARATOR "/"
#endif

// CUDA architecture settings (adjust based on target GPU)
#define CUDA_ARCH_SM60 1
#define CUDA_ARCH_SM70 1
#define CUDA_ARCH_SM75 1
#define CUDA_ARCH_SM80 1
#define CUDA_ARCH_SM86 1

// Compiler optimization flags
#ifdef NDEBUG
    #define OPTIMIZATION_LEVEL "-O3"
#else
    #define OPTIMIZATION_LEVEL "-O0"
#endif

// Memory management
#define USE_UNIFIED_MEMORY 0
#define USE_PINNED_MEMORY 1
#define USE_ASYNC_MEMORY_COPY 1

// Kernel configuration
#define USE_SHARED_MEMORY 1
#define USE_VECTORIZATION 1
#define USE_LOOP_UNROLLING 1

// Logging configuration
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_WARNING 1
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_DEBUG 0

// Performance thresholds
#define PERFORMANCE_THRESHOLD_MS 100.0f
#define MEMORY_USAGE_THRESHOLD_MB 1024

#endif // CONFIG_H 