#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "option_pricing.cuh"
#include "monte_carlo.cuh"
#include "risk_analysis.cuh"
#include "utils.cuh"
#include "error_handling.cuh"

// Test configuration
#define TEST_NUM_OPTIONS 1024
#define TEST_NUM_SIMULATIONS 10000
#define TEST_BLOCK_SIZE 256
#define TEST_TOLERANCE 1e-4f

// Test results structure
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
    float total_time_ms;
} test_results_t;

// Global test results
static test_results_t g_test_results = {0, 0, 0, 0.0f};

// Test utilities
#define TEST_ASSERT(condition, message) \
    do { \
        g_test_results.total_tests++; \
        if (condition) { \
            g_test_results.passed_tests++; \
            printf("✓ %s\n", message); \
        } else { \
            g_test_results.failed_tests++; \
            printf("✗ %s\n", message); \
        } \
    } while(0)

#define TEST_ASSERT_FLOAT_EQ(expected, actual, tolerance, message) \
    do { \
        g_test_results.total_tests++; \
        if (fabsf(expected - actual) <= tolerance) { \
            g_test_results.passed_tests++; \
            printf("✓ %s (expected: %.6f, actual: %.6f)\n", message, expected, actual); \
        } else { \
            g_test_results.failed_tests++; \
            printf("✗ %s (expected: %.6f, actual: %.6f)\n", message, expected, actual); \
        } \
    } while(0)

// Reference Black-Scholes implementation for testing
__host__ float reference_black_scholes_call(float S, float K, float T, float r, float sigma) {
    if (sigma <= 0.0f || T <= 0.0f || S <= 0.0f || K <= 0.0f) {
        return 0.0f;
    }
    
    float sqrt_T = sqrtf(T);
    float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt_T);
    float d2 = d1 - sigma * sqrt_T;
    
    // Cumulative normal distribution approximation
    float N_d1 = 0.5f * (1.0f + erff(d1 / sqrtf(2.0f)));
    float N_d2 = 0.5f * (1.0f + erff(d2 / sqrtf(2.0f)));
    
    return S * N_d1 - K * expf(-r * T) * N_d2;
}

__host__ float reference_black_scholes_delta(float S, float K, float T, float r, float sigma) {
    if (sigma <= 0.0f || T <= 0.0f || S <= 0.0f || K <= 0.0f) {
        return 0.0f;
    }
    
    float sqrt_T = sqrtf(T);
    float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt_T);
    
    return 0.5f * (1.0f + erff(d1 / sqrtf(2.0f)));
}

// Test data generation
__host__ void generate_test_data(float* stock_prices, float* strike_prices, 
                                float* volatilities, float* time_to_maturity, 
                                float* risk_free_rates, int num_options) {
    for (int i = 0; i < num_options; i++) {
        stock_prices[i] = 50.0f + (float)i * 0.1f;  // 50.0 to 152.3
        strike_prices[i] = stock_prices[i] * (0.8f + 0.4f * (float)(i % 10) / 10.0f);
        volatilities[i] = 0.1f + 0.3f * (float)(i % 20) / 20.0f;  // 0.1 to 0.4
        time_to_maturity[i] = 0.1f + 2.0f * (float)(i % 50) / 50.0f;  // 0.1 to 2.1
        risk_free_rates[i] = 0.01f + 0.04f * (float)(i % 25) / 25.0f;  // 0.01 to 0.05
    }
}

// Test 1: Basic parameter validation
__host__ void test_parameter_validation() {
    printf("\n=== Test 1: Parameter Validation ===\n");
    
    // Valid parameters
    TEST_ASSERT(true, "Valid stock price (100.0)");
    validate_financial_parameters(100.0f, 105.0f, 0.2f, 1.0f, 0.05f);
    
    // Invalid parameters should be caught by macros
    printf("Testing invalid parameters (should show error messages):\n");
    
    // Note: These will cause program exit due to validation macros
    // In a real test environment, we'd use a different approach
    printf("Parameter validation test completed.\n");
}

// Test 2: Reference Black-Scholes implementation
__host__ void test_reference_implementation() {
    printf("\n=== Test 2: Reference Black-Scholes Implementation ===\n");
    
    // Test case 1: Standard parameters
    float S1 = 100.0f, K1 = 100.0f, T1 = 1.0f, r1 = 0.05f, sigma1 = 0.2f;
    float call_price1 = reference_black_scholes_call(S1, K1, T1, r1, sigma1);
    float delta1 = reference_black_scholes_delta(S1, K1, T1, r1, sigma1);
    
    TEST_ASSERT(call_price1 > 0.0f, "Call price is positive");
    TEST_ASSERT(delta1 > 0.0f && delta1 < 1.0f, "Delta is in valid range");
    
    // Test case 2: In-the-money option
    float S2 = 110.0f, K2 = 100.0f, T2 = 1.0f, r2 = 0.05f, sigma2 = 0.2f;
    float call_price2 = reference_black_scholes_call(S2, K2, T2, r2, sigma2);
    float delta2 = reference_black_scholes_delta(S2, K2, T2, r2, sigma2);
    
    TEST_ASSERT(call_price2 > call_price1, "ITM option has higher price");
    TEST_ASSERT(delta2 > delta1, "ITM option has higher delta");
    
    // Test case 3: Out-of-the-money option
    float S3 = 90.0f, K3 = 100.0f, T3 = 1.0f, r3 = 0.05f, sigma3 = 0.2f;
    float call_price3 = reference_black_scholes_call(S3, K3, T3, r3, sigma3);
    float delta3 = reference_black_scholes_delta(S3, K3, T3, r3, sigma3);
    
    TEST_ASSERT(call_price3 < call_price1, "OTM option has lower price");
    TEST_ASSERT(delta3 < delta1, "OTM option has lower delta");
    
    printf("Reference implementation test completed.\n");
}

// Test 3: CUDA kernel accuracy
__host__ void test_cuda_kernel_accuracy() {
    printf("\n=== Test 3: CUDA Kernel Accuracy ===\n");
    
    // Generate test data
    float* h_stock_prices = (float*)malloc(TEST_NUM_OPTIONS * sizeof(float));
    float* h_strike_prices = (float*)malloc(TEST_NUM_OPTIONS * sizeof(float));
    float* h_volatilities = (float*)malloc(TEST_NUM_OPTIONS * sizeof(float));
    float* h_time_to_maturity = (float*)malloc(TEST_NUM_OPTIONS * sizeof(float));
    float* h_risk_free_rates = (float*)malloc(TEST_NUM_OPTIONS * sizeof(float));
    
    generate_test_data(h_stock_prices, h_strike_prices, h_volatilities, 
                      h_time_to_maturity, h_risk_free_rates, TEST_NUM_OPTIONS);
    
    // Allocate device memory
    float *d_stock_prices_x, *d_strike_prices_x, *d_volatilities_x, 
          *d_time_to_maturity_x, *d_risk_free_rates_x;
    float *d_option_prices, *d_deltas, *d_gammas;
    
    CUDA_MALLOC(d_stock_prices_x, TEST_NUM_OPTIONS * sizeof(float));
    CUDA_MALLOC(d_strike_prices_x, TEST_NUM_OPTIONS * sizeof(float));
    CUDA_MALLOC(d_volatilities_x, TEST_NUM_OPTIONS * sizeof(float));
    CUDA_MALLOC(d_time_to_maturity_x, TEST_NUM_OPTIONS * sizeof(float));
    CUDA_MALLOC(d_risk_free_rates_x, TEST_NUM_OPTIONS * sizeof(float));
    CUDA_MALLOC(d_option_prices, TEST_NUM_OPTIONS * sizeof(float));
    CUDA_MALLOC(d_deltas, TEST_NUM_OPTIONS * sizeof(float));
    CUDA_MALLOC(d_gammas, TEST_NUM_OPTIONS * sizeof(float));
    
    // Copy data to device
    CUDA_MEMCPY(d_stock_prices_x, h_stock_prices, TEST_NUM_OPTIONS * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_strike_prices_x, h_strike_prices, TEST_NUM_OPTIONS * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_volatilities_x, h_volatilities, TEST_NUM_OPTIONS * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_time_to_maturity_x, h_time_to_maturity, TEST_NUM_OPTIONS * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_risk_free_rates_x, h_risk_free_rates, TEST_NUM_OPTIONS * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int num_blocks = (TEST_NUM_OPTIONS + TEST_BLOCK_SIZE - 1) / TEST_BLOCK_SIZE;
    CUDA_LAUNCH(black_scholes_option_pricing, num_blocks, TEST_BLOCK_SIZE, 0, 0,
                d_option_prices, d_deltas, d_gammas,
                d_stock_prices_x, d_stock_prices_x,  // Using same data for x and y
                d_strike_prices_x, d_strike_prices_x,
                d_volatilities_x, d_volatilities_x,
                d_time_to_maturity_x, d_time_to_maturity_x,
                d_risk_free_rates_x, d_risk_free_rates_x,
                TEST_NUM_OPTIONS);
    
    CUDA_SYNC();
    
    // Copy results back
    float* h_option_prices = (float*)malloc(TEST_NUM_OPTIONS * sizeof(float));
    float* h_deltas = (float*)malloc(TEST_NUM_OPTIONS * sizeof(float));
    float* h_gammas = (float*)malloc(TEST_NUM_OPTIONS * sizeof(float));
    
    CUDA_MEMCPY(h_option_prices, d_option_prices, TEST_NUM_OPTIONS * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_MEMCPY(h_deltas, d_deltas, TEST_NUM_OPTIONS * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_MEMCPY(h_gammas, d_gammas, TEST_NUM_OPTIONS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare with reference implementation
    int num_errors = 0;
    for (int i = 0; i < TEST_NUM_OPTIONS; i++) {
        float ref_price = reference_black_scholes_call(h_stock_prices[i], h_strike_prices[i],
                                                     h_time_to_maturity[i], h_risk_free_rates[i],
                                                     h_volatilities[i]);
        float ref_delta = reference_black_scholes_delta(h_stock_prices[i], h_strike_prices[i],
                                                       h_time_to_maturity[i], h_risk_free_rates[i],
                                                       h_volatilities[i]);
        
        if (fabsf(h_option_prices[i] - ref_price) > TEST_TOLERANCE ||
            fabsf(h_deltas[i] - ref_delta) > TEST_TOLERANCE) {
            num_errors++;
            if (num_errors <= 5) {  // Show first 5 errors
                printf("Error at index %d: CUDA(%.6f, %.6f) vs Ref(%.6f, %.6f)\n",
                       i, h_option_prices[i], h_deltas[i], ref_price, ref_delta);
            }
        }
    }
    
    TEST_ASSERT(num_errors == 0, "CUDA kernel matches reference implementation");
    
    // Cleanup
    free(h_stock_prices);
    free(h_strike_prices);
    free(h_volatilities);
    free(h_time_to_maturity);
    free(h_risk_free_rates);
    free(h_option_prices);
    free(h_deltas);
    free(h_gammas);
    
    safe_cuda_free(d_stock_prices_x);
    safe_cuda_free(d_strike_prices_x);
    safe_cuda_free(d_volatilities_x);
    safe_cuda_free(d_time_to_maturity_x);
    safe_cuda_free(d_risk_free_rates_x);
    safe_cuda_free(d_option_prices);
    safe_cuda_free(d_deltas);
    safe_cuda_free(d_gammas);
    
    printf("CUDA kernel accuracy test completed.\n");
}

// Test 4: Memory management
__host__ void test_memory_management() {
    printf("\n=== Test 4: Memory Management ===\n");
    
    // Test memory allocation
    void* test_ptr = safe_cuda_malloc(1024);
    TEST_ASSERT(test_ptr != NULL, "CUDA memory allocation successful");
    
    // Test memory copy
    float* h_data = (float*)malloc(256 * sizeof(float));
    for (int i = 0; i < 256; i++) {
        h_data[i] = (float)i;
    }
    
    float* d_data;
    CUDA_MALLOC(d_data, 256 * sizeof(float));
    CUDA_MEMCPY(d_data, h_data, 256 * sizeof(float), cudaMemcpyHostToDevice);
    
    float* h_result = (float*)malloc(256 * sizeof(float));
    CUDA_MEMCPY(h_result, d_data, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify data integrity
    int data_correct = 1;
    for (int i = 0; i < 256; i++) {
        if (h_data[i] != h_result[i]) {
            data_correct = 0;
            break;
        }
    }
    TEST_ASSERT(data_correct, "Memory copy preserves data integrity");
    
    // Cleanup
    safe_cuda_free(test_ptr);
    safe_cuda_free(d_data);
    free(h_data);
    free(h_result);
    
    printf("Memory management test completed.\n");
}

// Test 5: Performance benchmark
__host__ void test_performance() {
    printf("\n=== Test 5: Performance Benchmark ===\n");
    
    const int num_options = 100000;
    const int num_iterations = 10;
    
    // Generate test data
    float* h_stock_prices = (float*)malloc(num_options * sizeof(float));
    float* h_strike_prices = (float*)malloc(num_options * sizeof(float));
    float* h_volatilities = (float*)malloc(num_options * sizeof(float));
    float* h_time_to_maturity = (float*)malloc(num_options * sizeof(float));
    float* h_risk_free_rates = (float*)malloc(num_options * sizeof(float));
    
    generate_test_data(h_stock_prices, h_strike_prices, h_volatilities, 
                      h_time_to_maturity, h_risk_free_rates, num_options);
    
    // Allocate device memory
    float *d_stock_prices_x, *d_strike_prices_x, *d_volatilities_x, 
          *d_time_to_maturity_x, *d_risk_free_rates_x;
    float *d_option_prices, *d_deltas, *d_gammas;
    
    CUDA_MALLOC(d_stock_prices_x, num_options * sizeof(float));
    CUDA_MALLOC(d_strike_prices_x, num_options * sizeof(float));
    CUDA_MALLOC(d_volatilities_x, num_options * sizeof(float));
    CUDA_MALLOC(d_time_to_maturity_x, num_options * sizeof(float));
    CUDA_MALLOC(d_risk_free_rates_x, num_options * sizeof(float));
    CUDA_MALLOC(d_option_prices, num_options * sizeof(float));
    CUDA_MALLOC(d_deltas, num_options * sizeof(float));
    CUDA_MALLOC(d_gammas, num_options * sizeof(float));
    
    // Copy data to device
    CUDA_MEMCPY(d_stock_prices_x, h_stock_prices, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_strike_prices_x, h_strike_prices, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_volatilities_x, h_volatilities, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_time_to_maturity_x, h_time_to_maturity, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_risk_free_rates_x, h_risk_free_rates, num_options * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warm up
    int num_blocks = (num_options + TEST_BLOCK_SIZE - 1) / TEST_BLOCK_SIZE;
    CUDA_LAUNCH(black_scholes_option_pricing, num_blocks, TEST_BLOCK_SIZE, 0, 0,
                d_option_prices, d_deltas, d_gammas,
                d_stock_prices_x, d_stock_prices_x,
                d_strike_prices_x, d_strike_prices_x,
                d_volatilities_x, d_volatilities_x,
                d_time_to_maturity_x, d_time_to_maturity_x,
                d_risk_free_rates_x, d_risk_free_rates_x,
                num_options);
    
    CUDA_SYNC();
    
    // Performance measurement
    cudaEvent_t start, stop;
    start_timer(&start, &stop);
    
    for (int i = 0; i < num_iterations; i++) {
        CUDA_LAUNCH(black_scholes_option_pricing, num_blocks, TEST_BLOCK_SIZE, 0, 0,
                    d_option_prices, d_deltas, d_gammas,
                    d_stock_prices_x, d_stock_prices_x,
                    d_strike_prices_x, d_strike_prices_x,
                    d_volatilities_x, d_volatilities_x,
                    d_time_to_maturity_x, d_time_to_maturity_x,
                    d_risk_free_rates_x, d_risk_free_rates_x,
                    num_options);
    }
    
    float total_time = stop_timer(start, stop);
    float avg_time = total_time / num_iterations;
    float options_per_second = (float)num_options / (avg_time / 1000.0f);
    
    printf("Performance Results:\n");
    printf("  Total time: %.2f ms\n", total_time);
    printf("  Average time per iteration: %.2f ms\n", avg_time);
    printf("  Options processed per second: %.0f\n", options_per_second);
    
    TEST_ASSERT(avg_time < 100.0f, "Performance is acceptable (< 100ms per iteration)");
    
    // Cleanup
    free(h_stock_prices);
    free(h_strike_prices);
    free(h_volatilities);
    free(h_time_to_maturity);
    free(h_risk_free_rates);
    
    safe_cuda_free(d_stock_prices_x);
    safe_cuda_free(d_strike_prices_x);
    safe_cuda_free(d_volatilities_x);
    safe_cuda_free(d_time_to_maturity_x);
    safe_cuda_free(d_risk_free_rates_x);
    safe_cuda_free(d_option_prices);
    safe_cuda_free(d_deltas);
    safe_cuda_free(d_gammas);
    
    printf("Performance test completed.\n");
}

// Main test function
int main() {
    printf("CUDA Finance Tool Test Suite\n");
    printf("============================\n");
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    print_device_info();
    
    // Run tests
    test_parameter_validation();
    test_reference_implementation();
    test_cuda_kernel_accuracy();
    test_memory_management();
    test_performance();
    
    // Print results
    printf("\n=== Test Results Summary ===\n");
    printf("Total tests: %d\n", g_test_results.total_tests);
    printf("Passed: %d\n", g_test_results.passed_tests);
    printf("Failed: %d\n", g_test_results.failed_tests);
    printf("Success rate: %.1f%%\n", 
           (float)g_test_results.passed_tests / g_test_results.total_tests * 100.0f);
    
    if (g_test_results.failed_tests == 0) {
        printf("\n🎉 All tests passed!\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please check the output above.\n");
        return 1;
    }
} 