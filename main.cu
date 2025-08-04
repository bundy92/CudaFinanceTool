// Include necessary libraries
#include <stdio.h>               // Standard input/output library
#include <stdlib.h>              // Standard library
#include "option_pricing.cuh"    // CUDA kernel for option pricing
#include "monte_carlo.cuh"       // CUDA kernel for Monte Carlo simulation
#include "risk_analysis.cuh"     // CUDA kernel for risk analysis
#include "utils.cuh"             // Utility functions
#include "error_handling.cuh"    // Error handling system

// Main CUDA program for option pricing, Monte Carlo simulation, and risk analysis
/*
 * Program Description:
 * --------------------
 * This CUDA program performs option pricing using the Black-Scholes model,
 * Monte Carlo simulation for option pricing, and risk analysis using VaR calculation.
 * It utilizes CUDA kernels for parallel computation on the GPU.
 *
 * Program Workflow:
 * -----------------
 * 1. Initialize sample data for option parameters.
 * 2. Allocate device memory for input data and copy sample data from host to device.
 * 3. Launch the Black-Scholes option pricing kernel to compute option prices, deltas, and gammas.
 * 4. Launch the Monte Carlo simulation kernel to perform option pricing through simulation.
 * 5. Launch the risk analysis kernel to calculate the VaR (Value at Risk) based on Monte Carlo results.
 * 6. Copy results back to host memory.
 * 7. Print the option pricing results, Monte Carlo simulation results, and VaR results.
 * 8. Free dynamically allocated memory and device memory.
 *
 * Input Parameters:
 * -----------------
 * - num_options: Total number of options to process.
 * - mem: Memory size for arrays holding sample data.
 * - confidence_level: Confidence level for VaR calculation.
 * - num_simulations: Number of simulations for Monte Carlo option pricing.
 * - block_size: CUDA block size for kernel execution.
 * - num_blocks: Number of CUDA blocks required based on the total number of options and block size.
 *
 * Host Functions:
 * ---------------
 * - mean_of_array: Computes the mean value of an array.
 *
 * Included Headers:
 * -----------------
 * - option_pricing.cuh: Contains the Black-Scholes option pricing kernel.
 * - monte_carlo.cuh: Contains the Monte Carlo simulation kernel.
 * - risk_analysis.cuh: Contains the VaR calculation kernel.
 * - utils.cuh: Contains utility functions.
 */

 // Main function
int main() {
    // Initialize CUDA and error handling
    CUDA_CHECK(cudaSetDevice(0));
    print_device_info();
    
    // Constants
    const int num_options = 8196;           // Number of options
    const int mem = 8196;                      // Memory allocation size
    const float confidence_level = 0.95;      // Confidence level for risk analysis
    const int num_simulations = 100000;         // Number of Monte Carlo simulations
    const int block_size = 512;                // CUDA block size
    const int num_blocks = (num_options + block_size - 1) / block_size; // Number of CUDA blocks
    
    // Validate parameters
    VALIDATE_CONFIDENCE_LEVEL(confidence_level);

    // Sample data for testing
    float sample_stock_prices_x[mem];    // Array to hold sample stock prices for x
    float sample_stock_prices_y[mem];    // Array to hold sample stock prices for y
    float sample_strike_prices_x[mem];   // Array to hold sample strike prices for x
    float sample_strike_prices_y[mem];   // Array to hold sample strike prices for y
    float sample_volatilities_x[mem];    // Array to hold sample volatilities for x
    float sample_volatilities_y[mem];    // Array to hold sample volatilities for y
    float sample_time_to_maturity_x[mem];// Array to hold sample time to maturity for x
    float sample_time_to_maturity_y[mem];// Array to hold sample time to maturity for y
    float sample_risk_free_rates_x[mem];// Array to hold sample risk-free rates for x
    float sample_risk_free_rates_y[mem];// Array to hold sample risk-free rates for y

    // Initialize sample data for each array
    for (int i = 0; i < num_options; ++i) {
        sample_stock_prices_x[i] = 100.0 + i;     // Sample stock prices ranging from 100 to 1123 for x
        sample_stock_prices_y[i] = 100.0 + i;     // Sample stock prices ranging from 100 to 1123 for y
        sample_strike_prices_x[i] = 105.0 + i;    // Sample strike prices ranging from 105 to 1128 for x
        sample_strike_prices_y[i] = 105.0 + i;    // Sample strike prices ranging from 105 to 1128 for y
        sample_volatilities_x[i] = 0.2;           // Constant volatility for testing for x
        sample_volatilities_y[i] = 0.2;           // Constant volatility for testing for y
        sample_time_to_maturity_x[i] = 1.0;       // Constant time to maturity for testing for x
        sample_time_to_maturity_y[i] = 1.0;       // Constant time to maturity for testing for y
        sample_risk_free_rates_x[i] = 0.05;      // Constant risk-free rate for testing for x
        sample_risk_free_rates_y[i] = 0.05;      // Constant risk-free rate for testing for y
    }

    // Check device memory requirements
    size_t total_memory_required = num_options * sizeof(float) * 10; // 10 arrays
    check_device_memory(total_memory_required);
    
    // Allocate device memory for input data
    float* d_stock_prices_x, * d_stock_prices_y, * d_strike_prices_x, * d_strike_prices_y,
        * d_volatilities_x, * d_volatilities_y, * d_time_to_maturity_x, * d_time_to_maturity_y,
        * d_risk_free_rates_x, * d_risk_free_rates_y;

    CUDA_MALLOC(d_stock_prices_x, num_options * sizeof(float));
    CUDA_MALLOC(d_stock_prices_y, num_options * sizeof(float));
    CUDA_MALLOC(d_strike_prices_x, num_options * sizeof(float));
    CUDA_MALLOC(d_strike_prices_y, num_options * sizeof(float));
    CUDA_MALLOC(d_volatilities_x, num_options * sizeof(float));
    CUDA_MALLOC(d_volatilities_y, num_options * sizeof(float));
    CUDA_MALLOC(d_time_to_maturity_x, num_options * sizeof(float));
    CUDA_MALLOC(d_time_to_maturity_y, num_options * sizeof(float));
    CUDA_MALLOC(d_risk_free_rates_x, num_options * sizeof(float));
    CUDA_MALLOC(d_risk_free_rates_y, num_options * sizeof(float));

    // Copy sample data from host to device
    CUDA_MEMCPY(d_stock_prices_x, sample_stock_prices_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_stock_prices_y, sample_stock_prices_y, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_strike_prices_x, sample_strike_prices_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_strike_prices_y, sample_strike_prices_y, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_volatilities_x, sample_volatilities_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_volatilities_y, sample_volatilities_y, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_time_to_maturity_x, sample_time_to_maturity_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_time_to_maturity_y, sample_time_to_maturity_y, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_risk_free_rates_x, sample_risk_free_rates_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_MEMCPY(d_risk_free_rates_y, sample_risk_free_rates_y, num_options * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate device memory for output data
    float* d_option_prices, * d_deltas, * d_gammas;
    CUDA_MALLOC(d_option_prices, num_options * sizeof(float));
    CUDA_MALLOC(d_deltas, num_options * sizeof(float));
    CUDA_MALLOC(d_gammas, num_options * sizeof(float));

    // Launch option pricing kernel with error checking
    CUDA_LAUNCH(black_scholes_option_pricing, num_blocks, block_size, 0, 0,
                d_option_prices, d_deltas, d_gammas,
                d_stock_prices_x, d_stock_prices_y,
                d_strike_prices_x, d_strike_prices_y,
                d_volatilities_x, d_volatilities_y,
                d_time_to_maturity_x, d_time_to_maturity_y,
                d_risk_free_rates_x, d_risk_free_rates_y,
                num_options);

    // Synchronize and check for errors
    CUDA_SYNC();

    // Allocate device memory for Monte Carlo output data
    float* d_monte_carlo_results;
    CUDA_MALLOC(d_monte_carlo_results, num_options * num_simulations * sizeof(float));

    // Launch Monte Carlo simulation kernel with error checking
    CUDA_LAUNCH(monte_carlo_option_pricing, num_blocks, block_size, 0, 0,
                d_monte_carlo_results, d_stock_prices_x,
                d_volatilities_x, d_time_to_maturity_x,
                d_risk_free_rates_x, d_strike_prices_x, num_options,
                num_simulations);

    // Synchronize and check for errors
    CUDA_SYNC();

    // Allocate device memory for VaR calculation
    float* d_var_values;
    CUDA_MALLOC(d_var_values, num_blocks * sizeof(float));

    // Launch risk analysis kernel with error checking
    CUDA_LAUNCH(calculate_var, num_blocks, block_size, block_size * sizeof(float), 0,
                d_monte_carlo_results, d_var_values,
                num_options, confidence_level);

    // Synchronize and check for errors
    CUDA_SYNC();

    // Copy results back to host memory
    float h_option_prices[mem];
    float h_deltas[mem];
    float h_gammas[mem];
    float h_var_values[mem];
    float* h_monte_carlo_results = (float*)malloc(num_options * num_simulations * sizeof(float));

    // Check if memory allocation was successful
    if (h_monte_carlo_results == NULL) {
        fprintf(stderr, "Failed to allocate memory for Monte Carlo results on host\n");
        return 1; // Return an error code
    }

    CUDA_MEMCPY(h_option_prices, d_option_prices, num_options * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_MEMCPY(h_deltas, d_deltas, num_options * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_MEMCPY(h_gammas, d_gammas, num_options * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_MEMCPY(h_var_values, d_var_values, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_MEMCPY(h_monte_carlo_results, d_monte_carlo_results, num_options * num_simulations * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate mean option price from Monte Carlo simulation results
    float mean_option_price = mean_of_array(h_monte_carlo_results, num_options * num_simulations);

    // Print the results in a professional format
    printf("Option Pricing Results:\n");
    printf("%-10s %-15s %-15s %-15s\n", "Option", "Price", "Delta", "Gamma");
    for (int i = 0; i < 10; ++i) {
        printf("%-10d %-15.4f %-15.4f %-15.4f\n", i + 1, h_option_prices[i], h_deltas[i], h_gammas[i]);
    }

    // Print Monte Carlo Simulation Results
    printf("\nMonte Carlo Simulation Results:\n");
    printf("Mean Option Price: %.4f\n", mean_option_price);

    printf("\nValue at Risk (VaR) Results:\n");
    printf("Confidence Level: %.2f\n", confidence_level);
    printf("Value at Risk (VaR): %.4f\n", h_var_values[0]);

    // Final error check
    check_device_status();

    // Free dynamically allocated memory
    free(h_monte_carlo_results);

    // Free device memory with error checking
    safe_cuda_free(d_stock_prices_x);
    safe_cuda_free(d_stock_prices_y);
    safe_cuda_free(d_strike_prices_x);
    safe_cuda_free(d_strike_prices_y);
    safe_cuda_free(d_volatilities_x);
    safe_cuda_free(d_volatilities_y);
    safe_cuda_free(d_time_to_maturity_x);
    safe_cuda_free(d_time_to_maturity_y);
    safe_cuda_free(d_risk_free_rates_x);
    safe_cuda_free(d_risk_free_rates_y);
    safe_cuda_free(d_option_prices);
    safe_cuda_free(d_deltas);
    safe_cuda_free(d_gammas);
    safe_cuda_free(d_monte_carlo_results);
    safe_cuda_free(d_var_values);

    return 0;
}
