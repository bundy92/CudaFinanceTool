// Include necessary libraries
#include <stdio.h>               // Standard input/output library
#include <stdlib.h>              // Standard library
#include "option_pricing.cuh"    // CUDA kernel for option pricing
#include "monte_carlo.cuh"       // CUDA kernel for Monte Carlo simulation
#include "risk_analysis.cuh"     // CUDA kernel for risk analysis
#include "utils.cuh"             // Utility functions

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
    // Constants
    const int num_options = 8196;           // Number of options
    const int mem = 8196;                      // Memory allocation size
    const float confidence_level = 0.95;      // Confidence level for risk analysis
    const int num_simulations = 100000;         // Number of Monte Carlo simulations
    const int block_size = 512;                // CUDA block size
    const int num_blocks = (num_options + block_size - 1) / block_size; // Number of CUDA blocks

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

    // Allocate device memory for input data
    float* d_stock_prices_x, * d_stock_prices_y, * d_strike_prices_x, * d_strike_prices_y,
        * d_volatilities_x, * d_volatilities_y, * d_time_to_maturity_x, * d_time_to_maturity_y,
        * d_risk_free_rates_x, * d_risk_free_rates_y;

    cudaMalloc((void**)&d_stock_prices_x, num_options * sizeof(float));
    cudaMalloc((void**)&d_stock_prices_y, num_options * sizeof(float));
    cudaMalloc((void**)&d_strike_prices_x, num_options * sizeof(float));
    cudaMalloc((void**)&d_strike_prices_y, num_options * sizeof(float));
    cudaMalloc((void**)&d_volatilities_x, num_options * sizeof(float));
    cudaMalloc((void**)&d_volatilities_y, num_options * sizeof(float));
    cudaMalloc((void**)&d_time_to_maturity_x, num_options * sizeof(float));
    cudaMalloc((void**)&d_time_to_maturity_y, num_options * sizeof(float));
    cudaMalloc((void**)&d_risk_free_rates_x, num_options * sizeof(float));
    cudaMalloc((void**)&d_risk_free_rates_y, num_options * sizeof(float));

    // Copy sample data from host to device
    cudaMemcpy(d_stock_prices_x, sample_stock_prices_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stock_prices_y, sample_stock_prices_y, num_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strike_prices_x, sample_strike_prices_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strike_prices_y, sample_strike_prices_y, num_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_volatilities_x, sample_volatilities_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_volatilities_y, sample_volatilities_y, num_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_time_to_maturity_x, sample_time_to_maturity_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_time_to_maturity_y, sample_time_to_maturity_y, num_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_risk_free_rates_x, sample_risk_free_rates_x, num_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_risk_free_rates_y, sample_risk_free_rates_y, num_options * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate device memory for output data
    float* d_option_prices, * d_deltas, * d_gammas;
    cudaMalloc((void**)&d_option_prices, num_options * sizeof(float));
    cudaMalloc((void**)&d_deltas, num_options * sizeof(float));
    cudaMalloc((void**)&d_gammas, num_options * sizeof(float));

    // Launch option pricing kernel
    black_scholes_option_pricing << <num_blocks, block_size >> > (d_option_prices, d_deltas, d_gammas,
        d_stock_prices_x, d_stock_prices_y,
        d_strike_prices_x, d_strike_prices_y,
        d_volatilities_x, d_volatilities_y,
        d_time_to_maturity_x, d_time_to_maturity_y,
        d_risk_free_rates_x, d_risk_free_rates_y,
        num_options);

    // Check for kernel launch errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    // Wait for kernel to finish
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    // Allocate device memory for Monte Carlo output data
    float* d_monte_carlo_results;
    cudaMalloc((void**)&d_monte_carlo_results, num_options * num_simulations * sizeof(float));

    // Launch Monte Carlo simulation kernel
    monte_carlo_option_pricing << <num_blocks, block_size >> > (d_monte_carlo_results, d_stock_prices_x,
        d_volatilities_x, d_time_to_maturity_x,
        d_risk_free_rates_x, d_strike_prices_x, num_options,
        num_simulations);

    // Check for kernel launch errors
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    // Wait for kernel to finish
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    // Allocate device memory for VaR calculation
    float* d_var_values;
    cudaMalloc((void**)&d_var_values, num_blocks * sizeof(float));

    // Launch risk analysis kernel
    calculate_var << <num_blocks, block_size, block_size * sizeof(float) >> > (d_monte_carlo_results, d_var_values,
        num_options, confidence_level);

    // Check for kernel launch errors
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    // Wait for kernel to finish
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

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

    cudaMemcpy(h_option_prices, d_option_prices, num_options * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_deltas, d_deltas, num_options * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gammas, d_gammas, num_options * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_var_values, d_var_values, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_monte_carlo_results, d_monte_carlo_results, num_options * num_simulations * sizeof(float), cudaMemcpyDeviceToHost);

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

    // Check for kernel launch errors
    cudaDeviceSynchronize();
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    // Free dynamically allocated memory
    free(h_monte_carlo_results);

    // Free device memory
    cudaFree(d_stock_prices_x);
    cudaFree(d_stock_prices_y);
    cudaFree(d_strike_prices_x);
    cudaFree(d_strike_prices_y);
    cudaFree(d_volatilities_x);
    cudaFree(d_volatilities_y);
    cudaFree(d_time_to_maturity_x);
    cudaFree(d_time_to_maturity_y);
    cudaFree(d_risk_free_rates_x);
    cudaFree(d_risk_free_rates_y);
    cudaFree(d_option_prices);
    cudaFree(d_deltas);
    cudaFree(d_gammas);
    cudaFree(d_monte_carlo_results);
    cudaFree(d_var_values);

    return 0;
}
