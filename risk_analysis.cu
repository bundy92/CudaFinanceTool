#ifndef RISK_ANALYSIS_CU
#define RISK_ANALYSIS_CU

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118
#endif

#include "risk_analysis.cuh"

// VaR calculation kernel implementation
/*
 * Kernel Description:
 * -------------------
 * This CUDA kernel implements the Value at Risk (VaR) calculation for risk analysis.
 * It calculates the VaR value for a set of option prices based on a given confidence level.
 * The VaR value represents the maximum potential loss with a certain confidence level.
 *
 * Input Parameters:
 * -----------------
 * - option_prices: Array of option prices for each option.
 * - var_values: Array to store the resulting VaR values.
 * - num_options: Total number of options.
 * - confidence_level: Confidence level for VaR calculation.
 *
 * Thread Organization:
 * --------------------
 * Each CUDA block is responsible for processing a subset of option prices.
 * Each CUDA thread loads an option price into shared memory.
 * Once all option prices are loaded, parallel reduction is used to find the maximum option price.
 * The VaR value is then computed based on the sorted option prices.
 */
__global__ void calculate_var(double* option_prices, double* var_values, const int num_options,
    const double confidence_level) {
    extern __shared__ double shared_buffer[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = threadIdx.x;

    if (tid < num_options) {
        shared_buffer[idx] = option_prices[tid];
    }
    else {
        shared_buffer[idx] = 0.0;
    }

    __syncthreads();

    if (idx == 0) {
        // Sort the option prices using parallel reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (idx < stride) {
                double temp = shared_buffer[idx];
                double other = shared_buffer[idx + stride];
                shared_buffer[idx] = (temp > other) ? temp : other;
            }
            __syncthreads();
        }

        // Compute VaR value
        int var_index = (int)ceil(num_options * (1.0 - confidence_level));
        var_values[blockIdx.x] = shared_buffer[var_index];
    }
}

#endif // RISK_ANALYSIS_CU
