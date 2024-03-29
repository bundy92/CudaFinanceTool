#ifndef OPTION_PRICING_CU
#define OPTION_PRICING_CU

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118f
#endif

#include "option_pricing.cuh"

// Function to calculate cumulative normal distribution
__device__ float cumulative_normal_distribution(float x) {
    return 0.5f * erfcf(-x * M_SQRT1_2);
}

// Black-Scholes option pricing kernel implementation with vectorization
/*
 * Kernel Description:
 * -------------------
 * This CUDA kernel implements the Black-Scholes option pricing model with vectorization.
 * It calculates the option price, delta, and gamma for a set of options based on their parameters.
 * The calculations are performed in parallel for multiple options using CUDA threads.
 * The Black-Scholes model assumes that the underlying asset follows geometric Brownian motion, and it's used extensively in financial markets to price European-style options.
 * The calculation involves computing the cumulative normal distribution, which represents the probability of the option expiring in-the-money.
 * The vectorization technique allows for efficient parallel computation of option prices, deltas, and gammas, improving performance on GPU architectures.
 *
 * Input Parameters:
 * -----------------
 * - option_prices: Array to store the resulting option prices.
 * - deltas: Array to store the resulting option deltas.
 * - gammas: Array to store the resulting option gammas.
 * - stock_prices_x: Array of stock prices for 'x' options.
 * - stock_prices_y: Array of stock prices for 'y' options.
 * - strike_prices_x: Array of strike prices for 'x' options.
 * - strike_prices_y: Array of strike prices for 'y' options.
 * - volatilities_x: Array of volatilities for 'x' options.
 * - volatilities_y: Array of volatilities for 'y' options.
 * - time_to_maturity_x: Array of time to maturity for 'x' options.
 * - time_to_maturity_y: Array of time to maturity for 'y' options.
 * - risk_free_rates_x: Array of risk-free rates for 'x' options.
 * - risk_free_rates_y: Array of risk-free rates for 'y' options.
 * - num_options: Total number of options to process.
 *
 * Thread Organization:
 * --------------------
 * Each CUDA thread is responsible for processing two options (using vectorization), hence the computation is divided into blocks and threads.
 * For each pair of options, it loads their parameters, calculates the Black-Scholes coefficients (d1 and d2), computes the option price, delta, and gamma, and stores the results in the corresponding output arrays.
 * The vectorization technique allows for efficient parallel computation of option prices, deltas, and gammas, improving performance on GPU architectures.
 * This kernel is designed to handle cases where the number of options is not necessarily a multiple of the block size, ensuring all options are processed correctly.
 */

// Black-Scholes option pricing kernel implementation with vectorization
__global__ void black_scholes_option_pricing(float* option_prices, float* deltas, float* gammas,
    const float* stock_prices_x, const float* stock_prices_y,
    const float* strike_prices_x, const float* strike_prices_y,
    const float* volatilities_x, const float* volatilities_y,
    const float* time_to_maturity_x, const float* time_to_maturity_y,
    const float* risk_free_rates_x, const float* risk_free_rates_y,
    const int num_options) {

    extern __shared__ float shared_data[]; // Shared memory for caching

    int tid = threadIdx.x + blockIdx.x * blockDim.x * 2;

    // Indices for accessing shared memory
    int local_index = threadIdx.x;
    int global_index = tid;

    // Load data into shared memory
    if (global_index < num_options) {
        shared_data[local_index] = stock_prices_x[global_index];
        shared_data[local_index + blockDim.x] = stock_prices_y[global_index];
        shared_data[local_index + 2 * blockDim.x] = strike_prices_x[global_index];
        shared_data[local_index + 3 * blockDim.x] = strike_prices_y[global_index];
        shared_data[local_index + 4 * blockDim.x] = volatilities_x[global_index];
        shared_data[local_index + 5 * blockDim.x] = volatilities_y[global_index];
        shared_data[local_index + 6 * blockDim.x] = time_to_maturity_x[global_index];
        shared_data[local_index + 7 * blockDim.x] = time_to_maturity_y[global_index];
        shared_data[local_index + 8 * blockDim.x] = risk_free_rates_x[global_index];
        shared_data[local_index + 9 * blockDim.x] = risk_free_rates_y[global_index];
    }

    __syncthreads(); // Ensure all threads have loaded data into shared memory

    if (tid < num_options) {
        // Load data from shared memory
        float S_x = shared_data[local_index];
        float S_y = shared_data[local_index + blockDim.x];
        float K_x = shared_data[local_index + 2 * blockDim.x];
        float K_y = shared_data[local_index + 3 * blockDim.x];
        float sigma_x = shared_data[local_index + 4 * blockDim.x];
        float sigma_y = shared_data[local_index + 5 * blockDim.x];
        float T_x = shared_data[local_index + 6 * blockDim.x];
        float T_y = shared_data[local_index + 7 * blockDim.x];
        float r_x = shared_data[local_index + 8 * blockDim.x];
        float r_y = shared_data[local_index + 9 * blockDim.x];

        // Calculate coefficients
        float sqrt_T_x = sqrtf(T_x);
        float d1_x = (logf(S_x / K_x) + (r_x + 0.5f * sigma_x * sigma_x) * T_x) / (sigma_x * sqrt_T_x);
        float d2_x = d1_x - sigma_x * sqrt_T_x;

        // Compute option price, delta, and gamma
        float Nd1_x = cumulative_normal_distribution(d1_x);
        float Nd2_x = cumulative_normal_distribution(d2_x);
        float exp_minus_r_T_x = expf(-r_x * T_x);
        float sqrt_T_S_sigma_x = S_x * sigma_x * sqrt_T_x;
        option_prices[tid] = S_x * Nd1_x - K_x * exp_minus_r_T_x * Nd2_x;
        deltas[tid] = Nd1_x;
        gammas[tid] = exp_minus_r_T_x / sqrt_T_S_sigma_x * Nd1_x;

        // Process the second element if within range
        if (tid + blockDim.x < num_options) {
            // Load data for the second element
            float S_x_next = shared_data[local_index + blockDim.x];
            float S_y_next = shared_data[local_index + blockDim.x + blockDim.x];
            float K_x_next = shared_data[local_index + 2 * blockDim.x + blockDim.x];
            float K_y_next = shared_data[local_index + 3 * blockDim.x + blockDim.x];
            float sigma_x_next = shared_data[local_index + 4 * blockDim.x + blockDim.x];
            float sigma_y_next = shared_data[local_index + 5 * blockDim.x + blockDim.x];
            float T_x_next = shared_data[local_index + 6 * blockDim.x + blockDim.x];
            float T_y_next = shared_data[local_index + 7 * blockDim.x + blockDim.x];
            float r_x_next = shared_data[local_index + 8 * blockDim.x + blockDim.x];
            float r_y_next = shared_data[local_index + 9 * blockDim.x + blockDim.x];

            // Calculate coefficients for the second element
            float sqrt_T_x_next = sqrtf(T_x_next);
            float d1_x_next = (logf(S_x_next / K_x_next) + (r_x_next + 0.5f * sigma_x_next * sigma_x_next) * T_x_next) / (sigma_x_next * sqrt_T_x_next);
            float d2_x_next = d1_x_next - sigma_x_next * sqrt_T_x_next;

            // Compute option price, delta, and gamma for the second element
            float Nd1_x_next = cumulative_normal_distribution(d1_x_next);
            float Nd2_x_next = cumulative_normal_distribution(d2_x_next);
            float exp_minus_r_T_x_next = expf(-r_x_next * T_x_next);
            float sqrt_T_S_sigma_x_next = S_x_next * sigma_x_next * sqrt_T_x_next;
            option_prices[tid + blockDim.x] = S_x_next * Nd1_x_next - K_x_next * exp_minus_r_T_x_next * Nd2_x_next;
            deltas[tid + blockDim.x] = Nd1_x_next;
            gammas[tid + blockDim.x] = exp_minus_r_T_x_next / sqrt_T_S_sigma_x_next * Nd1_x_next;
        }
    }
}

#endif // OPTION_PRICING_CU
