#ifndef OPTION_PRICING_CU
#define OPTION_PRICING_CU

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118
#endif

#include "option_pricing.cuh"

// Function to calculate cumulative normal distribution
__device__ double cumulative_normal_distribution(double x) {
    return 0.5 * erfc(-x * M_SQRT1_2);
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
__global__ void black_scholes_option_pricing(double* option_prices, double* deltas, double* gammas,
    const double* stock_prices_x, const double* stock_prices_y,
    const double* strike_prices_x, const double* strike_prices_y,
    const double* volatilities_x, const double* volatilities_y,
    const double* time_to_maturity_x, const double* time_to_maturity_y,
    const double* risk_free_rates_x, const double* risk_free_rates_y,
    const int num_options) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x * 2;
    if (tid < num_options) {
        // Load data for the first element of the vector
        double S_x = stock_prices_x[tid];
        double S_y = stock_prices_y[tid];
        double K_x = strike_prices_x[tid];
        double K_y = strike_prices_y[tid];
        double sigma_x = volatilities_x[tid];
        double sigma_y = volatilities_y[tid];
        double T_x = time_to_maturity_x[tid];
        double T_y = time_to_maturity_y[tid];
        double r_x = risk_free_rates_x[tid];
        double r_y = risk_free_rates_y[tid];

        // Calculate coefficients for the first element
        double sqrt_T_x = sqrt(T_x);
        double d1_x = (log(S_x / K_x) + (r_x + 0.5 * sigma_x * sigma_x) * T_x) / (sigma_x * sqrt_T_x);
        double d2_x = d1_x - sigma_x * sqrt_T_x;

        // Compute option price, delta, and gamma for the first element
        double Nd1_x = cumulative_normal_distribution(d1_x);
        double Nd2_x = cumulative_normal_distribution(d2_x);
        double exp_minus_r_T_x = exp(-r_x * T_x);
        double sqrt_T_S_sigma_x = S_x * sigma_x * sqrt_T_x;
        option_prices[tid] = S_x * Nd1_x - K_x * exp_minus_r_T_x * Nd2_x;
        deltas[tid] = Nd1_x;
        gammas[tid] = exp_minus_r_T_x / sqrt_T_S_sigma_x * Nd1_x;

        // Check if there's a second element to process
        if (tid + blockDim.x < num_options) {
            // Load data for the second element of the vector
            double S_x_next = stock_prices_x[tid + blockDim.x];
            double S_y_next = stock_prices_y[tid + blockDim.x];
            double K_x_next = strike_prices_x[tid + blockDim.x];
            double K_y_next = strike_prices_y[tid + blockDim.x];
            double sigma_x_next = volatilities_x[tid + blockDim.x];
            double sigma_y_next = volatilities_y[tid + blockDim.x];
            double T_x_next = time_to_maturity_x[tid + blockDim.x];
            double T_y_next = time_to_maturity_y[tid + blockDim.x];
            double r_x_next = risk_free_rates_x[tid + blockDim.x];
            double r_y_next = risk_free_rates_y[tid + blockDim.x];

            // Calculate coefficients for the second element
            double sqrt_T_x_next = sqrt(T_x_next);
            double d1_x_next = (log(S_x_next / K_x_next) + (r_x_next + 0.5 * sigma_x_next * sigma_x_next) * T_x_next) / (sigma_x_next * sqrt_T_x_next);
            double d2_x_next = d1_x_next - sigma_x_next * sqrt_T_x_next;

            // Compute option price, delta, and gamma for the second element
            double Nd1_x_next = cumulative_normal_distribution(d1_x_next);
            double Nd2_x_next = cumulative_normal_distribution(d2_x_next);
            double exp_minus_r_T_x_next = exp(-r_x_next * T_x_next);
            double sqrt_T_S_sigma_x_next = S_x_next * sigma_x_next * sqrt_T_x_next;
            option_prices[tid + blockDim.x] = S_x_next * Nd1_x_next - K_x_next * exp_minus_r_T_x_next * Nd2_x_next;
            deltas[tid + blockDim.x] = Nd1_x_next;
            gammas[tid + blockDim.x] = exp_minus_r_T_x_next / sqrt_T_S_sigma_x_next * Nd1_x_next;
        }
    }
}

#endif // OPTION_PRICING_CU
