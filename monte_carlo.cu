#ifndef MONTE_CARLO_CU
#define MONTE_CARLO_CU

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118
#endif

#include "monte_carlo.cuh"

// Monte Carlo simulation kernel implementation
/*
 * Kernel Description:
 * -------------------
 * This CUDA kernel implements the Monte Carlo simulation for option pricing.
 * It simulates multiple paths of the stock price using the geometric Brownian motion model and calculates the option price based on the average payoff over these paths.
 * The simulation is performed in parallel for multiple options using CUDA threads.
 *
 * Input Parameters:
 * -----------------
 * - option_prices: Array to store the resulting option prices.
 * - stock_prices: Array of current stock prices for each option.
 * - volatilities: Array of volatilities for each option.
 * - time_to_maturity: Array of time to maturity for each option.
 * - risk_free_rates: Array of risk-free rates for each option.
 * - strike_prices: Array of strike prices for each option.
 * - num_options: Total number of options to process.
 * - num_simulations: Number of Monte Carlo simulations to run for each option.
 *
 * Thread Organization:
 * --------------------
 * Each CUDA thread is responsible for processing one option.
 * For each option, it initializes a random number generator, simulates multiple paths of the stock price using the geometric Brownian motion model, calculates the option payoff for each path, and computes the average payoff over all paths.
 * The final option price is computed as the discounted average payoff over all simulations.
 */

// Monte Carlo simulation kernel implementation
__global__ void monte_carlo_option_pricing(float* option_prices, const float* stock_prices,
    const float* volatilities, const float* time_to_maturity,
    const float* risk_free_rates, const float* strike_prices, const int num_options,
    const int num_simulations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_options) {
        curandState state;
        curand_init(tid, 0, 0, &state);

        float S = stock_prices[tid];
        float sigma = volatilities[tid];
        float T = time_to_maturity[tid];
        float r = risk_free_rates[tid];
        float K = strike_prices[tid];

        float dt = T / num_simulations;
        float sum_payoffs = 0.0f;

        for (int i = 0; i < num_simulations; i++) {
            float Z = curand_normal(&state);
            S *= expf((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * Z);
            float payoff = fmaxf(S - K, 0.0f);
            sum_payoffs += payoff;
        }

        option_prices[tid] = expf(-r * T) * sum_payoffs / num_simulations;
    }
}

#endif // MONTE_CARLO_CU