#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <curand_kernel.h>

// Function prototype for the Monte Carlo simulation kernel
//extern __device__ double cumulative_normal_distribution(double x);
extern __global__ void monte_carlo_option_pricing(float* option_prices, const float* stock_prices,
    const float* volatilities, const float* time_to_maturity,
    const float* risk_free_rates, const float* strike_prices, const int num_options,
    const int num_simulations);

#endif // MONTE_CARLO_H
