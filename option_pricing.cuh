#ifndef OPTION_PRICING_H
#define OPTION_PRICING_H

#include <curand_kernel.h>

// Function prototype to calculate cumulative normal distribution
extern __device__ float cumulative_normal_distribution(float x);

// Function prototype for the Black-Scholes option pricing kernel
extern __global__ void black_scholes_option_pricing(float* option_prices, float* deltas, float* gammas,
    const float* stock_prices_x, const float* stock_prices_y,
    const float* strike_prices_x, const float* strike_prices_y,
    const float* volatilities_x, const float* volatilities_y,
    const float* time_to_maturity_x, const float* time_to_maturity_y,
    const float* risk_free_rates_x, const float* risk_free_rates_y,
    const int num_options);

#endif // OPTION_PRICING_H
