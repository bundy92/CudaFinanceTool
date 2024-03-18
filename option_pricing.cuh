#ifndef OPTION_PRICING_H
#define OPTION_PRICING_H

#pragma once

#ifdef KERNEL_EXPORTS
#define KERNEL_API __declspec(dllexport)
#else
#define KERNEL_API __declspec(dllimport)
#endif

#include <curand_kernel.h>

// Function prototype to calculate cumulative normal distribution
extern __device__ double cumulative_normal_distribution(double x);

// Function prototype for the Black-Scholes option pricing kernel
extern __global__ void black_scholes_option_pricing(double* option_prices, double* deltas, double* gammas,
    const double* stock_prices_x, const double* stock_prices_y,
    const double* strike_prices_x, const double* strike_prices_y,
    const double* volatilities_x, const double* volatilities_y,
    const double* time_to_maturity_x, const double* time_to_maturity_y,
    const double* risk_free_rates_x, const double* risk_free_rates_y,
    const int num_options);

#endif // OPTION_PRICING_H
