#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#pragma once

#ifdef KERNEL_EXPORTS
#define KERNEL_API __declspec(dllexport)
#else
#define KERNEL_API __declspec(dllimport)
#endif

#include <curand_kernel.h>

// Function prototype for the Monte Carlo simulation kernel
extern __device__ double cumulative_normal_distribution(double x);
extern __global__ void monte_carlo_option_pricing(double* option_prices, const double* stock_prices,
    const double* volatilities, const double* time_to_maturity,
    const double* risk_free_rates, const double* strike_prices, const int num_options,
    const double num_simulations);

#endif // MONTE_CARLO_H
