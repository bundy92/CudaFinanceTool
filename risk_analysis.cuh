#ifndef RISK_ANALYSIS_H
#define RISK_ANALYSIS_H

#pragma once

#ifdef KERNEL_EXPORTS
#define KERNEL_API __declspec(dllexport)
#else
#define KERNEL_API __declspec(dllimport)
#endif

#include <curand_kernel.h>

// Function prototype for the Value at Risk (VaR) calculation kernel
extern __device__ double cumulative_normal_distribution(double x);
extern __global__ void calculate_var(double* option_prices, double* var_values,
    int num_options, double confidence_level);

#endif // RISK_ANALYSIS_H
