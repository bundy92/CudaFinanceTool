#ifndef RISK_ANALYSIS_H
#define RISK_ANALYSIS_H


#include <curand_kernel.h>

// Function prototype for the Value at Risk (VaR) calculation kernel
//extern __device__ double cumulative_normal_distribution(double x);
extern __global__ void calculate_var(float* option_prices, float* var_values, const int num_options,
    const float confidence_level);

#endif // RISK_ANALYSIS_H
