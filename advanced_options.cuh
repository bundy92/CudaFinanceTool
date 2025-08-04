#ifndef ADVANCED_OPTIONS_H
#define ADVANCED_OPTIONS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "config.h"

// Option types enumeration
typedef enum {
    OPTION_TYPE_EUROPEAN_CALL = 0,
    OPTION_TYPE_EUROPEAN_PUT = 1,
    OPTION_TYPE_AMERICAN_CALL = 2,
    OPTION_TYPE_AMERICAN_PUT = 3,
    OPTION_TYPE_BARRIER_UP_AND_OUT = 4,
    OPTION_TYPE_BARRIER_DOWN_AND_OUT = 5,
    OPTION_TYPE_BARRIER_UP_AND_IN = 6,
    OPTION_TYPE_BARRIER_DOWN_AND_IN = 7,
    OPTION_TYPE_ASIAN_CALL = 8,
    OPTION_TYPE_ASIAN_PUT = 9,
    OPTION_TYPE_BASKET_CALL = 10,
    OPTION_TYPE_BASKET_PUT = 11
} option_type_t;

// American option pricing using binomial tree
__global__ void american_option_binomial_tree(
    float* option_prices, float* deltas, float* gammas,
    const float* stock_prices, const float* strike_prices,
    const float* volatilities, const float* time_to_maturity,
    const float* risk_free_rates, const int* option_types,
    const int num_options, const int num_steps);

// Barrier option pricing
__global__ void barrier_option_pricing(
    float* option_prices, float* deltas, float* gammas,
    const float* stock_prices, const float* strike_prices,
    const float* barrier_prices, const float* volatilities,
    const float* time_to_maturity, const float* risk_free_rates,
    const int* option_types, const int num_options);

// Asian option pricing using Monte Carlo
__global__ void asian_option_monte_carlo(
    float* option_prices, float* deltas, float* gammas,
    const float* stock_prices, const float* strike_prices,
    const float* volatilities, const float* time_to_maturity,
    const float* risk_free_rates, const int* option_types,
    const int num_options, const int num_simulations,
    const int num_time_steps);

// Basket option pricing
__global__ void basket_option_pricing(
    float* option_prices, float* deltas, float* gammas,
    const float* stock_prices, const float* strike_prices,
    const float* volatilities, const float* time_to_maturity,
    const float* risk_free_rates, const float* correlations,
    const int* option_types, const int num_options,
    const int num_assets);

// Helper functions
__device__ float calculate_early_exercise_value(float stock_price, float strike_price, 
                                              float time_to_maturity, float risk_free_rate,
                                              int option_type);
__device__ bool check_barrier_condition(float stock_price, float barrier_price, 
                                       int barrier_type);
__device__ float calculate_asian_average(float* price_path, int num_steps);
__device__ float calculate_basket_value(float* stock_prices, float* weights, 
                                       int num_assets);

// Option type validation
__host__ bool is_valid_option_type(int option_type);
__host__ const char* get_option_type_name(int option_type);

// Parameter validation for advanced options
__host__ void validate_barrier_parameters(float stock_price, float barrier_price, 
                                        float strike_price, int barrier_type);
__host__ void validate_basket_parameters(float* stock_prices, float* weights, 
                                       int num_assets, float* correlations);

#endif // ADVANCED_OPTIONS_H 