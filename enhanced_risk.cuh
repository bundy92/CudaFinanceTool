#ifndef ENHANCED_RISK_H
#define ENHANCED_RISK_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "config.h"

// Risk measures
typedef enum {
    RISK_MEASURE_VAR = 0,
    RISK_MEASURE_CVAR = 1,
    RISK_MEASURE_STRESS = 2,
    RISK_MEASURE_SCENARIO = 3
} risk_measure_t;

// Stress test scenarios
typedef enum {
    STRESS_SCENARIO_MARKET_CRASH = 0,
    STRESS_SCENARIO_VOLATILITY_SPIKE = 1,
    STRESS_SCENARIO_INTEREST_RATE_SHOCK = 2,
    STRESS_SCENARIO_CORRELATION_BREAKDOWN = 3,
    STRESS_SCENARIO_LIQUIDITY_CRISIS = 4
} stress_scenario_t;

// Portfolio structure
typedef struct {
    float* option_prices;
    float* weights;
    float* deltas;
    float* gammas;
    float* vegas;
    float* thetas;
    int num_positions;
    float total_value;
} portfolio_t;

// CVaR calculation kernel
__global__ void calculate_cvar(
    float* cvar_values, const float* portfolio_values,
    const float* confidence_levels, const int num_portfolios,
    const int num_simulations);

// Stress testing kernel
__global__ void stress_test_portfolio(
    float* stress_results, const float* option_prices,
    const float* stock_prices, const float* volatilities,
    const float* risk_free_rates, const int* stress_scenarios,
    const int num_options, const int num_scenarios);

// Scenario analysis kernel
__global__ void scenario_analysis(
    float* scenario_results, const float* option_prices,
    const float* stock_prices, const float* volatilities,
    const float* risk_free_rates, const float* scenario_shocks,
    const int num_options, const int num_scenarios);

// Portfolio risk calculation
__global__ void calculate_portfolio_risk(
    float* portfolio_var, float* portfolio_cvar,
    const portfolio_t* portfolios, const float* correlations,
    const int num_portfolios, const float confidence_level);

// Correlation matrix calculation
__global__ void calculate_correlation_matrix(
    float* correlation_matrix, const float* returns,
    const int num_assets, const int num_periods);

// Volatility surface calculation
__global__ void calculate_volatility_surface(
    float* volatility_surface, const float* option_prices,
    const float* stock_prices, const float* strike_prices,
    const float* time_to_maturity, const int num_options,
    const int num_strikes, const int num_maturities);

// Greeks calculation for all options
__global__ void calculate_all_greeks(
    float* deltas, float* gammas, float* vegas, float* thetas,
    const float* option_prices, const float* stock_prices,
    const float* strike_prices, const float* volatilities,
    const float* time_to_maturity, const float* risk_free_rates,
    const int num_options);

// Monte Carlo VaR with correlation
__global__ void monte_carlo_var_correlated(
    float* var_values, const float* portfolio_values,
    const float* correlations, const float confidence_level,
    const int num_portfolios, const int num_simulations);

// Historical simulation VaR
__global__ void historical_var(
    float* var_values, const float* historical_returns,
    const float* portfolio_weights, const float confidence_level,
    const int num_portfolios, const int num_historical_periods);

// Helper functions
__device__ float calculate_portfolio_value(const portfolio_t* portfolio);
__device__ float calculate_portfolio_delta(const portfolio_t* portfolio);
__device__ float calculate_portfolio_gamma(const portfolio_t* portfolio);
__device__ float calculate_portfolio_vega(const portfolio_t* portfolio);
__device__ float calculate_portfolio_theta(const portfolio_t* portfolio);

// Risk measure validation
__host__ void validate_risk_parameters(float confidence_level, int num_simulations);
__host__ void validate_stress_scenario(int scenario_type, float shock_magnitude);
__host__ void validate_portfolio_weights(float* weights, int num_positions);

// Risk reporting functions
__host__ void print_risk_report(float var, float cvar, float* stress_results, 
                               int num_scenarios, const char* portfolio_name);
__host__ void export_risk_data(const char* filename, float* risk_data, 
                              int num_portfolios, int num_measures);

#endif // ENHANCED_RISK_H 