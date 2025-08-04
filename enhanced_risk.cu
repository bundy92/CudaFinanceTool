#include "enhanced_risk.cuh"
#include "error_handling.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function implementations
__device__ float calculate_portfolio_value(const portfolio_t* portfolio) {
    float total_value = 0.0f;
    for (int i = 0; i < portfolio->num_positions; i++) {
        total_value += portfolio->option_prices[i] * portfolio->weights[i];
    }
    return total_value;
}

__device__ float calculate_portfolio_delta(const portfolio_t* portfolio) {
    float total_delta = 0.0f;
    for (int i = 0; i < portfolio->num_positions; i++) {
        total_delta += portfolio->deltas[i] * portfolio->weights[i];
    }
    return total_delta;
}

__device__ float calculate_portfolio_gamma(const portfolio_t* portfolio) {
    float total_gamma = 0.0f;
    for (int i = 0; i < portfolio->num_positions; i++) {
        total_gamma += portfolio->gammas[i] * portfolio->weights[i];
    }
    return total_gamma;
}

__device__ float calculate_portfolio_vega(const portfolio_t* portfolio) {
    float total_vega = 0.0f;
    for (int i = 0; i < portfolio->num_positions; i++) {
        total_vega += portfolio->vegas[i] * portfolio->weights[i];
    }
    return total_vega;
}

__device__ float calculate_portfolio_theta(const portfolio_t* portfolio) {
    float total_theta = 0.0f;
    for (int i = 0; i < portfolio->num_positions; i++) {
        total_theta += portfolio->thetas[i] * portfolio->weights[i];
    }
    return total_theta;
}

// CVaR calculation kernel
__global__ void calculate_cvar(
    float* cvar_values, const float* portfolio_values,
    const float* confidence_levels, const int num_portfolios,
    const int num_simulations) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_portfolios) return;
    
    float confidence_level = confidence_levels[tid];
    int var_index = (int)((1.0f - confidence_level) * num_simulations);
    
    // Sort portfolio values to find VaR threshold
    extern __shared__ float shared_values[];
    float* local_values = shared_values + threadIdx.x * num_simulations;
    
    // Copy portfolio values to shared memory
    for (int i = 0; i < num_simulations; i++) {
        local_values[i] = portfolio_values[tid * num_simulations + i];
    }
    
    // Simple bubble sort (in practice, use more efficient sorting)
    for (int i = 0; i < num_simulations - 1; i++) {
        for (int j = 0; j < num_simulations - i - 1; j++) {
            if (local_values[j] > local_values[j + 1]) {
                float temp = local_values[j];
                local_values[j] = local_values[j + 1];
                local_values[j + 1] = temp;
            }
        }
    }
    
    // Calculate CVaR (average of losses beyond VaR)
    float var_threshold = local_values[var_index];
    float cvar_sum = 0.0f;
    int count = 0;
    
    for (int i = 0; i < var_index; i++) {
        if (local_values[i] < var_threshold) {
            cvar_sum += local_values[i];
            count++;
        }
    }
    
    cvar_values[tid] = (count > 0) ? cvar_sum / (float)count : 0.0f;
}

// Stress testing kernel
__global__ void stress_test_portfolio(
    float* stress_results, const float* option_prices,
    const float* stock_prices, const float* volatilities,
    const float* risk_free_rates, const int* stress_scenarios,
    const int num_options, const int num_scenarios) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_options) return;
    
    int scenario = stress_scenarios[tid % num_scenarios];
    float original_price = option_prices[tid];
    float stressed_price = original_price;
    
    // Apply stress scenario
    switch (scenario) {
        case STRESS_SCENARIO_MARKET_CRASH:
            // Stock price drops by 20%
            stressed_price *= 0.8f;
            break;
            
        case STRESS_SCENARIO_VOLATILITY_SPIKE:
            // Volatility increases by 50%
            stressed_price *= 1.2f; // Simplified impact
            break;
            
        case STRESS_SCENARIO_INTEREST_RATE_SHOCK:
            // Interest rate increases by 200 basis points
            stressed_price *= 0.95f; // Simplified impact
            break;
            
        case STRESS_SCENARIO_CORRELATION_BREAKDOWN:
            // Correlation breakdown (simplified)
            stressed_price *= 1.1f;
            break;
            
        case STRESS_SCENARIO_LIQUIDITY_CRISIS:
            // Liquidity crisis (simplified)
            stressed_price *= 0.9f;
            break;
    }
    
    stress_results[tid] = stressed_price - original_price;
}

// Scenario analysis kernel
__global__ void scenario_analysis(
    float* scenario_results, const float* option_prices,
    const float* stock_prices, const float* volatilities,
    const float* risk_free_rates, const float* scenario_shocks,
    const int num_options, const int num_scenarios) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_options) return;
    
    float original_price = option_prices[tid];
    float total_impact = 0.0f;
    
    // Apply multiple scenario shocks
    for (int scenario = 0; scenario < num_scenarios; scenario++) {
        float shock = scenario_shocks[scenario];
        float scenario_impact = original_price * shock;
        total_impact += scenario_impact;
    }
    
    scenario_results[tid] = total_impact;
}

// Portfolio risk calculation
__global__ void calculate_portfolio_risk(
    float* portfolio_var, float* portfolio_cvar,
    const portfolio_t* portfolios, const float* correlations,
    const int num_portfolios, const float confidence_level) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_portfolios) return;
    
    portfolio_t portfolio = portfolios[tid];
    float portfolio_value = calculate_portfolio_value(&portfolio);
    float portfolio_delta = calculate_portfolio_delta(&portfolio);
    float portfolio_gamma = calculate_portfolio_gamma(&portfolio);
    
    // Simplified risk calculation using Greeks
    float var = portfolio_value * 0.02f; // 2% daily VaR (simplified)
    float cvar = var * 1.5f; // CVaR typically 1.5x VaR
    
    // Adjust for portfolio Greeks
    var += portfolio_delta * 0.01f; // Delta adjustment
    var += portfolio_gamma * 0.0001f; // Gamma adjustment
    
    portfolio_var[tid] = var;
    portfolio_cvar[tid] = cvar;
}

// Correlation matrix calculation
__global__ void calculate_correlation_matrix(
    float* correlation_matrix, const float* returns,
    const int num_assets, const int num_periods) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_pairs = num_assets * num_assets;
    if (tid >= total_pairs) return;
    
    int i = tid / num_assets;
    int j = tid % num_assets;
    
    if (i == j) {
        correlation_matrix[tid] = 1.0f;
        return;
    }
    
    // Calculate correlation between assets i and j
    float sum_i = 0.0f, sum_j = 0.0f, sum_ij = 0.0f;
    float sum_i_sq = 0.0f, sum_j_sq = 0.0f;
    
    for (int period = 0; period < num_periods; period++) {
        float ret_i = returns[period * num_assets + i];
        float ret_j = returns[period * num_assets + j];
        
        sum_i += ret_i;
        sum_j += ret_j;
        sum_ij += ret_i * ret_j;
        sum_i_sq += ret_i * ret_i;
        sum_j_sq += ret_j * ret_j;
    }
    
    float mean_i = sum_i / (float)num_periods;
    float mean_j = sum_j / (float)num_periods;
    
    float var_i = sum_i_sq / (float)num_periods - mean_i * mean_i;
    float var_j = sum_j_sq / (float)num_periods - mean_j * mean_j;
    float cov_ij = sum_ij / (float)num_periods - mean_i * mean_j;
    
    float correlation = cov_ij / sqrtf(var_i * var_j);
    correlation_matrix[tid] = correlation;
}

// Volatility surface calculation
__global__ void calculate_volatility_surface(
    float* volatility_surface, const float* option_prices,
    const float* stock_prices, const float* strike_prices,
    const float* time_to_maturity, const int num_options,
    const int num_strikes, const int num_maturities) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_strikes * num_maturities) return;
    
    int strike_idx = tid / num_maturities;
    int maturity_idx = tid % num_maturities;
    
    // Find options with matching strike and maturity
    float total_volatility = 0.0f;
    int count = 0;
    
    for (int opt = 0; opt < num_options; opt++) {
        int strike_bucket = (int)(strike_prices[opt] / 10.0f); // Simplified bucketing
        int maturity_bucket = (int)(time_to_maturity[opt] * 12.0f); // Monthly buckets
        
        if (strike_bucket == strike_idx && maturity_bucket == maturity_idx) {
            // Calculate implied volatility from option price
            float S = stock_prices[opt];
            float K = strike_prices[opt];
            float T = time_to_maturity[opt];
            float C = option_prices[opt];
            
            // Simplified implied volatility calculation
            float moneyness = logf(S / K);
            float time_factor = sqrtf(T);
            float implied_vol = sqrtf(-2.0f * logf(C / S) / (moneyness * moneyness + 2.0f * time_factor));
            
            total_volatility += implied_vol;
            count++;
        }
    }
    
    volatility_surface[tid] = (count > 0) ? total_volatility / (float)count : 0.0f;
}

// Greeks calculation for all options
__global__ void calculate_all_greeks(
    float* deltas, float* gammas, float* vegas, float* thetas,
    const float* option_prices, const float* stock_prices,
    const float* strike_prices, const float* volatilities,
    const float* time_to_maturity, const float* risk_free_rates,
    const int num_options) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_options) return;
    
    float S = stock_prices[tid];
    float K = strike_prices[tid];
    float sigma = volatilities[tid];
    float T = time_to_maturity[tid];
    float r = risk_free_rates[tid];
    
    float sqrt_T = sqrtf(T);
    float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt_T);
    float d2 = d1 - sigma * sqrt_T;
    
    // Delta
    deltas[tid] = 0.5f * (1.0f + erff(d1 / sqrtf(2.0f)));
    
    // Gamma
    gammas[tid] = expf(-d1 * d1 / 2.0f) / (S * sigma * sqrt_T * sqrtf(2.0f * M_PI));
    
    // Vega
    vegas[tid] = S * sqrt_T * expf(-d1 * d1 / 2.0f) / sqrtf(2.0f * M_PI);
    
    // Theta
    thetas[tid] = -(S * sigma * expf(-d1 * d1 / 2.0f)) / (2.0f * sqrt_T * sqrtf(2.0f * M_PI)) 
                   - r * K * expf(-r * T) * (0.5f * (1.0f + erff(d2 / sqrtf(2.0f))));
}

// Monte Carlo VaR with correlation
__global__ void monte_carlo_var_correlated(
    float* var_values, const float* portfolio_values,
    const float* correlations, const float confidence_level,
    const int num_portfolios, const int num_simulations) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_portfolios) return;
    
    // Initialize random number generator
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    
    float sum_losses = 0.0f;
    int var_index = (int)((1.0f - confidence_level) * num_simulations);
    
    // Generate correlated random returns
    for (int sim = 0; sim < num_simulations; sim++) {
        float correlated_return = 0.0f;
        
        // Apply correlation matrix to random returns
        for (int asset = 0; asset < num_portfolios; asset++) {
            float random_normal = curand_normal(&state);
            float correlation = correlations[tid * num_portfolios + asset];
            correlated_return += random_normal * correlation;
        }
        
        float portfolio_loss = portfolio_values[tid] * correlated_return;
        sum_losses += portfolio_loss;
    }
    
    // Calculate VaR
    var_values[tid] = sum_losses / (float)num_simulations;
}

// Historical simulation VaR
__global__ void historical_var(
    float* var_values, const float* historical_returns,
    const float* portfolio_weights, const float confidence_level,
    const int num_portfolios, const int num_historical_periods) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_portfolios) return;
    
    // Calculate portfolio returns using historical data
    float* portfolio_returns = (float*)malloc(num_historical_periods * sizeof(float));
    
    for (int period = 0; period < num_historical_periods; period++) {
        portfolio_returns[period] = 0.0f;
        for (int asset = 0; asset < num_portfolios; asset++) {
            portfolio_returns[period] += historical_returns[period * num_portfolios + asset] 
                                       * portfolio_weights[asset];
        }
    }
    
    // Sort returns to find VaR
    int var_index = (int)((1.0f - confidence_level) * num_historical_periods);
    
    // Simple sorting (in practice, use more efficient sorting)
    for (int i = 0; i < num_historical_periods - 1; i++) {
        for (int j = 0; j < num_historical_periods - i - 1; j++) {
            if (portfolio_returns[j] > portfolio_returns[j + 1]) {
                float temp = portfolio_returns[j];
                portfolio_returns[j] = portfolio_returns[j + 1];
                portfolio_returns[j + 1] = temp;
            }
        }
    }
    
    var_values[tid] = portfolio_returns[var_index];
    
    free(portfolio_returns);
}

// Host function implementations
__host__ void validate_risk_parameters(float confidence_level, int num_simulations) {
    VALIDATE_CONFIDENCE_LEVEL(confidence_level);
    
    if (num_simulations <= 0 || num_simulations > 1000000) {
        fprintf(stderr, "Invalid number of simulations: %d\n", num_simulations);
        exit(EXIT_FAILURE);
    }
}

__host__ void validate_stress_scenario(int scenario_type, float shock_magnitude) {
    if (scenario_type < STRESS_SCENARIO_MARKET_CRASH || 
        scenario_type > STRESS_SCENARIO_LIQUIDITY_CRISIS) {
        fprintf(stderr, "Invalid stress scenario type: %d\n", scenario_type);
        exit(EXIT_FAILURE);
    }
    
    if (shock_magnitude < -1.0f || shock_magnitude > 1.0f) {
        fprintf(stderr, "Invalid shock magnitude: %f\n", shock_magnitude);
        exit(EXIT_FAILURE);
    }
}

__host__ void validate_portfolio_weights(float* weights, int num_positions) {
    if (num_positions <= 0) {
        fprintf(stderr, "Invalid number of positions: %d\n", num_positions);
        exit(EXIT_FAILURE);
    }
    
    float weight_sum = 0.0f;
    for (int i = 0; i < num_positions; i++) {
        if (weights[i] < 0.0f) {
            fprintf(stderr, "Invalid weight: %f\n", weights[i]);
            exit(EXIT_FAILURE);
        }
        weight_sum += weights[i];
    }
    
    if (fabsf(weight_sum - 1.0f) > ERROR_TOLERANCE) {
        fprintf(stderr, "Weights must sum to 1.0, got: %f\n", weight_sum);
        exit(EXIT_FAILURE);
    }
}

__host__ void print_risk_report(float var, float cvar, float* stress_results, 
                               int num_scenarios, const char* portfolio_name) {
    printf("\n=== Risk Report for %s ===\n", portfolio_name);
    printf("Value at Risk (VaR): %.4f\n", var);
    printf("Conditional VaR (CVaR): %.4f\n", cvar);
    printf("Stress Test Results:\n");
    
    for (int i = 0; i < num_scenarios; i++) {
        printf("  Scenario %d: %.4f\n", i, stress_results[i]);
    }
    printf("========================\n");
}

__host__ void export_risk_data(const char* filename, float* risk_data, 
                              int num_portfolios, int num_measures) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    fprintf(file, "Portfolio,VaR,CVaR,Stress_Test,Scenario_Analysis\n");
    
    // Write data
    for (int i = 0; i < num_portfolios; i++) {
        fprintf(file, "%d", i);
        for (int j = 0; j < num_measures; j++) {
            fprintf(file, ",%.6f", risk_data[i * num_measures + j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    printf("Risk data exported to %s\n", filename);
} 