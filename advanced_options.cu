#include "advanced_options.cuh"
#include "error_handling.cuh"
#include <math.h>
#include <stdio.h>

// Helper function implementations
__device__ float calculate_early_exercise_value(float stock_price, float strike_price, 
                                              float time_to_maturity, float risk_free_rate,
                                              int option_type) {
    float intrinsic_value = 0.0f;
    
    if (option_type == OPTION_TYPE_AMERICAN_CALL) {
        intrinsic_value = fmaxf(stock_price - strike_price, 0.0f);
    } else if (option_type == OPTION_TYPE_AMERICAN_PUT) {
        intrinsic_value = fmaxf(strike_price - stock_price, 0.0f);
    }
    
    // Discount the intrinsic value
    return intrinsic_value * expf(-risk_free_rate * time_to_maturity);
}

__device__ bool check_barrier_condition(float stock_price, float barrier_price, 
                                       int barrier_type) {
    switch (barrier_type) {
        case OPTION_TYPE_BARRIER_UP_AND_OUT:
        case OPTION_TYPE_BARRIER_UP_AND_IN:
            return stock_price >= barrier_price;
        case OPTION_TYPE_BARRIER_DOWN_AND_OUT:
        case OPTION_TYPE_BARRIER_DOWN_AND_IN:
            return stock_price <= barrier_price;
        default:
            return false;
    }
}

__device__ float calculate_asian_average(float* price_path, int num_steps) {
    float sum = 0.0f;
    for (int i = 0; i < num_steps; i++) {
        sum += price_path[i];
    }
    return sum / (float)num_steps;
}

__device__ float calculate_basket_value(float* stock_prices, float* weights, 
                                       int num_assets) {
    float basket_value = 0.0f;
    for (int i = 0; i < num_assets; i++) {
        basket_value += stock_prices[i] * weights[i];
    }
    return basket_value;
}

// American option pricing using binomial tree
__global__ void american_option_binomial_tree(
    float* option_prices, float* deltas, float* gammas,
    const float* stock_prices, const float* strike_prices,
    const float* volatilities, const float* time_to_maturity,
    const float* risk_free_rates, const int* option_types,
    const int num_options, const int num_steps) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_options) return;
    
    // Load parameters
    float S = stock_prices[tid];
    float K = strike_prices[tid];
    float sigma = volatilities[tid];
    float T = time_to_maturity[tid];
    float r = risk_free_rates[tid];
    int option_type = option_types[tid];
    
    // Validate option type
    if (option_type != OPTION_TYPE_AMERICAN_CALL && option_type != OPTION_TYPE_AMERICAN_PUT) {
        return;
    }
    
    // Binomial tree parameters
    float dt = T / (float)num_steps;
    float u = expf(sigma * sqrtf(dt));
    float d = 1.0f / u;
    float p = (expf(r * dt) - d) / (u - d);
    
    // Allocate shared memory for the tree
    extern __shared__ float shared_tree[];
    float* tree = shared_tree;
    
    // Build the binomial tree
    for (int i = 0; i <= num_steps; i++) {
        for (int j = 0; j <= i; j++) {
            int index = i * (i + 1) / 2 + j;
            float stock_value = S * powf(u, i - j) * powf(d, j);
            
            if (i == num_steps) {
                // Terminal nodes
                if (option_type == OPTION_TYPE_AMERICAN_CALL) {
                    tree[index] = fmaxf(stock_value - K, 0.0f);
                } else {
                    tree[index] = fmaxf(K - stock_value, 0.0f);
                }
            }
        }
    }
    
    // Backward induction with early exercise
    for (int i = num_steps - 1; i >= 0; i--) {
        for (int j = 0; j <= i; j++) {
            int index = i * (i + 1) / 2 + j;
            int up_index = (i + 1) * (i + 2) / 2 + j;
            int down_index = (i + 1) * (i + 2) / 2 + j + 1;
            
            float stock_value = S * powf(u, i - j) * powf(d, j);
            float continuation_value = (p * tree[up_index] + (1.0f - p) * tree[down_index]) * expf(-r * dt);
            float intrinsic_value = calculate_early_exercise_value(stock_value, K, dt, r, option_type);
            
            tree[index] = fmaxf(continuation_value, intrinsic_value);
        }
    }
    
    // Store results
    option_prices[tid] = tree[0];
    
    // Calculate Delta (simplified)
    if (num_steps > 1) {
        float delta_up = tree[1];
        float delta_down = tree[2];
        deltas[tid] = (delta_up - delta_down) / (S * (u - d));
    } else {
        deltas[tid] = 0.0f;
    }
    
    // Calculate Gamma (simplified)
    if (num_steps > 2) {
        float gamma_up = tree[3];
        float gamma_mid = tree[4];
        float gamma_down = tree[5];
        gammas[tid] = (gamma_up - 2.0f * gamma_mid + gamma_down) / (S * S * (u - d) * (u - d));
    } else {
        gammas[tid] = 0.0f;
    }
}

// Barrier option pricing
__global__ void barrier_option_pricing(
    float* option_prices, float* deltas, float* gammas,
    const float* stock_prices, const float* strike_prices,
    const float* barrier_prices, const float* volatilities,
    const float* time_to_maturity, const float* risk_free_rates,
    const int* option_types, const int num_options) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_options) return;
    
    // Load parameters
    float S = stock_prices[tid];
    float K = strike_prices[tid];
    float B = barrier_prices[tid];
    float sigma = volatilities[tid];
    float T = time_to_maturity[tid];
    float r = risk_free_rates[tid];
    int option_type = option_types[tid];
    
    // Check if barrier is hit
    bool barrier_hit = check_barrier_condition(S, B, option_type);
    
    // Calculate option price based on barrier type
    float option_price = 0.0f;
    float delta = 0.0f;
    float gamma = 0.0f;
    
    if (option_type == OPTION_TYPE_BARRIER_UP_AND_OUT || 
        option_type == OPTION_TYPE_BARRIER_DOWN_AND_OUT) {
        // Knock-out options
        if (barrier_hit) {
            option_price = 0.0f;
        } else {
            // Use Black-Scholes with barrier adjustment
            float sqrt_T = sqrtf(T);
            float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt_T);
            float d2 = d1 - sigma * sqrt_T;
            
            float N_d1 = 0.5f * (1.0f + erff(d1 / sqrtf(2.0f)));
            float N_d2 = 0.5f * (1.0f + erff(d2 / sqrtf(2.0f)));
            
            if (option_type == OPTION_TYPE_BARRIER_UP_AND_OUT) {
                option_price = S * N_d1 - K * expf(-r * T) * N_d2;
            } else {
                option_price = K * expf(-r * T) * (1.0f - N_d2) - S * (1.0f - N_d1);
            }
        }
    } else if (option_type == OPTION_TYPE_BARRIER_UP_AND_IN || 
               option_type == OPTION_TYPE_BARRIER_DOWN_AND_IN) {
        // Knock-in options
        if (barrier_hit) {
            // Use Black-Scholes
            float sqrt_T = sqrtf(T);
            float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt_T);
            float d2 = d1 - sigma * sqrt_T;
            
            float N_d1 = 0.5f * (1.0f + erff(d1 / sqrtf(2.0f)));
            float N_d2 = 0.5f * (1.0f + erff(d2 / sqrtf(2.0f)));
            
            if (option_type == OPTION_TYPE_BARRIER_UP_AND_IN) {
                option_price = S * N_d1 - K * expf(-r * T) * N_d2;
            } else {
                option_price = K * expf(-r * T) * (1.0f - N_d2) - S * (1.0f - N_d1);
            }
        } else {
            option_price = 0.0f;
        }
    }
    
    // Store results
    option_prices[tid] = option_price;
    deltas[tid] = delta;
    gammas[tid] = gamma;
}

// Asian option pricing using Monte Carlo
__global__ void asian_option_monte_carlo(
    float* option_prices, float* deltas, float* gammas,
    const float* stock_prices, const float* strike_prices,
    const float* volatilities, const float* time_to_maturity,
    const float* risk_free_rates, const int* option_types,
    const int num_options, const int num_simulations,
    const int num_time_steps) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_options) return;
    
    // Load parameters
    float S = stock_prices[tid];
    float K = strike_prices[tid];
    float sigma = volatilities[tid];
    float T = time_to_maturity[tid];
    float r = risk_free_rates[tid];
    int option_type = option_types[tid];
    
    // Validate option type
    if (option_type != OPTION_TYPE_ASIAN_CALL && option_type != OPTION_TYPE_ASIAN_PUT) {
        return;
    }
    
    // Monte Carlo simulation
    float sum_payoffs = 0.0f;
    float dt = T / (float)num_time_steps;
    
    // Initialize random number generator
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    
    for (int sim = 0; sim < num_simulations; sim++) {
        float current_price = S;
        float price_path[256]; // Assuming max 256 time steps
        
        // Generate price path
        for (int step = 0; step < num_time_steps; step++) {
            float random_normal = curand_normal(&state);
            float drift = (r - 0.5f * sigma * sigma) * dt;
            float diffusion = sigma * sqrtf(dt) * random_normal;
            current_price *= expf(drift + diffusion);
            price_path[step] = current_price;
        }
        
        // Calculate Asian average
        float average_price = calculate_asian_average(price_path, num_time_steps);
        
        // Calculate payoff
        float payoff = 0.0f;
        if (option_type == OPTION_TYPE_ASIAN_CALL) {
            payoff = fmaxf(average_price - K, 0.0f);
        } else {
            payoff = fmaxf(K - average_price, 0.0f);
        }
        
        sum_payoffs += payoff;
    }
    
    // Calculate option price
    float option_price = (sum_payoffs / (float)num_simulations) * expf(-r * T);
    
    // Store results
    option_prices[tid] = option_price;
    deltas[tid] = 0.0f; // Simplified
    gammas[tid] = 0.0f; // Simplified
}

// Basket option pricing
__global__ void basket_option_pricing(
    float* option_prices, float* deltas, float* gammas,
    const float* stock_prices, const float* strike_prices,
    const float* volatilities, const float* time_to_maturity,
    const float* risk_free_rates, const float* correlations,
    const int* option_types, const int num_options,
    const int num_assets) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_options) return;
    
    // Load parameters for this option
    float K = strike_prices[tid];
    float T = time_to_maturity[tid];
    float r = risk_free_rates[tid];
    int option_type = option_types[tid];
    
    // Validate option type
    if (option_type != OPTION_TYPE_BASKET_CALL && option_type != OPTION_TYPE_BASKET_PUT) {
        return;
    }
    
    // Calculate basket value
    float basket_stocks[16]; // Assuming max 16 assets
    float weights[16];
    
    // Load stock prices and set equal weights
    for (int i = 0; i < num_assets; i++) {
        basket_stocks[i] = stock_prices[tid * num_assets + i];
        weights[i] = 1.0f / (float)num_assets;
    }
    
    float basket_value = calculate_basket_value(basket_stocks, weights, num_assets);
    
    // Calculate option price using Black-Scholes approximation
    float option_price = 0.0f;
    float delta = 0.0f;
    float gamma = 0.0f;
    
    // Simplified basket pricing (in practice, would need correlation matrix)
    float avg_volatility = 0.0f;
    for (int i = 0; i < num_assets; i++) {
        avg_volatility += volatilities[tid * num_assets + i];
    }
    avg_volatility /= (float)num_assets;
    
    float sqrt_T = sqrtf(T);
    float d1 = (logf(basket_value / K) + (r + 0.5f * avg_volatility * avg_volatility) * T) / (avg_volatility * sqrt_T);
    float d2 = d1 - avg_volatility * sqrt_T;
    
    float N_d1 = 0.5f * (1.0f + erff(d1 / sqrtf(2.0f)));
    float N_d2 = 0.5f * (1.0f + erff(d2 / sqrtf(2.0f)));
    
    if (option_type == OPTION_TYPE_BASKET_CALL) {
        option_price = basket_value * N_d1 - K * expf(-r * T) * N_d2;
        delta = N_d1;
    } else {
        option_price = K * expf(-r * T) * (1.0f - N_d2) - basket_value * (1.0f - N_d1);
        delta = N_d1 - 1.0f;
    }
    
    // Store results
    option_prices[tid] = option_price;
    deltas[tid] = delta;
    gammas[tid] = gamma;
}

// Host function implementations
__host__ bool is_valid_option_type(int option_type) {
    return option_type >= OPTION_TYPE_EUROPEAN_CALL && option_type <= OPTION_TYPE_BASKET_PUT;
}

__host__ const char* get_option_type_name(int option_type) {
    switch (option_type) {
        case OPTION_TYPE_EUROPEAN_CALL: return "European Call";
        case OPTION_TYPE_EUROPEAN_PUT: return "European Put";
        case OPTION_TYPE_AMERICAN_CALL: return "American Call";
        case OPTION_TYPE_AMERICAN_PUT: return "American Put";
        case OPTION_TYPE_BARRIER_UP_AND_OUT: return "Barrier Up-and-Out";
        case OPTION_TYPE_BARRIER_DOWN_AND_OUT: return "Barrier Down-and-Out";
        case OPTION_TYPE_BARRIER_UP_AND_IN: return "Barrier Up-and-In";
        case OPTION_TYPE_BARRIER_DOWN_AND_IN: return "Barrier Down-and-In";
        case OPTION_TYPE_ASIAN_CALL: return "Asian Call";
        case OPTION_TYPE_ASIAN_PUT: return "Asian Put";
        case OPTION_TYPE_BASKET_CALL: return "Basket Call";
        case OPTION_TYPE_BASKET_PUT: return "Basket Put";
        default: return "Unknown";
    }
}

__host__ void validate_barrier_parameters(float stock_price, float barrier_price, 
                                        float strike_price, int barrier_type) {
    VALIDATE_STOCK_PRICE(stock_price);
    VALIDATE_STRIKE_PRICE(strike_price);
    
    if (barrier_price <= 0.0f) {
        fprintf(stderr, "Invalid barrier price: %f\n", barrier_price);
        exit(EXIT_FAILURE);
    }
    
    // Check barrier logic
    if (barrier_type == OPTION_TYPE_BARRIER_UP_AND_OUT || 
        barrier_type == OPTION_TYPE_BARRIER_UP_AND_IN) {
        if (barrier_price <= stock_price) {
            fprintf(stderr, "Up barrier should be above current stock price\n");
            exit(EXIT_FAILURE);
        }
    } else if (barrier_type == OPTION_TYPE_BARRIER_DOWN_AND_OUT || 
               barrier_type == OPTION_TYPE_BARRIER_DOWN_AND_IN) {
        if (barrier_price >= stock_price) {
            fprintf(stderr, "Down barrier should be below current stock price\n");
            exit(EXIT_FAILURE);
        }
    }
}

__host__ void validate_basket_parameters(float* stock_prices, float* weights, 
                                       int num_assets, float* correlations) {
    if (num_assets <= 0 || num_assets > 16) {
        fprintf(stderr, "Invalid number of assets: %d\n", num_assets);
        exit(EXIT_FAILURE);
    }
    
    float weight_sum = 0.0f;
    for (int i = 0; i < num_assets; i++) {
        VALIDATE_STOCK_PRICE(stock_prices[i]);
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
    
    // Validate correlation matrix
    for (int i = 0; i < num_assets; i++) {
        for (int j = 0; j < num_assets; j++) {
            float corr = correlations[i * num_assets + j];
            if (corr < -1.0f || corr > 1.0f) {
                fprintf(stderr, "Invalid correlation: %f\n", corr);
                exit(EXIT_FAILURE);
            }
        }
    }
} 