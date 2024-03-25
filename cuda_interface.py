import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import yfinance as yf

# CUDA Kernel code
cuda_kernel_code = """
#ifndef OPTION_PRICING_CU
#define OPTION_PRICING_CU

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118
#endif

#include <math.h>

// Function to calculate cumulative normal distribution
__device__ double cumulative_normal_distribution(double x) {
    return 0.5 * erfc(-x * M_SQRT1_2);
}

__global__ void black_scholes_option_pricing(double* option_prices, double* deltas, double* gammas,
    const double* stock_prices_x, const double* stock_prices_y,
    const double* strike_prices_x, const double* strike_prices_y,
    const double* volatilities_x, const double* volatilities_y,
    const double* time_to_maturity_x, const double* time_to_maturity_y,
    const double* risk_free_rates_x, const double* risk_free_rates_y,
    const int num_options) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x * 2;
    if (tid < num_options) {
        // Load data for the first element of the vector
        double S_x = stock_prices_x[tid];
        double S_y = stock_prices_y[tid];
        double K_x = strike_prices_x[tid];
        double K_y = strike_prices_y[tid];
        double sigma_x = volatilities_x[tid];
        double sigma_y = volatilities_y[tid];
        double T_x = time_to_maturity_x[tid];
        double T_y = time_to_maturity_y[tid];
        double r_x = risk_free_rates_x[tid];
        double r_y = risk_free_rates_y[tid];

        // Calculate coefficients for the first element
        double sqrt_T_x = sqrt(T_x);
        double d1_x = (log(S_x / K_x) + (r_x + 0.5 * sigma_x * sigma_x) * T_x) / (sigma_x * sqrt_T_x);
        double d2_x = d1_x - sigma_x * sqrt_T_x;

        // Compute option price, delta, and gamma for the first element
        double Nd1_x = cumulative_normal_distribution(d1_x);
        double Nd2_x = cumulative_normal_distribution(d2_x);
        double exp_minus_r_T_x = exp(-r_x * T_x);
        double sqrt_T_S_sigma_x = S_x * sigma_x * sqrt_T_x;
        option_prices[tid] = S_x * Nd1_x - K_x * exp_minus_r_T_x * Nd2_x;
        deltas[tid] = Nd1_x;
        gammas[tid] = exp_minus_r_T_x / sqrt_T_S_sigma_x * Nd1_x;

        // Check if there's a second element to process
        if (tid + blockDim.x < num_options) {
            // Load data for the second element of the vector
            double S_x_next = stock_prices_x[tid + blockDim.x];
            double S_y_next = stock_prices_y[tid + blockDim.x];
            double K_x_next = strike_prices_x[tid + blockDim.x];
            double K_y_next = strike_prices_y[tid + blockDim.x];
            double sigma_x_next = volatilities_x[tid + blockDim.x];
            double sigma_y_next = volatilities_y[tid + blockDim.x];
            double T_x_next = time_to_maturity_x[tid + blockDim.x];
            double T_y_next = time_to_maturity_y[tid + blockDim.x];
            double r_x_next = risk_free_rates_x[tid + blockDim.x];
            double r_y_next = risk_free_rates_y[tid + blockDim.x];

            // Calculate coefficients for the second element
            double sqrt_T_x_next = sqrt(T_x_next);
            double d1_x_next = (log(S_x_next / K_x_next) + (r_x_next + 0.5 * sigma_x_next * sigma_x_next) * T_x_next) / (sigma_x_next * sqrt_T_x_next);
            double d2_x_next = d1_x_next - sigma_x_next * sqrt_T_x_next;

            // Compute option price, delta, and gamma for the second element
            double Nd1_x_next = cumulative_normal_distribution(d1_x_next);
            double Nd2_x_next = cumulative_normal_distribution(d2_x_next);
            double exp_minus_r_T_x_next = exp(-r_x_next * T_x_next);
            double sqrt_T_S_sigma_x_next = S_x_next * sigma_x_next * sqrt_T_x_next;
            option_prices[tid + blockDim.x] = S_x_next * Nd1_x_next - K_x_next * exp_minus_r_T_x_next * Nd2_x_next;
            deltas[tid + blockDim.x] = Nd1_x_next;
            gammas[tid + blockDim.x] = exp_minus_r_T_x_next / sqrt_T_S_sigma_x_next * Nd1_x_next;
        }
    }
}

#endif // OPTION_PRICING_CU
"""
# Compile CUDA code
mod = SourceModule(cuda_kernel_code)

# Get the CUDA kernel function
cuda_kernel_func = mod.get_function("black_scholes_option_pricing")

# Function to calculate option prices using CUDA
def calculate_option_prices():
    # Acquire data from yfinance
    data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")

    # Extract relevant data for options pricing
    stock_prices_x = data['Adj Close'].values.astype(np.float64)  # Use Adj Close prices as stock prices
    strike_prices_x = stock_prices_x * 1.1  # Example: Strike price is 10% higher than stock price
    volatilities_x = np.random.rand(len(stock_prices_x)) * 0.5  # Example: Random volatilities
    time_to_maturity_x = np.random.randint(1, 365, len(stock_prices_x)) / 365  # Example: Random time to maturity
    risk_free_rates_x = np.full_like(stock_prices_x, 0.02)  # Example: Constant risk-free rate

    # Allocate device memory for input data
    d_stock_prices_x = cuda.mem_alloc(stock_prices_x.nbytes)
    d_strike_prices_x = cuda.mem_alloc(strike_prices_x.nbytes)
    d_volatilities_x = cuda.mem_alloc(volatilities_x.nbytes)
    d_time_to_maturity_x = cuda.mem_alloc(time_to_maturity_x.nbytes)
    d_risk_free_rates_x = cuda.mem_alloc(risk_free_rates_x.nbytes)

    # Copy data from host to device
    cuda.memcpy_htod(d_stock_prices_x, stock_prices_x)
    cuda.memcpy_htod(d_strike_prices_x, strike_prices_x)
    cuda.memcpy_htod(d_volatilities_x, volatilities_x)
    cuda.memcpy_htod(d_time_to_maturity_x, time_to_maturity_x)
    cuda.memcpy_htod(d_risk_free_rates_x, risk_free_rates_x)

    # Allocate device memory for output data
    option_prices = np.zeros_like(stock_prices_x)
    deltas = np.zeros_like(stock_prices_x)
    gammas = np.zeros_like(stock_prices_x)
    d_option_prices = cuda.mem_alloc(option_prices.nbytes)
    d_deltas = cuda.mem_alloc(deltas.nbytes)
    d_gammas = cuda.mem_alloc(gammas.nbytes)

    # Define block and grid dimensions
    block_size = 256
    num_blocks = (len(stock_prices_x) + block_size - 1) // block_size

    # Launch CUDA kernel
    cuda_kernel_func(d_option_prices, d_deltas, d_gammas,
                      d_stock_prices_x, d_stock_prices_x,
                      d_strike_prices_x, d_strike_prices_x,
                      d_volatilities_x, d_volatilities_x,
                      d_time_to_maturity_x, d_time_to_maturity_x,
                      d_risk_free_rates_x, d_risk_free_rates_x,
                      np.int32(len(stock_prices_x)),
                      block=(block_size, 1, 1), grid=(num_blocks, 1))

    # Copy results from device to host
    cuda.memcpy_dtoh(option_prices, d_option_prices)
    cuda.memcpy_dtoh(deltas, d_deltas)
    cuda.memcpy_dtoh(gammas, d_gammas)

    # Print or further process the results
    for i in range(len(stock_prices_x)):
        print(f"Option {i + 1}: Price: {option_prices[i]}, Delta: {deltas[i]}, Gamma: {gammas[i]}")

# Example usage
calculate_option_prices()