import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import ctypes
# Load the CUDA DLL
cuda_lib = ctypes.cdll.LoadLibrary('./CudaFinanceTool.dll')

from CudaFinanceTool import option_pricing, monte_carlo_simulation, risk_analysis

# Sample data
num_options = 2048
mem = 2048
confidence_level = 0.95
num_simulations = 10000
block_size = 256
num_blocks = (num_options + block_size - 1) // block_size

# Create sample data arrays
sample_stock_prices_x = np.arange(100.0, 100.0 + num_options)
sample_stock_prices_y = np.arange(100.0, 100.0 + num_options)
sample_strike_prices_x = np.arange(105.0, 105.0 + num_options)
sample_strike_prices_y = np.arange(105.0, 105.0 + num_options)
sample_volatilities_x = np.full(num_options, 0.2)
sample_volatilities_y = np.full(num_options, 0.2)
sample_time_to_maturity_x = np.full(num_options, 1.0)
sample_time_to_maturity_y = np.full(num_options, 1.0)
sample_risk_free_rates_x = np.full(num_options, 0.05)
sample_risk_free_rates_y = np.full(num_options, 0.05)

# Call option pricing function
option_prices, deltas, gammas = cuda_lib.option_pricing(sample_stock_prices_x, sample_stock_prices_y,
                                               sample_strike_prices_x, sample_strike_prices_y,
                                               sample_volatilities_x, sample_volatilities_y,
                                               sample_time_to_maturity_x, sample_time_to_maturity_y,
                                               sample_risk_free_rates_x, sample_risk_free_rates_y,
                                               num_options, block_size, num_blocks)

# Call Monte Carlo simulation function
monte_carlo_results = cuda_lib.monte_carlo_simulation(sample_stock_prices_x, sample_volatilities_x,
                                              sample_time_to_maturity_x, sample_risk_free_rates_x,
                                              sample_strike_prices_x, num_options, num_simulations,
                                              block_size, num_blocks)

# Call risk analysis function
var_values = cuda_lib.risk_analysis(monte_carlo_results, num_options, confidence_level, block_size, num_blocks)

# Print results
print("Option Pricing Results:")
print("%-10s %-15s %-15s %-15s" % ("Option", "Price", "Delta", "Gamma"))
for i in range(10):
    print("%-10d %-15.4f %-15.4f %-15.4f" % (i + 1, option_prices[i], deltas[i], gammas[i]))

mean_option_price = np.mean(monte_carlo_results)
print("\nMonte Carlo Simulation Results:")
print("Mean Option Price:", mean_option_price)

print("\nValue at Risk (VaR) Results:")
print("Confidence Level:", confidence_level)
print("Value at Risk (VaR):", var_values[0])
