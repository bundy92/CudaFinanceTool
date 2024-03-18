# CUDA Option Pricing and Risk Analysis

This project demonstrates how to use CUDA (Compute Unified Device Architecture) for option pricing and risk analysis. It includes CUDA kernels for option pricing, Monte Carlo simulation, and risk analysis.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Option pricing is a fundamental problem in finance, and Monte Carlo simulation is a common technique used to estimate the value of financial derivatives. This project implements CUDA kernels for performing option pricing and Monte Carlo simulation in parallel on a GPU (Graphics Processing Unit). Additionally, it includes a CUDA kernel for calculating the Value at Risk (VaR) for a portfolio of options.

## Features

- Option pricing using the Black-Scholes model
- Monte Carlo simulation for estimating option prices
- Calculation of Value at Risk (VaR) for a portfolio of options
- Efficient parallel computation using CUDA

## Dependencies

- CUDA Toolkit: This project requires the CUDA Toolkit to be installed on your system. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/bundy92/cuda-option-pricing.git
    ```

2. Install the CUDA Toolkit following the instructions provided by NVIDIA.

3. Build the project using a compatible C++ compiler.

## Usage

1. Compile the CUDA kernels using nvcc (NVIDIA CUDA Compiler):

    ```bash
    nvcc -o option_pricing option_pricing.cu
    nvcc -o monte_carlo monte_carlo.cu
    nvcc -o risk_analysis risk_analysis.cu
    ```

2. Run the executable:

    ```bash
    ./option_pricing
    ```

3. Follow the on-screen instructions to input parameters and view the results.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, please open an issue or submit a pull request.

## Contributors

- [Thomas Bundy](https://github.com/bundy92) - Electrical Engineer

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
