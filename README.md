# CUDA Option Pricing and Risk Analysis

This project demonstrates how to use CUDA (Compute Unified Device Architecture) for option pricing and risk analysis. It includes CUDA kernels for option pricing, Monte Carlo simulation, and risk analysis.

## Table of Contents

- [CUDA Option Pricing and Risk Analysis](#cuda-option-pricing-and-risk-analysis)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
    - [**Core Functionality**](#core-functionality)
    - [**Technical Features**](#technical-features)
    - [**Advanced Capabilities**](#advanced-capabilities)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Usage](#usage)
    - [**Command Line Interface**](#command-line-interface)
    - [**Web Interface**](#web-interface)
    - [**API Usage**](#api-usage)
    - [**Program Features**](#program-features)
  - [Contributing](#contributing)
  - [Contributors](#contributors)
  - [License](#license)

## Introduction

Option pricing is a fundamental problem in finance, and Monte Carlo simulation is a common technique used to estimate the value of financial derivatives. This project implements CUDA kernels for performing option pricing and Monte Carlo simulation in parallel on a GPU (Graphics Processing Unit). Additionally, it includes a CUDA kernel for calculating the Value at Risk (VaR) for a portfolio of options.

## Features

### **Core Functionality**
- **Option Pricing**: Black-Scholes model with Delta and Gamma calculations
- **Monte Carlo Simulation**: Parallel simulation for option price estimation
- **Risk Analysis**: Value at Risk (VaR) calculation for portfolios
- **Advanced Options**: American, Barrier, Asian, and Basket options
- **Enhanced Risk Models**: CVaR, stress testing, and scenario analysis

### **Technical Features**
- **Error Handling**: Comprehensive CUDA error checking and validation
- **Testing**: Complete test suite with accuracy and performance benchmarks
- **Build System**: Multiple build options (Makefile, CMake, manual)
- **Web Interface**: Modern Flask-based web application with REST API
- **Python Interface**: PyCUDA integration for easy scripting
- **Performance**: Optimized CUDA kernels with shared memory usage
- **Validation**: Financial parameter validation and bounds checking

### **Advanced Capabilities**
- **Multiple Option Types**: European, American, Barrier, Asian, Basket options
- **Portfolio Risk Management**: VaR, CVaR, stress testing, correlation analysis
- **Machine Learning Integration**: Volatility forecasting and pattern recognition
- **Real-time Market Data**: Multi-source data feeds with caching and aggregation
- **Real-time Processing**: Asynchronous job management and progress tracking
- **Data Export**: CSV and JSON output formats
- **GPU Memory Management**: Automatic memory allocation and cleanup

## Dependencies

- CUDA Toolkit: This project requires the CUDA Toolkit to be installed on your system. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/bundy92/cuda-option-pricing.git
    cd cuda-option-pricing
    ```

2. Install the CUDA Toolkit following the instructions provided by NVIDIA.

3. Install Python dependencies (optional, for Python interface):

    ```bash
    pip install -r requirements.txt
    ```

4. Build the project:

    **Using Makefile (recommended):**
    ```bash
    make setup    # Check CUDA installation and GPU compatibility
    make all      # Build the main executable
    make test     # Run the test suite
    ```

    **Using CMake:**
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

    **Manual compilation:**
    ```bash
    nvcc -o cuda_finance_tool main.cu option_pricing.cu monte_carlo.cu risk_analysis.cu utils.cu error_handling.cu
    ```

## Usage

### **Command Line Interface**

1. Run the main executable:

    ```bash
    ./bin/cuda_finance_tool
    ```

2. Run the test suite to verify functionality:

    ```bash
    ./bin/test_suite
    ```

3. For Python interface:

    ```bash
    python cuda_interface.py
    ```

### **Web Interface**

1. Install web dependencies:

    ```bash
    make install-web
    ```

2. Start the web server:

    ```bash
    make web
    ```

3. Open your browser and navigate to `http://localhost:5000`

### **API Usage**

The web interface provides REST API endpoints:

- `POST /api/pricing/quick` - Quick option pricing
- `POST /api/pricing` - Create batch pricing jobs
- `POST /api/risk/var` - Calculate VaR/CVaR
- `GET /api/jobs` - List active jobs
- `GET /api/system/status` - System and GPU status

### **Program Features**

The program will automatically:
- Initialize CUDA and check device compatibility
- Validate all financial parameters
- Perform option pricing using Black-Scholes model
- Run Monte Carlo simulations
- Calculate Value at Risk (VaR)
- Display results in a formatted table
- Support multiple option types and risk models

## Contributing

We welcome contributions to the CUDA Finance Tool! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `make test`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions
- Include type hints where appropriate
- Write comprehensive tests for new features

### Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=src --cov-report=html
```

### Documentation

- Update README.md for new features
- Add docstrings to new functions
- Update API documentation
- Include usage examples

## Contributors

- [Thomas Bundy](https://github.com/bundy92) - Electrical Engineer and initial development
- [Contributor Name](https://github.com/contributor) - Advanced options and risk models
- [Contributor Name](https://github.com/contributor) - Web interface and API development

## Acknowledgments

- NVIDIA for CUDA technology
- The financial engineering community for mathematical models
- Open source contributors for various libraries and tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [DEPLOYMENT.md](DEPLOYMENT.md) for deployment guides
- **Issues**: [GitHub Issues](https://github.com/bundy92/cuda-option-pricing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bundy92/cuda-option-pricing/discussions)
- **Wiki**: [Project Wiki](https://github.com/bundy92/cuda-option-pricing/wiki)

## Roadmap

### Upcoming Features

- **Machine Learning Integration**: Volatility forecasting and pattern recognition
- **Real-time Market Data**: Live data feeds and streaming analytics
- **Multi-GPU Support**: Distributed computing across multiple GPUs
- **Advanced Risk Models**: Credit risk, liquidity risk, and regulatory compliance
- **Mobile Application**: iOS and Android apps for mobile access
- **Cloud Deployment**: AWS, Azure, and GCP deployment templates

### Long-term Goals

- **AI-Powered Trading**: Automated trading strategies and portfolio optimization
- **Blockchain Integration**: DeFi protocols and smart contract pricing
- **Quantum Computing**: Quantum algorithms for option pricing
- **Global Expansion**: Multi-currency and multi-region support
- **Cloud-Native Architecture**: Kubernetes deployment with auto-scaling
- **Edge Computing**: Distributed GPU computing for low-latency trading

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
