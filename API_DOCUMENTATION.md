# CUDA Finance Tool - API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Endpoints](#endpoints)
   - [Option Pricing](#option-pricing)
   - [Risk Analysis](#risk-analysis)
   - [Job Management](#job-management)
   - [System Status](#system-status)
   - [Market Data](#market-data)
   - [Machine Learning](#machine-learning)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Examples](#examples)

## Overview

The CUDA Finance Tool provides a comprehensive REST API for option pricing, risk analysis, and financial computations using GPU acceleration. All endpoints return JSON responses.

## Authentication

Currently, the API uses simple API key authentication. Include your API key in the request headers:

```bash
curl -H "X-API-Key: your-api-key" https://api.cuda-finance.com/v1/pricing/quick
```

## Base URL

- **Development**: `http://localhost:5000/api`
- **Production**: `https://api.cuda-finance.com/v1`

## Endpoints

### Option Pricing

#### Quick Option Pricing

**POST** `/pricing/quick`

Calculate option price for a single option.

**Request Body:**
```json
{
    "stock_price": 100.0,
    "strike_price": 100.0,
    "volatility": 0.2,
    "time_to_maturity": 1.0,
    "risk_free_rate": 0.05,
    "option_type": "european_call"
}
```

**Response:**
```json
{
    "option_price": 10.45,
    "delta": 0.52,
    "gamma": 0.02,
    "vega": 39.8,
    "theta": -6.2,
    "execution_time_ms": 2.5
}
```

#### Batch Option Pricing

**POST** `/pricing`

Create a batch job for pricing multiple options.

**Request Body:**
```json
{
    "num_options": 1000,
    "stock_prices": [100.0, 105.0, 95.0],
    "strike_prices": [100.0, 105.0, 95.0],
    "volatilities": [0.2, 0.25, 0.15],
    "time_to_maturity": [1.0, 0.5, 2.0],
    "risk_free_rates": [0.05, 0.04, 0.06],
    "option_types": ["european_call", "american_put", "barrier_up_and_out"]
}
```

**Response:**
```json
{
    "job_id": "job_12345",
    "status": "created",
    "message": "Job created successfully"
}
```

#### Advanced Option Pricing

**POST** `/pricing/advanced`

Price advanced option types with additional parameters.

**Request Body:**
```json
{
    "option_type": "barrier_up_and_out",
    "stock_price": 100.0,
    "strike_price": 100.0,
    "barrier_price": 120.0,
    "volatility": 0.2,
    "time_to_maturity": 1.0,
    "risk_free_rate": 0.05,
    "num_simulations": 10000
}
```

### Risk Analysis

#### Value at Risk (VaR)

**POST** `/risk/var`

Calculate VaR for a portfolio.

**Request Body:**
```json
{
    "portfolio_values": [100000, 105000, 95000, 110000],
    "confidence_level": 0.95,
    "time_horizon": 1
}
```

**Response:**
```json
{
    "var": 8500.0,
    "cvar": 12000.0,
    "confidence_level": 0.95,
    "time_horizon": 1
}
```

#### Stress Testing

**POST** `/risk/stress`

Perform stress testing on a portfolio.

**Request Body:**
```json
{
    "portfolio": {
        "positions": [
            {"symbol": "AAPL", "quantity": 100, "price": 150.0},
            {"symbol": "GOOGL", "quantity": 50, "price": 2800.0}
        ]
    },
    "scenarios": [
        {"name": "market_crash", "shock": -0.2},
        {"name": "volatility_spike", "shock": 0.5}
    ]
}
```

**Response:**
```json
{
    "scenarios": [
        {
            "name": "market_crash",
            "portfolio_value": 80000.0,
            "loss": 20000.0,
            "loss_percentage": 0.2
        },
        {
            "name": "volatility_spike",
            "portfolio_value": 95000.0,
            "loss": 5000.0,
            "loss_percentage": 0.05
        }
    ]
}
```

#### Correlation Analysis

**POST** `/risk/correlation`

Calculate correlation matrix for assets.

**Request Body:**
```json
{
    "returns": [
        [0.01, 0.02, -0.01, 0.03],
        [0.02, 0.01, 0.02, -0.01],
        [-0.01, 0.02, 0.01, 0.02]
    ],
    "assets": ["AAPL", "GOOGL", "MSFT"]
}
```

**Response:**
```json
{
    "correlation_matrix": [
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.4],
        [0.2, 0.4, 1.0]
    ],
    "assets": ["AAPL", "GOOGL", "MSFT"]
}
```

### Job Management

#### Get Job Status

**GET** `/jobs/{job_id}`

Get the status and results of a job.

**Response:**
```json
{
    "job_id": "job_12345",
    "status": "completed",
    "progress": 100,
    "result": {
        "option_prices": [10.45, 8.32, 12.67],
        "execution_time_ms": 150.5
    },
    "start_time": "2024-01-15T10:30:00Z",
    "end_time": "2024-01-15T10:30:15Z"
}
```

#### List All Jobs

**GET** `/jobs`

Get a list of all jobs.

**Query Parameters:**
- `status`: Filter by job status (pending, running, completed, failed)
- `limit`: Maximum number of jobs to return (default: 100)

**Response:**
```json
{
    "jobs": [
        {
            "job_id": "job_12345",
            "status": "completed",
            "progress": 100,
            "created_at": "2024-01-15T10:30:00Z"
        },
        {
            "job_id": "job_12346",
            "status": "running",
            "progress": 75,
            "created_at": "2024-01-15T10:35:00Z"
        }
    ]
}
```

#### Cancel Job

**DELETE** `/jobs/{job_id}`

Cancel a running job.

**Response:**
```json
{
    "job_id": "job_12345",
    "status": "cancelled",
    "message": "Job cancelled successfully"
}
```

### System Status

#### Health Check

**GET** `/health`

Check system health and status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0",
    "gpu_available": true,
    "gpu_memory_usage": 2048,
    "database_connected": true,
    "active_jobs": 5
}
```

#### System Information

**GET** `/system/info`

Get detailed system information.

**Response:**
```json
{
    "cuda_available": true,
    "gpu_info": {
        "name": "NVIDIA GeForce RTX 3080",
        "memory": 10240,
        "compute_capability": "8.6"
    },
    "system_resources": {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "disk_usage": 23.1
    },
    "performance_metrics": {
        "avg_option_pricing_time_ms": 2.5,
        "throughput_options_per_second": 400
    }
}
```

### Market Data

#### Get Latest Market Data

**GET** `/market/data/{symbol}`

Get latest market data for a symbol.

**Response:**
```json
{
    "symbol": "AAPL",
    "price": 150.25,
    "volume": 45000000,
    "high": 152.10,
    "low": 149.80,
    "open": 150.00,
    "timestamp": "2024-01-15T10:30:00Z",
    "sources": ["yfinance", "alpha_vantage"],
    "confidence": 0.95
}
```

#### Get Historical Data

**GET** `/market/history/{symbol}`

Get historical market data.

**Query Parameters:**
- `period`: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- `interval`: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

**Response:**
```json
{
    "symbol": "AAPL",
    "period": "1mo",
    "interval": "1d",
    "data": [
        {
            "date": "2024-01-15",
            "open": 150.00,
            "high": 152.10,
            "low": 149.80,
            "close": 150.25,
            "volume": 45000000
        }
    ]
}
```

### Machine Learning

#### Train Volatility Model

**POST** `/ml/train/volatility`

Train the volatility forecasting model.

**Request Body:**
```json
{
    "symbol": "AAPL",
    "period": "2y",
    "target_horizon": 5,
    "model_type": "random_forest"
}
```

**Response:**
```json
{
    "status": "training",
    "job_id": "ml_job_12345",
    "message": "Volatility model training started"
}
```

#### Get Trading Signals

**GET** `/ml/signals/{symbol}`

Get trading signals for a symbol.

**Response:**
```json
{
    "symbol": "AAPL",
    "volatility_forecast": 0.18,
    "patterns_detected": {
        "double_top": false,
        "double_bottom": true,
        "head_and_shoulders": false,
        "breakout": false
    },
    "trading_recommendation": "BUY - Double bottom pattern detected",
    "confidence_score": 0.75,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Update ML Models

**POST** `/ml/update`

Update ML models with new data.

**Request Body:**
```json
{
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "period": "1mo",
    "force_retrain": false
}
```

**Response:**
```json
{
    "status": "updating",
    "job_id": "ml_update_12345",
    "message": "ML models update started"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages.

### Error Response Format

```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid parameter: stock_price must be positive",
        "details": {
            "parameter": "stock_price",
            "value": -100.0,
            "constraint": "must be positive"
        }
    }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Free Tier**: 100 requests per hour
- **Pro Tier**: 1000 requests per hour
- **Enterprise Tier**: 10000 requests per hour

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1642248000
```

## Examples

### Python Client Example

```python
import requests
import json

class CUDAFinanceClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
    
    def price_option(self, stock_price, strike_price, volatility, 
                    time_to_maturity, risk_free_rate):
        """Price a single option"""
        data = {
            'stock_price': stock_price,
            'strike_price': strike_price,
            'volatility': volatility,
            'time_to_maturity': time_to_maturity,
            'risk_free_rate': risk_free_rate
        }
        
        response = requests.post(
            f"{self.base_url}/pricing/quick",
            headers=self.headers,
            json=data
        )
        
        return response.json()
    
    def calculate_var(self, portfolio_values, confidence_level=0.95):
        """Calculate VaR for a portfolio"""
        data = {
            'portfolio_values': portfolio_values,
            'confidence_level': confidence_level
        }
        
        response = requests.post(
            f"{self.base_url}/risk/var",
            headers=self.headers,
            json=data
        )
        
        return response.json()

# Usage
client = CUDAFinanceClient('https://api.cuda-finance.com/v1', 'your-api-key')

# Price an option
result = client.price_option(100, 100, 0.2, 1.0, 0.05)
print(f"Option price: ${result['option_price']:.2f}")

# Calculate VaR
var_result = client.calculate_var([100000, 105000, 95000])
print(f"VaR: ${var_result['var']:.2f}")
```

### JavaScript Client Example

```javascript
class CUDAFinanceClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
        };
    }
    
    async priceOption(stockPrice, strikePrice, volatility, timeToMaturity, riskFreeRate) {
        const data = {
            stock_price: stockPrice,
            strike_price: strikePrice,
            volatility: volatility,
            time_to_maturity: timeToMaturity,
            risk_free_rate: riskFreeRate
        };
        
        const response = await fetch(`${this.baseUrl}/pricing/quick`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        
        return await response.json();
    }
    
    async calculateVaR(portfolioValues, confidenceLevel = 0.95) {
        const data = {
            portfolio_values: portfolioValues,
            confidence_level: confidenceLevel
        };
        
        const response = await fetch(`${this.baseUrl}/risk/var`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        
        return await response.json();
    }
}

// Usage
const client = new CUDAFinanceClient('https://api.cuda-finance.com/v1', 'your-api-key');

// Price an option
client.priceOption(100, 100, 0.2, 1.0, 0.05)
    .then(result => console.log(`Option price: $${result.option_price.toFixed(2)}`));

// Calculate VaR
client.calculateVaR([100000, 105000, 95000])
    .then(result => console.log(`VaR: $${result.var.toFixed(2)}`));
```

### cURL Examples

```bash
# Quick option pricing
curl -X POST https://api.cuda-finance.com/v1/pricing/quick \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "stock_price": 100.0,
    "strike_price": 100.0,
    "volatility": 0.2,
    "time_to_maturity": 1.0,
    "risk_free_rate": 0.05
  }'

# Calculate VaR
curl -X POST https://api.cuda-finance.com/v1/risk/var \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_values": [100000, 105000, 95000],
    "confidence_level": 0.95
  }'

# Get system status
curl -X GET https://api.cuda-finance.com/v1/health \
  -H "X-API-Key: your-api-key"
```

## SDKs and Libraries

Official SDKs are available for:

- **Python**: `pip install cuda-finance-python`
- **JavaScript**: `npm install cuda-finance-js`
- **R**: `install.packages("cudaFinance")`
- **MATLAB**: Available through MathWorks File Exchange

## Support

For API support and questions:

- **Documentation**: https://docs.cuda-finance.com
- **API Status**: https://status.cuda-finance.com
- **Support Email**: api-support@cuda-finance.com
- **Community Forum**: https://community.cuda-finance.com 