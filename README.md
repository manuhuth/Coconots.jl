# Coconots

[![Build Status](https://github.com/manuhuth/Coconots.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/manuhuth/Coconots.jl/actions/workflows/CI.yml?query=branch%3Amain)

The `Coconots` package provides a robust and user-friendly Julia framework for analyzing time series consisting of low counts. It implements practical integer autoregressive (INAR) models capable of capturing both first-order and higher-order dependencies based on the foundational work by Joe (1996). The package supports modeling of both equidispersed and overdispersed marginal distributions and allows for the inclusion of regression effects.

## Key Features

- First- and higher-order integer autoregressive models
- Supports Poisson and generalized Poisson distributions
- Inclusion of covariates with customizable link functions
- Efficient likelihood-based inference and forecasting
- Comprehensive tools for model validation and diagnostics

## Installation

Install directly via Github:

```julia
using Pkg
#Pkg.add("Coconots") #Adding the package from the general repository will be available soon
Pkg.add(url="https://github.com/manuhuth/Coconots.jl")
```

## Usage Examples

### Simulating Integer Autoregressive Time Series

```julia
using Coconots
using Random

Random.seed!(3)

# First-order model with Generalized Poisson distribution
type = "GP"
order = 1
parameter = [0.3, 0.2, 0.7]  # alpha, eta, lambda
n = 40

# Simulate data
data = cocoSim(type, order, parameter, n)

# Second-order model with Generalized Poisson distribution
order2 = 2
parameter2 = [0.3, 0.05, 0.2, 0.2, 0.7]  # alpha1, alpha2, alpha3, eta, lambda

# Simulate second-order data
data2 = cocoSim(type, order2, parameter2, n)
```

### Model Fitting

Fit a first-order model to simulated data:

```julia
fit = cocoReg(type, order, data)
```

Fit a second-order model to simulated data:

```julia
fit2 = cocoReg(type, order2, data2)
```

### Model Diagnostics

Perform residual diagnostics and bootstrap:

```julia
# Bootstrap
cocoBoot(fit, 1:21, 10)

# Probability integral transform (PIT)
cocoPit(fit)

# Compute scores
compute_scores(fit)
```

### Forecasting

#### One-step-ahead forecast

```julia
forecast_range = 0:Int(ceil(maximum(data) * 1.5))
cocoPredictOneStep(fit)
```

#### K-step-ahead forecast

```julia
cocoPredictKsteps(fit, 1)
```

### Using Covariates with Link Functions

```julia
covariates = reshape(sin.(1:n) .+ 4.2, :, 1)
parameter_cov = [0.3, 0.2, 0.7]  # alpha, eta, lambda
link = "log"

# Simulate data with covariates
data_cov = cocoSim(type, order, parameter_cov, n, covariates, link)

# Fit the model with covariates
fit_cov = cocoReg(type, order, data_cov, covariates)
```

## Documentation

A detailed documentation is planned to be available soon.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## References
- Jung, R. C., & Tremayne, A. R. (2011). Convolution-closed models for count time series with applications. *Journal of Time Series Analysis*, 32(3), 268-280.
- Joe, H. (1996). Time Series Models with Univariate Margins in the Convolution-Closed Infinitely Divisible Class. *Journal of Applied Probability*, 33(3), 664-677.
