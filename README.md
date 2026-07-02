# Coconots

<a href="https://github.com/manuhuth/Coconots.jl/actions/workflows/CI.yml?query=branch%3Amain">
    <img src="https://github.com/manuhuth/Coconots.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="CI"/>
</a>
<a href="https://codecov.io/gh/manuhuth/Coconots.jl">
    <img src="https://codecov.io/gh/manuhuth/Coconots.jl/branch/main/graph/badge.svg" alt="Coverage"/>
</a>
<a href="https://github.com/JuliaTesting/Aqua.jl">
    <img src="https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg" alt="Aqua QA"/>
</a>
<a href="https://julialang.org">
    <img src="https://img.shields.io/badge/Julia-1.10%2B-9558B2.svg?logo=julia&logoColor=white" alt="Julia 1.10+"/>
</a>
<a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
</a>
<a href="https://www.repostatus.org/#active">
    <img src="https://www.repostatus.org/badges/latest/active.svg" alt="Project Status: Active"/>
</a>

The `Coconots` package provides a robust and user-friendly Julia framework for analyzing time series consisting of low counts. Likelihood-based methods for model fitting, assessment and prediction analysis of some convolution-closed count time series model are provided. The marginal distribution can be Poisson or Generalized Poisson. Regression effects can be modelled via time varying innovation rates. 

## Key Features

- First- and higher-order integer autoregressive models
- Supports Poisson and generalized Poisson distributions
- Inclusion of covariates with customizable link functions
- Efficient likelihood-based inference and forecasting: type-stable,
  allocation-light convolution kernels with precomputed lookup tables
- Optimization via the [Optimization.jl](https://github.com/SciML/Optimization.jl)
  interface with pluggable solvers and automatic-differentiation backends
  (ForwardDiff by default; Enzyme, ReverseDiff, etc. via `adtype`)
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

Parameter constraints are handled by smooth parameter transformations by
default, so the problem is solved unconstrained (with `LBFGS()`). The
optimizer and the automatic-differentiation backend are pluggable: any
Optimization.jl solver and any `ADTypes` backend can be used; extra keyword
arguments are forwarded to `Optimization.solve`:

```julia
fit = cocoReg(type, order, data; adtype = Optimization.AutoForwardDiff(),
    maxiters = 500)

# solver-enforced box constraints instead of transformations:
fit = cocoReg(type, order, data; box_constrained = true)  # Fminbox(LBFGS())

# with Enzyme loaded, reverse-mode AD:
# import Enzyme
# fit = cocoReg(type, order, data; adtype = AutoEnzyme())
```

The likelihood evaluations are multithreaded, but a Julia process starts
single-threaded by default: launch with `julia --threads=8` (or set
`JULIA_NUM_THREADS`) to activate the parallel paths. Second-order fits on
long or overdispersed series gain roughly 3x on 8 threads; when calling from
R via JuliaConnectoR, set `Sys.setenv(JULIA_NUM_THREADS = "8")` before the
first Julia call of the session.

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
