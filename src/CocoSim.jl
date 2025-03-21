using Random, Distributions, StatsBase, DataFrames

export cocoSim

"""
    draw_random_generalized_poisson_variable(u, lambda, eta)

Generates a random draw from the generalized Poisson distribution using inverse transform sampling.

# Arguments
- `u`: A uniform random value in [0, 1] used as the threshold for the cumulative probability.
- `lambda`: The rate parameter of the generalized Poisson distribution.
- `eta`: The dispersion parameter of the generalized Poisson distribution.

# Returns
An integer value sampled from the generalized Poisson distribution.
"""
function draw_random_generalized_poisson_variable(u, lambda, eta)
    sum = 0.0
    i = 0
    while sum < u
        sum = sum + generalized_poisson_distribution(i, lambda, eta)
        i = i + 1
    end
    return i - 1
end


"""
    draw_random_g_r_y_z_variable(u, y, z, lambda, alpha1, alpha2, alpha3, eta, max)

Generates a random draw from a custom discrete distribution defined by the function `compute_g_r_y_z` using inverse transform sampling. This function is designed for models incorporating two autoregressive lags.

# Arguments
- `u`: A uniform random value in [0, 1] serving as the threshold.
- `y`: A model parameter (e.g., a previous count) used in the computation.
- `z`: A second model parameter (e.g., another lagged count).
- `lambda`: The rate parameter used in the distribution.
- `alpha1`: The first autoregressive parameter.
- `alpha2`: The second autoregressive parameter.
- `alpha3`: The third autoregressive parameter.
- `eta`: A dispersion parameter.
- `max`: A parameter that may set an upper limit or truncation value in the computation.

# Returns
An integer sampled from the custom distribution defined by `compute_g_r_y_z`.
"""
function draw_random_g_r_y_z_variable(u, y, z, lambda, alpha1, alpha2, alpha3, eta, max)
    sum = 0.0
    i = 0
    while sum < u
        sum = sum + compute_g_r_y_z(i, y, z, lambda, alpha1, alpha2, alpha3, eta, max)
        i = i + 1
    end
    return i - 1
end


"""
    draw_random_g_r_y_variable(u, y, alpha, eta, lambda)

Generates a random draw from a custom discrete distribution defined by the function `compute_g_r_y` using inverse transform sampling. This is intended for models with a single autoregressive parameter.

# Arguments
- `u`: A uniform random value in [0, 1] used as the sampling threshold.
- `y`: A model parameter (typically representing a lagged count).
- `alpha`: The autoregressive parameter for the model.
- `eta`: A dispersion parameter.
- `lambda`: The rate parameter for the distribution.

# Returns
An integer value sampled from the distribution defined by `compute_g_r_y`.
"""
function draw_random_g_r_y_variable(u, y, alpha, eta, lambda)
    sum = 0.0
    i = 0
    while sum < u
        sum = sum + compute_g_r_y(y, i, alpha, eta, lambda)
        i = i + 1
    end
    return i - 1
end

"""
    draw_random_g(order, u, y, z, lambda, alpha1, alpha2, alpha3, alpha, eta, max)

Draws a random variable from one of two custom distributions depending on the specified model order.

- If `order == 1`, it calls `draw_random_g_r_y_variable`.
- If `order == 2`, it calls `draw_random_g_r_y_z_variable`.

# Arguments
- `order`: An integer specifying the order of the model (1 for single-lag, 2 for two-lag).
- `u`: A uniform random value in [0, 1] used as the threshold.
- `y`: A model parameter (typically a lagged count).
- `z`: A secondary model parameter required for order 2 models.
- `lambda`: The rate parameter for the distribution.
- `alpha1`: First autoregressive parameter (for order 2).
- `alpha2`: Second autoregressive parameter (for order 2).
- `alpha3`: Third autoregressive parameter (for order 2).
- `alpha`: Autoregressive parameter used for order 1.
- `eta`: A dispersion parameter.
- `max`: A parameter used in the custom distribution for order 2.

# Returns
An integer representing the sampled value from the chosen distribution.
"""
function draw_random_g(order, u, y, z, lambda, alpha1, alpha2, alpha3, alpha, eta, max)
    if order == 1
        return draw_random_g_r_y_variable(u, y, alpha, eta, lambda)
    elseif order == 2
        return draw_random_g_r_y_z_variable(u, y, z, lambda, alpha1, alpha2, alpha3, eta, max)
    end
end

"""
    cocoSim(type, order, parameter, n, covariates=nothing, link="log", n_burn_in=200, x=zeros(Int(n + n_burn_in)))

Simulates a count time series based on an integer autoregressive model. The function accommodates both first-order and second-order models, with optional covariate information via a specified link function.

# Arguments
- `type`: A string indicating the model type (e.g., `"GP"` for generalized Poisson). This parameter affects how the dispersion parameter `eta` is set.
- `order`: An integer (1 or 2) specifying the autoregressive model order.
- `parameter`: A vector of model parameters. For order 1 models, the first element is `alpha` and (if `type` is `"GP"`) the second is `eta`. For order 2 models, the first three elements are `alpha1`, `alpha2`, `alpha3`, and (if `type` is `"GP"`) the fourth is `eta`.
- `n`: The number of observations to simulate after the burn-in period.
- `covariates` (optional): A matrix of covariate values. If provided, these are used with the specified link function to compute the rate parameter `lambda`.
- `link` (optional): A string specifying the link function to use (default is `"log"`).
- `n_burn_in` (optional): The number of initial burn-in observations to discard (default is 200).
- `x` (optional): An initial vector for the time series with length `n + n_burn_in`. Defaults to an integer array initialized with zeros.

# Returns
A vector of simulated count data of length `n`, corresponding to the time series after the burn-in period.

# Details
- The rate parameter `lambda` is computed using a link function (obtained via `get_link_function(link)`) applied to the covariates if provided, or is set to the last element of `parameter` otherwise.
- Depending on the model order, the appropriate autoregressive parameters are extracted from `parameter`.
- For each time step (starting from t = 3), the simulated count is the sum of a draw from a custom autoregressive component (via `draw_random_g`) and a draw from a generalized Poisson component (via `draw_random_generalized_poisson_variable`).
"""
function cocoSim(type, order, parameter, n, covariates=nothing,
                 link="log", n_burn_in=200,
                 x=zeros(Int(n + n_burn_in)))
    
    link_function = get_link_function(link)
    
    if !isnothing(covariates)
        lambda = link_function.(repeat(covariates * parameter[Int((end-(size(covariates)[2]-1))):Int(end)], Int(ceil(1 + n_burn_in / n)))[Int((end-n_burn_in-n+1)):Int(end)])
    else
        lambda = repeat([last(parameter)], Int(n + n_burn_in))
    end
    
    if order == 2
        alpha1 = parameter[1]
        alpha2 = parameter[2]
        alpha3 = parameter[3]
        alpha = nothing
        if type == "GP"
            eta = parameter[4]
        else
            eta = 0
        end
    else
        alpha1 = nothing
        alpha2 = nothing
        alpha3 = nothing
        alpha = parameter[1]
        if type == "GP"
            eta = parameter[2]
        else
            eta = 0
        end
    end
    
    for t in 3:(Int(n + n_burn_in))
        x[t] = draw_random_g(order, rand(Uniform(0,1),1)[1], Int(x[t-1]), Int(x[t-2]), lambda[t], alpha1, alpha2, alpha3, alpha, eta, nothing) +
               draw_random_generalized_poisson_variable(rand(Uniform(0,1),1)[1], lambda[t], eta)
    end
    
    return x[Int((end-n+1)):Int(end)]
end
