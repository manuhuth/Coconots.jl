"""
    get_bounds_GP2(type, covariates, lower_bound_covariates)

Computes the lower and upper bounds for the parameter vector for second-order models.
For "GP" (generalized Poisson) models, the parameter vector is assumed to consist of three autoregressive parameters,
one dispersion parameter, and additional covariate parameters. For "Poisson" models, the dispersion parameter is omitted.
The covariate bounds are set to `lower_bound_covariates` for the lower bound and `Inf` for the upper bound.

# Arguments
- `type`: A string specifying the model type ("GP" or "Poisson").
- `covariates`: A matrix of covariate values, or `nothing` if there are no covariates.
- `lower_bound_covariates`: A numeric lower bound for each covariate parameter.

# Returns
A tuple `(lower, upper)` where:
- `lower` is a vector of lower bounds.
- `upper` is a vector of upper bounds.
For "GP" models, the bounds are constructed as `vcat([0, 0, 0], [0], lambda_lower)`,
and for "Poisson" models as `vcat([0, 0, 0], lambda_lower)`.
"""
function get_bounds_GP2(type, covariates, lower_bound_covariates)
    lambda_lower = [0]
    lambda_upper = [Inf]
    if !isnothing(covariates)
        lambda_lower = repeat([lower_bound_covariates], size(covariates, 2))
        lambda_upper = repeat([Inf], size(covariates, 2))
    end

    if type == "GP"
        lower = vcat([0, 0, 0], [0], lambda_lower)
        upper = vcat([1, 1, 1], [1], lambda_upper)
    elseif type == "Poisson"
        lower = vcat([0, 0, 0], lambda_lower)
        upper = vcat([1, 1, 1], lambda_upper)
    end
    return lower, upper
end

"""
    get_bounds_GP1(type, covariates, lower_bound_covariates)

Computes the lower and upper bounds for the parameter vector for first-order models.
For "GP" (generalized Poisson) models, the parameter vector is assumed to consist of one autoregressive parameter,
one dispersion parameter, and additional covariate parameters. For "Poisson" models, the dispersion parameter is omitted.
The covariate bounds are set to `lower_bound_covariates` for the lower bound and `Inf` for the upper bound.

# Arguments
- `type`: A string specifying the model type ("GP" or "Poisson").
- `covariates`: A matrix of covariate values, or `nothing` if there are no covariates.
- `lower_bound_covariates`: A numeric lower bound for each covariate parameter.

# Returns
A tuple `(lower, upper)` where:
- `lower` is a vector of lower bounds.
- `upper` is a vector of upper bounds.
For "GP" models, the bounds are constructed as `vcat([0], [0], lambda_lower)`,
and for "Poisson" models as `vcat([0], lambda_lower)`.
"""
function get_bounds_GP1(type, covariates, lower_bound_covariates)
    lambda_lower = [0]
    lambda_upper = [Inf]
    if !isnothing(covariates)
        lambda_lower = repeat([lower_bound_covariates], size(covariates, 2))
        lambda_upper = repeat([Inf], size(covariates, 2))
    end

    if type == "GP"
        lower = vcat([0], [0], lambda_lower)
        upper = vcat([1], [1], lambda_upper)
    elseif type == "Poisson"
        lower = vcat([0], lambda_lower)
        upper = vcat([1], lambda_upper)
    end
    return lower, upper
end

"""
    get_bounds(order, type, covariates, lower_bound_covariates)

Selects and returns the lower and upper bounds for the parameter vector based on the model order.
For first-order models, it calls `get_bounds_GP1`; for second-order models, it calls `get_bounds_GP2`.

# Arguments
- `order`: An integer indicating the autoregressive order (1 or 2).
- `type`: A string specifying the model type ("GP" or "Poisson").
- `covariates`: A matrix of covariate values, or `nothing` if none.
- `lower_bound_covariates`: A numeric lower bound for the covariate parameters.

# Returns
A tuple `(lower, upper)` where `lower` is a vector of lower bounds and `upper` is a vector of upper bounds appropriate
for the specified model order and type.
"""
function get_bounds(order, type, covariates, lower_bound_covariates)
    if order == 1
        return get_bounds_GP1(type, covariates, lower_bound_covariates)
    elseif order == 2
        return get_bounds_GP2(type, covariates, lower_bound_covariates)
    end
end
