"""
    get_starting_values!(type, order, data, covariates, starting_values, n_parameter_without, lower_bound_covariates)

If starting values are not provided, computes starting values for model parameters based on the specified model type ("GP" or "Poisson")
and order (1 or 2). For models with covariates, it adjusts the starting values for the covariate parameters by appending a constant value derived
from the lower bound.

# Arguments
- `type`: A string specifying the model type ("GP" for generalized Poisson or "Poisson").
- `order`: The autoregressive order (1 or 2).
- `data`: A vector of observed counts.
- `covariates`: A matrix of covariates (or `nothing` if none).
- `starting_values`: A vector of initial parameter values; if provided, these values are used.
- `n_parameter_without`: The number of parameters before including the covariate parameters.
- `lower_bound_covariates`: A numeric lower bound for the covariate parameters.

# Returns
A vector of starting values for the model parameters.
"""
function get_starting_values!(type, order, data, covariates, starting_values, n_parameter_without, lower_bound_covariates)
    if isnothing(starting_values)
        if (type == "GP") & (order == 2)
            starting_values = compute_starting_values_GP2_reparameterized("GP", data)
            if !isnothing(covariates)
                starting_values[n_parameter_without+1] = max(0, lower_bound_covariates + 1e-08)
                starting_values = vcat(starting_values, repeat([max(0, lower_bound_covariates + 1e-08)], size(covariates, 2)-1))
            end
        end

        if (type == "Poisson") & (order == 2)
            starting_values = compute_starting_values_GP2_reparameterized("Poisson", data)
            if !isnothing(covariates)
                starting_values[n_parameter_without+1] = max(0, lower_bound_covariates + 1e-08)
                starting_values = vcat(starting_values, repeat([max(0, lower_bound_covariates + 1e-08)], size(covariates, 2)-1))
            end
        end

        if (type == "GP") & (order == 1)
            starting_values = compute_starting_values_GP1("GP", data)
            if !isnothing(covariates)
                starting_values[n_parameter_without+1] = max(0, lower_bound_covariates + 1e-08)
                starting_values = vcat(starting_values, repeat([max(0, lower_bound_covariates + 1e-08)], size(covariates, 2)-1))
            end
        end

        if (type == "Poisson") & (order == 1)
            starting_values = compute_starting_values_GP1("Poisson", data)
            if !isnothing(covariates)
                starting_values[n_parameter_without+1] = max(0, lower_bound_covariates + 1e-08)
                starting_values = vcat(starting_values, repeat([max(0, lower_bound_covariates + 1e-08)], size(covariates, 2)-1))
            end
        end
    end
    return starting_values
end

"""
    compute_starting_values_GP2_reparameterized(type, data)

Computes starting values for a second-order model using a reparameterization approach.
It estimates the autoregressive parameters (alpha1, alpha2, alpha3) based on autocorrelations and a third moment measure,
computes a starting value for the dispersion parameter (eta) if applicable, and calculates the rate parameter lambda.

# Arguments
- `type`: A string indicating the model type ("GP" for generalized Poisson or "Poisson").
- `data`: A vector of observed counts.

# Returns
For "GP" models, returns a vector:  
`[2*alpha1 + alpha3, alpha3 / (2*alpha1+alpha3), alpha2 / (1-alpha1-alpha3), eta, lambda]`.  
For "Poisson" models, returns:  
`[2*alpha1 + alpha3, alpha3 / (2*alpha1+alpha3), alpha2 / (1-alpha1-alpha3), lambda]`.
"""
function compute_starting_values_GP2_reparameterized(type, data)
    if type == "GP"
        eta = compute_eta_starting_value(data)
    elseif type == "Poisson"
        eta = 0
    end
    alpha3 = set_to_unit_interval(compute_mu_hat_gmm(data) / mean(data) / (1 + 2 * eta) * (1 - eta)^4)
    alpha1 = set_to_unit_interval(compute_autocorrelation(data, 1))
    alpha2 = set_to_unit_interval(compute_autocorrelation(data, 2))

    if alpha3 + alpha2 + alpha1 >= 1
        while alpha3 + alpha2 + alpha1 >= 1
            alpha3 = alpha3 * 0.8
            alpha2 = alpha2 * 0.8
            alpha1 = alpha1 * 0.8
        end
    end

    if alpha3 + 2 * alpha1 >= 1
        while alpha3 + 2 * alpha1 >= 1
            alpha3 = alpha3 * 0.8
            alpha1 = alpha2 * 0.8
        end
    end

    lambda = mean(data) * (1 - alpha1 - alpha2 - alpha3) * (1 - eta)

    if type == "GP"
        return [2 * alpha1 + alpha3, alpha3 / (2 * alpha1 + alpha3), alpha2 / (1 - alpha1 - alpha3), eta, lambda]
    elseif type == "Poisson"
        return [2 * alpha1 + alpha3, alpha3 / (2 * alpha1 + alpha3), alpha2 / (1 - alpha1 - alpha3), lambda]
    end
end

"""
    compute_starting_values_GP1(type, data)

Computes starting values for a first-order model.
It estimates the autoregressive parameter (alpha) from the first-order autocorrelation and computes the rate parameter lambda accordingly.
For "GP" models, it also computes a starting value for the dispersion parameter (eta).

# Arguments
- `type`: A string indicating the model type ("GP" for generalized Poisson or "Poisson").
- `data`: A vector of observed counts.

# Returns
For "GP" models, returns a vector `[alpha, eta, lambda]`.  
For "Poisson" models, returns a vector `[alpha, lambda]`.
"""
function compute_starting_values_GP1(type, data)
    alpha = abs(compute_autocorrelation(data, 1))
    lambda = mean(data) * (1 - alpha)
    if type == "GP"
        eta = compute_eta_starting_value(data)
        return [alpha, eta, lambda]
    elseif type == "Poisson"
        eta = 0
        return [alpha, lambda]
    end
end

"""
    compute_eta_starting_value(data)

Computes a starting value for the dispersion parameter (eta) based on the observed data.
Eta is computed as `1 - sqrt(mean(data) / var(data))` and is set to a small positive value if the result is negative.

# Arguments
- `data`: A vector of observed counts.

# Returns
A non-negative value for eta, with a minimum of 0.0001.
"""
function compute_eta_starting_value(data)
    eta = 1 - (mean(data) / var(data))^0.5
    if eta < 0
        eta = 0.0001
    end
    return eta
end
