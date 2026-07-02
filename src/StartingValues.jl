"""
    get_starting_values!(type, order, data, covariates, starting_values,
                         n_parameter_without, lower_bound_covariates)

Returns `starting_values` unchanged when provided; otherwise computes
data-driven starting values for the requested model. For covariate models the
innovation-rate start is replaced by per-coefficient starts respecting
`lower_bound_covariates`.
"""
function get_starting_values!(type, order, data, covariates, starting_values,
        n_parameter_without, lower_bound_covariates)
    isnothing(starting_values) || return starting_values
    if Int(order) == 2
        starting_values = compute_starting_values_GP2_reparameterized(type, data)
    else
        starting_values = compute_starting_values_GP1(type, data)
    end
    if !isnothing(covariates)
        beta_start = max(0, lower_bound_covariates + 1e-08)
        starting_values[Int(n_parameter_without) + 1] = beta_start
        starting_values = vcat(starting_values,
            fill(beta_start, size(covariates, 2) - 1))
    end
    return starting_values
end

"""
    compute_starting_values_GP2_reparameterized(type, data)

Data-driven starting values for second-order models on the reparameterized
scale: `[2*alpha1 + alpha3, alpha3 / (2*alpha1 + alpha3),
alpha2 / (1 - alpha1 - alpha3)[, eta], lambda]`.
"""
function compute_starting_values_GP2_reparameterized(type, data)
    eta = type == "GP" ? compute_eta_starting_value(data) : 0.0
    alpha3 = set_to_unit_interval(compute_mu_hat_gmm(data) / mean(data) /
                                  (1 + 2 * eta) * (1 - eta)^4)
    alpha1 = set_to_unit_interval(compute_autocorrelation(data, 1))
    alpha2 = set_to_unit_interval(compute_autocorrelation(data, 2))

    while alpha3 + alpha2 + alpha1 >= 1
        alpha3 *= 0.8
        alpha2 *= 0.8
        alpha1 *= 0.8
    end
    while alpha3 + 2 * alpha1 >= 1
        alpha3 *= 0.8
        alpha1 *= 0.8
    end

    lambda = mean(data) * (1 - alpha1 - alpha2 - alpha3) * (1 - eta)

    reparameterized = [2 * alpha1 + alpha3, alpha3 / (2 * alpha1 + alpha3),
        alpha2 / (1 - alpha1 - alpha3)]
    if type == "GP"
        return vcat(reparameterized, [eta, lambda])
    end
    return vcat(reparameterized, [lambda])
end

"""
    compute_starting_values_GP1(type, data)

Data-driven starting values for first-order models: `[alpha[, eta], lambda]`.
"""
function compute_starting_values_GP1(type, data)
    alpha = abs(compute_autocorrelation(data, 1))
    lambda = mean(data) * (1 - alpha)
    if type == "GP"
        return [alpha, compute_eta_starting_value(data), lambda]
    end
    return [alpha, lambda]
end

"""
    compute_eta_starting_value(data)

Dispersion starting value `1 - sqrt(mean / var)`, floored at `0.0001`.
"""
function compute_eta_starting_value(data)
    eta = 1 - sqrt(mean(data) / var(data))
    return eta < 0 ? 0.0001 : eta
end
