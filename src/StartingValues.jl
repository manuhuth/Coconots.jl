"""
    get_starting_values!(type, order, data, covariates, starting_values,
                         n_parameter_without, lower_bound_covariates,
                         link_function = "log")

Returns `starting_values` unchanged when provided; otherwise computes
data-driven starting values for the requested model. For covariate models the
innovation-rate start is replaced by the covariate coefficient starts of
[`compute_covariate_starting_values`](@ref).
"""
function get_starting_values!(type, order, data, covariates, starting_values,
        n_parameter_without, lower_bound_covariates, link_function = "log")
    isnothing(starting_values) || return starting_values
    if Int(order) == 2
        starting_values = compute_starting_values_GP2_reparameterized(type, data)
    else
        starting_values = compute_starting_values_GP1(type, data)
    end
    if !isnothing(covariates)
        n = Int(n_parameter_without)
        scale = _innovation_scale(type, order, starting_values, n)
        beta_start = compute_covariate_starting_values(data, covariates,
            link_function, scale, lower_bound_covariates)
        starting_values = vcat(starting_values[1:n], beta_start)
    end
    return starting_values
end

"""
    _innovation_scale(type, order, starting_values, n_parameter_without)

`(1 - sum(alpha)) * (1 - eta)` at the starting values (`eta = 0` for Poisson);
for order 2 the alphas are recovered from the reparameterized starts.
"""
function _innovation_scale(type, order, starting_values, n_parameter_without)
    eta = type == "GP" ? starting_values[n_parameter_without] : 0.0
    if Int(order) == 1
        return (1 - starting_values[1]) * (1 - eta)
    end
    alpha1, alpha2, alpha3 = reparameterize_alpha(starting_values)
    return (1 - alpha1 - alpha2 - alpha3) * (1 - eta)
end

"""
    compute_covariate_starting_values(data, covariates, link_function, scale,
                                      lower_bound)

Starting values of the covariate coefficients beta. The (GP-)INAR models have
`E[X_t] = lambda / ((1 - sum(alpha)) (1 - eta))`, so for slowly varying
`lambda_t` the innovation rate is `lambda_t = scale * mu_t` with
`scale = (1 - sum(alpha)) (1 - eta)` and `mu_t` the marginal mean. Per link:

- `"log"`: Poisson GLM of the counts on `covariates` with offset `-log(scale)`;
- `"identity"`, `"relu"`: `scale` times the least-squares fit of the counts;
- `"softplus"`: least-squares fit of `softplus^{-1}(lambda_t)` with
  `lambda_t = exp(covariates * beta_log)` from the log-link GLM above.

Falls back to the marginal fit (Poisson GLM for `"log"`, otherwise least
squares with coefficients `<= 0` set to `1e-5`) if `scale` is not in `(0, 1]`
or the result is not admissible (non-finite, `beta <= lower_bound`, or
`lambda_t <= 0` for some row). Mirrors `covariateStartingValues` and the
marginal fit in coconots' `cocoReg_cov.R`.
"""
function compute_covariate_starting_values(data, covariates, link_function, scale,
        lower_bound)
    X = Matrix{Float64}(covariates)
    y = Float64.(data)
    if link_function == "log"
        fallback = _poisson_glm_coefficients(X, y, 0.0)
    else
        fallback = X \ y
        fallback[fallback .<= 0] .= 1e-5
    end
    (isfinite(scale) && 0 < scale <= 1) || return fallback
    if link_function in ("identity", "relu")
        beta = scale .* (X \ y)
    else
        beta_log = _poisson_glm_coefficients(X, y, -log(scale))
        if link_function == "log"
            beta = beta_log
        else
            lambda = exp.(X * beta_log)
            beta = X \ (lambda .+ log.(-expm1.(-lambda)))
        end
    end
    return _admissible_covariate_start(beta, X, link_function, lower_bound) ?
           beta : fallback
end

function _admissible_covariate_start(beta, X, link_function, lower_bound)
    (all(isfinite, beta) && all(beta .> lower_bound)) || return false
    linear_predictor = X * beta
    all(isfinite, linear_predictor) || return false
    link_function in ("identity", "relu") && return all(linear_predictor .> 0)
    link_function == "log" && return all(exp.(linear_predictor) .> 0)
    return true
end

"""
    _poisson_glm_coefficients(X, y, offset)

Coefficients of the Poisson log-link GLM `log(mu) = offset + X * beta` by
iteratively reweighted least squares as in R's `glm.fit` (start
`eta = log(y + 0.1)`, `mu = max(exp(eta), eps)`, at most 25 iterations,
stop when `|dev - dev_old| / (|dev| + 0.1) < 1e-8`). Returns `NaN`s if the
deviance becomes non-finite.
"""
function _poisson_glm_coefficients(X::AbstractMatrix, y::AbstractVector, offset::Real)
    eta = log.(y .+ 0.1)
    mu = max.(exp.(eta), eps(Float64))
    dev_old = _poisson_deviance(y, mu)
    beta = fill(NaN, size(X, 2))
    for _ in 1:25
        z = (eta .- offset) .+ (y .- mu) ./ mu
        w = sqrt.(mu)
        beta = (X .* w) \ (z .* w)
        eta = X * beta .+ offset
        mu = max.(exp.(eta), eps(Float64))
        dev = _poisson_deviance(y, mu)
        isfinite(dev) || return fill(NaN, size(X, 2))
        abs(dev - dev_old) / (abs(dev) + 0.1) < 1e-8 && break
        dev_old = dev
    end
    return beta
end

function _poisson_deviance(y, mu)
    return 2 * sum(yi > 0 ? yi * log(yi / mi) - (yi - mi) : mi
                   for (yi, mi) in zip(y, mu))
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
