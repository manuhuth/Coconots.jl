"""
    compute_distribution_convolution_x_r_y(x, y, lambda, alpha, eta)

Computes the cumulative distribution for the first-order convolution likelihood.
It sums the convolution probability mass function from 0 to `x`.

# Arguments
- `x`: The count up to which the cumulative probability is computed.
- `y`: The lagged count (conditioning variable).
- `lambda`: The rate parameter.
- `alpha`: The autoregressive parameter.
- `eta`: The dispersion parameter.

# Returns
A scalar representing the cumulative probability up to `x`. Returns 0 if `x` is negative.
"""
function compute_distribution_convolution_x_r_y(x, y, lambda, alpha, eta)
    if x < 0
        return 0
    end
    return sum([compute_convolution_x_r_y(i, y, lambda, alpha, eta) for i in 0:x])
end

"""
    compute_distribution_convolution_x_r_y_z(x, y, z, alpha1, alpha2, alpha3, lambda, eta, max_loop=nothing)

Computes the cumulative distribution for the second-order convolution likelihood.
It sums the convolution probability mass function from 0 to `x`.

# Arguments
- `x`: The count up to which the cumulative probability is computed.
- `y`: The first lagged count.
- `z`: The second lagged count.
- `alpha1`, `alpha2`, `alpha3`: The autoregressive parameters for the second-order model.
- `lambda`: The rate parameter.
- `eta`: The dispersion parameter.
- `max_loop` (optional): An optional parameter controlling iteration limits.

# Returns
A scalar representing the cumulative probability up to `x`. Returns 0 if `x` is negative.
"""
function compute_distribution_convolution_x_r_y_z(x, y, z, alpha1, alpha2, alpha3, lambda, eta, max_loop=nothing)
    if x < 0
        return 0
    end
    return sum([compute_convolution_x_r_y_z(i, y, z, lambda, alpha1, alpha2, alpha3, eta, max_loop) for i in 0:x])
end

"""
    compute_g_r_y_z(r, y, z, lambda, alpha1, alpha2, alpha3, eta, max)

Computes the convolution probability mass function for a second-order model component.

# Arguments
- `r`: The target value in the convolution sum.
- `y`: The first lagged count.
- `z`: The second lagged count.
- `lambda`: The rate parameter.
- `alpha1`, `alpha2`, `alpha3`: The autoregressive parameters.
- `eta`: The dispersion parameter.
- `max`: The maximum summation index for the inner loops (if not provided, defaults to `y`).

# Returns
The computed probability mass value, normalized by the bivariate generalized Poisson probability.
"""
function compute_g_r_y_z(r, y, z, lambda, alpha1, alpha2, alpha3, eta, max)
    U = compute_U(alpha1, alpha2, alpha3)
    beta3 = compute_beta_i(lambda, U, alpha3)
    beta2 = compute_beta_i(lambda, U, alpha2)
    beta1 = compute_beta_i(lambda, U, alpha1)
    zeta = compute_zeta(lambda, U, alpha1, alpha3)
    
    if isnothing(max)
        max = y
    end

    sum = 0.0
    for s in 0:max, v in 0:max, w in 0:max
        if ((r - s - v) >= 0) & ((z - r + v - w) >= 0) & ((y - s - v - w) >= 0)
            sum = sum + generalized_poisson_distribution(s, beta3, eta) *
                        generalized_poisson_distribution(v, beta1, eta) *
                        generalized_poisson_distribution(w, beta1, eta) *
                        generalized_poisson_distribution(r - s - v, beta2, eta) *
                        generalized_poisson_distribution(z - r + v - w, lambda, eta) *
                        generalized_poisson_distribution(y - s - v - w, zeta, eta)
        end
    end

    return sum / bivariate_generalized_poisson(y, z, lambda, alpha1, alpha2, alpha3, eta)
end

"""
    compute_g_r_y(y, r, alpha, eta, lambda)

Computes the convolution probability mass function for a first-order model component using a closed-form expression.

# Arguments
- `y`: The lagged count.
- `r`: The target value in the convolution sum.
- `alpha`: The autoregressive parameter.
- `eta`: The dispersion parameter.
- `lambda`: The rate parameter.

# Returns
The computed probability mass value. For small `y`, standard factorials are used; for larger `y`, big integer factorials are applied.
"""
function compute_g_r_y(y, r, alpha, eta, lambda)
    psi = eta * (1 - alpha) / lambda
    if y < 20
        return factorial(Int(y)) / factorial(Int(r)) / factorial(Int(y - r)) * alpha * (1 - alpha) *
               (alpha + psi * r)^(r - 1) * (1 - alpha + psi * (y - r))^(y - r - 1) /
               (1 + psi * y)^(y - 1)
    else
        return Float64(factorial(big(Int(y)))) / Float64(factorial(big(Int(r)))) /
               Float64(factorial(big(Int(y - r)))) * alpha * (1 - alpha) *
               (alpha + psi * r)^(r - 1) * (1 - alpha + psi * (y - r))^(y - r - 1) /
               (1 + psi * y)^(y - 1)
    end
end

"""
    compute_convolution_x_r_y(x, y, lambda, alpha, eta)

Computes the convolution likelihood for a first-order model by summing contributions over possible latent counts.

# Arguments
- `x`: The observed count.
- `y`: The lagged count.
- `lambda`: The rate parameter.
- `alpha`: The autoregressive parameter.
- `eta`: The dispersion parameter.

# Returns
The sum of the convolution probability over `r` from 0 to `min(x, y)`.
"""
function compute_convolution_x_r_y(x, y, lambda, alpha, eta)
    sum = 0.0
    for r in 0:min(x, y)
        if y >= r
            sum = sum + compute_g_r_y(y, r, alpha, eta, lambda) * generalized_poisson_distribution(x - r, lambda, eta)
        end
    end
    return sum
end

"""
    compute_negative_log_likelihood_GP1(lambdas, alpha, eta, data, array_fill=Array{Union{Float64, ForwardDiff.Dual}}(undef, length(data) - 1))

Computes the negative log-likelihood for a first-order generalized Poisson model via convolution likelihood.

# Arguments
- `lambdas`: A vector of rate parameters corresponding to each observation.
- `alpha`: The autoregressive parameter.
- `eta`: The dispersion parameter.
- `data`: A vector of observed counts.
- `array_fill` (optional): Preallocated array for storing individual negative log-likelihood values.

# Returns
The total negative log-likelihood computed as the sum of the log-likelihood contributions for each observation (from the second observation onward).
"""
function compute_negative_log_likelihood_GP1(lambdas, alpha, eta, data, array_fill=Array{Union{Float64, ForwardDiff.Dual}}(undef, length(data) - 1))
    Threads.@threads for t in eachindex(data)[2:end]
        array_fill[t - 1] = -log(compute_convolution_x_r_y(data[t], data[t - 1], lambdas[t], alpha, eta))
    end
    return sum(array_fill)
end

"""
    compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, eta, data, max=nothing, array_fill=Array{Union{Float64, ForwardDiff.Dual}}(undef, length(data) - 2))

Computes the negative log-likelihood for a second-order generalized Poisson model via convolution likelihood.

# Arguments
- `lambdas`: A vector of rate parameters for each observation.
- `alpha1`, `alpha2`, `alpha3`: The autoregressive parameters.
- `eta`: The dispersion parameter.
- `data`: A vector of observed counts.
- `max` (optional): A parameter to control iteration limits in the convolution calculation.
- `array_fill` (optional): Preallocated array for storing individual negative log-likelihood values.

# Returns
The total negative log-likelihood computed as the sum of the log-likelihood contributions for each observation (from the third observation onward).
"""
function compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, eta, data, max=nothing, array_fill=Array{Union{Float64, ForwardDiff.Dual}}(undef, length(data) - 2))
    Threads.@threads for t in eachindex(data)[3:end]
        array_fill[t - 2] = -log(compute_convolution_x_r_y_z(data[t], data[t - 1], data[t - 2], lambdas[t], alpha1, alpha2, alpha3, eta, max))
    end
    return sum(array_fill)
end

"""
    compute_convolution_x_r_y_z(x, y, z, lambda, alpha1, alpha2, alpha3, eta, max=nothing)

Computes the convolution likelihood for a second-order model by summing contributions over latent counts.

# Arguments
- `x`: The observed count.
- `y`: The first lagged count.
- `z`: The second lagged count.
- `lambda`: The rate parameter.
- `alpha1`, `alpha2`, `alpha3`: The autoregressive parameters.
- `eta`: The dispersion parameter.
- `max` (optional): An optional parameter controlling the maximum iteration in the convolution sum.

# Returns
The summed convolution probability value.
"""
function compute_convolution_x_r_y_z(x, y, z, lambda, alpha1, alpha2, alpha3, eta, max=nothing)
    sum = 0.0
    for r in 0:min(x, y + z)
        sum = sum + compute_g_r_y_z(r, y, z, lambda, alpha1, alpha2, alpha3, eta, max) * generalized_poisson_distribution(x - r, lambda, eta)
    end
    return sum
end

"""
    get_lambda(cocoReg_fit, last_val=false)

Extracts or computes the rate parameter `lambda` from a fitted count regression model.

# Arguments
- `cocoReg_fit`: A dictionary containing the model fit details. Expected keys include `"link"`, `"covariates"`, and `"parameter"`.
- `last_val` (optional): A boolean indicating whether to compute `lambda` using only the last row of covariates (default is `false`).

# Returns
The computed rate parameter(s) after applying the specified link function.
"""
function get_lambda(cocoReg_fit, last_val=false)
    link_function = get_link_function(cocoReg_fit["link"])
    if isnothing(cocoReg_fit["covariates"])
        return last(cocoReg_fit["parameter"])
    else
        if last_val
            return link_function.(sum(cocoReg_fit["covariates"][end, :] .* cocoReg_fit["parameter"][(end - size(cocoReg_fit["covariates"])[2] + 1):end]))
        else
            return link_function.(cocoReg_fit["covariates"] * cocoReg_fit["parameter"][(end - size(cocoReg_fit["covariates"])[2] + 1):end])
        end
    end
end
