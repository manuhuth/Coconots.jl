using LinearAlgebra

"""
    compute_beta_i(lambda, U, alpha_i)

Computes beta_i as the product of lambda, U, and alpha_i.

# Arguments
- `lambda`: The rate parameter.
- `U`: A scaling factor.
- `alpha_i`: An autoregressive parameter.

# Returns
The computed beta_i value.
"""
function compute_beta_i(lambda, U, alpha_i)
    return lambda * U * alpha_i
end

"""
    compute_U(alpha1, alpha2, alpha3)

Computes the scaling factor U based on the autoregressive parameters.

# Arguments
- `alpha1`: The first autoregressive parameter.
- `alpha2`: The second autoregressive parameter.
- `alpha3`: The third autoregressive parameter.

# Returns
The computed value of U, defined as 1 / (1 - alpha1 - alpha2 - alpha3).
"""
function compute_U(alpha1, alpha2, alpha3)
    return 1 / (1 - alpha1 - alpha2 - alpha3)
end

"""
    compute_zeta(lambda, U, alpha1, alpha3)

Computes zeta using the provided parameters.

# Arguments
- `lambda`: The rate parameter.
- `U`: A scaling factor computed from autoregressive parameters.
- `alpha1`: The first autoregressive parameter.
- `alpha3`: The third autoregressive parameter.

# Returns
The computed value of zeta, defined as lambda * U * (1 - 2*alpha1 - alpha3).
"""
function compute_zeta(lambda, U, alpha1, alpha3)
    return lambda * U * (1 - 2 * alpha1 - alpha3)
end

"""
    compute_inverse_matrix(M)

Computes the inverse of the given matrix M.

# Arguments
- `M`: A square matrix.

# Returns
The inverse of matrix M.
"""
function compute_inverse_matrix(M)
    return inv(M)
end

"""
    compute_hessian(f, x)

Computes the Hessian matrix of a function f at the point x using automatic differentiation.

# Arguments
- `f`: A function for which the Hessian is to be computed.
- `x`: The point (vector) at which the Hessian is evaluated.

# Returns
The Hessian matrix of f evaluated at x.
"""
function compute_hessian(f, x)
    return ForwardDiff.hessian(f, x)
end

"""
    compute_mu_hat_gmm(data)

Computes the GMM estimator (mu_hat) based on third-order moments of the data.

# Arguments
- `data`: A vector of data points.

# Returns
The computed mu_hat, defined as the average of the product of deviations 
(data[t] - x̄) * (data[t-1] - x̄) * (data[t-2] - x̄) over t = 3 to length(data).
"""
function compute_mu_hat_gmm(data)
    sum = 0.0
    x_bar = mean(data)
    for t in eachindex(data)[3:end]
        sum = sum + (data[t] - x_bar) * (data[t-1] - x_bar) * (data[t-2] - x_bar)
    end
    return sum / length(data)
end

"""
    compute_autocorrelation(data, order)

Computes the autocorrelation of the data with a specified lag order.

# Arguments
- `data`: A vector of data points.
- `order`: The lag order to compute the autocorrelation.

# Returns
The Pearson correlation coefficient between the original data and its lagged version.
"""
function compute_autocorrelation(data, order)
    x_t = data[1:(length(data)-order)]
    x_lag = data[(order+1):length(data)]
    return cor(x_t, x_lag)
end

"""
    set_to_unit_interval(x)

Constrains the value x to be within the unit interval [0.0001, 0.9999].

# Arguments
- `x`: A numeric value.

# Returns
x clamped to the interval [0.0001, 0.9999].
"""
function set_to_unit_interval(x)
    return max(min(x, 0.9999), 0.0001)
end

"""
    reparameterize_alpha(parameter)

Reparameterizes the alpha parameters from a given parameter vector.

# Arguments
- `parameter`: A vector containing at least three elements used for reparameterization.

# Returns
A tuple `(alpha1, alpha2, alpha3)` computed as follows:
- `alpha3` = parameter[1] * parameter[2]
- `alpha1` = (parameter[1] - alpha3) / 2
- `alpha2` = (1 - alpha1 - alpha3) * parameter[3]
"""
function reparameterize_alpha(parameter)
    alpha3 = parameter[1] * parameter[2]
    alpha1 = (parameter[1] - alpha3) / 2
    alpha2 = (1 - alpha1 - alpha3) * parameter[3]
    return alpha1, alpha2, alpha3
end
