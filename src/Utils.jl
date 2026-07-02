"""
    compute_beta_i(lambda, U, alpha_i)

Derived rate `beta_i = lambda * U * alpha_i` of the second-order model.
"""
compute_beta_i(lambda::Real, U::Real, alpha_i::Real) = lambda * U * alpha_i

"""
    compute_U(alpha1, alpha2, alpha3)

Scaling factor `U = 1 / (1 - alpha1 - alpha2 - alpha3)`.
"""
compute_U(alpha1::Real, alpha2::Real, alpha3::Real) = 1 / (1 - alpha1 - alpha2 - alpha3)

"""
    compute_zeta(lambda, U, alpha1, alpha3)

Derived rate `zeta = lambda * U * (1 - 2*alpha1 - alpha3)`.
"""
compute_zeta(lambda::Real, U::Real, alpha1::Real, alpha3::Real) = lambda * U *
                                                                  (1 - 2 * alpha1 - alpha3)

"""
    compute_inverse_matrix(M)

Inverse of the square matrix `M`.
"""
compute_inverse_matrix(M::AbstractMatrix) = inv(M)

"""
    compute_hessian(f, x)

Hessian of `f` at `x` via ForwardDiff.
"""
compute_hessian(f, x::AbstractVector) = ForwardDiff.hessian(f, x)

"""
    compute_mu_hat_gmm(data)

GMM estimator based on centered third-order moments of `data`.
"""
function compute_mu_hat_gmm(data::AbstractVector)
    total = 0.0
    x_bar = mean(data)
    for t in 3:length(data)
        total += (data[t] - x_bar) * (data[t - 1] - x_bar) * (data[t - 2] - x_bar)
    end
    return total / length(data)
end

"""
    compute_autocorrelation(data, order)

Pearson correlation between `data` and its `order`-lagged version.
"""
function compute_autocorrelation(data::AbstractVector, order::Integer)
    x_t = @view data[1:(length(data) - order)]
    x_lag = @view data[(order + 1):length(data)]
    return cor(x_t, x_lag)
end

"""
    set_to_unit_interval(x)

Clamps `x` to `[0.0001, 0.9999]`.
"""
set_to_unit_interval(x::Real) = clamp(x, 0.0001, 0.9999)

"""
    reparameterize_alpha(parameter)

Maps the stationarity-preserving optimization parameters of the second-order
model back to `(alpha1, alpha2, alpha3)`:
`alpha3 = p1 * p2`, `alpha1 = (p1 - alpha3) / 2`,
`alpha2 = (1 - alpha1 - alpha3) * p3`.
"""
function reparameterize_alpha(parameter::AbstractVector)
    alpha3 = parameter[1] * parameter[2]
    alpha1 = (parameter[1] - alpha3) / 2
    alpha2 = (1 - alpha1 - alpha3) * parameter[3]
    return alpha1, alpha2, alpha3
end
