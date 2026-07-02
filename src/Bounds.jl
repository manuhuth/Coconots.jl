function _covariate_bounds(covariates, lower_bound_covariates)
    covariates === nothing && return [0.0], [Inf]
    n_betas = size(covariates, 2)
    return fill(Float64(lower_bound_covariates), n_betas), fill(Inf, n_betas)
end

"""
    get_bounds_GP1(type, covariates, lower_bound_covariates)

Box constraints `(lower, upper)` of the first-order parameter vector:
`alpha[, eta], lambda` (or the covariate coefficients in place of `lambda`).
"""
function get_bounds_GP1(type, covariates, lower_bound_covariates)
    lambda_lower, lambda_upper = _covariate_bounds(covariates, lower_bound_covariates)
    n_base = type == "GP" ? 2 : 1
    return vcat(zeros(n_base), lambda_lower), vcat(ones(n_base), lambda_upper)
end

"""
    get_bounds_GP2(type, covariates, lower_bound_covariates)

Box constraints `(lower, upper)` of the second-order parameter vector:
`alpha1, alpha2, alpha3[, eta], lambda` (or the covariate coefficients in
place of `lambda`).
"""
function get_bounds_GP2(type, covariates, lower_bound_covariates)
    lambda_lower, lambda_upper = _covariate_bounds(covariates, lower_bound_covariates)
    n_base = type == "GP" ? 4 : 3
    return vcat(zeros(n_base), lambda_lower), vcat(ones(n_base), lambda_upper)
end

"""
    get_bounds(order, type, covariates, lower_bound_covariates)

Box constraints for the parameter vector of the given model order and type.
"""
function get_bounds(order, type, covariates, lower_bound_covariates)
    if Int(order) == 1
        return get_bounds_GP1(type, covariates, lower_bound_covariates)
    end
    return get_bounds_GP2(type, covariates, lower_bound_covariates)
end
