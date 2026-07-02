export cocoReg

function _validate_reg_inputs(type, order)
    type in ("GP", "Poisson") ||
        throw(ArgumentError("type must be \"GP\" or \"Poisson\", got \"$type\""))
    Int(order) in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
    return Int(order)
end

function _n_parameters_without_covariates(type, order)
    Int(order) == 1 ? (type == "GP" ? 2 : 1) : (type == "GP" ? 4 : 3)
end

function _lambda_vector(theta, first_index::Int, covariates, link_function::F) where {F}
    n_betas = size(covariates, 2)
    return link_function.(covariates * view(theta, first_index:(first_index + n_betas - 1)))
end

"""
    _make_objective(type, order, data, covariates, link_function, max_loop, reparameterized)

Builds the negative log-likelihood objective `theta -> nll(theta)` for the
requested model. For constant-rate models the unique data transitions and
their multiplicities are precomputed once here, outside the optimization
loop, so every objective evaluation only pays for distinct transition
probabilities. `reparameterized` selects the stationarity-preserving
reparameterization of the second-order alphas used during optimization.
"""
function _make_objective(type, order, data::Vector{Int}, covariates, link_function,
        max_loop, reparameterized::Bool)
    is_gp = type == "GP"
    if covariates === nothing
        if Int(order) == 1
            transitions, counts = _transition_pair_counts(data)
            if is_gp
                return theta -> _nll_gp1_transitions(theta[3], theta[1], theta[2],
                    transitions, counts)
            end
            return theta -> _nll_gp1_transitions(theta[2], theta[1],
                zero(eltype(theta)), transitions, counts)
        end
        transitions, counts = _transition_triple_counts(data)
        max_index = _gp2_max_index(transitions)
        groups = _group_transitions(transitions, counts)
        lambda_index = is_gp ? 5 : 4
        return function (theta)
            alpha1, alpha2, alpha3 = reparameterized ? reparameterize_alpha(theta) :
                                     (theta[1], theta[2], theta[3])
            eta = is_gp ? theta[4] : zero(eltype(theta))
            return _nll_gp2_transitions(theta[lambda_index], alpha1, alpha2, alpha3,
                eta, groups, max_index, max_loop)
        end
    end
    link = get_link_function(link_function)
    if Int(order) == 1
        lambda_index = is_gp ? 3 : 2
        return function (theta)
            lambdas = _lambda_vector(theta, lambda_index, covariates, link)
            eta = is_gp ? theta[2] : zero(eltype(theta))
            return compute_negative_log_likelihood_GP1(lambdas, theta[1], eta, data)
        end
    end
    lambda_index = is_gp ? 5 : 4
    return function (theta)
        lambdas = _lambda_vector(theta, lambda_index, covariates, link)
        alpha1, alpha2, alpha3 = reparameterized ? reparameterize_alpha(theta) :
                                 (theta[1], theta[2], theta[3])
        eta = is_gp ? theta[4] : zero(eltype(theta))
        return compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, eta,
            data, max_loop)
    end
end

# Backward-compatible objective wrappers (parameter layouts documented in cocoReg).
function minimize_pars_GP1(theta, data, covariates = nothing, link = "log",
        max_loop = nothing)
    if covariates === nothing
        return compute_negative_log_likelihood_GP1(theta[3], theta[1], theta[2],
            Int.(data))
    end
    lambdas = _lambda_vector(theta, 3, covariates, get_link_function(link))
    return compute_negative_log_likelihood_GP1(lambdas, theta[1], theta[2], Int.(data))
end

function minimize_pars_Poisson1(theta, data, covariates = nothing, link = "log",
        max_loop = nothing)
    eta = zero(eltype(theta))
    if covariates === nothing
        return compute_negative_log_likelihood_GP1(theta[2], theta[1], eta, Int.(data))
    end
    lambdas = _lambda_vector(theta, 2, covariates, get_link_function(link))
    return compute_negative_log_likelihood_GP1(lambdas, theta[1], eta, Int.(data))
end

function minimize_pars_GP2(theta, data, covariates = nothing, link = "log",
        max_loop = nothing)
    if covariates === nothing
        return compute_negative_log_likelihood_GP2(theta[5], theta[1], theta[2],
            theta[3], theta[4], Int.(data), max_loop)
    end
    lambdas = _lambda_vector(theta, 5, covariates, get_link_function(link))
    return compute_negative_log_likelihood_GP2(lambdas, theta[1], theta[2], theta[3],
        theta[4], Int.(data), max_loop)
end

function minimize_pars_Poisson2(theta, data, covariates = nothing, link = "log",
        max_loop = nothing)
    eta = zero(eltype(theta))
    if covariates === nothing
        return compute_negative_log_likelihood_GP2(theta[4], theta[1], theta[2],
            theta[3], eta, Int.(data), max_loop)
    end
    lambdas = _lambda_vector(theta, 4, covariates, get_link_function(link))
    return compute_negative_log_likelihood_GP2(lambdas, theta[1], theta[2], theta[3],
        eta, Int.(data), max_loop)
end

function minimize_pars_reparameterization_GP2(theta, data, covariates = nothing,
        link = "log", max_loop = nothing)
    alpha1, alpha2, alpha3 = reparameterize_alpha(theta)
    if covariates === nothing
        return compute_negative_log_likelihood_GP2(theta[5], alpha1, alpha2, alpha3,
            theta[4], Int.(data), max_loop)
    end
    lambdas = _lambda_vector(theta, 5, covariates, get_link_function(link))
    return compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3,
        theta[4], Int.(data), max_loop)
end

function minimize_pars_reparameterization_Poisson2(theta, data, covariates = nothing,
        link = "log", max_loop = nothing)
    alpha1, alpha2, alpha3 = reparameterize_alpha(theta)
    eta = zero(eltype(theta))
    if covariates === nothing
        return compute_negative_log_likelihood_GP2(theta[4], alpha1, alpha2, alpha3,
            eta, Int.(data), max_loop)
    end
    lambdas = _lambda_vector(theta, 4, covariates, get_link_function(link))
    return compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, eta,
        Int.(data), max_loop)
end

_logistic(u::Real) = 1 / (1 + exp(-u))
_logit(x::Real) = log(x) - log1p(-x)

"""
    _to_bounded(u, lower, upper)

Maps the unconstrained value `u` into `(lower, upper)`: logistic scaling for
two-sided bounds, `lower + exp(u)` / `upper - exp(u)` for one-sided bounds,
identity when unbounded. Smooth in `u`, so any AD backend differentiates
through it.
"""
function _to_bounded(u::Real, lower::Real, upper::Real)
    if isfinite(lower) && isfinite(upper)
        return lower + (upper - lower) * _logistic(u)
    elseif isfinite(lower)
        return lower + exp(u)
    elseif isfinite(upper)
        return upper - exp(u)
    end
    return u
end

"""
    _to_unconstrained(x, lower, upper)

Inverse of [`_to_bounded`](@ref); values at (or numerically outside) the
bounds are pulled strictly inside before transforming.
"""
function _to_unconstrained(x::Real, lower::Real, upper::Real)
    margin = 1e-8
    if isfinite(lower) && isfinite(upper)
        return _logit(clamp((x - lower) / (upper - lower), margin, 1 - margin))
    elseif isfinite(lower)
        return log(max(x - lower, margin))
    elseif isfinite(upper)
        return log(max(upper - x, margin))
    end
    return x
end

"""
    cocoReg(type, order, data, covariates=nothing, starting_values=nothing,
            link_function="log", lower_bound_covariates=-Inf, max_loop=nothing,
            optimizer=nothing; adtype=Optimization.AutoForwardDiff(),
            box_constrained=false, solve_kwargs...)

Fits a (Generalized) Poisson autoregressive count model of the given `order`
by maximum likelihood.

The optimization runs through the Optimization.jl interface: the objective is
wrapped in an `OptimizationFunction` with the automatic-differentiation
backend `adtype` (any `ADTypes.AbstractADType`, e.g.
`Optimization.AutoForwardDiff()` or — with Enzyme loaded — `AutoEnzyme()`).
Extra keyword arguments are forwarded to `Optimization.solve` (e.g.
`maxiters`, `reltol`).

By default the parameter constraints are handled by smooth parameter
transformations (logistic for two-sided bounds, shifted `exp` for one-sided
bounds) and the problem is solved unconstrained with `optimizer` (default:
`LBFGS()` from Optim.jl; any unconstrained Optimization.jl solver works).
With `box_constrained = true` the constraints are passed to the solver as box
bounds instead (default solver then: `Fminbox(LBFGS())`).

For second-order models the optimizer additionally works on a
stationarity-preserving reparameterization of the alphas; the returned
parameters, covariance matrix and standard errors are always on the original
scale.

Returns a `Dict` with keys `"parameter"`, `"covariance_matrix"`, `"se"`,
`"log_likelihood"`, `"type"`, `"order"`, `"data"`, `"covariates"`, `"link"`,
`"starting_values"`, `"optimizer"`, `"lower_bounds"`, `"upper_bounds"`,
`"optimization"` and `"max_loop"`.
"""
function cocoReg(type, order, data, covariates = nothing, starting_values = nothing,
        link_function = "log", lower_bound_covariates = -Inf, max_loop = nothing,
        optimizer = nothing; adtype = Optimization.AutoForwardDiff(),
        box_constrained::Bool = false, solve_kwargs...)
    order = _validate_reg_inputs(type, order)
    data_int = Int.(data)

    starting_values = get_starting_values!(type, order, data_int, covariates,
        starting_values, _n_parameters_without_covariates(type, order),
        lower_bound_covariates)
    lower, upper = get_bounds(order, type, covariates, lower_bound_covariates)

    objective = _make_objective(type, order, data_int, covariates, link_function,
        max_loop, order == 2)
    f_alphas = _make_objective(type, order, data_int, covariates, link_function,
        max_loop, false)

    if box_constrained
        optimizer = optimizer === nothing ? Fminbox(LBFGS()) : optimizer
        optimization_function = OptimizationFunction((theta, _) -> objective(theta),
            adtype)
        problem = OptimizationProblem(optimization_function, Float64.(starting_values);
            lb = Float64.(lower), ub = Float64.(upper))
        solution = solve(problem, optimizer; solve_kwargs...)
        parameter = copy(solution.u)
    else
        optimizer = optimizer === nothing ? LBFGS() : optimizer
        lo, hi = Float64.(lower), Float64.(upper)
        u0 = _to_unconstrained.(Float64.(starting_values), lo, hi)
        optimization_function = OptimizationFunction(
            (u, _) -> objective(_to_bounded.(u, lo, hi)), adtype)
        problem = OptimizationProblem(optimization_function, u0)
        solution = solve(problem, optimizer; solve_kwargs...)
        parameter = _to_bounded.(solution.u, lo, hi)
    end

    if order == 2
        alpha1, alpha2, alpha3 = reparameterize_alpha(parameter)
        parameter[1:3] = [alpha1, alpha2, alpha3]
    end

    inv_hessian = compute_inverse_matrix(compute_hessian(f_alphas, parameter))
    for i in 1:size(inv_hessian, 1)
        if inv_hessian[i, i] < 0
            inv_hessian[i, i] = 1e-12
        end
    end

    out = Dict{String, Any}("parameter" => parameter,
        "covariance_matrix" => inv_hessian,
        "log_likelihood" => -f_alphas(parameter),
        "type" => type,
        "order" => order,
        "data" => data,
        "covariates" => covariates,
        "link" => link_function,
        "starting_values" => starting_values,
        "optimizer" => optimizer,
        "lower_bounds" => lower,
        "upper_bounds" => upper,
        "optimization" => solution,
        "max_loop" => max_loop)
    out["se"] = sqrt.(diag(inv_hessian))
    return out
end
