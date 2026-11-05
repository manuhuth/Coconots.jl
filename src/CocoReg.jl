export cocoReg

function _validate_reg_inputs(type, order)
    type in ("GP", "Poisson") ||
        throw(ArgumentError("type must be \"GP\" or \"Poisson\", got \"$type\""))
    Int(order) in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
    return Int(order)
end

"""
    _check_covariates_full_rank(covariates)

Throws an `ArgumentError` naming the aliased columns if the covariate matrix is
rank deficient: the regression coefficients are then not identified and the
GLM starting values are undefined. The rank is that of the column-pivoted QR
decomposition with relative tolerance `1e-7` (the tolerance of R's `lm`); the
aliased columns are those pivoted behind the rank. Matrices with non-finite
entries are left to the later checks.
"""
function _check_covariates_full_rank(covariates)
    X = covariates isa AbstractVector ? reshape(Float64.(covariates), :, 1) :
        Matrix{Float64}(covariates)
    k = size(X, 2)
    (k == 0 || !all(isfinite, X)) && return nothing
    F = qr(X, ColumnNorm())
    d = abs.(diag(F.R))
    r = (isempty(d) || d[1] == 0) ? 0 : count(v -> v > 1e-7 * d[1], d)
    if r < k
        aliased = sort(F.p[(r + 1):k])
        throw(ArgumentError("xreg is rank deficient: column(s) $(join(aliased, ", ")) " *
                            "are linear combinations of others (rank $r < $k columns); " *
                            "the model is not identified. Remove the aliased column(s)."))
    end
    return nothing
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

"""
    _solve_default_with_fallback(problem, optimizer, allow_fallback; solve_kwargs...)

Solves `problem` with `optimizer`. With `allow_fallback` (the default
optimizer `LBFGS()`, Hager-Zhang line search), a run that throws, e.g. when
the line search reaches a non-finite objective or gradient inside its
bracket, or that ends at a non-finite objective is repeated from the same
start with `LBFGS` and a backtracking line search, which halves the step
until the objective is finite.
"""
function _solve_default_with_fallback(problem, optimizer, allow_fallback::Bool;
        solve_kwargs...)
    allow_fallback || return solve(problem, optimizer; solve_kwargs...)
    solution = try
        solve(problem, optimizer; solve_kwargs...)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    if solution !== nothing && isfinite(solution.objective)
        return solution
    end
    return solve(problem, LBFGS(linesearch = LineSearches.BackTracking());
        solve_kwargs...)
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
    _pull_inside(x, lower, upper, margin)

Moves `x` at least `margin` (relative to the width for two-sided bounds,
absolute for one-sided bounds) inside its bounds.
"""
function _pull_inside(x::Real, lower::Real, upper::Real, margin::Real)
    if isfinite(lower) && isfinite(upper)
        width = (upper - lower) * margin
        return clamp(x, lower + width, upper - width)
    elseif isfinite(lower)
        return max(x, lower + margin)
    elseif isfinite(upper)
        return min(x, upper - margin)
    end
    return x
end

const _RESTART_MARGIN = 1e-4
const _MAX_RESTARTS = 3

"""
    _restart_near_bounds(solution, optimization_function, optimizer, allow_fallback,
                         lo, hi; solve_kwargs...)

Restarts the unconstrained solve from its solution (at most `_MAX_RESTARTS`
times). On the transformed scale the gradient of a coordinate vanishes as it
approaches its bound, so a solve can stop at a bound although the optimum is
interior (the coordinate cannot move back), or stall near a bound while the
objective still decreases. The first restart therefore starts from the
solution with every coordinate within `_RESTART_MARGIN` of a bound moved that
far inside; later restarts start from the current solution as it is, with
fresh curvature memory, and stop once the objective no longer decreases by
more than `1e-10`. A restart is kept only if it lowers the objective. At an
interior stationary point a restart stops at its first gradient check. A
restart that started from the current solution and did not improve ends the
restarts (the next one would repeat the same deterministic solve).
Restarts use [`_solve_default_with_fallback`](@ref) with `allow_fallback`; a
restart that still throws ends the restarts and keeps the current solution.
"""
function _restart_near_bounds(solution, optimization_function, optimizer,
        allow_fallback::Bool, lo, hi; solve_kwargs...)
    for k in 1:_MAX_RESTARTS
        u_restart = copy(solution.u)
        if k == 1
            x = _to_bounded.(solution.u, lo, hi)
            x_inside = _pull_inside.(x, lo, hi, _RESTART_MARGIN)
            # no coordinate near a bound: start from solution.u itself instead of
            # its (rounding-level different) transform round trip
            x_inside == x || (u_restart = _to_unconstrained.(x_inside, lo, hi))
        end
        restart = try
            _solve_default_with_fallback(
                OptimizationProblem(optimization_function, u_restart), optimizer,
                allow_fallback; solve_kwargs...)
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
        restart === nothing && break
        improvement = isfinite(solution.objective) ?
                      solution.objective - restart.objective :
                      (isfinite(restart.objective) ? Inf : 0.0)
        # The next restart starts from solution.u; if this one started there too
        # (k = 1 with no coordinate near a bound) and did not improve, the
        # deterministic solve would only repeat itself.
        !(improvement > 0) && u_restart == solution.u && break
        improvement > 0 && (solution = restart)
        k > 1 && !(improvement > 1e-10) && break
    end
    return solution
end

const _KKT_INSIDE_MARGIN = 1e-2
const _KKT_GRADIENT_TOL = 1e-6
const _MAX_KKT_RESTARTS = 2

"""
    _bound_kkt_violations(objective, x, lo, hi)

Coordinates of `x` (bounded scale of [`_to_bounded`](@ref)) that lie within
`_RESTART_MARGIN` of a bound (relative to the width for two-sided bounds,
absolute for one-sided bounds) although the objective decreases into the
interior there: gradient below `-_KKT_GRADIENT_TOL` at a lower bound, above
`_KKT_GRADIENT_TOL` at an upper bound. At a minimum on a bound the gradient
points out of the box (Karush-Kuhn-Tucker condition); a violated coordinate
is a point where the unconstrained solve stalled because the gradient on the
transformed scale vanishes at the bound. Returns an empty vector if the
gradient is not finite.
"""
function _bound_kkt_violations(objective, x, lo, hi)
    g = try
        ForwardDiff.gradient(objective, x)
    catch err
        err isa InterruptException && rethrow()
        return Int[]
    end
    all(isfinite, g) || return Int[]
    near_lower(i) = isfinite(lo[i]) &&
                    x[i] - lo[i] <= _RESTART_MARGIN * (isfinite(hi[i]) ? hi[i] - lo[i] : 1.0)
    near_upper(i) = isfinite(hi[i]) &&
                    hi[i] - x[i] <= _RESTART_MARGIN * (isfinite(lo[i]) ? hi[i] - lo[i] : 1.0)
    return [i for i in eachindex(x)
            if (near_lower(i) && g[i] < -_KKT_GRADIENT_TOL) ||
               (near_upper(i) && g[i] > _KKT_GRADIENT_TOL)]
end

"""
    _restart_kkt_bounds(solution, objective, optimization_function, optimizer,
                        allow_fallback, lo, hi; solve_kwargs...)

Restarts the unconstrained solve (at most `_MAX_KKT_RESTARTS` times) while
its solution has coordinates at a bound at which the objective still
decreases into the interior ([`_bound_kkt_violations`](@ref)). Such a
coordinate is moved `_KKT_INSIDE_MARGIN` inside its bound (relative to the
width for two-sided bounds, absolute for one-sided bounds), the other
coordinates are kept, and the problem is solved from there followed by
[`_restart_near_bounds`](@ref). The near-bound restarts of
[`_restart_near_bounds`](@ref) run before the solve has located the
interior coordinates, so a coordinate can return to its bound and stall
there; once the other coordinates are (near) optimal, the gradient at the
bound tells whether the bound is optimal. A restart is kept only if it lowers
the objective; a restart that throws ends the restarts.
"""
function _restart_kkt_bounds(solution, objective, optimization_function, optimizer,
        allow_fallback::Bool, lo, hi; solve_kwargs...)
    for _ in 1:_MAX_KKT_RESTARTS
        isfinite(solution.objective) || break
        x = _to_bounded.(solution.u, lo, hi)
        violated = _bound_kkt_violations(objective, x, lo, hi)
        isempty(violated) && break
        x_restart = copy(x)
        for i in violated
            x_restart[i] = _pull_inside(x[i], lo[i], hi[i], _KKT_INSIDE_MARGIN)
        end
        restart = try
            r = _solve_default_with_fallback(
                OptimizationProblem(optimization_function,
                    _to_unconstrained.(x_restart, lo, hi)), optimizer, allow_fallback;
                solve_kwargs...)
            _restart_near_bounds(r, optimization_function, optimizer, allow_fallback,
                lo, hi; solve_kwargs...)
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
        restart === nothing && break
        restart.objective < solution.objective || break
        solution = restart
    end
    return solution
end

"""
    _PrimalValueCache(f, n)

Callable wrapper of the objective `f` for the optimizer. OptimizationOptimJL
evaluates `fg!` (no fused `fg` is generated for Optim.jl solvers) as a
ForwardDiff gradient followed by a separate plain evaluation of `f` at the
same point, although the gradient pass already carries that value as the
primal part of its result. A first-order dual evaluation therefore records
its point and primal value, and a plain `Float64` evaluation at exactly that
point returns the recorded value instead of evaluating `f` again (one
objective pass fewer per gradient call). Any other input, and any plain
evaluation at a different point, is passed through to `f`.
"""
mutable struct _PrimalValueCache{F}
    f::F
    u::Vector{Float64}
    value::Float64
    valid::Bool
end

_PrimalValueCache(f::F, n::Integer) where {F} = _PrimalValueCache{F}(f,
    Vector{Float64}(undef, n), NaN, false)

function (cache::_PrimalValueCache)(u::AbstractVector{<:ForwardDiff.Dual{<:Any, Float64}})
    y = cache.f(u)
    if y isa ForwardDiff.Dual && ForwardDiff.value(y) isa Float64 &&
       length(u) == length(cache.u)
        @inbounds for i in eachindex(cache.u, u)
            cache.u[i] = ForwardDiff.value(u[i])
        end
        cache.value = ForwardDiff.value(y)
        cache.valid = true
    else
        cache.valid = false
    end
    return y
end

function (cache::_PrimalValueCache)(u::AbstractVector{Float64})
    if cache.valid && length(u) == length(cache.u) && u == cache.u
        return cache.value
    end
    return cache.f(u)
end

(cache::_PrimalValueCache)(u::AbstractVector) = cache.f(u)

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
    # same input checks as coconots::cocoReg: starting values from the lag-order
    # autocorrelation and at least two transitions; counts up to 170; a full-rank
    # covariate matrix
    _check_min_length(length(data), order + 2, "cocoReg",
        "order-$order model: starting values from the lag-$order autocorrelation, " *
        "at least two transitions in the likelihood")
    _check_count_limit(data)
    isnothing(covariates) || _check_covariates_full_rank(covariates)
    data_int = Int.(data)

    starting_values = get_starting_values!(type, order, data_int, covariates,
        starting_values, _n_parameters_without_covariates(type, order),
        lower_bound_covariates, link_function)
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
        default_optimizer = optimizer === nothing
        optimizer = default_optimizer ? LBFGS() : optimizer
        lo, hi = Float64.(lower), Float64.(upper)
        u0 = _to_unconstrained.(Float64.(starting_values), lo, hi)
        cached = _PrimalValueCache(u -> objective(_to_bounded.(u, lo, hi)), length(u0))
        optimization_function = OptimizationFunction((u, _) -> cached(u), adtype)
        problem = OptimizationProblem(optimization_function, u0)
        solution = _solve_default_with_fallback(problem, optimizer, default_optimizer;
            solve_kwargs...)
        solution = _restart_near_bounds(solution, optimization_function, optimizer,
            default_optimizer, lo, hi; solve_kwargs...)
        solution = _restart_kkt_bounds(solution, objective, optimization_function,
            optimizer, default_optimizer, lo, hi; solve_kwargs...)
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
