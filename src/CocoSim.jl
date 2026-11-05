export cocoSim

const _MAX_INVERSE_TRANSFORM_ITERATIONS = 1_000_000

"""
    draw_random_generalized_poisson_variable(u, lambda, eta)

Draws from the Generalized Poisson distribution by inverse-transform sampling
of the uniform variate `u`.
"""
function draw_random_generalized_poisson_variable(u::Real, lambda::Real, eta::Real)
    cumulative = generalized_poisson_distribution(0, lambda, eta)
    i = 0
    while cumulative < u && i < _MAX_INVERSE_TRANSFORM_ITERATIONS
        i += 1
        cumulative += generalized_poisson_distribution(i, lambda, eta)
    end
    return i
end

"""
    draw_random_g_r_y_variable(u, y, alpha, eta, lambda)

Draws from the first-order convolution distribution `g(r | y)` by
inverse-transform sampling. The support is `0:y`.
"""
function draw_random_g_r_y_variable(u::Real, y::Integer, alpha::Real, eta::Real,
        lambda::Real)
    cumulative = compute_g_r_y(y, 0, alpha, eta, lambda)
    i = 0
    while cumulative < u && i < y
        i += 1
        cumulative += compute_g_r_y(y, i, alpha, eta, lambda)
    end
    return i
end

"""
    draw_random_g_r_y_z_variable(u, y, z, lambda, alpha1, alpha2, alpha3, eta, max_loop)

Draws from the second-order convolution distribution `g(r | y, z)` by
inverse-transform sampling. The kernel tables and the bivariate normalizer are
built once per draw instead of once per candidate value. The support is
`0:(y + z)`.
"""
function draw_random_g_r_y_z_variable(u::Real, y::Integer, z::Integer, lambda::Real,
        alpha1::Real, alpha2::Real, alpha3::Real, eta::Real, max_loop)
    kernel = GP2Kernel(lambda, alpha1, alpha2, alpha3, eta, y + z)
    smax = max_loop === nothing ? y : Int(max_loop)
    normalizer = bivariate_generalized_poisson(y, z, lambda, alpha1, alpha2, alpha3,
        eta)
    cumulative = compute_g_r_y_z(kernel, 0, y, z, normalizer, smax)
    i = 0
    while cumulative < u && i < y + z
        i += 1
        cumulative += compute_g_r_y_z(kernel, i, y, z, normalizer, smax)
    end
    return i
end

"""
    draw_random_g(order, u, y, z, lambda, alpha1, alpha2, alpha3, alpha, eta, max_loop)

Draws the autoregressive component: from `g(r | y)` for `order == 1`, from
`g(r | y, z)` for `order == 2`.
"""
function draw_random_g(order, u, y, z, lambda, alpha1, alpha2, alpha3, alpha, eta,
        max_loop)
    if Int(order) == 1
        return draw_random_g_r_y_variable(u, y, alpha, eta, lambda)
    end
    return draw_random_g_r_y_z_variable(u, y, z, lambda, alpha1, alpha2, alpha3, eta,
        max_loop)
end

#--------------------------------------------------------------------------------------
# Cached inverse-transform sampling
#
# At fixed parameters and a fixed innovation rate lambda, the conditional law of
# X_t given the history depends only on (X_{t-1}, X_{t-2}). The samplers below
# tabulate the cumulative distributions of the innovation and of g(r | y[, z])
# once per rate and history and invert them with the same uniform and the same
# stopping rule as the scalar functions above, so the draws follow the same law;
# only the order of the floating-point summation inside g(r | y, z) differs.
#--------------------------------------------------------------------------------------

# Upper bound on the number of cached cumulative-distribution entries per rate
# (Float64 values); histories beyond it are evaluated in a scratch buffer.
const _SIM_CACHE_MAX_ENTRIES = 4_194_304

"""
    _invert_cdf(cdf, u, upper)

Smallest `i` in `0:(upper - 1)` with `!(cdf[i + 1] < u)`, else `upper`. This is
the stopping rule of the sequential inverse-transform loops above.
"""
@inline function _invert_cdf(cdf::Vector{Float64}, u::Float64, upper::Int)
    @inbounds for i in 0:(upper - 1)
        cdf[i + 1] < u || return i
    end
    return upper
end

mutable struct _TransitionSampler
    order::Int
    alpha::Float64
    alpha1::Float64
    alpha2::Float64
    alpha3::Float64
    eta::Float64
    lambda::Float64
    innovation_cdf::Vector{Float64}
    g_cdfs::Dict{Tuple{Int, Int}, Vector{Float64}}
    cached_entries::Int
    tables::NTuple{5, Vector{Float64}}
    kernel::Union{Nothing, GP2Kernel{Float64}}
    kernel_max::Int
    gbuf::Vector{Float64}
    hbuf::Vector{Float64}
    scratch::Vector{Float64}
end

function _TransitionSampler(order::Int, alpha::Float64, alpha1::Float64, alpha2::Float64,
        alpha3::Float64, eta::Float64)
    return _TransitionSampler(order, alpha, alpha1, alpha2, alpha3, eta, NaN,
        Float64[], Dict{Tuple{Int, Int}, Vector{Float64}}(), 0,
        (Float64[], Float64[], Float64[], Float64[], Float64[]), nothing, -1,
        Float64[], Float64[], Float64[])
end

"""
    _set_rate!(sampler, lambda)

Switches the sampler to innovation rate `lambda`, dropping all cached tables
when the rate changes.
"""
function _set_rate!(s::_TransitionSampler, lambda::Float64)
    lambda == s.lambda && return s
    s.lambda = lambda
    empty!(s.innovation_cdf)
    empty!(s.g_cdfs)
    s.cached_entries = 0
    s.kernel = nothing
    s.kernel_max = -1
    return s
end

function _extend_innovation_cdf!(s::_TransitionSampler, new_length::Int)
    cdf = s.innovation_cdf
    k = length(cdf)
    total = k == 0 ? 0.0 : cdf[k]
    for i in k:(new_length - 1)
        total += generalized_poisson_distribution(i, s.lambda, s.eta)
        push!(cdf, total)
    end
    return cdf
end

"""
    _draw_innovation!(sampler, u)

Same draw as [`draw_random_generalized_poisson_variable`](@ref) at the
sampler's rate, from a lazily extended cumulative table.
"""
function _draw_innovation!(s::_TransitionSampler, u::Float64)
    upper = _MAX_INVERSE_TRANSFORM_ITERATIONS
    i = 0
    while i < upper
        if i >= length(s.innovation_cdf)
            _extend_innovation_cdf!(s,
                min(upper, max(32, 2 * length(s.innovation_cdf))))
        end
        @inbounds s.innovation_cdf[i + 1] < u || return i
        i += 1
    end
    return upper
end

function _ensure_kernel!(s::_TransitionSampler, needed::Int)
    if s.kernel === nothing || s.kernel_max < needed
        s.kernel = _gp2_kernel!(s.tables, s.lambda, s.alpha1, s.alpha2, s.alpha3, s.eta,
            needed)
        s.kernel_max = needed
    end
    return s.kernel::GP2Kernel{Float64}
end

"""
    _g_cdf!(cdf, sampler, y, z)

Cumulative distribution of `g(r | y)` (order 1, support `0:y`) or
`g(r | y, z)` (order 2, support `0:(y + z)`) at the sampler's rate, written
into `cdf`.
"""
function _g_cdf!(cdf::Vector{Float64}, s::_TransitionSampler, y::Int, z::Int)
    total = 0.0
    if s.order == 1
        @inbounds for r in 0:y
            total += compute_g_r_y(y, r, s.alpha, s.eta, s.lambda)
            cdf[r + 1] = total
        end
        return cdf
    end
    upper = y + z
    kernel = _ensure_kernel!(s, upper)
    g = _gp2_g_vector!(s.gbuf, s.hbuf, kernel, y, z, upper, nothing)
    @inbounds for r in 0:upper
        total += g[r + 1]
        cdf[r + 1] = total
    end
    return cdf
end

"""
    _draw_g!(sampler, u, y, z)

Same draw as [`draw_random_g`](@ref) at the sampler's rate, from a cached
cumulative distribution per history `(y[, z])`.
"""
function _draw_g!(s::_TransitionSampler, u::Float64, y::Int, z::Int)
    upper = s.order == 1 ? y : y + z
    key = s.order == 1 ? (y, 0) : (y, z)
    cdf = get(s.g_cdfs, key, nothing)
    if cdf === nothing
        if s.cached_entries + upper + 1 <= _SIM_CACHE_MAX_ENTRIES
            cdf = _g_cdf!(Vector{Float64}(undef, upper + 1), s, y, z)
            s.g_cdfs[key] = cdf
            s.cached_entries += upper + 1
        else
            cdf = _g_cdf!(resize!(s.scratch, upper + 1), s, y, z)
        end
    end
    return _invert_cdf(cdf, u, upper)
end

"""
    _simulation_setup(type, order, parameter, n, covariates, link, n_burn_in)

Rates `lambda[1:(n + n_burn_in)]` (with covariates the burn-in cycles through
the rows) and the sampler holding the autoregressive parameters.
"""
function _simulation_setup(type, order::Int, parameter, n::Int, covariates, link,
        n_burn_in::Int)
    n_total = n + n_burn_in
    if !isnothing(covariates)
        link_function = get_link_function(link)
        n_betas = size(covariates, 2)
        lambda_data = link_function.(covariates * parameter[(end - n_betas + 1):end])
        lambda = Float64.(repeat(lambda_data,
            Int(ceil(1 + n_burn_in / n)))[(end - n_total + 1):end])
    else
        lambda = fill(Float64(last(parameter)), n_total)
    end

    if order == 2
        eta = type == "GP" ? Float64(parameter[4]) : 0.0
        sampler = _TransitionSampler(2, 0.0, Float64(parameter[1]),
            Float64(parameter[2]), Float64(parameter[3]), eta)
    else
        eta = type == "GP" ? Float64(parameter[2]) : 0.0
        sampler = _TransitionSampler(1, Float64(parameter[1]), 0.0, 0.0, 0.0, eta)
    end
    return lambda, sampler
end

"""
    _simulate_paths!(X, sampler, lambda, n_burn_in)

Simulates the rows of `X` (replicates x time; columns `1:2` hold the start
values) as independent paths, time step by time step, so that all replicates
share the cached conditional distributions of the current rate. Draw order per
step and replicate: first the autoregressive part, then the innovation.
"""
function _simulate_paths!(X::AbstractMatrix{Int}, s::_TransitionSampler,
        lambda::Vector{Float64}, n_burn_in::Int)
    n_rep, n_total = size(X)
    order = s.order
    # With a burn-in of at least two steps the start values X[:, 1:2] are
    # discarded. Otherwise they are part of the output: they are then drawn from
    # the innovation distribution at lambda_1, ..., lambda_order (the RCPP
    # convention for simulations without burn-in, e.g. with covariates) instead
    # of being returned as zeros, and the recursion starts at t = order + 1.
    if n_burn_in < 2
        for t in 1:min(order, n_total)
            _set_rate!(s, lambda[t])
            for b in 1:n_rep
                X[b, t] = _draw_innovation!(s, rand())
            end
        end
        t_first = order + 1
    else
        t_first = 3
    end

    for t in t_first:n_total
        _set_rate!(s, lambda[t])
        for b in 1:n_rep
            y = X[b, t - 1]
            z = t > 2 ? X[b, t - 2] : 0
            r = _draw_g!(s, rand(), y, z)
            X[b, t] = r + _draw_innovation!(s, rand())
        end
    end
    return X
end

"""
    cocoSim(type, order, parameter, n, covariates=nothing, link="log", n_burn_in=200,
            x=zeros(Int(n + n_burn_in)))

Simulates a count time series of length `n` from a (Generalized) Poisson
autoregressive model, discarding `n_burn_in` initial observations. With
covariates, observation `t` of the output uses covariate row `t` (the burn-in
cycles through the rows). With `n_burn_in = 0` the first `order` observations
are drawn from the innovation distribution at their covariate rows. Parameter
layout: `alpha[, eta], lambda` for order 1 and
`alpha1, alpha2, alpha3[, eta], lambda` for order 2; with covariates the
trailing `lambda` is replaced by the regression coefficients.
"""
function cocoSim(type, order, parameter, n, covariates = nothing, link = "log",
        n_burn_in = 200, x = zeros(Int(n + n_burn_in)))
    n = Int(n)
    n_burn_in = Int(n_burn_in)
    order = Int(order)
    n_total = n + n_burn_in
    lambda, sampler = _simulation_setup(type, order, parameter, n, covariates, link,
        n_burn_in)

    X = Matrix{Int}(undef, 1, n_total)
    for t in 1:n_total
        X[1, t] = Int(x[t])
    end
    _simulate_paths!(X, sampler, lambda, n_burn_in)
    for t in 1:n_total
        x[t] = X[1, t]
    end

    return x[(end - n + 1):end]
end
