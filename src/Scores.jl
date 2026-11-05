export cocoPit, compute_scores

function _fit_eta(cocoReg_fit)
    order = Int(cocoReg_fit["order"])
    cocoReg_fit["type"] == "Poisson" && return 0.0
    return Float64(cocoReg_fit["parameter"][order == 1 ? 2 : 4])
end

function _fit_lambdas(cocoReg_fit, n::Integer)
    lambda = get_lambda(cocoReg_fit, false)
    isnothing(cocoReg_fit["covariates"]) && return fill(Float64(lambda), n)
    return Float64.(lambda)
end

"""
    _constant_kernel(cocoReg_fit, data, lambdas, eta, xmax_hint)

One shared [`GP2Kernel`](@ref) for all observations of a constant-rate
second-order fit (sized to cover every transition and `xmax_hint`), `nothing`
otherwise.
"""
function _constant_kernel(cocoReg_fit, data::Vector{Int}, lambdas, eta::Real,
        xmax_hint::Integer)
    (Int(cocoReg_fit["order"]) == 2 && isnothing(cocoReg_fit["covariates"])) ||
        return nothing
    parameter = cocoReg_fit["parameter"]
    return GP2Kernel(lambdas[1], parameter[1], parameter[2], parameter[3], eta,
        max(xmax_hint, 2 * maximum(data)))
end

"""
    _transition_pmf(cocoReg_fit, t, xmax, data, lambdas, eta, kernel)

Transition pmf `P(x | past)` for observation `t` on `0:xmax`, dispatching on
the model order. Reuses `kernel` when one was prebuilt by
[`_constant_kernel`](@ref).
"""
function _transition_pmf(cocoReg_fit, t::Integer, xmax::Integer,
        data::Vector{Int}, lambdas, eta::Real, kernel)
    parameter = cocoReg_fit["parameter"]
    if Int(cocoReg_fit["order"]) == 1
        return convolution_pmf_vector(xmax, data[t - 1], lambdas[t], parameter[1], eta)
    end
    y, z = data[t - 1], data[t - 2]
    if kernel === nothing
        kernel = GP2Kernel(lambdas[t], parameter[1], parameter[2], parameter[3], eta,
            max(xmax, y + z))
    end
    return convolution_pmf_vector(kernel, xmax, y, z, cocoReg_fit["max_loop"])
end

_history_key(data::Vector{Int}, t::Integer, order::Integer) =
    order == 1 ? (data[t - 1], 0) : (data[t - 1], data[t - 2])

"""
    _history_pmfs(cocoReg_fit, data, lambdas, eta, kernel, xmax_of)

For a constant-rate fit the transition pmf of observation `t` depends only on
its history (`x_{t-1}`, and `x_{t-2}` for order 2). Returns a `Dict` from the
history (see `_history_key`) to its pmf on `0:xmax`, with `xmax` the largest
`xmax_of(t)` over the observations sharing that history, so each pmf is
computed once per history instead of once per observation. The entries of
[`convolution_pmf_vector`](@ref) on `0:m` do not depend on the length of the
requested support beyond `m` (same terms, same summation order), so every
value equals the per-observation computation. Returns `nothing` for covariate
fits (observation-specific rates).
"""
function _history_pmfs(cocoReg_fit, data::Vector{Int}, lambdas, eta::Real, kernel,
        xmax_of::F) where {F}
    isnothing(cocoReg_fit["covariates"]) || return nothing
    order = Int(cocoReg_fit["order"])
    first_t = order + 1
    xmax_needed = Dict{NTuple{2, Int}, Int}()
    for t in first_t:length(data)
        key = _history_key(data, t, order)
        xmax_needed[key] = max(get(xmax_needed, key, 0), Int(xmax_of(t)))
    end
    pmfs = Dict{NTuple{2, Int}, Any}()
    for t in first_t:length(data)
        key = _history_key(data, t, order)
        haskey(pmfs, key) && continue
        pmfs[key] = _transition_pmf(cocoReg_fit, t, xmax_needed[key], data, lambdas, eta,
            kernel)
    end
    return pmfs
end

"""
    _score_terms(pmf, x, max_x)

Per-observation terms of [`compute_scores`](@ref) from the transition pmf
(support at least `0:max(max_x, x)`): `log(pmf[x])`, `sum(pmf^2) - 2 pmf[x]`
and the ranked probability score, both on `0:max_x`. A function barrier: the
pmf comes out of the untyped fit `Dict`, so the summation loop is compiled
for its concrete type here. Same operations in the same order as the former
inline loop.
"""
function _score_terms(pmf::AbstractVector, x::Int, max_x::Int)
    log_term = log(pmf[x + 1])
    h_index = 0.0
    cdf = 0.0
    rps = 0.0
    @inbounds for s in 0:max_x
        p = pmf[s + 1]
        h_index += p^2
        cdf += p
        rps += (x <= s ? 1 - cdf : cdf)^2
    end
    return log_term, h_index - 2 * pmf[x + 1], rps
end

"""
    cocoPit(cocoReg_fit, n_bins=21)

Non-randomized Probability Integral Transform histogram of a fitted model
(Czado et al., 2009). Each observation's cumulative transition probabilities
are obtained from a single pmf pass. The PIT uses every observation with a
full history, `t = order + 1, ..., n`, and is the mean over these `n - order`
time points (as coconots' `cocoPit`). Returns a `Dict` with keys
`"Pit_values"` and `"bins"`.
"""
function cocoPit(cocoReg_fit, n_bins = 21)
    n_bins = Int(n_bins)
    data = Int.(cocoReg_fit["data"])
    n = length(data)
    order = Int(cocoReg_fit["order"])
    _check_count_limit(data)
    _check_min_length(n, order + 1, "cocoPit",
        "an order-$order model transforms the observations after the first $order")
    eta = _fit_eta(cocoReg_fit)
    lambdas = _fit_lambdas(cocoReg_fit, n)

    first_t = order + 1
    kernel = _constant_kernel(cocoReg_fit, data, lambdas, eta, maximum(data))
    Px = Vector{Float64}(undef, n - order)
    Pxm1 = Vector{Float64}(undef, n - order)
    pmfs = _history_pmfs(cocoReg_fit, data, lambdas, eta, kernel, t -> data[t])
    for t in first_t:n
        # same values as the per-observation pmf on 0:data[t] (prefix copy)
        pmf = pmfs === nothing ?
              _transition_pmf(cocoReg_fit, t, data[t], data, lambdas, eta, kernel) :
              pmfs[_history_key(data, t, order)][1:(data[t] + 1)]
        all(isfinite, pmf) || _throw_nonfinite_pmf("predictive pmf")
        cdf_x = sum(pmf)
        Px[t - order] = cdf_x
        Pxm1[t - order] = cdf_x - pmf[data[t] + 1]
    end

    u = collect(range(0, stop = 1, length = n_bins + 1))
    uniform_distribution = [get_pit_value(Px, Pxm1, u[s]) for s in 1:(n_bins + 1)]

    return Dict{String, Any}(
        "Pit_values" => [uniform_distribution[s] - uniform_distribution[s - 1]
                         for s in 2:(n_bins + 1)],
        "bins" => u[2:end])
end

"""
    get_pit_value(Px, Pxm1, u)

Mean of the conditional PIT at threshold `u` given the cumulative
probabilities at (`Px`) and just below (`Pxm1`) each observation. As in
coconots' `cocoPit`, an undefined ratio (`0 / 0`, zero predictive mass at the
observation) counts as `1`.
"""
function get_pit_value(Px, Pxm1, u)
    value = (u .- Pxm1) ./ (Px .- Pxm1)
    value[isnan.(value)] .= 1
    value[value .< 0] .= 0
    value[value .> 1] .= 1
    return mean(value)
end

"""
    compute_scores(cocoReg_fit, max_x=50)

Logarithmic, quadratic and ranked probability scores of a fitted model. For
each observation the transition pmf on `0:max_x` is computed once and all
three scores are derived from it (previously each score recomputed the pmf
from scratch, making this quadratic in `max_x`).

Returns a `Dict` with keys `"logarithmic_score"`, `"quadratic_score"` and
`"ranked_probability_score"`.
"""
function compute_scores(cocoReg_fit, max_x = 50)
    max_x = Int(max_x)
    data = Int.(cocoReg_fit["data"])
    n = length(data)
    order = Int(cocoReg_fit["order"])
    _check_support_limit(max_x, "max_x")
    _check_count_limit(data)
    _check_min_length(n, order + 1, "compute_scores",
        "an order-$order model scores the observations after the first $order")
    eta = _fit_eta(cocoReg_fit)
    lambdas = _fit_lambdas(cocoReg_fit, n)

    first_t = order + 1
    n_terms = n - order
    kernel = _constant_kernel(cocoReg_fit, data, lambdas, eta, max_x)
    log_score_sum = 0.0
    quad_score_sum = 0.0
    rps_sum = 0.0

    pmfs = _history_pmfs(cocoReg_fit, data, lambdas, eta, kernel,
        t -> max(max_x, data[t]))
    for t in first_t:n
        x = data[t]
        pmf = pmfs === nothing ?
              _transition_pmf(cocoReg_fit, t, max(max_x, x), data, lambdas, eta, kernel) :
              pmfs[_history_key(data, t, order)]

        (isfinite(pmf[x + 1]) && all(isfinite, view(pmf, 1:(max_x + 1)))) ||
            _throw_nonfinite_pmf("predictive pmf")
        log_term, quad_term, rps = _score_terms(pmf, x, max_x)
        log_score_sum -= log_term
        quad_score_sum += quad_term
        rps_sum += rps
    end

    return Dict{String, Any}("logarithmic_score" => log_score_sum / n_terms,
        "quadratic_score" => quad_score_sum / n_terms,
        "ranked_probability_score" => rps_sum / n_terms)
end
