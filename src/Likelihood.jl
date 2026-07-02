"""
    compute_g_r_y(y, r, alpha, eta, lambda)

Convolution probability mass `g(r | y)` of the first-order model (closed
form). Uses the factorial lookup table for `y <= 170` and log-space
evaluation beyond, so no `BigInt`s are ever allocated.
"""
function compute_g_r_y(y::Real, r::Real, alpha::Real, eta::Real, lambda::Real)
    yi, ri = Int(y), Int(r)
    psi = eta * (1 - alpha) / lambda
    if yi <= MAX_FACTORIAL_ARGUMENT
        return float_factorial(yi) / float_factorial(ri) / float_factorial(yi - ri) *
               alpha * (1 - alpha) * (alpha + psi * ri)^(ri - 1) *
               (1 - alpha + psi * (yi - ri))^(yi - ri - 1) / (1 + psi * yi)^(yi - 1)
    end
    return exp(logfactorial(yi) - logfactorial(ri) - logfactorial(yi - ri) + log(alpha) +
               log(1 - alpha) + (ri - 1) * log(alpha + psi * ri) +
               (yi - ri - 1) * log(1 - alpha + psi * (yi - ri)) -
               (yi - 1) * log(1 + psi * yi))
end

"""
    compute_convolution_x_r_y(x, y, lambda, alpha, eta)

Transition probability `P(x | y)` of the first-order model: the convolution of
`g(r | y)` with the Generalized Poisson innovation, summed over
`r = 0, ..., min(x, y)`.
"""
function compute_convolution_x_r_y(x::Real, y::Real, lambda::Real, alpha::Real, eta::Real)
    xi, yi = Int(x), Int(y)
    total = zero(promote_type(typeof(lambda), typeof(alpha), typeof(eta)))
    for r in 0:min(xi, yi)
        total += compute_g_r_y(yi, r, alpha, eta, lambda) *
                 generalized_poisson_distribution(xi - r, lambda, eta)
    end
    return total
end

"""
    convolution_pmf_vector(xmax, y, lambda, alpha, eta)

Vector of first-order transition probabilities `P(x | y)` for
`x = 0, ..., xmax`, computed in one pass: `g(r | y)` and the innovation pmf are
tabulated once and reused across all `x` (instead of recomputing them for every
cumulative-distribution evaluation).
"""
function convolution_pmf_vector(xmax::Integer, y::Integer, lambda::Real, alpha::Real,
        eta::Real)
    T = promote_type(typeof(lambda), typeof(alpha), typeof(eta))
    rmax = min(xmax, y)
    g = T[compute_g_r_y(y, r, alpha, eta, lambda) for r in 0:rmax]
    innovation = T[generalized_poisson_distribution(i, lambda, eta) for i in 0:xmax]
    pmf = Vector{T}(undef, xmax + 1)
    @inbounds for x in 0:xmax
        total = zero(T)
        for r in 0:min(x, rmax)
            total += g[r + 1] * innovation[x - r + 1]
        end
        pmf[x + 1] = total
    end
    return pmf
end

"""
    compute_distribution_convolution_x_r_y(x, y, lambda, alpha, eta)

Cumulative distribution of the first-order transition probability, summed from
0 to `x`. Returns 0 for negative `x`.
"""
function compute_distribution_convolution_x_r_y(x::Real, y::Real, lambda::Real,
        alpha::Real, eta::Real)
    T = promote_type(typeof(lambda), typeof(alpha), typeof(eta))
    x < 0 && return zero(T)
    return sum(convolution_pmf_vector(Int(x), Int(y), lambda, alpha, eta))
end

#--------------------------------------------------------------------------------------
# Second-order kernel
#--------------------------------------------------------------------------------------

"""
    GP2Kernel(lambda, alpha1, alpha2, alpha3, eta, max_index)

Precomputed state for second-order convolution evaluations at fixed parameter
values: the derived rates (`beta1`, `beta2`, `beta3`, `zeta`) and Generalized
Poisson pmf lookup tables for each of them on `0:max_index`. Building the
kernel once and reusing it across observations removes the dominant redundant
work of the second-order likelihood.
"""
struct GP2Kernel{T <: Real}
    lambda::T
    alpha1::T
    alpha2::T
    alpha3::T
    eta::T
    gp_beta1::Vector{T}
    gp_beta2::Vector{T}
    gp_beta3::Vector{T}
    gp_lambda::Vector{T}
    gp_zeta::Vector{T}
end

function GP2Kernel(lambda::Real, alpha1::Real, alpha2::Real, alpha3::Real, eta::Real,
        max_index::Integer)
    U = compute_U(alpha1, alpha2, alpha3)
    beta1 = compute_beta_i(lambda, U, alpha1)
    beta2 = compute_beta_i(lambda, U, alpha2)
    beta3 = compute_beta_i(lambda, U, alpha3)
    zeta = compute_zeta(lambda, U, alpha1, alpha3)
    T = promote_type(typeof(beta1), typeof(eta))
    table(rate) = T[generalized_poisson_distribution(i, rate, eta) for i in 0:max_index]
    return GP2Kernel{T}(lambda, alpha1, alpha2, alpha3, eta, table(beta1), table(beta2),
        table(beta3), table(lambda), table(zeta))
end

"""
    compute_g_r_y_z(kernel, r, y, z, normalizer, smax)

Convolution probability mass `g(r | y, z)` of the second-order model, using
the kernel's pmf tables. The triple sum runs only over the feasible set
(`r - s - v >= 0`, `z - r + v - w >= 0`, `y - s - v - w >= 0` intersected with
`s, v, w <= smax`) instead of a guarded full cube. `normalizer` is the
bivariate Generalized Poisson mass of `(y, z)`, computed once by the caller
and hoisted out of any loop over `r`.
"""
function compute_g_r_y_z(kernel::GP2Kernel{T}, r::Integer, y::Integer, z::Integer,
        normalizer::Real, smax::Integer) where {T}
    total = zero(T)
    @inbounds for s in 0:min(smax, r, y)
        ps = kernel.gp_beta3[s + 1]
        for v in 0:min(smax, r - s, y - s)
            pv = ps * kernel.gp_beta1[v + 1] * kernel.gp_beta2[r - s - v + 1]
            zrv = z - r + v
            ysv = y - s - v
            for w in 0:min(smax, ysv, zrv)
                total += pv * kernel.gp_beta1[w + 1] * kernel.gp_lambda[zrv - w + 1] *
                         kernel.gp_zeta[ysv - w + 1]
            end
        end
    end
    return total / normalizer
end

function _check_kernel_size(kernel::GP2Kernel, required_index::Integer)
    length(kernel.gp_lambda) > required_index || throw(ArgumentError(
        "GP2Kernel tables cover 0:$(length(kernel.gp_lambda) - 1) but index " *
        "$required_index is required; construct the kernel with a larger max_index"))
    return nothing
end

function convolution_x_r_y_z(kernel::GP2Kernel{T}, x::Integer, y::Integer, z::Integer,
        max_loop) where {T}
    _check_kernel_size(kernel, max(x, y + z))
    smax = max_loop === nothing ? y : Int(max_loop)
    normalizer = bivariate_generalized_poisson(y, z, kernel.lambda, kernel.alpha1,
        kernel.alpha2, kernel.alpha3, kernel.eta)
    total = zero(T)
    @inbounds for r in 0:min(x, y + z)
        total += compute_g_r_y_z(kernel, r, y, z, normalizer, smax) *
                 kernel.gp_lambda[x - r + 1]
    end
    return total
end

"""
    convolution_pmf_vector(kernel, xmax, y, z, max_loop)

Vector of second-order transition probabilities `P(x | y, z)` for
`x = 0, ..., xmax`. The `g(r | y, z)` values are computed once and reused for
every `x`, turning the previously quadratic cumulative-distribution work into a
single linear convolution pass.
"""
function convolution_pmf_vector(kernel::GP2Kernel{T}, xmax::Integer, y::Integer,
        z::Integer, max_loop) where {T}
    _check_kernel_size(kernel, max(xmax, y + z))
    smax = max_loop === nothing ? y : Int(max_loop)
    normalizer = bivariate_generalized_poisson(y, z, kernel.lambda, kernel.alpha1,
        kernel.alpha2, kernel.alpha3, kernel.eta)
    rmax = min(xmax, y + z)
    g = T[compute_g_r_y_z(kernel, r, y, z, normalizer, smax) for r in 0:rmax]
    pmf = Vector{T}(undef, xmax + 1)
    @inbounds for x in 0:xmax
        total = zero(T)
        for r in 0:min(x, rmax)
            total += g[r + 1] * kernel.gp_lambda[x - r + 1]
        end
        pmf[x + 1] = total
    end
    return pmf
end

#--------------------------------------------------------------------------------------
# Second-order public entry points (kept API-compatible with earlier versions)
#--------------------------------------------------------------------------------------

"""
    compute_g_r_y_z(r, y, z, lambda, alpha1, alpha2, alpha3, eta, max_loop=nothing)

Convolution probability mass `g(r | y, z)` of the second-order model. Builds a
one-shot [`GP2Kernel`](@ref); prefer constructing the kernel once when
evaluating many `(r, y, z)` combinations at fixed parameters.
"""
function compute_g_r_y_z(r::Real, y::Real, z::Real, lambda::Real, alpha1::Real,
        alpha2::Real, alpha3::Real, eta::Real, max_loop = nothing)
    ri, yi, zi = Int(r), Int(y), Int(z)
    kernel = GP2Kernel(lambda, alpha1, alpha2, alpha3, eta, max(ri, yi + zi))
    smax = max_loop === nothing ? yi : Int(max_loop)
    normalizer = bivariate_generalized_poisson(yi, zi, lambda, alpha1, alpha2, alpha3,
        eta)
    return compute_g_r_y_z(kernel, ri, yi, zi, normalizer, smax)
end

"""
    compute_convolution_x_r_y_z(x, y, z, lambda, alpha1, alpha2, alpha3, eta, max_loop=nothing)

Transition probability `P(x | y, z)` of the second-order model.
"""
function compute_convolution_x_r_y_z(x::Real, y::Real, z::Real, lambda::Real,
        alpha1::Real, alpha2::Real, alpha3::Real, eta::Real, max_loop = nothing)
    xi, yi, zi = Int(x), Int(y), Int(z)
    kernel = GP2Kernel(lambda, alpha1, alpha2, alpha3, eta, max(xi, yi + zi))
    return convolution_x_r_y_z(kernel, xi, yi, zi, max_loop)
end

"""
    compute_distribution_convolution_x_r_y_z(x, y, z, alpha1, alpha2, alpha3, lambda, eta, max_loop=nothing)

Cumulative distribution of the second-order transition probability, summed
from 0 to `x`. Returns 0 for negative `x`. (Note the historical argument
order: the alphas precede `lambda` here.)
"""
function compute_distribution_convolution_x_r_y_z(x::Real, y::Real, z::Real,
        alpha1::Real, alpha2::Real, alpha3::Real, lambda::Real, eta::Real,
        max_loop = nothing)
    T = promote_type(typeof(lambda), typeof(alpha1), typeof(eta))
    x < 0 && return zero(T)
    xi, yi, zi = Int(x), Int(y), Int(z)
    kernel = GP2Kernel(lambda, alpha1, alpha2, alpha3, eta, max(xi, yi + zi))
    return sum(convolution_pmf_vector(kernel, xi, yi, zi, max_loop))
end

#--------------------------------------------------------------------------------------
# Negative log-likelihoods
#--------------------------------------------------------------------------------------

function _transition_counts(data::AbstractVector, order::Integer)
    if Int(order) == 1
        counts = Dict{NTuple{2, Int}, Int}()
        for t in 2:length(data)
            key = (Int(data[t]), Int(data[t - 1]))
            counts[key] = get(counts, key, 0) + 1
        end
    else
        counts = Dict{NTuple{3, Int}, Int}()
        for t in 3:length(data)
            key = (Int(data[t]), Int(data[t - 1]), Int(data[t - 2]))
            counts[key] = get(counts, key, 0) + 1
        end
    end
    transitions = sort!(collect(keys(counts)))
    return transitions, Int[counts[k] for k in transitions]
end

function _nll_gp1_transitions(lambda::Real, alpha::Real, eta::Real,
        transitions::Vector{NTuple{2, Int}}, counts::Vector{Int})
    T = promote_type(typeof(lambda), typeof(alpha), typeof(eta))
    if nthreads() > 1 && length(transitions) > 32
        parts = Vector{T}(undef, length(transitions))
        @threads for i in eachindex(transitions)
            x, y = transitions[i]
            parts[i] = -counts[i] * log(compute_convolution_x_r_y(x, y, lambda, alpha,
                eta))
        end
        return sum(parts)
    end
    total = zero(T)
    for i in eachindex(transitions)
        x, y = transitions[i]
        total -= counts[i] * log(compute_convolution_x_r_y(x, y, lambda, alpha, eta))
    end
    return total
end

function _nll_gp2_transitions(lambda::Real, alpha1::Real, alpha2::Real, alpha3::Real,
        eta::Real, transitions::Vector{NTuple{3, Int}}, counts::Vector{Int},
        max_index::Integer, max_loop)
    kernel = GP2Kernel(lambda, alpha1, alpha2, alpha3, eta, max_index)
    T = eltype(kernel.gp_lambda)
    if nthreads() > 1 && length(transitions) > 32
        parts = Vector{T}(undef, length(transitions))
        @threads for i in eachindex(transitions)
            x, y, z = transitions[i]
            parts[i] = -counts[i] * log(convolution_x_r_y_z(kernel, x, y, z, max_loop))
        end
        return sum(parts)
    end
    total = zero(T)
    for i in eachindex(transitions)
        x, y, z = transitions[i]
        total -= counts[i] * log(convolution_x_r_y_z(kernel, x, y, z, max_loop))
    end
    return total
end

function _gp2_max_index(transitions::Vector{NTuple{3, Int}})
    maximum(t -> max(t[1], t[2] + t[3]), transitions)
end

"""
    compute_negative_log_likelihood_GP1(lambda, alpha, eta, data)

Negative log-likelihood of the first-order model at a constant innovation rate
`lambda`. Aggregates over the unique `(x_t, x_{t-1})` transitions of `data`
(with multiplicities), so each distinct transition probability is evaluated
exactly once.
"""
function compute_negative_log_likelihood_GP1(lambda::Real, alpha::Real, eta::Real,
        data::AbstractVector)
    transitions, counts = _transition_counts(data, 1)
    return _nll_gp1_transitions(lambda, alpha, eta, transitions, counts)
end

"""
    compute_negative_log_likelihood_GP1(lambdas, alpha, eta, data)

Negative log-likelihood of the first-order model with an observation-specific
innovation-rate vector `lambdas` (covariate models). Falls back to the
constant-rate fast path when all rates coincide.
"""
function compute_negative_log_likelihood_GP1(lambdas::AbstractVector, alpha::Real,
        eta::Real, data::AbstractVector)
    allequal(lambdas) &&
        return compute_negative_log_likelihood_GP1(first(lambdas), alpha, eta, data)
    T = promote_type(eltype(lambdas), typeof(alpha), typeof(eta))
    n = length(data)
    if nthreads() > 1
        parts = Vector{T}(undef, n - 1)
        @threads for t in 2:n
            parts[t - 1] = -log(compute_convolution_x_r_y(data[t], data[t - 1],
                lambdas[t], alpha, eta))
        end
        return sum(parts)
    end
    total = zero(T)
    for t in 2:n
        total -= log(compute_convolution_x_r_y(data[t], data[t - 1], lambdas[t], alpha,
            eta))
    end
    return total
end

function _gp2_nll_term(lambda::Real, alpha1::Real, alpha2::Real, alpha3::Real,
        eta::Real, x::Integer, y::Integer, z::Integer, max_loop)
    kernel = GP2Kernel(lambda, alpha1, alpha2, alpha3, eta, max(x, y + z))
    return -log(convolution_x_r_y_z(kernel, x, y, z, max_loop))
end

"""
    compute_negative_log_likelihood_GP2(lambda, alpha1, alpha2, alpha3, eta, data, max_loop=nothing)

Negative log-likelihood of the second-order model at a constant innovation
rate `lambda`. Builds one [`GP2Kernel`](@ref) and aggregates over the unique
`(x_t, x_{t-1}, x_{t-2})` transitions of `data` with multiplicities.
"""
function compute_negative_log_likelihood_GP2(lambda::Real, alpha1::Real, alpha2::Real,
        alpha3::Real, eta::Real, data::AbstractVector, max_loop = nothing)
    transitions, counts = _transition_counts(data, 2)
    return _nll_gp2_transitions(lambda, alpha1, alpha2, alpha3, eta, transitions,
        counts, _gp2_max_index(transitions), max_loop)
end

"""
    compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, eta, data, max_loop=nothing)

Negative log-likelihood of the second-order model with an
observation-specific innovation-rate vector `lambdas` (covariate models).
Falls back to the constant-rate fast path when all rates coincide.
"""
function compute_negative_log_likelihood_GP2(lambdas::AbstractVector, alpha1::Real,
        alpha2::Real, alpha3::Real, eta::Real, data::AbstractVector, max_loop = nothing)
    allequal(lambdas) &&
        return compute_negative_log_likelihood_GP2(first(lambdas), alpha1, alpha2,
            alpha3, eta, data, max_loop)
    T = promote_type(eltype(lambdas), typeof(alpha1), typeof(eta))
    n = length(data)
    if nthreads() > 1
        parts = Vector{T}(undef, n - 2)
        @threads for t in 3:n
            parts[t - 2] = _gp2_nll_term(lambdas[t], alpha1, alpha2, alpha3, eta,
                Int(data[t]), Int(data[t - 1]), Int(data[t - 2]), max_loop)
        end
        return sum(parts)
    end
    total = zero(T)
    for t in 3:n
        total += _gp2_nll_term(lambdas[t], alpha1, alpha2, alpha3, eta, Int(data[t]),
            Int(data[t - 1]), Int(data[t - 2]), max_loop)
    end
    return total
end

"""
    get_lambda(cocoReg_fit, last_val=false)

Extracts or computes the innovation rate(s) `lambda` from a fitted model
dictionary. With covariates, applies the link function to the linear
predictor; `last_val=true` uses only the last covariate row.
"""
function get_lambda(cocoReg_fit, last_val = false)
    link_function = get_link_function(cocoReg_fit["link"])
    covariates = cocoReg_fit["covariates"]
    parameter = cocoReg_fit["parameter"]
    isnothing(covariates) && return last(parameter)
    betas = parameter[(end - size(covariates, 2) + 1):end]
    if last_val
        return link_function(sum(covariates[end, :] .* betas))
    end
    return link_function.(covariates * betas)
end
