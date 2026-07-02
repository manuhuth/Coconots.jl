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

"""
    cocoPit(cocoReg_fit, n_bins=21)

Non-randomized Probability Integral Transform histogram of a fitted model
(Czado et al., 2009). Each observation's cumulative transition probabilities
are obtained from a single pmf pass. Returns a `Dict` with keys
`"Pit_values"` and `"bins"`.
"""
function cocoPit(cocoReg_fit, n_bins = 21)
    n_bins = Int(n_bins)
    data = Int.(cocoReg_fit["data"])
    n = length(data)
    order = Int(cocoReg_fit["order"])
    eta = _fit_eta(cocoReg_fit)
    lambdas = _fit_lambdas(cocoReg_fit, n)

    first_t = order + 1
    kernel = _constant_kernel(cocoReg_fit, data, lambdas, eta, maximum(data))
    Px = Vector{Float64}(undef, n - order)
    Pxm1 = Vector{Float64}(undef, n - order)
    for t in first_t:n
        pmf = _transition_pmf(cocoReg_fit, t, data[t], data, lambdas, eta, kernel)
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
probabilities at (`Px`) and just below (`Pxm1`) each observation.
"""
function get_pit_value(Px, Pxm1, u)
    value = (u .- Pxm1) ./ (Px .- Pxm1)
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
    eta = _fit_eta(cocoReg_fit)
    lambdas = _fit_lambdas(cocoReg_fit, n)

    first_t = order + 1
    n_terms = n - order
    kernel = _constant_kernel(cocoReg_fit, data, lambdas, eta, max_x)
    log_score_sum = 0.0
    quad_score_sum = 0.0
    rps_sum = 0.0

    for t in first_t:n
        x = data[t]
        pmf = _transition_pmf(cocoReg_fit, t, max(max_x, x), data, lambdas, eta, kernel)

        log_score_sum -= log(pmf[x + 1])

        h_index = 0.0
        cdf = 0.0
        rps = 0.0
        @inbounds for s in 0:max_x
            p = pmf[s + 1]
            h_index += p^2
            cdf += p
            rps += (x <= s ? 1 - cdf : cdf)^2
        end
        quad_score_sum += h_index - 2 * pmf[x + 1]
        rps_sum += rps
    end

    return Dict{String, Any}("logarithmic_score" => log_score_sum / n_terms,
        "quadratic_score" => quad_score_sum / n_terms,
        "ranked_probability_score" => rps_sum / n_terms)
end
