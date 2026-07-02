export cocoPredictOneStep, cocoForwardSim, cocoPredictKsteps

"""
    cocoPredictOneStep(cocoReg_fit, x=0:10, covariates=nothing)

One-step-ahead predictive distribution of a fitted model over the candidate
counts `x`. The full transition pmf is computed in a single pass and reused
for the mode and median.

Returns a `Dict` with keys `"probabilities"`, `"x"`, `"prediction_mode"` and
`"prediction_median"`.
"""
function cocoPredictOneStep(cocoReg_fit, x = 0:10, covariates = nothing)
    lambda = get_lambda(cocoReg_fit, true)
    if !isnothing(covariates)
        link_function = get_link_function(cocoReg_fit["link"])
        n_betas = size(cocoReg_fit["covariates"], 2)
        betas = cocoReg_fit["parameter"][(end - n_betas + 1):end]
        lambda = link_function(sum(covariates .* betas))
    end

    parameter = cocoReg_fit["parameter"]
    data = cocoReg_fit["data"]
    xs = collect(Int, x)
    xmax = maximum(xs)

    if Int(cocoReg_fit["order"]) == 2
        eta = cocoReg_fit["type"] == "Poisson" ? 0.0 : parameter[4]
        y, z = Int(data[end]), Int(data[end - 1])
        kernel = GP2Kernel(lambda, parameter[1], parameter[2], parameter[3], eta,
            max(xmax, y + z))
        pmf = convolution_pmf_vector(kernel, xmax, y, z, cocoReg_fit["max_loop"])
    else
        eta = cocoReg_fit["type"] == "Poisson" ? 0.0 : parameter[2]
        pmf = convolution_pmf_vector(xmax, Int(data[end]), lambda, parameter[1], eta)
    end

    probabilities = pmf[xs .+ 1]
    return Dict{String, Any}("probabilities" => probabilities,
        "x" => x,
        "prediction_mode" => x[argmax(probabilities)],
        "prediction_median" => x[findfirst(cumsum(probabilities) .>= 0.5)])
end

"""
    cocoForwardSim(n, x_prev, type, order, parameter, covariates=nothing, link="log",
                   add_help=order * -1 + 2, x=zeros(Int(n + length(x_prev) + add_help)))

Simulates `n` steps forward from the trailing observations `x_prev` at the
given parameter values.
"""
function cocoForwardSim(n, x_prev, type, order, parameter, covariates = nothing,
        link = "log", add_help = Int(order) * -1 + 2,
        x = zeros(Int(n + length(x_prev) + add_help)))
    n = Int(n)
    order = Int(order)
    n_burn_in = length(x_prev)
    n_total = n + n_burn_in + Int(add_help)

    if order == 2
        x[end - n - 1] = x_prev[end - 1]
        x[end - n] = x_prev[end]
    else
        x[end - n] = x_prev[end]
    end

    if order == 2
        alpha1, alpha2, alpha3 = parameter[1], parameter[2], parameter[3]
        alpha = nothing
        eta = type == "GP" ? parameter[4] : 0.0
    else
        alpha1 = alpha2 = alpha3 = nothing
        alpha = parameter[1]
        eta = type == "GP" ? parameter[2] : 0.0
    end

    link_function = get_link_function(link)
    if !isnothing(covariates)
        n_betas = size(covariates, 2)
        lambda = link_function.(covariates * parameter[(end - n_betas + 1):end])
    else
        lambda = fill(float(last(parameter)), n_total)
    end

    for t in 3:n_total
        x[t] = draw_random_g(order, rand(), Int(x[t - 1]), Int(x[t - 2]), lambda[t - 2],
            alpha1, alpha2, alpha3, alpha, eta, nothing) +
               draw_random_generalized_poisson_variable(rand(), lambda[t - 2], eta)
    end

    return x[(end - n + 1):end]
end

"""
    cocoPredictKsteps(cocoReg_fit, k, number_simulations=500, covariates=nothing,
                      link="log", matrix_fill=zeros(Int(number_simulations), Int(k)))

k-step-ahead predictive distribution by parametric forward simulation.
Returns a `Dict` with a `DataFrame` of values and relative frequencies per
step (keys `"prediction_1"`, ..., plus `"length"`).
"""
function cocoPredictKsteps(cocoReg_fit, k, number_simulations = 500,
        covariates = nothing, link = "log",
        matrix_fill = zeros(Int(number_simulations), Int(k)))
    type = cocoReg_fit["type"]
    order = Int(cocoReg_fit["order"])
    parameter = cocoReg_fit["parameter"]
    link = cocoReg_fit["link"]
    data = cocoReg_fit["data"]

    x_prev = order == 2 ? data[(end - 1):end] : data[end]

    for i in 1:Int(number_simulations)
        matrix_fill[i, :] = cocoForwardSim(k, x_prev, type, order, parameter,
            covariates, link)
    end

    return get_relative_frequencies(matrix_fill)
end

"""
    get_relative_frequencies(matrix)

Relative frequencies of the simulated predictions per step (matrix column).
Returns a `Dict` with a `DataFrame` per step plus a `"length"` entry.
"""
function get_relative_frequencies(matrix)
    out = Dict{String, Any}()
    n_simulations = size(matrix, 1)
    for i in 1:size(matrix, 2)
        freq_table = freqtable(matrix[:, i])
        out["prediction_$i"] = DataFrame(
            hcat(names(freq_table)[1], freq_table ./ n_simulations),
            ["value", "frequency"])
    end
    out["length"] = length(out)
    return out
end
