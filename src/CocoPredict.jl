using FreqTables

export cocoPredictOneStep, cocoForwardSim, cocoPredictKsteps

"""
    cocoPredictOneStep(cocoReg_fit, x=0:10, covariates=nothing, safe_array = Array{Float64}(undef, length(x)))

Predicts one-step ahead count probabilities from a fitted count regression model.

# Arguments
- `cocoReg_fit`: A dictionary containing fitted model information (e.g., data, parameters, order, type, link, etc.).
- `x` (optional): A range or vector of possible future count values over which the prediction is computed (default is `0:10`).
- `covariates` (optional): Optional covariate information used for prediction.
- `safe_array` (optional): A preallocated array to store computed probabilities.

# Returns
A dictionary with:
- `"probabilities"`: A vector of predicted probabilities for each count in `x`.
- `"x"`: The input range of counts.
- `"prediction_mode"`: The mode (most likely count) based on the maximum probability.
- `"prediction_median"`: The median prediction computed from the cumulative probability.
"""
function cocoPredictOneStep(cocoReg_fit, x=0:10, covariates=nothing, safe_array = Array{Float64}(undef, length(x)))
    lambda = get_lambda(cocoReg_fit, true)
  
    link_function = get_link_function(cocoReg_fit["link"])
    if (!isnothing(covariates))
        lambda = link_function(sum(covariates .* cocoReg_fit["parameter"][(end - size(cocoReg_fit["covariates"])[2] + 1):end]))
    end
  
    if cocoReg_fit["order"] == 2
        if cocoReg_fit["type"] == "Poisson"
            eta = 0
        else
            eta = cocoReg_fit["parameter"][4]
        end
  
        output = Dict("probabilities" => [compute_convolution_x_r_y_z(i, Int(cocoReg_fit["data"][end]),
                                               Int(cocoReg_fit["data"][end-1]), lambda,
                                               cocoReg_fit["parameter"][1], cocoReg_fit["parameter"][2],
                                               cocoReg_fit["parameter"][3], eta,
                                               cocoReg_fit["max_loop"]) for i in x],
                      "prediction_mode" => -3.0)
    elseif cocoReg_fit["order"] == 1
        if cocoReg_fit["type"] == "Poisson"
            eta = 0
        else
            eta = cocoReg_fit["parameter"][2]
        end
  
        output = Dict("probabilities" => [compute_convolution_x_r_y(i, Int(cocoReg_fit["data"][end]),
                                               lambda, cocoReg_fit["parameter"][1], eta) for i in x],
                      "prediction_mode" => -3.0)
    end
  
    output["x"] = x
    output["prediction_mode"] = x[argmax(output["probabilities"])]
    output["prediction_median"] = x[findfirst(cumsum(output["probabilities"]) .>= 0.5)]
  
    return output
end

"""
    cocoForwardSim(n, x_prev, type, order, parameter, covariates=nothing, link="log", add_help=order * -1 + 2, x=zeros(Int(n + length(x_prev) + add_help)))

Simulates forward count data for `n` steps given previous counts, model parameters, and optional covariates.

# Arguments
- `n`: The number of future steps to simulate.
- `x_prev`: A vector of previous count values used for initializing the simulation.
- `type`: A string indicating the model type (e.g., `"GP"` for generalized Poisson).
- `order`: The autoregressive model order (1 or 2).
- `parameter`: A vector of model parameters.
- `covariates` (optional): A matrix of covariate data used in simulation.
- `link` (optional): A string specifying the link function to use (default is `"log"`).
- `add_help` (optional): A helper parameter for simulation (default computed as `order * -1 + 2`).
- `x` (optional): A preallocated vector of integer counts with length `n + length(x_prev) + add_help`.

# Returns
A vector of simulated counts for the `n` forward steps.
"""
function cocoForwardSim(n, x_prev, type, order, parameter, covariates=nothing,
                        link="log", add_help=order * -1 + 2,
                        x=zeros(Int(n + length(x_prev) + add_help)))
    n_burn_in = length(x_prev)
    
    if order == 2
        x[end-1] = x_prev[end]
        x[end-2] = x_prev[end-1]
    else
        x[end-1] = x_prev[end]
    end
  
    link_function = get_link_function(link)
    if order == 2
        alpha1 = parameter[1]
        alpha2 = parameter[2]
        alpha3 = parameter[3]
        alpha = nothing
        if type == "GP"
            eta = parameter[4]
        else
            eta = 0
        end
    else
        alpha1 = nothing
        alpha2 = nothing
        alpha3 = nothing
        alpha = parameter[1]
        if type == "GP"
            eta = parameter[2]
        else
            eta = 0
        end
    end
  
    if !isnothing(covariates)
        lambda = link_function.(covariates * parameter[Int((end - (size(covariates)[2] - 1))):Int(end)])
    else
        lambda = repeat([last(parameter)], Int(n + n_burn_in + add_help))
    end
  
    for t in 3:(Int(n + n_burn_in + add_help))
        x[t] = draw_random_g(order, rand(Uniform(0, 1), 1)[1], Int(x[t-1]), Int(x[t-2]), lambda[t-2], alpha1, alpha2, alpha3, alpha, eta, nothing) +
               draw_random_generalized_poisson_variable(rand(Uniform(0, 1), 1)[1], lambda[t-2], eta)
    end
  
    return x[Int((end - n + 1)):Int(end)]
end

"""
    cocoPredictKsteps(cocoReg_fit, k, number_simulations=500, covariates=nothing, link="log", matrix_fill=zeros(Int64(number_simulations), Int64(k)))

Generates a k-step ahead predictive distribution by simulating the model multiple times.

# Arguments
- `cocoReg_fit`: A dictionary containing the fitted model details.
- `k`: The number of steps ahead to predict.
- `number_simulations` (optional): The number of simulation runs to generate the predictive distribution (default is 500).
- `covariates` (optional): Optional covariate data to use in the simulations.
- `link` (optional): A string specifying the link function (default is `"log"`).
- `matrix_fill` (optional): A preallocated matrix to store simulation results (default is a zeros matrix with dimensions `number_simulations` x `k`).

# Returns
A dictionary where each key `"prediction_i"` contains a DataFrame of count values and their relative frequencies for step `i`. Also includes a key `"length"` indicating the number of prediction steps.
"""
function cocoPredictKsteps(cocoReg_fit, k, number_simulations=500,
                           covariates=nothing,
                           link="log",
                           matrix_fill = zeros(Int64(number_simulations), Int64(k)))
    type = cocoReg_fit["type"]
    order = cocoReg_fit["order"]
    parameter = cocoReg_fit["parameter"]
    link = cocoReg_fit["link"]
    
    x_prev = cocoReg_fit["data"][end]
    if order == 2
        x_prev = cocoReg_fit["data"][(end-1):end]
    end
  
    for i in 1:Int64(number_simulations)
        matrix_fill[i, :] = cocoForwardSim(k, x_prev, type, order, parameter, covariates, link)
    end
  
    return get_relative_frequencies(matrix_fill)
end

"""
    get_relative_frequencies(matrix)

Computes the relative frequencies of simulated count predictions for each prediction step.

# Arguments
- `matrix`: A matrix of simulated count predictions, where each column corresponds to a prediction step.

# Returns
A dictionary where each key `"prediction_i"` contains a DataFrame with two columns:
- `"value"`: The unique count values.
- `"frequency"`: The relative frequency of each count (normalized by the number of simulations).
Also includes a key `"length"` indicating the number of prediction steps.
"""
function get_relative_frequencies(matrix)
    out = Dict()
    for i in 1:size(matrix)[2]
        freq_table = freqtable(matrix[:, i])
        out["prediction_$i"] = DataFrame(hcat(names(freq_table)[1], freq_table ./ size(matrix)[1]), ["value", "frequency"])
    end
    out["length"] = length(out)
    return out
end
