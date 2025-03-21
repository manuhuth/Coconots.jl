export cocoPit, compute_scores

"""
    cocoPit(cocoReg_fit, n_bins=21)

Calculates the Probability Integral Transform (PIT) histogram for a fitted count regression model.

# Arguments
- `cocoReg_fit`: A dictionary containing the fitted model results. Expected keys include `"data"`, `"parameter"`, `"order"`, `"type"`, and optionally `"covariates"` and `"max_loop"`.
- `n_bins` (optional): An integer specifying the number of bins to use for the PIT histogram (default is 21).

# Returns
A dictionary with:
- `"Pit_values"`: A vector of differences between consecutive PIT values, representing the estimated histogram probabilities.
- `"bins"`: A vector of bin edge values (excluding the initial zero).
"""
function cocoPit(cocoReg_fit, n_bins=21)
    cocoReg_fit["data"] = Int.(cocoReg_fit["data"])
    u = collect( range(0, stop = 1, length = Int(n_bins+1)) )
  
    lambda = get_lambda(cocoReg_fit, false)
    if isnothing(cocoReg_fit["covariates"])
      lambda = repeat([lambda], length(cocoReg_fit["data"]) )
    end
  
    if cocoReg_fit["order"] == 1
        if cocoReg_fit["type"] == "Poisson"
            eta = 0
        else
            eta = cocoReg_fit["parameter"][2]
        end
        Px = [compute_distribution_convolution_x_r_y(cocoReg_fit["data"][t],
                    cocoReg_fit["data"][t-1], lambda[t], cocoReg_fit["parameter"][1],
                     eta) for t in 2:length(cocoReg_fit["data"])]
        Pxm1 = [compute_distribution_convolution_x_r_y(cocoReg_fit["data"][t]-1,
                cocoReg_fit["data"][t-1], lambda[t], cocoReg_fit["parameter"][1],
                eta) for t in 2:length(cocoReg_fit["data"])]
    elseif cocoReg_fit["order"] == 2
        if cocoReg_fit["type"] == "Poisson"
            eta = 0
        else
            eta = cocoReg_fit["parameter"][4]
        end
        Px = [compute_distribution_convolution_x_r_y_z(cocoReg_fit["data"][t],
                                cocoReg_fit["data"][t-1], cocoReg_fit["data"][t-2],
                                cocoReg_fit["parameter"][1], cocoReg_fit["parameter"][2],
                                cocoReg_fit["parameter"][3], lambda[t],
                                eta, cocoReg_fit["max_loop"]) for t in 3:length(cocoReg_fit["data"])]
        Pxm1 = [compute_distribution_convolution_x_r_y_z(cocoReg_fit["data"][t]-1,
                    cocoReg_fit["data"][t-1], cocoReg_fit["data"][t-2],
                    cocoReg_fit["parameter"][1], cocoReg_fit["parameter"][2],
                    cocoReg_fit["parameter"][3], lambda[t],
                    eta, cocoReg_fit["max_loop"]) for t in 3:length(cocoReg_fit["data"])]
    end
  
    uniform_distribution = [get_pit_value(Px, Pxm1, u[s]) for s in 1:(Int(n_bins+1))]
  
    return Dict("Pit_values" => [uniform_distribution[s] - uniform_distribution[s-1] for s in 2:(Int(n_bins+1))],
                "bins" => u[2:end])
end


"""
    get_pit_value(Px, Pxm1, u)

Computes a PIT (Probability Integral Transform) value for a given threshold using the cumulative probabilities.

# Arguments
- `Px`: The cumulative probability at the observed count.
- `Pxm1`: The cumulative probability at one less than the observed count.
- `u`: A threshold value from a uniform grid [0, 1].

# Returns
The mean PIT value after constraining the computed values to the [0, 1] interval.
"""
function get_pit_value(Px, Pxm1, u)
    value = (u .- Pxm1) ./ (Px .- Pxm1)
    value[value .<  0] .= 0
    value[value .>  1] .= 1
  
    return mean(value)
end


"""
    compute_scores(cocoReg_fit, max_x = 50)

Computes scoring metrics for a fitted count regression model. The scores include the logarithmic score, quadratic score, and ranked probability score.

# Arguments
- `cocoReg_fit`: A dictionary containing the fitted model results with keys such as `"data"`, `"parameter"`, `"order"`, `"type"`, and optionally `"covariates"` and `"max_loop"`.
- `max_x` (optional): The maximum count value to use when summing the probability mass function (default is 50).

# Returns
A dictionary with the following keys:
- `"logarithmic_score"`: The average negative log-probability of the observed counts.
- `"quadratic_score"`: The average quadratic score based on computed probabilities and the h-index.
- `"ranked_probability_score"`: The average ranked probability score.
"""
function compute_scores(cocoReg_fit, max_x = 50)
  
    lambda = get_lambda(cocoReg_fit, false)
  
    if isnothing(cocoReg_fit["covariates"])
      lambda = repeat([lambda], length(cocoReg_fit["data"]) )
    end
  
    if Int(cocoReg_fit["order"]) == 1
        if cocoReg_fit["type"] == "Poisson"
            eta = 0
        else
            eta = cocoReg_fit["parameter"][2]
        end
  
        probabilities = [compute_convolution_x_r_y(cocoReg_fit["data"][t],
                         cocoReg_fit["data"][t-1], lambda[t], cocoReg_fit["parameter"][1],
                         eta) for t in 2:length(cocoReg_fit["data"])]
                               
        h_index = [compute_h_index_1(cocoReg_fit["data"][t-1], lambda[t],
                        cocoReg_fit["parameter"][1],
                         eta, max_x) for t in 2:length(cocoReg_fit["data"])]
  
        rbs = [compute_ranked_probability_helper_1(cocoReg_fit["data"][t],
                         cocoReg_fit["data"][t-1], lambda[t], cocoReg_fit["parameter"][1],
                         eta, max_x) for t in 2:length(cocoReg_fit["data"])]
  
    elseif Int(cocoReg_fit["order"]) == 2
        if cocoReg_fit["type"] == "Poisson"
            eta = 0
        else
            eta = cocoReg_fit["parameter"][4]
        end
  
        probabilities = [compute_convolution_x_r_y_z(cocoReg_fit["data"][t],
                        cocoReg_fit["data"][t-1], cocoReg_fit["data"][t-2], lambda[t],
                        cocoReg_fit["parameter"][1],
                        cocoReg_fit["parameter"][2], cocoReg_fit["parameter"][3],
                         eta, cocoReg_fit["max_loop"]) for t in 3:length(cocoReg_fit["data"])]
  
        h_index = [compute_h_index_2(cocoReg_fit["data"][t-1], cocoReg_fit["data"][t-2],
                                    cocoReg_fit["parameter"][1],
                        cocoReg_fit["parameter"][2], cocoReg_fit["parameter"][3],
                        lambda[t],
                         eta, cocoReg_fit["max_loop"], max_x) for t in 3:length(cocoReg_fit["data"])]
  
        rbs = [compute_ranked_probability_helper_2(cocoReg_fit["data"][t],
                        cocoReg_fit["data"][t-1], cocoReg_fit["data"][t-2],
                        cocoReg_fit["parameter"][1],
                        cocoReg_fit["parameter"][2], cocoReg_fit["parameter"][3],
                        lambda[t],
                        eta, cocoReg_fit["max_loop"], max_x) for t in 3:length(cocoReg_fit["data"])]
    end
  
    return Dict("logarithmic_score" => Float64(- sum(log.(probabilities)) / length(probabilities)),
                "quadratic_score" => Float64(sum(- 2 .* probabilities .+ h_index) / length(probabilities)),
                "ranked_probability_score" => Float64(sum(rbs) / length(probabilities))
                )
end


"""
    compute_h_index_1(y, lambda, alpha, eta, max_x = 50)

Computes the h-index for a first-order count regression model by summing the squared probabilities of the convolution distribution over a range of count values.

# Arguments
- `y`: The lagged observed count (previous time step).
- `lambda`: The rate parameter for the current observation.
- `alpha`: The autoregressive parameter.
- `eta`: The dispersion parameter.
- `max_x` (optional): The maximum count value to consider (default is 50).

# Returns
A scalar representing the h-index as the sum of squared probabilities.
"""
function compute_h_index_1(y, lambda, alpha, eta, max_x = 50)
    return sum(([compute_convolution_x_r_y(s, y, lambda, alpha, eta) for s in 0:Int(max_x)]).^2)
end


"""
    compute_ranked_probability_helper_1(x, y, lambda, alpha, eta, max_x = 50)

Computes a helper value for the ranked probability score in a first-order count regression model. The helper is obtained by summing the squared differences between the cumulative distribution and the observed count indicator.

# Arguments
- `x`: The observed count value.
- `y`: The lagged count value.
- `lambda`: The rate parameter for the current observation.
- `alpha`: The autoregressive parameter.
- `eta`: The dispersion parameter.
- `max_x` (optional): The maximum count value to consider (default is 50).

# Returns
A scalar value representing the ranked probability score helper.
"""
function compute_ranked_probability_helper_1(x, y, lambda, alpha, eta, max_x = 50)
    dist_val = [compute_distribution_convolution_x_r_y(s, y, lambda, alpha, eta) for s in 0:Int(max_x)]
    return sum(((x .<= 0:(length(dist_val)-1)) .* (1 .- dist_val) .+ (x .> 0:(length(dist_val)-1)) .* dist_val).^2)
end


"""
    compute_h_index_2(y, z, alpha1, alpha2, alpha3, lambda, eta, max_loop=nothing, max_x = 50)

Computes the h-index for a second-order count regression model by summing the squared probabilities of the convolution distribution over a range of count values.

# Arguments
- `y`: The first lagged count value.
- `z`: The second lagged count value.
- `alpha1`: The first autoregressive parameter.
- `alpha2`: The second autoregressive parameter.
- `alpha3`: The third autoregressive parameter.
- `lambda`: The rate parameter for the current observation.
- `eta`: The dispersion parameter.
- `max_loop` (optional): A parameter controlling the maximum iterations or truncation for the convolution calculation (default is `nothing`).
- `max_x` (optional): The maximum count value to consider (default is 50).

# Returns
A scalar representing the h-index as the sum of squared probabilities.
"""
function compute_h_index_2(y, z, alpha1, alpha2, alpha3, lambda, eta, max_loop=nothing, max_x = 50)
    return sum(([compute_convolution_x_r_y_z(s, y, z, lambda, alpha1, alpha2, alpha3,
                                                 eta, max_loop) for s in 0:Int(max_x)]).^2)
end


"""
    compute_ranked_probability_helper_2(x, y, z, alpha1, alpha2, alpha3,
                                        lambda, eta, max_loop=nothing, max_x = 50)

Computes a helper value for the ranked probability score in a second-order count regression model. The helper is computed by summing the squared differences based on the convolution distribution for counts from 0 to `max_x`.

# Arguments
- `x`: The observed count value.
- `y`: The first lagged count value.
- `z`: The second lagged count value.
- `alpha1`: The first autoregressive parameter.
- `alpha2`: The second autoregressive parameter.
- `alpha3`: The third autoregressive parameter.
- `lambda`: The rate parameter for the current observation.
- `eta`: The dispersion parameter.
- `max_loop` (optional): A parameter controlling the maximum iterations or truncation in the convolution calculation (default is `nothing`).
- `max_x` (optional): The maximum count value to consider (default is 50).

# Returns
A scalar representing the ranked probability score helper for the second-order model.
"""
function compute_ranked_probability_helper_2(x, y, z, alpha1, alpha2, alpha3,
                                             lambda, eta, max_loop=nothing, max_x = 50)
    dist_val = [compute_distribution_convolution_x_r_y_z(s, y, z, alpha1, alpha2, alpha3,
                                                lambda, eta, max_loop) for s in 0:Int(max_x)]
    return sum(((x .<= 0:(length(dist_val)-1)) .* (dist_val .- 1) .+ (x .> 0:(length(dist_val)-1)) .* dist_val).^2)
end


