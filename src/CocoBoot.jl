export compute_partial_autocorrelation, cocoBoot, compute_random_pacfs

"""
    compute_partial_autocorrelation(x, lags)

Computes the (partial) autocorrelation of the input vector `x` at the specified `lags`.
This function calls `autocor` with demeaning enabled.

# Arguments
- `x`: A vector of numerical observations.
- `lags`: A scalar or vector indicating the lag(s) at which to compute the autocorrelation.

# Returns
A vector of autocorrelation coefficients corresponding to the specified lags.
"""
function compute_partial_autocorrelation(x, lags)
    return autocor(x, lags, demean=true)
end

"""
    cocoBoot(cocoReg_fit, lags=[1:1:21;], n_bootstrap=400, alpha=0.05, n_burn_in=200, store_matrix = Array{Float64}(undef, length(lags), 2))

Bootstraps the partial autocorrelation function (PACF) for a fitted count regression model.
It generates bootstrap samples of PACF values, computes quantiles to form confidence intervals,
and then compares these intervals with the PACF computed from the observed data.

# Arguments
- `cocoReg_fit`: A dictionary containing the fitted model results.
- `lags` (optional): A vector of lag values for which the PACF is computed (default is `[1:1:21;]`).
- `n_bootstrap` (optional): The number of bootstrap replications (default is 400).
- `alpha` (optional): The significance level for the confidence intervals (default is 0.05).
- `n_burn_in` (optional): The number of burn-in samples to discard when simulating data (default is 200).
- `store_matrix` (optional): A preallocated matrix to store the lower and upper quantiles for each lag.

# Returns
A dictionary with the following keys:
- `"upper"`: The lower quantile values (first column) of the bootstrapped PACF for each lag.
- `"lower"`: The upper quantile values (second column) of the bootstrapped PACF for each lag.
- `"in_interval"`: A Boolean vector indicating if the observed PACF values lie within the bootstrap confidence intervals.
- `"pacf_data"`: The PACF computed from the observed data.
- `"pacfs"`: The matrix of bootstrapped PACF values.
- `"lags"`: The vector of lag values.
"""
function cocoBoot(cocoReg_fit, lags=[1:1:21;], n_bootstrap=400, alpha=0.05, n_burn_in=200, store_matrix = Array{Float64}(undef, length(lags), 2))
    pacfs = compute_random_pacfs(cocoReg_fit, lags, Int(n_bootstrap), n_burn_in, Array{Float64}(undef, Int(n_bootstrap), length(lags)))
    for i in 1:length(lags)
        store_matrix[i, :] = quantile!(pacfs[:, i], [alpha/2, (1 - alpha)/2])
    end
    pacf_data = compute_partial_autocorrelation(Int.(cocoReg_fit["data"]), lags)
    return Dict("upper" => store_matrix[:, 1],
                "lower" => store_matrix[:, 2],
                "in_interval" => (store_matrix[:, 1] .< pacf_data) .& (store_matrix[:, 2] .> pacf_data),
                "pacf_data" => pacf_data,
                "pacfs" => pacfs,
                "lags" => lags)
end

"""
    compute_random_pacfs(cocoReg_fit, lags, n_bootstrap=400, n_burn_in=200, pacfs=Array{Float64}(undef, n_bootstrap, length(lags)))

Generates bootstrapped partial autocorrelation coefficients by simulating new datasets from the fitted model.
For each bootstrap iteration, it simulates data using `cocoSim` and computes the PACF for the specified lags.

# Arguments
- `cocoReg_fit`: A dictionary containing the fitted model parameters and observed data.
- `lags`: A vector of lag values at which to compute the PACF.
- `n_bootstrap` (optional): The number of bootstrap samples (default is 400).
- `n_burn_in` (optional): The number of burn-in samples to discard during simulation (default is 200).
- `pacfs` (optional): A preallocated matrix to store the PACF values from each bootstrap sample.

# Returns
A matrix of bootstrapped PACF values with dimensions `(n_bootstrap, length(lags))`.
"""
function compute_random_pacfs(cocoReg_fit, lags, n_bootstrap=400, n_burn_in=200, pacfs=Array{Float64}(undef, n_bootstrap, length(lags)))
    for b in 1:Int(n_bootstrap)
        pacfs[b, :] = compute_partial_autocorrelation(cocoSim(cocoReg_fit["type"], Int(cocoReg_fit["order"]), cocoReg_fit["parameter"],
                                                             length(cocoReg_fit["data"]), cocoReg_fit["covariates"], cocoReg_fit["link"],
                                                             n_burn_in, zeros(length(cocoReg_fit["data"]) + n_burn_in)), lags)
    end
    return pacfs
end

"""
    create_julia_dict(keys, values)

Creates a Julia dictionary from arrays of keys and corresponding values.

# Arguments
- `keys`: An array containing the keys.
- `values`: An array containing the values; each element corresponds to a key in `keys`.

# Returns
A dictionary mapping each key in `keys` to the corresponding value in `values`.
"""
function create_julia_dict(keys, values)
    D = Dict()
    for i in eachindex(keys)
        D[keys[i]] = values[i]
    end
    return D
end
