export compute_partial_autocorrelation, cocoBoot, compute_random_pacfs

"""
    compute_partial_autocorrelation(x, lags)

Autocorrelation of `x` at the given `lags` (demeaned). Kept under its
historical name for interface stability.
"""
compute_partial_autocorrelation(x, lags) = autocor(x, lags, demean = true)

"""
    cocoBoot(cocoReg_fit, lags=[1:1:21;], n_bootstrap=400, alpha=0.05, n_burn_in=200)

Parametric-bootstrap acceptance envelope of the autocorrelation function of a
fitted model. Returns a `Dict` with keys `"lower"`, `"upper"`,
`"in_interval"`, `"pacf_data"`, `"pacfs"` and `"lags"`.
"""
function cocoBoot(cocoReg_fit, lags = [1:1:21;], n_bootstrap = 400, alpha = 0.05,
        n_burn_in = 200, store_matrix = Array{Float64}(undef, length(lags), 2))
    pacfs = compute_random_pacfs(cocoReg_fit, lags, Int(n_bootstrap), Int(n_burn_in))
    for i in eachindex(lags)
        store_matrix[i, :] = quantile!(pacfs[:, i], [alpha / 2, 1 - alpha / 2])
    end
    pacf_data = compute_partial_autocorrelation(Int.(cocoReg_fit["data"]), lags)
    return Dict{String, Any}("lower" => store_matrix[:, 1],
        "upper" => store_matrix[:, 2],
        "in_interval" => (store_matrix[:, 1] .< pacf_data) .&
                         (store_matrix[:, 2] .> pacf_data),
        "pacf_data" => pacf_data,
        "pacfs" => pacfs,
        "lags" => lags)
end

"""
    compute_random_pacfs(cocoReg_fit, lags, n_bootstrap=400, n_burn_in=200)

Autocorrelations of `n_bootstrap` datasets simulated from the fitted model.
Returns an `(n_bootstrap, length(lags))` matrix.
"""
function compute_random_pacfs(cocoReg_fit, lags, n_bootstrap = 400, n_burn_in = 200,
        pacfs = Array{Float64}(undef, Int(n_bootstrap), length(lags)))
    n = length(cocoReg_fit["data"])
    for b in 1:Int(n_bootstrap)
        simulated = cocoSim(cocoReg_fit["type"], Int(cocoReg_fit["order"]),
            cocoReg_fit["parameter"], n, cocoReg_fit["covariates"],
            cocoReg_fit["link"], n_burn_in, zeros(n + Int(n_burn_in)))
        pacfs[b, :] = compute_partial_autocorrelation(simulated, lags)
    end
    return pacfs
end

"""
    create_julia_dict(keys, values)

Dictionary from parallel arrays of keys and values (used by the R bridge to
synthesize a Julia fit object from an R-side fit).
"""
function create_julia_dict(keys, values)
    D = Dict{Any, Any}()
    for i in eachindex(keys)
        D[keys[i]] = values[i]
    end
    return D
end
