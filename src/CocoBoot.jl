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

# Upper bound on the number of simulated counts held in memory at once; the
# replicates are simulated in blocks of at most this many entries.
const _BOOT_BLOCK_ENTRIES = 4_194_304

"""
    compute_random_pacfs(cocoReg_fit, lags, n_bootstrap=400, n_burn_in=200)

Autocorrelations of `n_bootstrap` datasets simulated from the fitted model.
Returns an `(n_bootstrap, length(lags))` matrix. The replicates are simulated
jointly, time step by time step (in memory-bounded blocks), so that the
conditional distributions of each step are tabulated once and shared by all
replicates with the same history. The autocorrelations of a block are computed
jointly as well (see [`_block_autocor!`](@ref)).
"""
function compute_random_pacfs(cocoReg_fit, lags, n_bootstrap = 400, n_burn_in = 200,
        pacfs = Array{Float64}(undef, Int(n_bootstrap), length(lags)))
    n = length(cocoReg_fit["data"])
    n_bootstrap = Int(n_bootstrap)
    n_burn_in = Int(n_burn_in)
    n_total = n + n_burn_in
    lambda, sampler = _simulation_setup(cocoReg_fit["type"], Int(cocoReg_fit["order"]),
        cocoReg_fit["parameter"], n, cocoReg_fit["covariates"], cocoReg_fit["link"],
        n_burn_in)

    # joint computation for valid integer lags; otherwise per replicate through
    # compute_partial_autocorrelation (which also raises its errors)
    joint = eltype(lags) <: Integer && all(l -> 0 <= l < n, lags)
    block = clamp(div(_BOOT_BLOCK_ENTRIES, max(n_total, 1)), 1, max(n_bootstrap, 1))
    X = Matrix{Int}(undef, block, n_total)
    # buffers of the joint path (empty otherwise) and of the per-replicate path
    lag_vector = joint ? Int[l for l in lags] : Int[]
    Z = Matrix{Float64}(undef, joint ? block : 0, n)
    num = Matrix{Float64}(undef, joint ? block : 0, length(lag_vector))
    zz = Vector{Float64}(undef, joint ? block : 0)
    simulated = Vector{Float64}(undef, joint ? 0 : n)
    first_rep = 1
    while first_rep <= n_bootstrap
        n_rep = min(block, n_bootstrap - first_rep + 1)
        Xb = view(X, 1:n_rep, :)
        fill!(Xb, 0)
        _simulate_paths!(Xb, sampler, lambda, n_burn_in)
        if joint
            _block_autocor!(pacfs, first_rep - 1, Xb, n_burn_in, n, lag_vector, Z, num,
                zz)
        else
            for b in 1:n_rep
                for i in 1:n
                    simulated[i] = Xb[b, n_burn_in + i]
                end
                pacfs[first_rep + b - 1, :] = compute_partial_autocorrelation(simulated,
                    lags)
            end
        end
        first_rep += n_rep
    end
    return pacfs
end

"""
    _block_autocor!(pacfs, row0, Xb, n_burn_in, n, lags, Z, num, zz)

For every replicate `b` (row of `Xb`, observations in columns
`n_burn_in + 1 : n_burn_in + n`) writes the demeaned autocorrelation at
`lags` (`0 <= lag < n`) into `pacfs[row0 + b, :]`: the same estimator as
[`compute_partial_autocorrelation`](@ref),
`sum_{t <= n - l} z_t z_{t + l} / sum_t z_t^2` with `z = x - mean(x)`, but
evaluated for all replicates of the block at once along contiguous columns
(no per-replicate copies or allocations). The mean is exact (integer sum);
only the order of the floating-point summation of the dot products differs.
`Z`, `num` and `zz` are caller-owned buffers with at least `size(Xb, 1)` rows.
"""
function _block_autocor!(pacfs::AbstractMatrix{Float64}, row0::Int,
        Xb::AbstractMatrix{Int}, n_burn_in::Int, n::Int, lags::Vector{Int},
        Z::Matrix{Float64}, num::Matrix{Float64}, zz::Vector{Float64})
    n_rep = size(Xb, 1)
    m = length(lags)
    # means (held in zz until the sums of squares): sums of counts are exact
    # in Float64, so the mean equals that of compute_partial_autocorrelation
    @inbounds for b in 1:n_rep
        zz[b] = 0.0
    end
    @inbounds for t in 1:n
        @simd for b in 1:n_rep
            zz[b] += Xb[b, n_burn_in + t]
        end
    end
    @inbounds for b in 1:n_rep
        zz[b] /= n
    end
    @inbounds for t in 1:n
        @simd for b in 1:n_rep
            Z[b, t] = Xb[b, n_burn_in + t] - zz[b]
        end
    end
    @inbounds for b in 1:n_rep
        zz[b] = 0.0
    end
    @inbounds for k in 1:m, b in 1:n_rep
        num[b, k] = 0.0
    end
    # t outer: the columns t, ..., t + max(lags) stay in cache
    @inbounds for t in 1:n
        @simd for b in 1:n_rep
            zz[b] += Z[b, t] * Z[b, t]
        end
        for k in 1:m
            tl = t + lags[k]
            tl <= n || continue
            @simd for b in 1:n_rep
                num[b, k] += Z[b, t] * Z[b, tl]
            end
        end
    end
    @inbounds for k in 1:m, b in 1:n_rep
        pacfs[row0 + b, k] = num[b, k] / zz[b]
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
