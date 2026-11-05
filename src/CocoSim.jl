export cocoSim

# Termination of the Generalized Poisson inversion (as `GPInvCdf` in the C++
# backend). In floating point the accumulated cdf F can stay below u forever
# (its increments round to zero once they fall below half an ulp of F). Two
# rules bound the inversion:
#  - when the next value F(i) is still < u, the increment vanished
#    (F(i) == F(i - 1)) and 1 - F(i - 1) <= `_GP_INVERSION_SATURATION_TOL`, F
#    has saturated: beyond the bulk the pmf decreases, so F keeps this value for
#    all larger i, the remaining mass is at rounding level, and i is returned;
#  - a draw above `MAX_FACTORIAL_ARGUMENT` (170) is outside the supported range
#    of counts (as on the RCPP backend): once F(170) < u the inversion throws the
#    count-limit `ArgumentError` (`_gp_count_limit_error`).
# Draws <= 170 are otherwise those of the plain inversion; no extra uniforms are
# used.
const _GP_INVERSION_SATURATION_TOL = 1e-8

"""
    _gp_inversion_stop(previous, next)

Saturation rule of the inversion with `F(i - 1) = previous` and
`F(i) = next < u` (see the comment above): `true` if `i` is returned, `false`
to continue.
"""
@inline function _gp_inversion_stop(previous::Real, next::Real)
    return !(next > previous) && 1 - previous <= _GP_INVERSION_SATURATION_TOL
end

"""
    _gp_count_limit_error(lambda, eta)

Throws the count-limit `ArgumentError` for a Generalized Poisson innovation
above `MAX_FACTORIAL_ARGUMENT` (`F(170) < u` in the inversion).
"""
@noinline function _gp_count_limit_error(lambda::Real, eta::Real)
    throw(ArgumentError("coconots: counts above $(MAX_FACTORIAL_ARGUMENT) are not " *
                        "supported (factorial overflow above $(MAX_FACTORIAL_ARGUMENT)); " *
                        "a simulated Generalized Poisson innovation exceeds " *
                        "$(MAX_FACTORIAL_ARGUMENT) (lambda = $(lambda), eta = $(eta))"))
end

"""
    draw_random_generalized_poisson_variable(u, lambda, eta)

Draws from the Generalized Poisson distribution by inverse-transform sampling
of the uniform variate `u` (smallest `i` with `F(i) >= u`; see
`_gp_inversion_stop` for the termination when `F` saturates below `u`).
"""
function draw_random_generalized_poisson_variable(u::Real, lambda::Real, eta::Real)
    cumulative = generalized_poisson_distribution(0, lambda, eta)
    i = 0
    while cumulative < u
        # F(i) < u: the draw is above i
        i >= MAX_FACTORIAL_ARGUMENT && _gp_count_limit_error(lambda, eta)
        i += 1
        next = cumulative + generalized_poisson_distribution(i, lambda, eta)
        if next < u && _gp_inversion_stop(cumulative, next)
            return i
        end
        cumulative = next
    end
    return i
end

"""
    _check_simulation_rates(lambda)

Simulation needs a finite, positive innovation rate `lambda_t = link(x_t' beta)`
for every `t` (the identity link gives `lambda_t <= 0` for `x_t' beta <= 0`).
"""
function _check_simulation_rates(lambda)
    for (t, l) in enumerate(lambda)
        if !(isfinite(l) && l > 0)
            error("The innovation mean lambda_t = link(x_t' beta) must be finite " *
                  "and positive for every t, but lambda_t = $(l) at t = $(t). With " *
                  "the identity link choose covariates and beta such that x_t' beta > 0.")
        end
    end
    return nothing
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
#
# The cumulative distributions of g are stored back to back in one flat buffer
# and located through a dense (y, z) index whose cells are valid only if their
# stamp equals the sampler's current stamp; a rate change invalidates all of
# them in O(1) by incrementing the stamp (no hashing, no per-history vectors).
#
# Second order: the pmf tables of the derived rates are built once per rate by
# recurrences (`_fill_gp_table_rec!`), sized to the largest history of the
# current step, and the inner sum H(a, b) of g(r | y, z) (see `_gp2_h_cell`),
# which does not depend on (y, z) when the s-range is not truncated, is
# tabulated once per rate on the needed block and shared by all histories.
#--------------------------------------------------------------------------------------

# Upper bound on the number of cached cumulative-distribution entries per rate
# (Float64 values); histories beyond it are evaluated in a scratch buffer.
const _SIM_CACHE_MAX_ENTRIES = 4_194_304

# Histories with y or z at or above this bound are not indexed (scratch buffer).
const _SIM_CACHE_MAX_DIM = 1024

# Upper bound on the number of cells of the shared inner-sum table H(a, b);
# histories outside it compute their own block of H.
const _SHARED_H_MAX_CELLS = 65_536

"""
    _invert_cdf(cdf, u, upper)

Smallest `i` in `0:(upper - 1)` with `!(cdf[i + 1] < u)`, else `upper`. This is
the stopping rule of the sequential inverse-transform loops above.
"""
@inline function _invert_cdf(cdf::Vector{Float64}, u::Float64, upper::Int)
    return _invert_cdf(cdf, 1, u, upper)
end

"""
    _invert_cdf(store, offset, u, upper)

As `_invert_cdf(cdf, u, upper)` for the cumulative distribution stored at
`store[offset:(offset + upper)]`.
"""
@inline function _invert_cdf(store::Vector{Float64}, offset::Int, u::Float64, upper::Int)
    @inbounds for i in 0:(upper - 1)
        store[offset + i] < u || return i
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
    U::Float64
    lambda::Float64
    innovation_cdf::Vector{Float64}
    # cell (y + 1, z + 1): offset of the cdf of g(. | y[, z]) in g_store, valid
    # iff g_stamp[y + 1, z + 1] == stamp
    g_offset::Matrix{Int}
    g_stamp::Matrix{Int}
    stamp::Int
    g_store::Vector{Float64}
    tables::NTuple{5, Vector{Float64}}
    kernel::Union{Nothing, GP2Kernel{Float64}}
    kernel_max::Int
    # shared inner sums H(a, b) = h[b * (h_amax + 1) + a + 1], a in 0:h_amax,
    # b in 0:h_bmax (empty if h_amax < 0)
    h::Vector{Float64}
    h_amax::Int
    h_bmax::Int
    # shared products C(t, s) = P_beta3(s) P_beta2(t - s) = c[t * (h_bmax + 1) + s + 1],
    # t in 0:h_amax, s in 0:min(t, h_bmax), nonzero only on c_lo[t + 1]:c_hi[t + 1]
    # (see `_GP2CTable`); built together with h
    c::Vector{Float64}
    c_lo::Vector{Int}
    c_hi::Vector{Int}
    ebuf::Vector{Float64}
    gbuf::Vector{Float64}
    hbuf::Vector{Float64}
    scratch::Vector{Float64}
end

function _TransitionSampler(order::Int, alpha::Float64, alpha1::Float64, alpha2::Float64,
        alpha3::Float64, eta::Float64)
    U = order == 2 ? Float64(compute_U(alpha1, alpha2, alpha3)) : NaN
    return _TransitionSampler(order, alpha, alpha1, alpha2, alpha3, eta, U, NaN,
        Float64[], zeros(Int, 0, 0), zeros(Int, 0, 0), 1, Float64[],
        (Float64[], Float64[], Float64[], Float64[], Float64[]), nothing, -1,
        Float64[], -1, -1, Float64[], Int[], Int[], Float64[], Float64[], Float64[],
        Float64[])
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
    s.stamp += 1
    empty!(s.g_store)
    s.kernel = nothing
    s.kernel_max = -1
    s.h_amax = -1
    s.h_bmax = -1
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
sampler's rate (including its termination rule), from a lazily extended
cumulative table.
"""
function _draw_innovation!(s::_TransitionSampler, u::Float64)
    cdf = s.innovation_cdf
    i = 0
    while true
        if i >= length(cdf)
            _extend_innovation_cdf!(s, max(32, 2 * length(cdf)))
        end
        @inbounds c = cdf[i + 1]
        c < u || return i
        if i > 0 && _gp_inversion_stop(@inbounds(cdf[i]), c)
            return i
        end
        # F(i) < u: the draw is above i
        i >= MAX_FACTORIAL_ARGUMENT && _gp_count_limit_error(s.lambda, s.eta)
        i += 1
    end
end

"""
    _ensure_kernel!(sampler, needed)

Second-order pmf tables of `beta1`, `beta2`, `beta3`, `lambda` and `zeta` at
the sampler's rate on (at least) `0:needed`, built by the recurrences of
[`_fill_gp_table_rec!`](@ref) (equal to the per-entry pmf up to rounding).
"""
function _ensure_kernel!(s::_TransitionSampler, needed::Int)
    if s.kernel === nothing || s.kernel_max < needed
        lambda = s.lambda
        eta = s.eta
        q = exp(-eta)
        eta_is_zero = iszero(eta)
        lambda_U = lambda * s.U
        tb1, tb2, tb3, tl, tz = s.tables
        _fill_gp_table_rec!(tb1, needed, lambda_U * s.alpha1, eta, q, eta_is_zero)
        _fill_gp_table_rec!(tb2, needed, lambda_U * s.alpha2, eta, q, eta_is_zero)
        _fill_gp_table_rec!(tb3, needed, lambda_U * s.alpha3, eta, q, eta_is_zero)
        _fill_gp_table_rec!(tl, needed, lambda, eta, q, eta_is_zero)
        _fill_gp_table_rec!(tz, needed, lambda_U * (1 - 2 * s.alpha1 - s.alpha3), eta,
            q, eta_is_zero)
        s.kernel = GP2Kernel{Float64}(lambda, s.alpha1, s.alpha2, s.alpha3, eta, tb1,
            tb2, tb3, tl, tz)
        s.kernel_max = needed
    end
    return s.kernel::GP2Kernel{Float64}
end

"""
    _prepare_order2_step!(sampler, ymax, zmax)

Called once per time step (after `_set_rate!`) with the largest `y = X_{t-1}`
and `z = X_{t-2}` over the replicates: sizes the pmf tables to `ymax + zmax`
and extends the shared inner-sum table to `a in 0:zmax`, `b in 0:ymax` (within
`_SHARED_H_MAX_CELLS`). Under a fixed rate both only grow.
"""
function _prepare_order2_step!(s::_TransitionSampler, ymax::Int, zmax::Int)
    kernel = _ensure_kernel!(s, ymax + zmax)
    (zmax <= s.h_amax && ymax <= s.h_bmax) && return s
    amax = max(zmax, s.h_amax)
    bmax = max(ymax, s.h_bmax)
    na = amax + 1
    na * (bmax + 1) > _SHARED_H_MAX_CELLS && return s
    h = s.h
    length(h) < na * (bmax + 1) && resize!(h, na * (bmax + 1))
    # cap = b: min(cap, a, b) = min(a, b), the untruncated inner sum, which is
    # what g(r | y, z) reads for every b <= y (s-range 0:y)
    for b in 0:bmax
        _gp2_h_column!(h, b * na, s.ebuf, kernel, b, 0, amax, b)
    end
    # C rows t in 0:amax (t <= min(z, upper) = z <= amax) of width bmax + 1, filled
    # on s in 0:min(t, bmax), which covers min(t, smax, y) = min(t, y) for y <= bmax
    nb = bmax + 1
    c = s.c
    length(c) < na * nb && resize!(c, na * nb)
    length(s.c_lo) < na && resize!(s.c_lo, na)
    length(s.c_hi) < na && resize!(s.c_hi, na)
    for t in 0:amax
        l, u = _gp2_c_row!(c, t * nb, kernel, t, min(t, bmax))
        s.c_lo[t + 1] = l
        s.c_hi[t + 1] = u
    end
    s.h_amax = amax
    s.h_bmax = bmax
    return s
end

"""
    _gp2_sampler_normalizer(sampler, y, z)

Bivariate Generalized Poisson mass of `(y, z)` at the sampler's rate
([`_gp2_normalizer`](@ref) for `max(y, z) <= 170`, else
[`bivariate_generalized_poisson`](@ref)).
"""
function _gp2_sampler_normalizer(s::_TransitionSampler, y::Int, z::Int)
    if max(y, z) <= MAX_FACTORIAL_ARGUMENT
        lambda_U = s.lambda * s.U
        return _gp2_normalizer(y, z, lambda_U * (1 - s.alpha1 - s.alpha3),
            lambda_U * (s.alpha1 + s.alpha3), lambda_U * s.alpha1 + lambda_U * s.alpha3,
            s.eta, exp(s.eta), iszero(s.eta))
    end
    return bivariate_generalized_poisson(y, z, s.lambda, s.alpha1, s.alpha2, s.alpha3,
        s.eta)
end

"""
    _g_cdf!(store, offset, sampler, y, z)

Cumulative distribution of `g(r | y)` (order 1, support `0:y`) or
`g(r | y, z)` (order 2, support `0:(y + z)`) at the sampler's rate, written
into `store[offset:(offset + upper)]`.
"""
function _g_cdf!(store::Vector{Float64}, offset::Int, s::_TransitionSampler, y::Int,
        z::Int)
    total = 0.0
    if s.order == 1
        @inbounds for r in 0:y
            total += compute_g_r_y(y, r, s.alpha, s.eta, s.lambda)
            store[offset + r] = total
        end
        return store
    end
    upper = y + z
    kernel = _ensure_kernel!(s, upper)
    if y <= s.h_bmax && z <= s.h_amax
        gbuf = s.gbuf
        length(gbuf) < upper + 1 && resize!(gbuf, upper + 1)
        normalizer = _gp2_sampler_normalizer(s, y, z)
        base = range(0; step = s.h_amax + 1, length = y + 1)
        ct = _GP2CTable(s.c, range(0; step = s.h_bmax + 1, length = s.h_amax + 1),
            s.c_lo, s.c_hi)
        g = _gp2_g_vector!(gbuf, kernel, s.h, base, ct, y, z, upper, normalizer, y)
    else
        g = _gp2_g_vector!(s.gbuf, s.hbuf, kernel, y, z, upper, nothing)
    end
    @inbounds for r in 0:upper
        total += g[r + 1]
        store[offset + r] = total
    end
    return store
end

function _grow_cdf_index!(s::_TransitionSampler, ny::Int, nz::Int)
    oy, oz = size(s.g_stamp)
    (ny <= oy && nz <= oz) && return s
    my = ny <= oy ? oy : min(_SIM_CACHE_MAX_DIM, max(ny, 2 * oy, 16))
    mz = nz <= oz ? oz : min(_SIM_CACHE_MAX_DIM, max(nz, 2 * oz, s.order == 1 ? 1 : 16))
    stamp = zeros(Int, my, mz)
    offset = zeros(Int, my, mz)
    stamp[1:oy, 1:oz] .= s.g_stamp
    offset[1:oy, 1:oz] .= s.g_offset
    s.g_stamp = stamp
    s.g_offset = offset
    return s
end

"""
    _draw_g!(sampler, u, y, z)

Same draw as [`draw_random_g`](@ref) at the sampler's rate, from a cached
cumulative distribution per history `(y[, z])`.
"""
function _draw_g!(s::_TransitionSampler, u::Float64, y::Int, z::Int)
    if s.order == 1
        upper = y
        zk = 0
    else
        upper = y + z
        zk = z
    end
    nonnegative = y >= 0 && zk >= 0
    if nonnegative && y < size(s.g_stamp, 1) && zk < size(s.g_stamp, 2) &&
       @inbounds(s.g_stamp[y + 1, zk + 1]) == s.stamp
        return _invert_cdf(s.g_store, @inbounds(s.g_offset[y + 1, zk + 1]), u, upper)
    end
    offset = length(s.g_store) + 1
    if nonnegative && offset + upper <= _SIM_CACHE_MAX_ENTRIES &&
       y < _SIM_CACHE_MAX_DIM && zk < _SIM_CACHE_MAX_DIM
        _grow_cdf_index!(s, y + 1, zk + 1)
        resize!(s.g_store, offset + upper)
        _g_cdf!(s.g_store, offset, s, y, zk)
        @inbounds s.g_offset[y + 1, zk + 1] = offset
        @inbounds s.g_stamp[y + 1, zk + 1] = s.stamp
        return _invert_cdf(s.g_store, offset, u, upper)
    end
    _g_cdf!(resize!(s.scratch, upper + 1), 1, s, y, zk)
    return _invert_cdf(s.scratch, 1, u, upper)
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
        _check_simulation_rates(lambda_data)
        lambda = Float64.(repeat(lambda_data,
            Int(ceil(1 + n_burn_in / n)))[(end - n_total + 1):end])
    else
        _check_simulation_rates(last(parameter))
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

    # colmax[t]: largest count of column t over the replicates (sizes the
    # second-order tables of the next two steps)
    colmax = zeros(Int, n_total)
    for t in 1:min(t_first - 1, n_total)
        for b in 1:n_rep
            colmax[t] = max(colmax[t], X[b, t])
        end
    end

    for t in t_first:n_total
        _set_rate!(s, lambda[t])
        if order == 2
            _prepare_order2_step!(s, colmax[t - 1], t > 2 ? colmax[t - 2] : 0)
        end
        m = 0
        for b in 1:n_rep
            y = X[b, t - 1]
            z = t > 2 ? X[b, t - 2] : 0
            r = _draw_g!(s, rand(), y, z)
            x = r + _draw_innovation!(s, rand())
            X[b, t] = x
            m = max(m, x)
        end
        colmax[t] = m
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
    # counts above 170 are outside the supported range (as on the RCPP backend)
    _check_count_limit(X, "simulated")
    for t in 1:n_total
        x[t] = X[1, t]
    end

    return x[(end - n + 1):end]
end
