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
        @simd for r in 0:min(x, rmax)
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

"""
    _gp2_kernel!(tables, lambda, alpha1, alpha2, alpha3, eta, max_index)

Same as [`GP2Kernel`](@ref), but fills the five caller-owned `tables`
(resized to `max_index + 1`) instead of allocating new ones. The returned
kernel aliases `tables`, so it is only valid until the next call.
"""
function _gp2_kernel!(tables::NTuple{5, Vector{T}}, lambda::Real, alpha1::Real,
        alpha2::Real, alpha3::Real, eta::Real, max_index::Integer) where {T}
    U = compute_U(alpha1, alpha2, alpha3)
    _fill_gp_table!(tables[1], compute_beta_i(lambda, U, alpha1), eta, max_index)
    _fill_gp_table!(tables[2], compute_beta_i(lambda, U, alpha2), eta, max_index)
    _fill_gp_table!(tables[3], compute_beta_i(lambda, U, alpha3), eta, max_index)
    _fill_gp_table!(tables[4], lambda, eta, max_index)
    _fill_gp_table!(tables[5], compute_zeta(lambda, U, alpha1, alpha3), eta, max_index)
    return GP2Kernel{T}(lambda, alpha1, alpha2, alpha3, eta, tables...)
end

function _fill_gp_table!(table::Vector, rate::Real, eta::Real, max_index::Integer)
    resize!(table, max_index + 1)
    @inbounds for i in 0:max_index
        table[i + 1] = generalized_poisson_distribution(i, rate, eta)
    end
    return table
end

"""
    _gp2_h_cell(kernel, a, b, cap)

Innermost sum of `g(r | y, z)`:
`H(a, b) = sum_{w = 0}^{min(cap, a, b)} P_beta1(w) P_lambda(a - w) P_zeta(b - w)`.
It depends on the outer indices `(r, s, v)` only through `a = z - r + v` and
`b = y - s - v`, so tabulating it once per `(a, b)` removes the repeated
evaluation of identical inner sums.
"""
@inline function _gp2_h_cell(kernel::GP2Kernel{T}, a::Int, b::Int, cap::Int) where {T}
    acc = zero(T)
    @inbounds for w in 0:min(cap, a, b)
        acc += kernel.gp_beta1[w + 1] * kernel.gp_lambda[a - w + 1] *
               kernel.gp_zeta[b - w + 1]
    end
    return acc
end

#--------------------------------------------------------------------------------------
# Factored evaluation of g(r | y, z)
#
# With the kernel tables P1 = gp_beta1, P2 = gp_beta2, P3 = gp_beta3 and the inner
# sums H(a, b) (see `_gp2_h_cell`), the sum of `compute_g_r_y_z` is
#
#   N g(r) = sum_{s, v} P3(s) P1(v) P2(r - s - v) H(z - r + v, y - s - v)
#
# over the feasible set F(r) = {s, v >= 0 : s <= smax, v <= smax, s + v <= r,
# s + v <= y, v >= r - z} (u = r - s - v >= 0 indexes P2, a = z - r + v >= 0 and
# b = y - s - v >= 0 index H; s <= r and s <= y follow from v >= 0). Substituting
# t = r - v (so u = t - s and a = z - t) gives, for every r in 0:rmax,
#
#   N g(r) = sum_{v = max(0, r - z)}^{min(smax, r, y)} P1(v) *
#            sum_{s = 0}^{min(smax, t, y - v)} C(t, s) H(z - t, y - v - s),  t = r - v,
#
# with C(t, s) = P3(s) P2(t - s): the constraints v >= r - z <=> t <= z, v <= r <=>
# t >= 0, s <= t <=> u >= 0, s <= y - v <=> b >= 0, so (r, s, v) <-> (r, v, s) is a
# bijection of F(r). C does not depend on r, so it is tabulated once (rows
# t in 0:min(z, rmax), row t holding s in 0:S_t with S_t >= min(t, smax, y)), and
# each term of the triple sum costs one multiplication instead of three. The inner
# sum of H is factored the same way (see `_gp2_h_column!`).
#
# Exact-zero trimming: every row of C (and every column factor E of H) stores the
# first and last index whose entry is not exactly zero (`iszero`, i.e. value and all
# partials, recursively for nested Duals). Entries outside that range are exactly
# zero and are skipped: their product with a finite factor is +-0 in every
# component, and adding +-0 to an accumulator that starts at +0 never changes it
# (an IEEE sum is -0 only if both summands are -0). The result is therefore the
# same bits as summing all terms in the same order. This removes the subnormal and
# zero arithmetic of the long tails of pmf tables at tiny rates (e.g. beta2, beta3
# of a second-order fit to first-order data).
#--------------------------------------------------------------------------------------

"""
    _GP2CTable

Products `C(t, s) = P_beta3(s) P_beta2(t - s)` stored as
`C(t, s) = c[base[t + 1] + s + 1]`. Row `t` is nonzero (exactly) at most on
`lo[t + 1]:hi[t + 1]` (empty if `hi < lo`), a subrange of `0:min(t, S_t)` where
`S_t` is the last stored index of row `t`.
"""
struct _GP2CTable{T, B <: AbstractVector{Int}}
    c::Vector{T}
    base::B
    lo::Vector{Int}
    hi::Vector{Int}
end

"""
    _gp2_c_row!(c, off, kernel, t, smax_t)

Writes `C(t, s) = P_beta3(s) P_beta2(t - s)` into `c[off + s + 1]` for
`s in 0:smax_t` (`smax_t <= t`; nothing if `smax_t < 0`) and returns `(lo, hi)`,
the first and last `s` with `C(t, s)` not exactly zero (`(0, -1)` if none).
"""
@inline function _gp2_c_row!(c::Vector{T}, off::Int, kernel::GP2Kernel{T}, t::Int,
        smax_t::Int) where {T}
    lo, hi = 0, -1
    @inbounds for s in 0:smax_t
        cs = kernel.gp_beta3[s + 1] * kernel.gp_beta2[t - s + 1]
        c[off + s + 1] = cs
        if !iszero(cs)
            hi < 0 && (lo = s)
            hi = s
        end
    end
    return lo, hi
end

"""
    _gp2_c_block!(c, lo, hi, kernel, y, z, rmax, smax)

[`_GP2CTable`](@ref) for one `(y, z)` history in the caller-owned buffers
(grown if needed): rows `t in 0:min(z, rmax)`, each of width
`max(min(smax, y), 0) + 1` and filled on `s in 0:min(t, smax, y)`. Reads
`gp_beta3[0:min(y, rmax)]` and `gp_beta2[0:rmax]` only.
"""
function _gp2_c_block!(c::Vector{T}, lo::Vector{Int}, hi::Vector{Int},
        kernel::GP2Kernel{T}, y::Int, z::Int, rmax::Int, smax::Int) where {T}
    tmax = min(z, rmax)
    sw = min(smax, y)
    width = max(sw, 0) + 1
    length(c) < (tmax + 1) * width && resize!(c, (tmax + 1) * width)
    length(lo) < tmax + 1 && resize!(lo, tmax + 1)
    length(hi) < tmax + 1 && resize!(hi, tmax + 1)
    for t in 0:tmax
        l, u = _gp2_c_row!(c, t * width, kernel, t, min(t, sw))
        @inbounds lo[t + 1] = l
        @inbounds hi[t + 1] = u
    end
    return _GP2CTable(c, range(0; step = width, length = tmax + 1), lo, hi)
end

"""
    _gp2_h_column!(h, off, e, kernel, b, a_lo, a_hi, cap)

Writes `H(a, b)` (see [`_gp2_h_cell`](@ref)) into `h[off + a + 1]` for
`a in a_lo:a_hi` (`a_lo >= 0`). The factor `E(w) = P_beta1(w) P_zeta(b - w)`
does not depend on `a`, so it is tabulated once per column in the buffer `e`
for `w in 0:wmax`, `wmax = min(cap, b, a_hi)`, and
`H(a, b) = sum_{w = 0}^{min(wmax, a)} E(w) P_lambda(a - w)`, the same range as
`min(cap, a, b)` in `_gp2_h_cell` because `a <= a_hi`. Leading and trailing
exactly-zero `E(w)` are skipped (exact, see the note above `_GP2CTable`).
"""
function _gp2_h_column!(h::Vector{T}, off::Int, e::Vector{T}, kernel::GP2Kernel{T},
        b::Int, a_lo::Int, a_hi::Int, cap::Int) where {T}
    wmax = min(cap, b, a_hi)
    length(e) < wmax + 1 && resize!(e, wmax + 1)
    wlo, whi = 0, -1
    @inbounds for w in 0:wmax
        ew = kernel.gp_beta1[w + 1] * kernel.gp_zeta[b - w + 1]
        e[w + 1] = ew
        if !iszero(ew)
            whi < 0 && (wlo = w)
            whi = w
        end
    end
    @inbounds for a in a_lo:a_hi
        acc = zero(T)
        for w in wlo:min(whi, a)
            acc += e[w + 1] * kernel.gp_lambda[a - w + 1]
        end
        h[off + a + 1] = acc
    end
    return h
end

"""
    _gp2_g_vector!(g, kernel, h, base, ct, y, z, rmax, normalizer, smax)

Writes `g(r | y, z)` for `r = 0, ..., rmax` into `g[1:rmax+1]`, reading the
inner sums `H(a, b)` (see [`_gp2_h_cell`](@ref)) from `h[base[b + 1] + a + 1]`
and the products `C(t, s)` from the [`_GP2CTable`](@ref) `ct`. The H table
must cover `a in max(0, z - rmax):z` for every `b in 0:y`, with a `cap` that is
equivalent to `smax` on `b <= y`; `ct` must cover rows `t in 0:min(z, rmax)`
with `S_t >= min(t, smax, y)`. Same sum as [`compute_g_r_y_z`](@ref), factored
as in the note above `_GP2CTable` (equal up to rounding).
"""
function _gp2_g_vector!(g::AbstractVector{T}, kernel::GP2Kernel{T}, h::Vector{T},
        base::AbstractVector{Int}, ct::_GP2CTable{T}, y::Int, z::Int, rmax::Int,
        normalizer::Real, smax::Int) where {T}
    c, cbase, clo, chi = ct.c, ct.base, ct.lo, ct.hi
    p1 = kernel.gp_beta1
    @inbounds for r in 0:rmax
        total = zero(T)
        for v in max(0, r - z):min(smax, r, y)
            t = r - v               # 0 <= t <= min(z, rmax)
            a1 = z - t + 1          # a + 1 with a = z - r + v >= 0
            bv = y - v + 1          # b + 1 = bv - s with b = y - v - s >= 0
            co = cbase[t + 1] + 1   # C(t, s) = c[co + s]
            shi = min(chi[t + 1], smax, y - v)
            slo = clo[t + 1]
            slo > shi && continue   # every term of this v is exactly zero
            acc = zero(T)
            for s in slo:shi
                acc += c[co + s] * h[base[bv - s] + a1]
            end
            total += p1[v + 1] * acc
        end
        g[r + 1] = total / normalizer
    end
    return g
end

"""
    _gp2_g_vector!(g, h, kernel, y, z, rmax, max_loop)

Stand-alone version for one `(y, z)` history: computes the normalizer, the
needed block `a in max(0, z - rmax):z`, `b in 0:y` of the inner-sum table in
the caller-owned buffer `h` and the products `C(t, s)` (both resized if
needed) and returns `g` holding `g(r | y, z)` for `r = 0, ..., rmax`.
"""
function _gp2_g_vector!(g::Vector{T}, h::Vector{T}, kernel::GP2Kernel{T}, y::Int,
        z::Int, rmax::Int, max_loop) where {T}
    rmax < 0 && return g
    smax = max_loop === nothing ? y : Int(max_loop)
    normalizer = bivariate_generalized_poisson(y, z, kernel.lambda, kernel.alpha1,
        kernel.alpha2, kernel.alpha3, kernel.eta)
    length(g) < rmax + 1 && resize!(g, rmax + 1)
    base = _gp2_h_block!(h, T[], kernel, y, z, rmax, smax)
    ct = _gp2_c_block!(T[], Int[], Int[], kernel, y, z, rmax, smax)
    return _gp2_g_vector!(g, kernel, h, base, ct, y, z, rmax, normalizer, smax)
end

"""
    _gp2_h_block!(h, e, kernel, y, z, rmax, smax)

Fills the block `a in max(0, z - rmax):z`, `b in 0:y` of the inner-sum table
`H(a, b)` (see [`_gp2_h_cell`](@ref)) into the caller-owned buffer `h`
(grown if needed; `e` is scratch for [`_gp2_h_column!`](@ref)) and returns
`base` with `H(a, b) = h[base[b + 1] + a + 1]`. Reads `gp_beta1[0:y]`,
`gp_lambda[0:z]` and `gp_zeta[0:y]` only.
"""
function _gp2_h_block!(h::Vector{T}, e::Vector{T}, kernel::GP2Kernel{T}, y::Int, z::Int,
        rmax::Int, smax::Int) where {T}
    a_lo = max(0, z - rmax)
    na = z - a_lo + 1
    length(h) < na * (y + 1) && resize!(h, na * (y + 1))
    for b in 0:y
        # h[b * na + (a - a_lo) + 1] == h[(b * na - a_lo) + a + 1]
        _gp2_h_column!(h, b * na - a_lo, e, kernel, b, a_lo, z, smax)
    end
    # h[b * na + (a - a_lo) + 1] == h[base[b + 1] + a + 1] with this base
    return range(-a_lo; step = na, length = y + 1)
end

function _convolution_x_r_y_z!(g::Vector{T}, h::Vector{T}, kernel::GP2Kernel{T},
        x::Integer, y::Integer, z::Integer, max_loop) where {T}
    _check_kernel_size(kernel, max(x, y + z))
    rmax = min(Int(x), Int(y) + Int(z))
    _gp2_g_vector!(g, h, kernel, Int(y), Int(z), rmax, max_loop)
    total = zero(T)
    @inbounds for r in 0:rmax
        total += g[r + 1] * kernel.gp_lambda[x - r + 1]
    end
    return total
end

function convolution_x_r_y_z(kernel::GP2Kernel{T}, x::Integer, y::Integer, z::Integer,
        max_loop) where {T}
    return _convolution_x_r_y_z!(T[], T[], kernel, x, y, z, max_loop)
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
    rmax = min(Int(xmax), Int(y) + Int(z))
    g = _gp2_g_vector!(T[], T[], kernel, Int(y), Int(z), rmax, max_loop)
    pmf = Vector{T}(undef, xmax + 1)
    @inbounds for x in 0:xmax
        total = zero(T)
        @simd for r in 0:min(x, rmax)
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

function _transition_pair_counts(data::AbstractVector)
    counts = Dict{NTuple{2, Int}, Int}()
    for t in 2:length(data)
        key = (Int(data[t]), Int(data[t - 1]))
        counts[key] = get(counts, key, 0) + 1
    end
    transitions = sort!(collect(keys(counts)))
    return transitions, Int[counts[k] for k in transitions]
end

function _transition_triple_counts(data::AbstractVector)
    counts = Dict{NTuple{3, Int}, Int}()
    for t in 3:length(data)
        key = (Int(data[t]), Int(data[t - 1]), Int(data[t - 2]))
        counts[key] = get(counts, key, 0) + 1
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
    isempty(transitions) && return zero(T)
    # Serial path: tabulate g(r | y) once per (y, r) and the innovation pmf once
    # per count instead of once per transition. Same factors and the same
    # summation order as compute_convolution_x_r_y, so the result is identical.
    xmax, ymax = 0, 0
    for (x, y) in transitions
        xmax, ymax = max(xmax, x), max(ymax, y)
    end
    need = zeros(Int, ymax + 1)
    for (x, y) in transitions
        need[y + 1] = max(need[y + 1], min(x, y) + 1)
    end
    offsets = Vector{Int}(undef, ymax + 1)
    len = 0
    for y in 0:ymax
        offsets[y + 1] = len
        len += need[y + 1]
    end
    g = Vector{T}(undef, len)
    for y in 0:ymax, r in 0:(need[y + 1] - 1)
        g[offsets[y + 1] + r + 1] = compute_g_r_y(y, r, alpha, eta, lambda)
    end
    innovation = Vector{promote_type(typeof(lambda), typeof(eta))}(undef, xmax + 1)
    for i in 0:xmax
        innovation[i + 1] = generalized_poisson_distribution(i, lambda, eta)
    end
    total = zero(T)
    @inbounds for i in eachindex(transitions)
        x, y = transitions[i]
        offset = offsets[y + 1]
        convolution = zero(T)
        for r in 0:min(x, y)
            convolution += g[offset + r + 1] * innovation[x - r + 1]
        end
        total -= counts[i] * log(convolution)
    end
    return total
end

"""
    _GP2TransitionGroup

Unique transitions of one `(y, z)` history: the observed next counts `xs`
with multiplicities `counts`, and the largest convolution index `rmax`
any of them needs. Grouping lets one `g(r | y, z)` vector (and one bivariate
normalizer) serve every `x` that shares the history.
"""
struct _GP2TransitionGroup
    y::Int
    z::Int
    rmax::Int
    xs::Vector{Int}
    counts::Vector{Int}
end

function _group_transitions(transitions::Vector{NTuple{3, Int}}, counts::Vector{Int})
    order = sortperm(transitions; by = t -> (t[2], t[3], t[1]))
    groups = Vector{_GP2TransitionGroup}()
    for i in order
        x, y, z = transitions[i]
        rmax = min(x, y + z)
        if isempty(groups) || last(groups).y != y || last(groups).z != z
            push!(groups, _GP2TransitionGroup(y, z, rmax, [x], [counts[i]]))
        else
            group = last(groups)
            push!(group.xs, x)
            push!(group.counts, counts[i])
            groups[end] = _GP2TransitionGroup(y, z, max(group.rmax, rmax), group.xs,
                group.counts)
        end
    end
    return groups
end

function _nll_gp2_group(kernel::GP2Kernel{T}, group::_GP2TransitionGroup, max_loop,
        h::Vector{T}, base::Vector{Int}, ct::_GP2CTable{T},
        g_buffer::Vector{T} = Vector{T}(undef, group.rmax + 1)) where {T}
    smax = max_loop === nothing ? group.y : Int(max_loop)
    normalizer = bivariate_generalized_poisson(group.y, group.z, kernel.lambda,
        kernel.alpha1, kernel.alpha2, kernel.alpha3, kernel.eta)
    _gp2_g_vector!(g_buffer, kernel, h, base, ct, group.y, group.z, group.rmax,
        normalizer, smax)
    total = zero(T)
    @inbounds for (x, count) in zip(group.xs, group.counts)
        convolution = zero(T)
        for r in 0:min(x, group.rmax)
            convolution += g_buffer[r + 1] * kernel.gp_lambda[x - r + 1]
        end
        total -= count * log(convolution)
    end
    return total
end

"""
    _gp2_shared_h_table(kernel, groups, max_loop)

One inner-sum table `H(a, b)` (see [`_gp2_h_cell`](@ref)) shared by all
groups. With `max_loop === nothing` the per-group cap `smax = y` never binds
because only `b <= y` is read, and otherwise the cap `max_loop` is the same
for all groups, so one table serves every group. Column `b` stores only the
hull of the `a` ranges `max(0, z - rmax):z` of the groups with `y >= b`
(compact storage, so a single large count does not create a large dense
table). Returns `(h, base)` with `H(a, b) = h[base[b + 1] + a + 1]`.
"""
function _gp2_shared_h_table(kernel::GP2Kernel{T}, groups::Vector{_GP2TransitionGroup},
        max_loop) where {T}
    bmax = maximum(group -> group.y, groups)
    lo = fill(typemax(Int), bmax + 1)
    hi = fill(-1, bmax + 1)
    for group in groups
        a_lo = max(0, group.z - group.rmax)
        for b in 0:(group.y)
            lo[b + 1] = min(lo[b + 1], a_lo)
            hi[b + 1] = max(hi[b + 1], group.z)
        end
    end
    base = Vector{Int}(undef, bmax + 1)
    len = 0
    for b in 0:bmax
        base[b + 1] = len - lo[b + 1]
        len += hi[b + 1] - lo[b + 1] + 1
    end
    cap = max_loop === nothing ? typemax(Int) : Int(max_loop)
    h = Vector{T}(undef, len)
    e = T[]
    for b in 0:bmax
        _gp2_h_column!(h, base[b + 1], e, kernel, b, lo[b + 1], hi[b + 1], cap)
    end
    return h, base
end

"""
    _gp2_shared_c_table(kernel, groups, max_loop)

One [`_GP2CTable`](@ref) of `C(t, s) = P_beta3(s) P_beta2(t - s)` shared by all
groups (C depends on the parameters only). A group `(y, z, rmax)` with
`smax = max_loop === nothing ? y : max_loop` reads rows `t in 0:min(z, rmax)`
at `s in 0:min(t, smax, y)`, so row `t` stores `s in 0:S_t` with `S_t` the
largest `min(t, smax, y)` over the groups that read row `t` (compact storage,
as for the H table). Reads `gp_beta3` and `gp_beta2` on `0:max(min(z, rmax))`
only (covered by the kernel's `max_index >= z`).
"""
function _gp2_shared_c_table(kernel::GP2Kernel{T}, groups::Vector{_GP2TransitionGroup},
        max_loop) where {T}
    tmax = maximum(group -> min(group.z, group.rmax), groups)
    width = fill(-1, tmax + 1)   # S_t (-1: empty row)
    for group in groups
        smax = max_loop === nothing ? group.y : Int(max_loop)
        sw = min(smax, group.y)
        for t in 0:min(group.z, group.rmax)
            width[t + 1] = max(width[t + 1], min(t, sw))
        end
    end
    base = Vector{Int}(undef, tmax + 1)
    len = 0
    for t in 0:tmax
        base[t + 1] = len
        len += width[t + 1] + 1
    end
    c = Vector{T}(undef, len)
    lo = Vector{Int}(undef, tmax + 1)
    hi = Vector{Int}(undef, tmax + 1)
    for t in 0:tmax
        l, u = _gp2_c_row!(c, base[t + 1], kernel, t, width[t + 1])
        lo[t + 1] = l
        hi[t + 1] = u
    end
    return _GP2CTable(c, base, lo, hi)
end

function _nll_gp2_transitions(lambda::Real, alpha1::Real, alpha2::Real, alpha3::Real,
        eta::Real, groups::Vector{_GP2TransitionGroup}, max_index::Integer, max_loop)
    kernel = GP2Kernel(lambda, alpha1, alpha2, alpha3, eta, max_index)
    T = eltype(kernel.gp_lambda)
    h, base = _gp2_shared_h_table(kernel, groups, max_loop)
    ct = _gp2_shared_c_table(kernel, groups, max_loop)
    if nthreads() > 1 && length(groups) > 16
        parts = Vector{T}(undef, length(groups))
        @threads for i in eachindex(groups)
            parts[i] = _nll_gp2_group(kernel, groups[i], max_loop, h, base, ct)
        end
        return sum(parts)
    end
    total = zero(T)
    g_buffer = Vector{T}(undef, maximum(group -> group.rmax, groups) + 1)
    for group in groups
        total += _nll_gp2_group(kernel, group, max_loop, h, base, ct, g_buffer)
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
    transitions, counts = _transition_pair_counts(data)
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
    transitions, counts = _transition_triple_counts(data)
    return _nll_gp2_transitions(lambda, alpha1, alpha2, alpha3, eta,
        _group_transitions(transitions, counts), _gp2_max_index(transitions), max_loop)
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
    return _nll_gp2_covariates_serial(lambdas, alpha1, alpha2, alpha3, eta, data,
        max_loop)
end

"""
    _fill_gp_table_rec!(table, n, rate, eta, q, eta_is_zero)

Writes the Generalized Poisson pmf at `0:n` into `table[1:n+1]` (grown if
needed; entries beyond `n + 1` are left untouched). Same formula as
[`generalized_poisson_distribution`](@ref), but with the exponential factor
`exp(-rate - k eta)` carried as a running product with the hoisted
`q = exp(-eta)` (one `exp` per table instead of one per entry). With
`eta_is_zero` (eta and all its derivative parts exactly zero, i.e. the
Poisson model) the pmf is the Poisson pmf and is built by the ratio
`p(k) = p(k - 1) rate / k` without any power. Entries above
`MAX_FACTORIAL_ARGUMENT` use the log-space path of
`generalized_poisson_distribution`, as before. Equal to the per-entry
evaluation up to rounding.
"""
function _fill_gp_table_rec!(table::Vector, n::Int, rate::Real, eta::Real, q::Real,
        eta_is_zero::Bool)
    length(table) < n + 1 && resize!(table, n + 1)
    nrec = min(n, MAX_FACTORIAL_ARGUMENT)
    p0 = exp(-rate - zero(eta))
    @inbounds table[1] = p0
    if eta_is_zero
        p = p0
        @inbounds for k in 1:nrec
            p = p * rate / k
            table[k + 1] = p
        end
    elseif nrec >= 1
        e = p0 * q
        @inbounds table[2] = e * rate
        @inbounds for k in 2:nrec
            e *= q
            table[k + 1] = e * rate * (rate + k * eta)^(k - 1) / float_factorial(k)
        end
    end
    @inbounds for k in (MAX_FACTORIAL_ARGUMENT + 1):n
        table[k + 1] = generalized_poisson_distribution(k, rate, eta)
    end
    return table
end

"""
    _gp2_normalizer(y, z, c, d, beta13, eta, exp_eta, eta_is_zero)

[`bivariate_generalized_poisson`](@ref) of `(y, z)` for `max(y, z) <= 170`
with the rate-derived quantities `c = lambda U (1 - alpha1 - alpha3)`,
`d = lambda U (alpha1 + alpha3)` and `beta13 = beta1 + beta3` passed in.
`exp(j eta)` is a running product with the hoisted `exp_eta = exp(eta)`; with
`eta_is_zero` the summands are built by their ratio
`term(j + 1) / term(j) = d (y - j) (z - j) / (c^2 (j + 1))` instead of three
powers each. Equal to `bivariate_generalized_poisson` up to rounding.
"""
function _gp2_normalizer(y::Int, z::Int, c::Real, d::Real, beta13::Real, eta::Real,
        exp_eta::Real, eta_is_zero::Bool)
    m = min(y, z)
    if eta_is_zero
        j0 = 0  # the j = 0 summand, written exactly as in bivariate_generalized_poisson
        term = c^(y - j0 - 1) * c^(z - j0 - 1) * d^(j0 - 1) / float_factorial(j0) /
               float_factorial(y - j0) / float_factorial(z - j0)
        total = term
        ratio = d / (c * c)
        for j in 0:(m - 1)
            term = term * ratio * ((y - j) * (z - j) / (j + 1))
            total += term
        end
    else
        total = zero(promote_type(typeof(c), typeof(eta)))
        ej = one(exp_eta)
        for j in 0:m
            j > 0 && (ej *= exp_eta)
            total += (c + eta * (y - j))^(y - j - 1) * (c + eta * (z - j))^(z - j - 1) *
                     (d + eta * j)^(j - 1) / float_factorial(j) / float_factorial(y - j) /
                     float_factorial(z - j) * ej
        end
    end
    return total * beta13 * c^2 * exp(-beta13 - 2 * c - y * eta - z * eta)
end

"""
    _nll_gp2_covariates_serial(lambdas, alpha1, alpha2, alpha3, eta, data, max_loop)

Serial second-order negative log-likelihood with observation-specific rates.
Every pmf table depends on `lambda_t`, so the tables are rebuilt per `t`, but
(i) only up to the index each one is read at for `(x, y, z)`: `beta1`, `zeta`
on `0:y`, `beta3` on `0:min(y, rmax)`, `beta2` on `0:rmax`, `lambda` on
`0:max(x, z)` with `rmax = min(x, y + z)` (previously all five on
`0:max(x, y + z)`); (ii) the parameter-only factors (`U`, the alpha
combinations, `exp(-eta)`, `exp(eta)`) are computed once per call; (iii) the
tables and the normalizer avoid per-entry `exp` (and, for the Poisson model,
all powers) via recurrences. Same factors and summation order as the generic
kernel path otherwise.
"""
function _nll_gp2_covariates_serial(lambdas::AbstractVector, alpha1::Real, alpha2::Real,
        alpha3::Real, eta::Real, data::AbstractVector, max_loop)
    S = promote_type(eltype(lambdas), typeof(alpha1), typeof(alpha2), typeof(alpha3),
        typeof(eta))
    n = length(data)
    U = compute_U(alpha1, alpha2, alpha3)
    # parenthesized factors of compute_zeta and bivariate_generalized_poisson
    zeta_factor = 1 - 2 * alpha1 - alpha3
    c_factor = 1 - alpha1 - alpha3
    d_factor = alpha1 + alpha3
    eta_is_zero = iszero(eta)
    q = exp(-eta)
    exp_eta = exp(eta)
    tb1, tb2, tb3, tl, tz = S[], S[], S[], S[], S[]
    g, h, e, cbuf = S[], S[], S[], S[]
    clo, chi = Int[], Int[]
    total = zero(S)
    for t in 3:n
        x, y, z = Int(data[t]), Int(data[t - 1]), Int(data[t - 2])
        lambda = lambdas[t]
        rmax = min(x, y + z)
        lambda_U = lambda * U
        beta1 = lambda_U * alpha1
        beta3 = lambda_U * alpha3
        _fill_gp_table_rec!(tb1, y, beta1, eta, q, eta_is_zero)
        _fill_gp_table_rec!(tb2, rmax, lambda_U * alpha2, eta, q, eta_is_zero)
        _fill_gp_table_rec!(tb3, min(y, rmax), beta3, eta, q, eta_is_zero)
        _fill_gp_table_rec!(tl, max(x, z), lambda, eta, q, eta_is_zero)
        _fill_gp_table_rec!(tz, y, lambda_U * zeta_factor, eta, q, eta_is_zero)
        kernel = GP2Kernel{S}(lambda, alpha1, alpha2, alpha3, eta, tb1, tb2, tb3, tl, tz)
        if max(y, z) <= MAX_FACTORIAL_ARGUMENT
            normalizer = _gp2_normalizer(y, z, lambda_U * c_factor, lambda_U * d_factor,
                beta1 + beta3, eta, exp_eta, eta_is_zero)
        else
            normalizer = bivariate_generalized_poisson(y, z, lambda, alpha1, alpha2,
                alpha3, eta)
        end
        smax = max_loop === nothing ? y : Int(max_loop)
        length(g) < rmax + 1 && resize!(g, rmax + 1)
        base = _gp2_h_block!(h, e, kernel, y, z, rmax, smax)
        ct = _gp2_c_block!(cbuf, clo, chi, kernel, y, z, rmax, smax)
        _gp2_g_vector!(g, kernel, h, base, ct, y, z, rmax, normalizer, smax)
        convolution = zero(S)
        @inbounds for r in 0:rmax
            convolution += g[r + 1] * tl[x - r + 1]
        end
        total -= log(convolution)
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
