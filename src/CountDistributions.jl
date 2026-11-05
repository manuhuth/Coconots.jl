const MAX_FACTORIAL_ARGUMENT = 170
const FACTORIAL_TABLE = Float64[factorial(big(k)) for k in 0:MAX_FACTORIAL_ARGUMENT]

"""
    _check_count_limit(data, what = "observed")

Throws an `ArgumentError` if a count in `data` exceeds `MAX_FACTORIAL_ARGUMENT`
(170). Counts above it are outside the supported range of the package: Float64
factorials overflow at 171, and the RCPP backend of coconots rejects such counts,
so both backends accept and reject the same inputs. `what` names the counts in
the message (`"observed"`, `"simulated"`).
"""
function _check_count_limit(data, what::AbstractString = "observed")
    isempty(data) && return nothing
    m = maximum(data)
    if m > MAX_FACTORIAL_ARGUMENT
        throw(ArgumentError("coconots: counts above $(MAX_FACTORIAL_ARGUMENT) are not " *
                            "supported (factorial overflow above $(MAX_FACTORIAL_ARGUMENT)); " *
                            "$what count = $m"))
    end
    return nothing
end

"""
    _check_support_limit(value, name)

Throws an `ArgumentError` if the largest support point `value` of a requested
pmf (argument `name`, e.g. `max_x` of [`compute_scores`](@ref)) exceeds
`MAX_FACTORIAL_ARGUMENT`.
"""
function _check_support_limit(value::Integer, name::AbstractString)
    if value > MAX_FACTORIAL_ARGUMENT
        throw(ArgumentError("coconots: '$name' = $value exceeds the largest supported " *
                            "count $(MAX_FACTORIAL_ARGUMENT) (counts above " *
                            "$(MAX_FACTORIAL_ARGUMENT) are not supported: factorial overflow)"))
    end
    return nothing
end

"""
    _check_min_length(n, min_n, fun, reason)

Throws an `ArgumentError` if a series of length `n` is shorter than `min_n`.
"""
function _check_min_length(n::Integer, min_n::Integer, fun::AbstractString,
        reason::AbstractString)
    if n < min_n
        throw(ArgumentError("$fun: the time series has $n observation(s), but at least " *
                            "$min_n are required ($reason)"))
    end
    return nothing
end

"""
    _throw_nonfinite_pmf(what)

Throws an error for a non-finite (NaN or Inf) probability, which arises from
floating-point overflow in the density recursions.
"""
function _throw_nonfinite_pmf(what::AbstractString)
    throw(DomainError(what, "coconots: the $what is not finite (floating-point " *
                            "overflow in the density recursions for these counts and " *
                            "parameters; this can happen for counts close to the " *
                            "supported maximum of $(MAX_FACTORIAL_ARGUMENT) with a large eta)"))
end

"""
    float_factorial(x)

Factorial of `x` as a `Float64`, read from a precomputed table. Valid for
`0 <= x <= 170` (Float64 factorials overflow at 171).
"""
@inline function float_factorial(x::Integer)
    return @inbounds FACTORIAL_TABLE[Int(x) + 1]
end

"""
    log_generalized_poisson_pdf(x, lambda, eta)

Log probability mass function of the Generalized Poisson distribution at the
non-negative integer `x` with rate `lambda` and dispersion `eta`. Used instead
of the direct pmf whenever factorials would overflow.
"""
@inline function log_generalized_poisson_pdf(x::Integer, lambda::Real, eta::Real)
    return log(lambda) + (x - 1) * log(lambda + x * eta) - lambda - x * eta -
           logfactorial(x)
end

"""
    generalized_poisson_distribution(x, lambda, eta)

Probability mass function of the Generalized Poisson distribution at count `x`
with rate `lambda` and dispersion `eta`. Uses a factorial lookup table for
`x <= 170` and log-space evaluation beyond, so it never allocates `BigInt`s and
is differentiable in `lambda` and `eta`.
"""
@inline function generalized_poisson_distribution(x::Real, lambda::Real, eta::Real)
    xi = Int(x)
    # x = 0 and x = 1 in closed form: the general expression evaluates
    # lambda * lambda^(-1) at x = 0, which is NaN at lambda = 0 and whose
    # ForwardDiff Hessian overflows (Inf - Inf) for tiny lambda, e.g. the rate
    # beta2 = lambda * U * alpha2 when alpha2 is fitted to (nearly) zero.
    xi == 0 && return exp(-lambda - zero(eta))
    xi == 1 && return exp(-lambda - eta) * lambda
    if xi <= MAX_FACTORIAL_ARGUMENT
        return exp(-lambda - xi * eta) * lambda * (lambda + xi * eta)^(xi - 1) /
               float_factorial(xi)
    end
    return exp(log_generalized_poisson_pdf(xi, lambda, eta))
end

"""
    _primal_value(x)

Innermost primal value of `x` (for nested `ForwardDiff.Dual`s the value of the
innermost number), `x` itself for other numbers.
"""
@inline _primal_value(x::Real) = x
@inline _primal_value(x::ForwardDiff.Dual) = _primal_value(ForwardDiff.value(x))

"""
    _is_zero_value(x)

True iff the primal value of `x` is exactly zero, whatever its partials.
"""
@inline _is_zero_value(x::Real) = _primal_value(x) == 0

"""
    _bivariate_gp_zero_rate(y, z, c, d, beta13, eta)

Bivariate Generalized Poisson mass of `(y, z)` when
`d = lambda U (alpha1 + alpha3)` is exactly zero, i.e. `alpha1 = alpha3 = 0`
(boundary estimates are returned exactly at 0). With `c` and
`beta13 = beta1 + beta3` as in [`bivariate_generalized_poisson`](@ref). The
`j = 0` summand there carries the factor `beta13 * d^(-1)` of the GP pmf of rate
`d` at 0, which is `0 * Inf = NaN` at `d = 0`; since `beta13` equals `d`
identically in the parameters, that factor is its limit 1 and the `j = 0`
summand is evaluated without it. The summands `j >= 1` are those of
`bivariate_generalized_poisson` (finite at `d = 0`: the exponent of
`(d + eta j)^(j - 1)` is non-negative; in the log-space branch a zero base is
raised directly instead of via `log`). Both parts are the analytic expression at
every parameter value, so `ForwardDiff.Dual`s carry the exact partials.
"""
function _bivariate_gp_zero_rate(y::Int, z::Int, c::Real, d::Real, beta13::Real,
        eta::Real)
    S = promote_type(typeof(c), typeof(d), typeof(beta13), typeof(eta))
    direct = max(y, z) <= MAX_FACTORIAL_ARGUMENT
    if direct
        head = (c + eta * y)^(y - 1) * (c + eta * z)^(z - 1) / float_factorial(y) /
               float_factorial(z)
    else
        head = exp((y - 1) * log(c + eta * y) + (z - 1) * log(c + eta * z) -
                   logfactorial(y) - logfactorial(z))
    end
    tail = zero(S)
    for j in 1:min(y, z)
        base = d + eta * j
        if direct
            tail += (c + eta * (y - j))^(y - j - 1) * (c + eta * (z - j))^(z - j - 1) *
                    base^(j - 1) / float_factorial(j) / float_factorial(y - j) /
                    float_factorial(z - j) * exp(j * eta)
        else
            rest = (y - j - 1) * log(c + eta * (y - j)) +
                   (z - j - 1) * log(c + eta * (z - j)) - logfactorial(j) -
                   logfactorial(y - j) - logfactorial(z - j) + j * eta
            tail += _is_zero_value(base) ? base^(j - 1) * exp(rest) :
                    exp((j - 1) * log(base) + rest)
        end
    end
    return (head + beta13 * tail) * c^2 * exp(-beta13 - 2 * c - y * eta - z * eta)
end

"""
    bivariate_generalized_poisson(y, z, lambda, alpha1, alpha2, alpha3, eta)

Joint probability mass function of the bivariate Generalized Poisson
distribution of two consecutive counts `(y, z)`. All loop-invariant quantities
are hoisted; factorials come from the lookup table (log-space beyond 170).
With `alpha1 + alpha3` exactly zero, [`_bivariate_gp_zero_rate`](@ref).
"""
function bivariate_generalized_poisson(y::Real, z::Real, lambda::Real, alpha1::Real,
        alpha2::Real, alpha3::Real, eta::Real)
    yi, zi = Int(y), Int(z)
    U = compute_U(alpha1, alpha2, alpha3)
    beta1 = compute_beta_i(lambda, U, alpha1)
    beta3 = compute_beta_i(lambda, U, alpha3)
    c = lambda * U * (1 - alpha1 - alpha3)
    d = lambda * U * (alpha1 + alpha3)
    _is_zero_value(d) && return _bivariate_gp_zero_rate(yi, zi, c, d, beta1 + beta3, eta)

    total = zero(promote_type(typeof(c), typeof(eta)))
    if max(yi, zi) <= MAX_FACTORIAL_ARGUMENT
        for j in 0:min(yi, zi)
            total += (c + eta * (yi - j))^(yi - j - 1) * (c + eta * (zi - j))^(zi - j - 1) *
                     (d + eta * j)^(j - 1) / float_factorial(j) / float_factorial(yi - j) /
                     float_factorial(zi - j) * exp(j * eta)
        end
    else
        for j in 0:min(yi, zi)
            total += exp((yi - j - 1) * log(c + eta * (yi - j)) +
                         (zi - j - 1) * log(c + eta * (zi - j)) +
                         (j - 1) * log(d + eta * j) - logfactorial(j) -
                         logfactorial(yi - j) - logfactorial(zi - j) + j * eta)
        end
    end

    return total * (beta1 + beta3) * c^2 *
           exp(-(beta1 + beta3) - 2 * c - yi * eta - zi * eta)
end
