const MAX_FACTORIAL_ARGUMENT = 170
const FACTORIAL_TABLE = Float64[factorial(big(k)) for k in 0:MAX_FACTORIAL_ARGUMENT]

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
    poisson_distribution(x, lambda)

Probability mass function of the Poisson distribution at count `x` with rate
`lambda`.
"""
function poisson_distribution(x::Real, lambda::Real)
    xi = Int(x)
    if xi <= MAX_FACTORIAL_ARGUMENT
        return exp(-lambda) * lambda^xi / float_factorial(xi)
    end
    return exp(xi * log(lambda) - lambda - logfactorial(xi))
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
    if xi <= MAX_FACTORIAL_ARGUMENT
        return exp(-lambda - xi * eta) * lambda * (lambda + xi * eta)^(xi - 1) /
               float_factorial(xi)
    end
    return exp(log_generalized_poisson_pdf(xi, lambda, eta))
end

"""
    bivariate_generalized_poisson(y, z, lambda, alpha1, alpha2, alpha3, eta)

Joint probability mass function of the bivariate Generalized Poisson
distribution of two consecutive counts `(y, z)`. All loop-invariant quantities
are hoisted; factorials come from the lookup table (log-space beyond 170).
"""
function bivariate_generalized_poisson(y::Real, z::Real, lambda::Real, alpha1::Real,
        alpha2::Real, alpha3::Real, eta::Real)
    yi, zi = Int(y), Int(z)
    U = compute_U(alpha1, alpha2, alpha3)
    beta1 = compute_beta_i(lambda, U, alpha1)
    beta3 = compute_beta_i(lambda, U, alpha3)
    c = lambda * U * (1 - alpha1 - alpha3)
    d = lambda * U * (alpha1 + alpha3)

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
