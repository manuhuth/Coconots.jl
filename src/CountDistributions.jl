"""
    poisson_distribution(x, lambda)

Computes the probability mass function (PMF) of the Poisson distribution at a given count `x` with parameter `λ`.

# Arguments
- `x::Integer`: Non-negative integer value at which to evaluate the PMF.
- `lambda::Real`: Positive rate parameter of the Poisson distribution.

# Returns
- `Real`: Probability of observing the value `x`.
"""
function poisson_distribution(x, lambda)
    return exp(-lambda) * lambda^x / factorial(Int(x))
end

"""
    generalized_poisson_distribution(x, lambda, eta)

Computes the probability mass function of the Generalized Poisson distribution.

# Arguments
- `x::Integer`: Non-negative integer value at which to evaluate the PMF.
- `lambda::Real`: Positive rate parameter.
- `eta::Real`: Dispersion parameter.

# Returns
- `Real`: Probability of observing the value `x`.
"""
function generalized_poisson_distribution(x, lambda, eta)
    if x < 20
        return exp(-lambda - x*eta) * lambda*(lambda + x*eta)^(x - 1) / factorial(Int(x))
    else
        return exp(-lambda - x*eta) * lambda*(lambda + x*eta)^(x - 1) / Float64(factorial(big(Int(x))))
    end
end

"""
    bivariate_generalized_poisson(y, z, lambda, alpha1, alpha2, alpha3, eta)

Computes the joint probability mass function of the bivariate generalized Poisson distribution.

# Arguments
- `y::Integer`: First observed count.
- `z::Integer`: Second observed count.
- `lambda::Real`: Base rate parameter.
- `alpha1::Real`: Interaction parameter.
- `alpha2::Real`: Interaction parameter.
- `alpha3::Real`: Interaction parameter.
- `eta::Real`: Dispersion parameter.

# Returns
- `Real`: Joint probability of observing counts `(y, z)`.
"""
function bivariate_generalized_poisson(y, z, lambda, alpha1, alpha2, alpha3, eta)
    U = compute_U(alpha1, alpha2, alpha3)
    beta3 = compute_beta_i(lambda, U, alpha3)
    beta2 = compute_beta_i(lambda, U, alpha2)
    beta1 = compute_beta_i(lambda, U, alpha1)

    sum = 0.0
    if max(y,z) >= 20
        for j in 0:min(y,z)
            sum += (lambda*U*(1 - alpha1 - alpha3) + eta*(y - j))^(y - j - 1) *
                   (lambda*U*(1 - alpha1 - alpha3) + eta*(z - j))^(z - j - 1) *
                   (lambda*U*(alpha1 + alpha3) + eta*j)^(j - 1) /
                   Float64(factorial(big(Int(j)))) / Float64(factorial(big(Int(y - j)))) /
                   Float64(factorial(big(Int(z - j)))) * exp(j*eta)
        end
    else
        for j in 0:min(y,z)
            sum += (lambda*U*(1 - alpha1 - alpha3) + eta*(y - j))^(y - j - 1) *
                   (lambda*U*(1 - alpha1 - alpha3) + eta*(z - j))^(z - j - 1) *
                   (lambda*U*(alpha1 + alpha3) + eta*j)^(j - 1) /
                   factorial(Int(j)) / factorial(Int(y - j)) / factorial(Int(z - j)) * exp(j*eta)
        end
    end

    return sum * (beta1 + beta3) * (lambda*U*(1 - alpha1 - alpha3))^2 * exp(-(beta1 + beta3) - 2*(lambda*U*(1 - alpha1 - alpha3)) - y*eta - z*eta)
end