export cocoSim

const _MAX_INVERSE_TRANSFORM_ITERATIONS = 1_000_000

"""
    draw_random_generalized_poisson_variable(u, lambda, eta)

Draws from the Generalized Poisson distribution by inverse-transform sampling
of the uniform variate `u`.
"""
function draw_random_generalized_poisson_variable(u::Real, lambda::Real, eta::Real)
    cumulative = generalized_poisson_distribution(0, lambda, eta)
    i = 0
    while cumulative < u && i < _MAX_INVERSE_TRANSFORM_ITERATIONS
        i += 1
        cumulative += generalized_poisson_distribution(i, lambda, eta)
    end
    return i
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

"""
    cocoSim(type, order, parameter, n, covariates=nothing, link="log", n_burn_in=200,
            x=zeros(Int(n + n_burn_in)))

Simulates a count time series of length `n` from a (Generalized) Poisson
autoregressive model, discarding `n_burn_in` initial observations. Parameter
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
    link_function = get_link_function(link)

    if !isnothing(covariates)
        n_betas = size(covariates, 2)
        lambda_data = link_function.(covariates * parameter[(end - n_betas + 1):end])
        lambda = repeat(lambda_data, Int(ceil(1 + n_burn_in / n)))[(end - n_total + 1):end]
    else
        lambda = fill(float(last(parameter)), n_total)
    end

    if order == 2
        alpha1, alpha2, alpha3 = parameter[1], parameter[2], parameter[3]
        alpha = nothing
        eta = type == "GP" ? parameter[4] : zero(eltype(lambda))
    else
        alpha1 = alpha2 = alpha3 = nothing
        alpha = parameter[1]
        eta = type == "GP" ? parameter[2] : zero(eltype(lambda))
    end

    for t in 3:n_total
        x[t] = draw_random_g(order, rand(), Int(x[t - 1]), Int(x[t - 2]), lambda[t],
            alpha1, alpha2, alpha3, alpha, eta, nothing) +
               draw_random_generalized_poisson_variable(rand(), lambda[t], eta)
    end

    return x[(end - n + 1):end]
end
