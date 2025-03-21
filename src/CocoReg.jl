using Optim, LineSearches, ForwardDiff, Base.Threads

export cocoReg

"""
    minimize_pars_reparameterization_GP2(theta, data, covariates=nothing, link="log", max=nothing)

Computes the negative log-likelihood for a second-order generalized Poisson (GP2) model using reparameterized alpha parameters.
This function constructs the rate parameter vector `lambdas` using a link function and the provided covariates (if any), 
reparameterizes the autoregressive parameters via `reparameterize_alpha(theta)`, and then evaluates the negative log-lelihood.

# Arguments
- `theta`: A vector of parameters. For GP2, `theta[4]` represents the dispersion parameter (eta) and `theta[5]` onward are used for lambda.
- `data`: A vector of observed counts.
- `covariates` (optional): A matrix of covariates used to adjust the rate parameter.
- `link`: (optional) A string specifying the link function (default is `"log"`).
- `max`: (optional) A parameter controlling the iteration limit in the likelihood computation.

# Returns
The negative log-likelihood value computed by `compute_negative_log_likelihood_GP2`.
"""
function minimize_pars_reparameterization_GP2(theta, data, covariates=nothing, link="log", max=nothing)
    link_function = get_link_function(link)
    lambdas = repeat([theta[5]], length(data))
    if !isnothing(covariates)
        lambdas .= link_function.(covariates * theta[5:5+(size(covariates, 2)-1)])
    end

    alpha1, alpha2, alpha3 = reparameterize_alpha(theta)
    
    return compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, theta[4], data, max)
end

"""
    minimize_pars_reparameterization_Poisson2(theta, data, covariates=nothing, link="log", max=nothing)

Computes the negative log-likelihood for a second-order Poisson model using reparameterized alpha parameters.
The function constructs the rate parameter vector `lambdas` using the link function and covariates (if provided),
reparameterizes the autoregressive parameters, and evaluates the negative log-likelihood with dispersion set to zero.

# Arguments
- `theta`: A parameter vector where `theta[4]` is used for the base lambda.
- `data`: A vector of observed counts.
- `covariates` (optional): A matrix of covariates.
- `link`: (optional) The link function to use (default is `"log"`).
- `max`: (optional) Parameter to control iteration limits in the likelihood computation.

# Returns
The negative log-likelihood value computed by `compute_negative_log_likelihood_GP2` with eta set to 0.
"""
function minimize_pars_reparameterization_Poisson2(theta, data, covariates=nothing, link="log", max=nothing)
    link_function = get_link_function(link)
    lambdas = repeat([theta[4]], length(data))
    if !isnothing(covariates)
        lambdas .= link_function.(covariates * theta[4:4+(size(covariates, 2)-1)])
    end

    alpha1, alpha2, alpha3 = reparameterize_alpha(theta)
    
    return compute_negative_log_likelihood_GP2(lambdas, alpha1, alpha2, alpha3, 0, data, max)
end

"""
    minimize_pars_GP2(theta, data, covariates=nothing, link="log", max=nothing)

Computes the negative log-likelihood for a second-order generalized Poisson (GP2) model using the provided parameters.
The lambda vector is computed from `theta[5]` and adjusted by covariates if available, and the likelihood is evaluated using 
`compute_negative_log_likelihood_GP2`.

# Arguments
- `theta`: A vector of parameters where `theta[1:3]` are the autoregressive parameters, `theta[4]` is the dispersion parameter (eta),
  and `theta[5]` onward are used for lambda computation.
- `data`: A vector of observed counts.
- `covariates` (optional): A matrix of covariates.
- `link`: (optional) The link function to apply (default is `"log"`).
- `max`: (optional) A parameter controlling iteration limits in likelihood computation.

# Returns
The negative log-likelihood value for the GP2 model.
"""
function minimize_pars_GP2(theta, data, covariates=nothing, link="log", max=nothing)
    link_function = get_link_function(link)
    lambdas = repeat([theta[5]], length(data))
    if !isnothing(covariates)
        lambdas .= link_function.(covariates * theta[5:5+(size(covariates, 2)-1)])
    end

    return compute_negative_log_likelihood_GP2(lambdas, theta[1], theta[2], theta[3], theta[4], data, max)
end

"""
    minimize_pars_Poisson2(theta, data, covariates=nothing, link="log", max=nothing)

Computes the negative log-likelihood for a second-order Poisson model using the provided parameters.
The lambda vector is computed from `theta[4]` (and adjusted by covariates if provided) while the dispersion is set to 0.

# Arguments
- `theta`: A vector of parameters where `theta[1:3]` are the autoregressive parameters and `theta[4]` is used for the base lambda.
- `data`: A vector of observed counts.
- `covariates` (optional): A matrix of covariates.
- `link`: (optional) The link function to use (default is `"log"`).
- `max`: (optional) Parameter to control iteration limits in the likelihood computation.

# Returns
The negative log-likelihood value for the second-order Poisson model.
"""
function minimize_pars_Poisson2(theta, data, covariates=nothing, link="log", max=nothing)
    link_function = get_link_function(link)
    lambdas = repeat([theta[4]], length(data))
    if !isnothing(covariates)
        lambdas .= link_function.(covariates * theta[4:4+(size(covariates, 2)-1)])
    end

    return compute_negative_log_likelihood_GP2(lambdas, theta[1], theta[2], theta[3], 0, data, max)
end

"""
    minimize_pars_GP1(theta, data, covariates=nothing, link="log", max=nothing)

Computes the negative log-likelihood for a first-order generalized Poisson (GP1) model.
The lambda vector is computed from `theta[3]` and adjusted by covariates if available. 
It then evaluates the likelihood using `compute_negative_log_likelihood_GP1` with `theta[1]` as the autoregressive parameter 
and `theta[2]` as the dispersion parameter.

# Arguments
- `theta`: A vector of parameters where `theta[1]` is the autoregressive parameter, `theta[2]` is the dispersion parameter (eta),
  and `theta[3]` onward are used for lambda computation.
- `data`: A vector of observed counts.
- `covariates` (optional): A matrix of covariates.
- `link`: (optional) The link function (default is `"log"`).
- `max`: (optional) Not used in this model (included for consistency).

# Returns
The negative log-likelihood for the first-order GP model.
"""
function minimize_pars_GP1(theta, data, covariates=nothing, link="log", max=nothing)
    link_function = get_link_function(link)
    lambdas = repeat([theta[3]], length(data))
    if !isnothing(covariates)
        lambdas .= link_function.(covariates * theta[3:3+(size(covariates, 2)-1)])
    end

    return compute_negative_log_likelihood_GP1(lambdas, theta[1], theta[2], data)
end

"""
    minimize_pars_Poisson1(theta, data, covariates=nothing, link="log", max=nothing)

Computes the negative log-likelihood for a first-order Poisson model.
The lambda vector is computed from `theta[2]` (and adjusted by covariates if provided) while the dispersion is set to 0.

# Arguments
- `theta`: A vector of parameters where `theta[1]` is the autoregressive parameter and `theta[2]` onward are used for lambda computation.
- `data`: A vector of observed counts.
- `covariates` (optional): A matrix of covariates.
- `link`: (optional) The link function to use (default is `"log"`).
- `max`: (optional) Not used in this model (included for consistency).

# Returns
The negative log-likelihood for the first-order Poisson model.
"""
function minimize_pars_Poisson1(theta, data, covariates=nothing, link="log", max=nothing)
    link_function = get_link_function(link)
    lambdas = repeat([theta[2]], length(data))
    if !isnothing(covariates)
        lambdas .= link_function.(covariates * theta[2:2+(size(covariates, 2)-1)])
    end

    return compute_negative_log_likelihood_GP1(lambdas, theta[1], 0, data)
end

"""
    cocoReg(type, order, data, covariates=nothing, starting_values=nothing, link_function="log", lower_bound_covariates=-Inf, max_loop=nothing, optimizer=Fminbox(LBFGS()))

Fits a count regression model using either a generalized Poisson or Poisson likelihood based on the specified type and order.
The function sets up the optimization problem, determines starting values and parameter bounds, 
and minimizes the negative log-likelihood using an automatic differentiation optimizer.

# Arguments
- `type`: A string indicating the model type ("GP" for generalized Poisson, "Poisson" for Poisson).
- `order`: The autoregressive order (1 or 2).
- `data`: A vector of observed counts.
- `covariates` (optional): A matrix of covariate values.
- `starting_values` (optional): Initial values for the optimizer.
- `link_function` (optional): A string specifying the link function (default is `"log"`).
- `lower_bound_covariates` (optional): Lower bound for covariate parameters (default is `-Inf`).
- `max_loop` (optional): Parameter controlling the maximum iteration or truncation in likelihood computation.
- `optimizer` (optional): The optimization algorithm to use (default is `Fminbox(LBFGS())`).

# Returns
A dictionary containing:
- `"parameter"`: The fitted parameter vector.
- `"covariance_matrix"`: The inverse Hessian matrix as an estimate of the parameter covariance.
- `"log_likelihood"`: The log-likelihood value at the optimum.
- Model details including `type`, `order`, `data`, `covariates`, `link`, starting values, optimizer settings, and parameter bounds.
- `"se"`: The standard errors of the estimated parameters.
"""
function cocoReg(type, order, data, covariates=nothing, starting_values=nothing,
                  link_function="log", lower_bound_covariates=-Inf, max_loop=nothing,
                  optimizer=Fminbox(LBFGS()))
    
    #-------------------------start dependent on type----------------------------------------------------------
    if order == 2
        if type == "GP"
            starting_values = get_starting_values!(type, order, Int.(data), covariates, starting_values, 4, lower_bound_covariates)
            fn = OnceDifferentiable(theta -> minimize_pars_reparameterization_GP2(theta, Int.(data),
                                                                                covariates,
                                                                                link_function,
                                                                                max_loop),
                                    starting_values,
                                    autodiff=:forward)
            f_alphas = theta -> minimize_pars_GP2(theta, Int.(data), covariates,
                                                  link_function, max_loop)
        elseif type == "Poisson"
            starting_values = get_starting_values!(type, order, Int.(data), covariates, starting_values, 3, lower_bound_covariates)
            fn = OnceDifferentiable(theta -> minimize_pars_reparameterization_Poisson2(theta, Int.(data),
                                                                                covariates,
                                                                                link_function,
                                                                                max_loop),
                                    starting_values,
                                    autodiff=:forward)
            f_alphas = theta -> minimize_pars_Poisson2(theta, Int.(data), covariates,
                                                      link_function, max_loop)
        end
    end
    #-----------------------------------------Order 1 models-------------------------
    if order == 1
        if type == "GP"
            starting_values = get_starting_values!(type, order, Int.(data), covariates, starting_values, 2, lower_bound_covariates)
            fn = OnceDifferentiable(theta -> minimize_pars_GP1(theta, Int.(data),
                                                                covariates,
                                                                link_function),
                                    starting_values,
                                    autodiff=:forward)
            f_alphas = theta -> minimize_pars_GP1(theta, Int.(data), covariates,
                                                  link_function)
        elseif type == "Poisson"
            starting_values = get_starting_values!(type, order, Int.(data), covariates, starting_values, 1, lower_bound_covariates)
            fn = OnceDifferentiable(theta -> minimize_pars_Poisson1(theta, Int.(data),
                                                                     covariates,
                                                                     link_function),
                                    starting_values,
                                    autodiff=:forward)
            f_alphas = theta -> minimize_pars_Poisson1(theta, Int.(data), covariates,
                                                      link_function)
        end
    end
    #--------------------------end dependent on type

    # Write down constraints
    lower, upper = get_bounds(order, type, covariates, lower_bound_covariates)
    
    # Obtain fit
    fit = optimize(fn, lower, upper, starting_values, optimizer)
    parameter = Optim.minimizer(fit)
    
    # Get alphas from reparameterized results
    if order == 2
        alpha1, alpha2, alpha3 = reparameterize_alpha(parameter)
        parameter[1:3] = [alpha1, alpha2, alpha3]
    end
    

    inv_hessian = compute_inverse_matrix(compute_hessian(f_alphas, parameter))
    if any(diag(inv_hessian) .< 0)
        for i in 1:size(inv_hessian, 1)
            if inv_hessian[i, i] < 0
                inv_hessian[i, i] = 10^-12
            end
        end
    end  
    
    # Construct output
    out = Dict("parameter" => parameter,
               "covariance_matrix" => inv_hessian,
               "log_likelihood" => -f_alphas(parameter),
               "type" => type,
               "order" => order,
               "data" => data,
               "covariates" => covariates,
               "link" => link_function,
               "starting_values" => starting_values,
               "optimizer" => optimizer,
               "lower_bounds" => lower,
               "upper_bounds" => upper,
               "optimization" => fit,
               "max_loop" => max_loop)
    
    # Compute standard errors
    out["se"] = diag(out["covariance_matrix"]).^0.5
    return out
end
