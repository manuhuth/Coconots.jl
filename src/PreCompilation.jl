using Random, PrecompileTools
using Base.Iterators: product


@compile_workload begin
    types = ["Poisson", "GP"]
    orders = [1, 2]
    links = ["identity", "log", "relu"]
    covariates = [false, true]
    lambdas = [0.7]
    n = 40

    #------------------------------------------No Covariates-----------------------------------------------------------------------
    parameter_1 = [0.3]
    parameter_2 = [0.3, 0.05, 0.2]

    k = n

    cov_x = reshape(sin.(1:k) .+ 4.2, :, 1)
    cov_x_p1 = reshape(sin.((k+1):(k+3)) .+ 4.2, :, 1)

    pars = []

    Random.seed!(3)

    for (type, order, link, covs, lambda) in product(types, orders, links, covariates, lambdas)

        if order == 1
            pars = copy(parameter_1)
        elseif order == 2
            pars = copy(parameter_2)
        end

        append!(pars, lambda)

        if type == "GP"
            insert!(pars, length(pars), 0.2)
        end

        if covs
            pars[end] = lambda
            cov_sim = copy(cov_x)
            cov_reg = copy(cov_x)
            cov_pred = copy(cov_x[1:1, :])
        else
            cov_sim = nothing
            cov_reg = nothing
            cov_pred = nothing
        end

        if link == "log"
            cov_sim = log.(copy(cov_x))
            cov_reg = log.(copy(cov_x))
            cov_pred = log.(copy(cov_x[1:1, :]))
        end

        data = cocoSim(type, order, pars, n, cov_sim,
                            link, 50)

        fit = cocoReg(type, order, data, cov_reg)

        cocoBoot(fit, [1:1:21;], 10)

        cocoPit(fit)

        compute_scores(fit)

        cocoPredictOneStep(fit, 0:Int(ceil(maximum(data) * 1.5)), cov_pred)

        cocoPredictKsteps(fit, 1, 50, cov_pred)
    end
end