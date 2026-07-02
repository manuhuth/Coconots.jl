using PrecompileTools: @compile_workload

@compile_workload begin
    Random.seed!(1)
    let
        data1 = cocoSim("Poisson", 1, [0.3, 1.0], 60, nothing, "log", 20)
        fit1 = cocoReg("Poisson", 1, data1)
        compute_scores(fit1, 10)
        cocoPit(fit1, 5)
        cocoBoot(fit1, [1:1:5;], 4)
        cocoPredictOneStep(fit1, 0:5)
        cocoPredictKsteps(fit1, 2, 10)

        data2 = cocoSim("GP", 2, [0.25, 0.05, 0.1, 0.1, 1.0], 60, nothing, "log", 20)
        fit2 = cocoReg("GP", 2, data2)
        compute_scores(fit2, 10)
        cocoPit(fit2, 5)
        cocoPredictOneStep(fit2, 0:5)

        covariates = hcat(ones(60), sin.((1:60) ./ 6))
        data3 = cocoSim("GP", 1, [0.3, 0.1, 0.5, 0.2], 60, covariates, "log", 0)
        cocoReg("GP", 1, data3, covariates)
    end
end
