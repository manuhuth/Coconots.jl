using Coconots
using Test
using Random
using ForwardDiff
using Statistics: mean

@testset "convolution distributions (order 1)" begin
    y = 10
    λ = 2.0
    α = 0.5
    η = 1.0
    @test Coconots.compute_distribution_convolution_x_r_y(-1, y, λ, α, η) == 0

    result_cum = Coconots.compute_distribution_convolution_x_r_y(0, y, λ, α, η)
    result_single = Coconots.compute_convolution_x_r_y(0, y, λ, α, η)
    @test isapprox(result_cum, result_single; atol = 1e-8)

    x = 3
    expected = sum([Coconots.compute_convolution_x_r_y(i, y, λ, α, η) for i in 0:x])
    @test isapprox(Coconots.compute_distribution_convolution_x_r_y(x, y, λ, α, η),
        expected; atol = 1e-8)

    # transition pmf sums to one over its support
    @test isapprox(Coconots.compute_distribution_convolution_x_r_y(200, 4, 1.5, 0.4,
            0.2),
        1.0; atol = 1e-8)

    # pmf vector matches pointwise evaluation
    pmf = Coconots.convolution_pmf_vector(6, 4, 1.5, 0.4, 0.2)
    @test pmf ≈
          [Coconots.compute_convolution_x_r_y(i, 4, 1.5, 0.4, 0.2) for i in 0:6]
end

@testset "convolution distributions (order 2)" begin
    y = 10
    z = 5
    λ = 2.0
    α1 = 0.3
    α2 = 0.2
    α3 = 0.1
    η = 1.0
    @test Coconots.compute_distribution_convolution_x_r_y_z(-1, y, z, α1, α2, α3, λ,
        η) == 0

    result_cum = Coconots.compute_distribution_convolution_x_r_y_z(0, y, z, α1, α2, α3,
        λ, η)
    result_single = Coconots.compute_convolution_x_r_y_z(0, y, z, λ, α1, α2, α3, η)
    @test isapprox(result_cum, result_single; atol = 1e-8)

    x = 3
    expected = sum([Coconots.compute_convolution_x_r_y_z(i, y, z, λ, α1, α2, α3, η)
                    for i in 0:x])
    @test isapprox(
        Coconots.compute_distribution_convolution_x_r_y_z(x, y, z, α1, α2,
            α3, λ, η),
        expected; atol = 1e-8)

    # kernel-based pmf vector matches pointwise evaluation, and g(r | y, z) is
    # unchanged by the one-shot wrapper
    kernel = Coconots.GP2Kernel(λ, 0.3, 0.2, 0.1, 0.2, y + z)
    pmf = Coconots.convolution_pmf_vector(kernel, 8, y, z, nothing)
    @test pmf ≈ [Coconots.compute_convolution_x_r_y_z(i, y, z, λ, 0.3, 0.2, 0.1, 0.2)
           for i in 0:8]
    # an undersized kernel is rejected instead of reading past its tables
    @test_throws ArgumentError Coconots.convolution_pmf_vector(kernel, 300, y, z,
        nothing)
    wide_kernel = Coconots.GP2Kernel(λ, 0.3, 0.2, 0.1, 0.2, 300)
    @test isapprox(sum(Coconots.convolution_pmf_vector(wide_kernel, 300, y, z,
            nothing)),
        1.0; atol = 1e-8)
end

@testset "generalized Poisson pmf" begin
    # log-space branch agrees with the table branch scaling behavior and both
    # accept Dual numbers
    @test Coconots.generalized_poisson_distribution(3, 2.0, 0.2) ≈
          exp(Coconots.log_generalized_poisson_pdf(3, 2.0, 0.2))
    d = ForwardDiff.Dual(2.0, 1.0)
    @test Coconots.generalized_poisson_distribution(3, d, 0.2) isa ForwardDiff.Dual
    # pmf of large counts stays finite (previously required BigInt factorials)
    @test 0 <=
          Coconots.generalized_poisson_distribution(250, 2.0, 0.2) <= 1
end

@testset "negative log-likelihood" begin
    Random.seed!(11)
    data = cocoSim("GP", 2, [0.25, 0.1, 0.1, 0.15, 1.5], 120)

    # constant-rate fast path (unique transitions) equals the per-observation path
    lambdas = fill(1.5, length(data))
    nll_scalar = Coconots.compute_negative_log_likelihood_GP2(1.5, 0.25, 0.1, 0.1,
        0.15, data)
    nll_direct = sum(-log(Coconots.compute_convolution_x_r_y_z(data[t], data[t - 1],
                         data[t - 2], 1.5, 0.25, 0.1, 0.1, 0.15)) for t in 3:length(data))
    @test isapprox(nll_scalar, nll_direct; rtol = 1e-10)
    @test isapprox(
        Coconots.compute_negative_log_likelihood_GP2(lambdas, 0.25, 0.1,
            0.1, 0.15, data),
        nll_scalar; rtol = 1e-12)

    nll1_scalar = Coconots.compute_negative_log_likelihood_GP1(1.5, 0.4, 0.15, data)
    nll1_direct = sum(-log(Coconots.compute_convolution_x_r_y(data[t], data[t - 1],
                          1.5, 0.4, 0.15)) for t in 2:length(data))
    @test isapprox(nll1_scalar, nll1_direct; rtol = 1e-10)

    # differentiable end to end
    grad = ForwardDiff.gradient(
        t -> Coconots.compute_negative_log_likelihood_GP2(t[5], t[1], t[2], t[3], t[4],
            data), [0.25, 0.1, 0.1, 0.15, 1.5])
    @test all(isfinite, grad)
end

@testset "cocoReg" begin
    Random.seed!(7)
    data = cocoSim("GP", 1, [0.4, 0.2, 2.0], 300)
    fit = cocoReg("GP", 1, data)
    @test length(fit["parameter"]) == 3
    @test all(isfinite, fit["se"])
    @test isapprox(fit["parameter"][1], 0.4; atol = 0.15)
    @test isapprox(fit["parameter"][3], 2.0; atol = 0.5)
    @test fit["log_likelihood"] ≈
          -Coconots.compute_negative_log_likelihood_GP1(fit["parameter"][3],
        fit["parameter"][1], fit["parameter"][2], Int.(data))

    # assessment tools run on the fit and return the contract keys
    scores = compute_scores(fit, 30)
    @test all(haskey.(Ref(scores),
        ["logarithmic_score", "quadratic_score", "ranked_probability_score"]))
    pit = cocoPit(fit, 10)
    @test length(pit["Pit_values"]) == 10
    @test isapprox(sum(pit["Pit_values"]), 1.0; atol = 1e-6)
    boot = cocoBoot(fit, [1:1:5;], 20)
    @test size(boot["pacfs"]) == (20, 5)
    pred = cocoPredictOneStep(fit, 0:20)
    @test isapprox(sum(pred["probabilities"]), 1.0; atol = 1e-3)
    predk = cocoPredictKsteps(fit, 2, 50)
    @test predk["length"] == 2

    # default transformation-based fit respects the constraints and agrees
    # with the box-constrained solver
    @test all(0 .<= fit["parameter"][1:2] .<= 1)
    @test fit["parameter"][3] >= 0
    fit_box = cocoReg("GP", 1, data; box_constrained = true)
    @test isapprox(fit_box["parameter"], fit["parameter"]; atol = 1e-3)
    @test isapprox(fit_box["log_likelihood"], fit["log_likelihood"]; atol = 1e-4)

    @test_throws ArgumentError cocoReg("NegBin", 1, data)
    @test_throws ArgumentError cocoReg("GP", 3, data)
end

@testset "bound transformations" begin
    for (lo, hi) in [(0.0, 1.0), (0.0, Inf), (-Inf, Inf), (-2.0, Inf), (-Inf, 3.0)]
        for x in [-1.5, 0.3, 2.5]
            (lo < x < hi) || continue
            u = Coconots._to_unconstrained(x, lo, hi)
            @test isapprox(Coconots._to_bounded(u, lo, hi), x; rtol = 1e-6)
        end
        for u in [-4.0, 0.0, 4.0]
            x = Coconots._to_bounded(u, lo, hi)
            @test lo <= x <= hi
        end
    end
end

@testset "cocoReg with covariates" begin
    Random.seed!(21)
    n = 250
    covariates = hcat(ones(n), sin.((1:n) ./ 12))
    data = cocoSim("Poisson", 1, [0.3, 0.7, 0.4], n, covariates, "log", 0)
    fit = cocoReg("Poisson", 1, data, covariates)
    @test length(fit["parameter"]) == 3
    @test isapprox(fit["parameter"][2], 0.7; atol = 0.4)
    @test all(isfinite, fit["se"])
end

@testset "cocoForwardSim conditions on the last observations" begin
    # With alpha near 1 and tiny lambda, forecasts must inherit the level of
    # x_prev. (A previous version seeded x_prev at the end of the state vector
    # where the simulation loop immediately overwrote it for k >= 2.)
    Random.seed!(5)
    forecasts = [Coconots.cocoForwardSim(2, 30, "Poisson", 1, [0.95, 0.01])
                 for _ in 1:100]
    @test mean(first.(forecasts)) > 20
    @test mean(last.(forecasts)) > 15
end

@testset "get_lambda" begin
    fit = Dict("link" => "identity", "covariates" => nothing,
        "parameter" => [1.0, 2.0, 3.0])
    @test Coconots.get_lambda(fit) == 3.0

    fit = Dict("link" => "identity", "covariates" => [1.0 2.0; 3.0 4.0],
        "parameter" => [0.5, 1.5])
    @test all(Coconots.get_lambda(fit) .== fit["covariates"] * fit["parameter"])
    @test isapprox(Coconots.get_lambda(fit, true),
        sum(fit["covariates"][end, :] .* fit["parameter"]); atol = 1e-8)
end

@testset "link functions" begin
    f = Coconots.get_link_function("softplus")
    @test isapprox(f(0.0), log(2.0); atol = 1e-10)
    @test f(-10.0) > 0
    @test isapprox(f(100.0), 100.0; atol = 1e-6)
    @test Coconots.get_link_function("log") === exp
    @test Coconots.get_link_function("relu")(-1.0) == 1e-10
    @test Coconots.get_link_function("identity")(2.5) == 2.5
    @test_throws ErrorException Coconots.get_link_function("unknown")
end
