using Coconots
using Test
using Random
using ForwardDiff
using Statistics: mean
using Aqua
using JET

@testset "Aqua" begin
    Aqua.test_all(Coconots; ambiguities = false)
    Aqua.test_ambiguities(Coconots)
end

@testset "JET" begin
    JET.test_package(Coconots; target_defined_modules = true)
end

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

@testset "restart from bounds at which the objective decreases inward (KKT)" begin
    # objective decreasing into the interior at the lower bound of x[1] only
    f(x) = (x[1] - 0.3)^2 + (x[2] + 1.0)^2
    lo, hi = [0.0, 0.0], [1.0, Inf]
    @test Coconots._bound_kkt_violations(f, [1e-12, 1e-9], lo, hi) == [1]
    @test isempty(Coconots._bound_kkt_violations(f, [0.3, 1e-9], lo, hi))
    # at an upper bound: a violation iff the objective decreases downward
    @test Coconots._bound_kkt_violations(x -> x[1], [1 - 1e-9, 2.0], lo, hi) == [1]
    @test isempty(Coconots._bound_kkt_violations(x -> -x[1], [1 - 1e-9, 2.0], lo, hi))

    # count series (benchmark problem PAR1_softplus_n200_s5) on which the GP2
    # fit from the default start (the coconots start: alpha1 = alpha2 = 1e-5,
    # alpha3 = 0.515) stalled with alpha2 at its bound 0 (loglik
    # -468.538005) although the objective decreases in alpha2 there; the
    # maximum is interior in alpha2 (loglik -468.524202)
    data = [
        6, 2, 7, 4, 4, 5, 4, 8, 7, 10, 11, 7, 6, 3, 8, 6, 6, 9, 6, 11, 12, 9, 6, 11,
        13, 7, 9, 6, 5, 9, 7, 5, 8, 7, 5, 9, 7, 5, 8, 9, 10, 15, 10, 11, 7, 11, 6, 5,
        10, 4, 6, 5, 5, 4, 10, 6, 8, 6, 12, 11, 12, 8, 9, 8, 10, 6, 8, 8, 8, 5, 6, 8,
        12, 10, 12, 14, 6, 7, 4, 3, 6, 10, 9, 8, 12, 11, 6, 7, 6, 3, 6, 7, 7, 4, 8, 10,
        10, 5, 6, 9, 9, 6, 4, 6, 9, 7, 8, 10, 9, 3, 7, 17, 11, 12, 7, 7, 10, 6, 7, 8,
        14, 16, 16, 10, 14, 6, 6, 8, 7, 6, 10, 3, 9, 11, 9, 7, 7, 7, 7, 5, 7, 8, 3, 6,
        7, 5, 6, 10, 6, 9, 6, 4, 6, 8, 9, 6, 8, 2, 6, 8, 8, 10, 13, 8, 8, 3, 10, 10, 9,
        14, 12, 13, 9, 10, 10, 14, 8, 9, 13, 13, 8, 6, 7, 8, 7, 9, 9, 5, 6, 8, 7, 9, 5,
        10, 5, 6, 2, 4, 7, 9]
    fit = cocoReg("GP", 2, data)
    @test fit["log_likelihood"] >= -468.524202241155 - 1e-6
    @test isapprox(fit["parameter"][2], 0.0399690904; atol = 1e-4)
    f_alpha = Coconots._make_objective("GP", 2, data, nothing, "log", nothing, false)
    lo2, hi2 = Coconots.get_bounds(2, "GP", nothing, -Inf)
    @test isempty(Coconots._bound_kkt_violations(f_alpha, fit["parameter"], lo2, hi2))
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

@testset "input validation (counts, short series, covariate rank)" begin
    # counts above 170 are rejected on both backends
    big_data = vcat(fill(3, 20), 171, fill(4, 20))
    @test_throws ArgumentError cocoReg("Poisson", 1, big_data)
    err = try
        cocoReg("GP", 2, big_data)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("counts above 170 are not supported", err.msg)
    @test occursin("observed count = 171", err.msg)

    # too short series
    @test_throws ArgumentError cocoReg("Poisson", 1, [1, 2])
    @test_throws ArgumentError cocoReg("GP", 2, [1, 2, 3])

    # rank-deficient covariates
    n = 60
    x1 = sin.((1:n) ./ 6)
    err = try
        cocoReg("Poisson", 1, fill(2, n), hcat(ones(n), x1, 2 .* x1))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("xreg is rank deficient", err.msg)
    @test_throws ArgumentError Coconots._check_covariates_full_rank(hcat(x1, zeros(n)))
    @test Coconots._check_covariates_full_rank(hcat(ones(n), x1)) === nothing

    # assessment tools on a fitted model
    Random.seed!(31)
    data = cocoSim("Poisson", 1, [0.3, 1.0], 80)
    fit = cocoReg("Poisson", 1, data)
    @test_throws ArgumentError compute_scores(fit, 171)
    @test_throws ArgumentError cocoPredictOneStep(fit, 0:171)
    @test_throws ArgumentError cocoBoot(fit, [1:1:80;], 5)
    bad = copy(fit)
    bad["data"] = vcat(data[1:(end - 1)], 200)
    @test_throws ArgumentError compute_scores(bad, 50)
    @test_throws ArgumentError cocoPit(bad, 10)
    @test_throws ArgumentError cocoBoot(bad, [1:1:5;], 5)
    @test_throws ArgumentError cocoPredictOneStep(bad, 0:10)
    short = copy(fit)
    short["data"] = data[1:1]
    @test_throws ArgumentError compute_scores(short, 50)
    @test_throws ArgumentError cocoPit(short, 10)
end

@testset "default starting values equal the coconots (R) default start" begin
    # Expected values are the formulas of coconots' cocoReg_base.R (cocoReg
    # defaults: seasonality c(1, 2), replace.start.val 1e-5,
    # iteration.start.val 0.6; GP2 shrinks by 0.99), evaluated by hand.
    start(type, order, x) = Coconots.get_starting_values!(type, order, x, nothing,
        nothing, Coconots._n_parameters_without_covariates(type, order), -Inf)

    # x = (0,0,0,0,0,0,3,3,3): mean 1, z = x - 1 = (-1 x 6, 2 x 3),
    # sum z^2 = 18, var = 18 / 8 = 9/4, so eta = 1 - (4/9)^0.5 = 1/3;
    # acf(1) = 11/18, acf(2) = 4/18 = 2/9 (lag sums 11 and 4 over 18);
    # mu111 = (-1 -1 -1 -1 +2 -4 +8) / 9 = 2/9.
    x = [0, 0, 0, 0, 0, 0, 3, 3, 3]
    # Poisson 1: alpha = 11/18, lambda = 1 * (1 - 11/18) = 7/18
    @test isapprox(start("Poisson", 1, x), [11 / 18, 7 / 18]; rtol = 1e-12)
    # GP 1: lambda = 1 * (2/3) * (7/18) = 7/27
    @test isapprox(start("GP", 1, x), [11 / 18, 1 / 3, 7 / 27]; rtol = 1e-12)
    # GP 2: alpha3 = (2/9) (2/3)^4 / (5/3) = 32/1215 ~ 0.026337,
    # alpha2 = 2/9 - 32/1215 = 238/1215 ~ 0.195885,
    # alpha1 = 11/18 - 32/1215 = 1421/2430 ~ 0.584774;
    # sum ~ 0.807 <= 1, but 2 alpha1 + alpha3 = 2906/2430 ~ 1.1959 > 1, so
    # alpha1 and alpha3 are multiplied by 0.99 eighteen times
    # (0.99^17 * 1.1959 ~ 1.008 > 1, 0.99^18 * 1.1959 ~ 0.998 <= 1).
    # Expected ~ (0.997985, 0.022023, 0.399751, 0.333333, 0.196089).
    s = 0.99^18
    a1 = 1421 / 2430 * s
    a3 = 32 / 1215 * s
    a2 = 238 / 1215
    @test isapprox(start("GP", 2, x),
        [2 * a1 + a3, a3 / (2 * a1 + a3), a2 / (1 - a1 - a3), 1 / 3,
            (2 / 3) * (1 - a1 - a2 - a3)]; rtol = 1e-12)

    # x = (0,0,0,0,0,2,2,2,2,2): mean 1, z = (-1 x 5, 1 x 5), acf(1) = 7/10,
    # acf(2) = 4/10, mu111 = (-3 + 1 - 1 + 3) / 10 = 0.
    # Poisson 2: alpha3 = 0, alpha2 = 0.4, alpha1 = 0.7; the sum 1.1 > 1, so all
    # three are multiplied by 0.6 once: (0.42, 0.24, 0); lambda = 1 - 0.66.
    x = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
    @test isapprox(start("Poisson", 2, x), [0.84, 0.0, 0.24 / 0.58, 0.34];
        rtol = 1e-12, atol = 1e-14)

    # negative acf(1) is replaced by 1e-5 (not its absolute value):
    # x = (0,2,0,2,0,2), acf(1) = -5/6
    x = [0, 2, 0, 2, 0, 2]
    @test isapprox(start("Poisson", 1, x), [1e-5, 1 - 1e-5]; rtol = 1e-12)
    # underdispersion: x = (0,1,2,1,0,1,2,1,0,1,2,1) has mean 1, var = 6/11,
    # mean / var = 11/6 > 1, so 1 - (11/6)^0.5 < 0 is replaced by eta = 1e-5
    @test start("GP", 1, [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1])[2] == 1e-5

    # user-supplied starts are returned unchanged
    @test Coconots.get_starting_values!("GP", 1, x, nothing, [0.1, 0.2, 0.3], 2,
        -Inf) == [0.1, 0.2, 0.3]
end

@testset "cocoPit uses t = order + 1, ..., n" begin
    Random.seed!(3)
    data = cocoSim("GP", 1, [0.4, 0.2, 2.0], 60)
    fit = cocoReg("GP", 1, data)
    alpha, eta, lambda = fit["parameter"]
    x = Int.(data)
    n = length(x)
    Px = [Coconots.compute_distribution_convolution_x_r_y(x[t], x[t - 1], lambda,
              alpha, eta) for t in 2:n]
    Pxm1 = [Coconots.compute_distribution_convolution_x_r_y(x[t] - 1, x[t - 1],
                lambda, alpha, eta) for t in 2:n]
    F = [mean(clamp.((ui .- Pxm1) ./ (Px .- Pxm1), 0, 1))
         for ui in range(0, 1, length = 11)]
    pit = cocoPit(fit, 10)
    @test isapprox(pit["Pit_values"], diff(F); atol = 1e-8)
    @test isapprox(sum(pit["Pit_values"]), 1.0; atol = 1e-10)
end

include("simulation.jl")
