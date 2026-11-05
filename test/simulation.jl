# Simulation: covariate alignment, start values, termination of the inversion
# samplers and rate checks.

const _SPIKE_BETA = [log(0.1), log(2000.0)]  # lambda = 0.1, and 200 at the spike

_spike_covariates(n, t0) = hcat(ones(n), [t == t0 ? 1.0 : 0.0 for t in 1:n])

# plain inversion of the Generalized Poisson cdf (no termination rule)
function _plain_gp_inversion(u, lambda, eta)
    c = Coconots.generalized_poisson_distribution(0, lambda, eta)
    i = 0
    while c < u
        i += 1
        c += Coconots.generalized_poisson_distribution(i, lambda, eta)
    end
    return i
end

@testset "simulation: covariate row t drives observation t" begin
    n = 20
    for (type, order, dyn) in (("Poisson", 1, [0.01]), ("GP", 1, [0.01, 0.1]),
            ("Poisson", 2, [0.01, 0.01, 0.01]), ("GP", 2, [0.01, 0.01, 0.01, 0.1]))
        for t0 in (1, order, 10, n)
            Random.seed!(100 + t0)
            x = cocoSim(type, order, vcat(dyn, _SPIKE_BETA), n,
                _spike_covariates(n, t0), "log", 0)
            @test length(x) == n
            @test argmax(x) == t0
            @test x[t0] > 100
        end
    end
end

@testset "simulation: order-2 GP start values are GP(lambda_t, eta)" begin
    eta = 0.6
    par = [0.2, 0.1, 0.2, eta, log(1.5), 0.5]
    covariates = hcat(ones(5), collect(range(0, 1; length = 5)))
    lambda_t = exp.(covariates * par[5:6])
    Random.seed!(3)
    x = cocoSim("GP", 2, par, 5, covariates, "log", 0)
    Random.seed!(3)
    u1 = rand()
    u2 = rand()
    @test x[1] == Coconots.draw_random_generalized_poisson_variable(u1, lambda_t[1], eta)
    @test x[2] == Coconots.draw_random_generalized_poisson_variable(u2, lambda_t[2], eta)

    lambda, sampler = Coconots._simulation_setup("GP", 2, par, 2, covariates[1:2, :],
        "log", 0)
    X = zeros(Int, 4000, 2)
    Random.seed!(4)
    Coconots._simulate_paths!(X, sampler, lambda, 0)
    @test isapprox(mean(X[:, 1]), lambda_t[1] / (1 - eta); rtol = 0.1)
end

@testset "simulation: the inversion samplers terminate" begin
    for eta in (0.0, 0.4), lam in (2.0, 5.0)
        # u = 1: the accumulated cdf may never exceed u
        x = Coconots.draw_random_generalized_poisson_variable(1.0, lam, eta)
        @test 10 < x < 1000
        s = Coconots._TransitionSampler(1, 0.3, 0.0, 0.0, 0.0, eta)
        Coconots._set_rate!(s, lam)
        @test Coconots._draw_innovation!(s, 1.0) == x
        @test Coconots._draw_innovation!(s, prevfloat(1.0)) ==
              Coconots.draw_random_generalized_poisson_variable(prevfloat(1.0), lam, eta)
    end
    # ordinary draws are those of the plain inversion
    Random.seed!(11)
    s = Coconots._TransitionSampler(1, 0.3, 0.0, 0.0, 0.0, 0.3)
    Coconots._set_rate!(s, 3.0)
    for u in rand(50)
        @test Coconots.draw_random_generalized_poisson_variable(u, 3.0, 0.3) ==
              _plain_gp_inversion(u, 3.0, 0.3)
        @test Coconots._draw_innovation!(s, u) == _plain_gp_inversion(u, 3.0, 0.3)
    end
    # a cdf that stays far below 1 beyond the bound is an error
    @test_throws ErrorException Coconots._gp_inversion_stop(5000, 0.5, 0.5, 100, 1.0,
        0.1)
    @test !Coconots._gp_inversion_stop(50, 0.5, 0.6, 100, 1.0, 0.1)
    # thinning parts: u = 1 gives the support maximum
    @test Coconots.draw_random_g_r_y_variable(1.0, 5, 0.5, 0.2, 1.0) == 5
    @test Coconots.draw_random_g_r_y_z_variable(1.0, 3, 2, 1.0, 0.3, 0.1, 0.2, 0.2,
        nothing) == 5
end

@testset "simulation: non-positive rates are an error" begin
    n = 30
    covariates = hcat(ones(n), vcat(ones(n - 1), -10.0))  # 1 + 0.5 x = -4 at t = n
    @test_throws ErrorException cocoSim("Poisson", 1, [0.2, 1.0, 0.5], n, covariates,
        "identity", 0)
    @test_throws ErrorException cocoSim("GP", 2, [0.2, 0.1, 0.2, 0.2, 1.0, 0.5], n,
        covariates, "identity", 0)
    @test_throws ErrorException cocoSim("Poisson", 1, [0.2, -1.0], n)
end
