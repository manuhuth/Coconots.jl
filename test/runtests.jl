using Coconots
using Test
using Random
using Base.Iterators: product

@testset "compute_distribution_convolution_x_r_y" begin
    # Test that negative x returns 0
    @test Coconots.compute_distribution_convolution_x_r_y(-1, 10, 1.0, 0.5, 1.0) == 0

    # Test that for x = 0 the cumulative equals the likelihood at 0
    y = 10; λ = 2.0; α = 0.5; η = 1.0
    result_cum = Coconots.compute_distribution_convolution_x_r_y(0, y, λ, α, η)
    result_single = Coconots.compute_convolution_x_r_y(0, y, λ, α, η)
    @test isapprox(result_cum, result_single; atol=1e-8)

    # For a positive x, check that the cumulative is the sum of individual likelihoods.
    x = 3
    expected = sum([Coconots.compute_convolution_x_r_y(i, y, λ, α, η) for i in 0:x])
    @test isapprox(Coconots.compute_distribution_convolution_x_r_y(x, y, λ, α, η), expected; atol=1e-8)
end

@testset "compute_distribution_convolution_x_r_y_z" begin
    # Negative x should return 0
    @test Coconots.compute_distribution_convolution_x_r_y_z(-1, 10, 5, 0.3, 0.2, 0.1, 2.0, 1.0) == 0

    # For x = 0, the cumulative should match the single convolution likelihood at 0.
    y = 10; z = 5; λ = 2.0; α1 = 0.3; α2 = 0.2; α3 = 0.1; η = 1.0
    result_cum = Coconots.compute_distribution_convolution_x_r_y_z(0, y, z, α1, α2, α3, λ, η)
    result_single = Coconots.compute_convolution_x_r_y_z(0, y, z, λ, α1, α2, α3, η)
    @test isapprox(result_cum, result_single; atol=1e-8)

    # For x > 0, check that the cumulative equals the sum of individual likelihoods.
    x = 3
    expected = sum([Coconots.compute_convolution_x_r_y_z(i, y, z, λ, α1, α2, α3, η) for i in 0:x])
    @test isapprox(Coconots.compute_distribution_convolution_x_r_y_z(x, y, z, α1, α2, α3, λ, η), expected; atol=1e-8)
end


@testset "get_lambda" begin
    # When there are no covariates, get_lambda should return the last parameter.
    cocoReg_fit = Dict("link" => "identity", "covariates" => nothing, "parameter" => [1.0, 2.0, 3.0])
    @test Coconots.get_lambda(cocoReg_fit) == 3.0

    # When covariates are provided, get_lambda should compute using the link function.
    # Here we simulate a simple linear model where the link function is identity.
    cocoReg_fit = Dict("link" => "identity",
                       "covariates" => [1.0 2.0; 3.0 4.0],
                       "parameter" => [0.5, 1.5])
    expected_all = cocoReg_fit["covariates"] * cocoReg_fit["parameter"]
    result_all = Coconots.get_lambda(cocoReg_fit)
    @test all(result_all .== expected_all)

    # Test the `last_val` option: only the last row is used.
    expected_last = sum(cocoReg_fit["covariates"][end, :] .* cocoReg_fit["parameter"])
    result_last = Coconots.get_lambda(cocoReg_fit, true)
    @test isapprox(result_last, expected_last; atol=1e-8)
end
