module Coconots

using LinearAlgebra: diag, inv
using Random
using Statistics: cor, mean, quantile!, var
using StatsBase: autocor
using DataFrames: DataFrame
using FreqTables
using SpecialFunctions: logfactorial
using ForwardDiff
using Optimization
using OptimizationOptimJL
using Optim: Fminbox, LBFGS, NelderMead
using Base.Threads: @threads, nthreads

include("Utils.jl")
include("LinkFunctions.jl")
include("CountDistributions.jl")
include("Likelihood.jl")
include("Bounds.jl")
include("StartingValues.jl")
include("CocoReg.jl")
include("CocoSim.jl")
include("CocoPredict.jl")
include("CocoBoot.jl")
include("Scores.jl")
include("PreCompilation.jl")

end
