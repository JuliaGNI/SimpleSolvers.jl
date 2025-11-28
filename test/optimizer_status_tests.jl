
using SimpleSolvers
using Test

include("optimizers_problems.jl")


n = 1
x = ones(n)
nl = QuasiNewtonOptimizer(x, F)

@test config(nl) == nl.config

solve!(x, nl)