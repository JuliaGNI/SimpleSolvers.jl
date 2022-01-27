
using SimpleSolvers
using Test

include("optimizers_problems.jl")


n = 1
x = ones(n)
nl = QuasiNewtonOptimizer(x, F)

@test config(nl) == nl.config
@test status(nl) == nl.status

solve!(x, nl)
println(status(nl))
