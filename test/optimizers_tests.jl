
using LinearAlgebra
using SimpleSolvers
using SimpleSolvers: initialize!, objective, solver_step!
using Test

include("optimizers_problems.jl")


struct OptimizerTest{T} <: Optimizer{T} end

test_optim = OptimizerTest{Float64}()

@test_throws ErrorException config(test_optim)
@test_throws ErrorException status(test_optim)
@test_throws ErrorException objective(test_optim)
@test_throws ErrorException initialize!(test_optim)
@test_throws ErrorException solver_step!(test_optim)


for optim in (QuasiNewtonOptimizer,BFGSOptimizer,DFPOptimizer)
    n = 1
    x = ones(n)
    y = zero(eltype(x))
    nl = optim(x, F)

    @test config(nl) == nl.config
    @test status(nl) == nl.status

    solve!(x, nl)
    # println(status(nl))
    @test norm(nl.status.x) ≈ 0 atol=1E-7

    x = ones(n)
    nl = optim(x, F; ∇F! = ∇F!)
    solve!(x, nl)
    # println(status(nl))
    @test norm(nl.status.x) ≈ 0 atol=1E-7
end
