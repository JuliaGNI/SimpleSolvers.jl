
using LinearAlgebra
using SimpleSolvers
using SimpleSolvers: initialize!, objective, solver_step!
using Test

include("optimizers_problems.jl")


# struct OptimizerTest{T} <: Optimizer end

# test_optim = OptimizerTest{Float64}()

# @test_throws ErrorException config(test_optim)
# @test_throws ErrorException status(test_optim)
# @test_throws ErrorException objective(test_optim)
# @test_throws ErrorException initialize!(test_optim)
# @test_throws ErrorException solver_step!(test_optim)

# function test_optimizer(opt) end

for method in (Newton(), BFGS(), DFP())
    for linesearch in (Static(0.8), Backtracking(), Quadratic(), Bisection())
        n = 1
        x = ones(n)
        opt = Optimizer(x, F; algorithm = method, linesearch = linesearch)

        @test config(opt) == opt.config
        @test status(opt) == opt.result.status

        solve!(x, opt)
        # println(opt)
        @test norm(minimizer(opt)) ≈ 0 atol=1E-7
        @test norm(minimum(opt)) ≈ F(0) atol=1E-7

        x = ones(n)
        opt = Optimizer(x, F; ∇F! = ∇F!, algorithm = method, linesearch = linesearch)
        solve!(x, opt)
        # println(opt)
        @test norm(minimizer(opt)) ≈ 0 atol=1E-7
        @test norm(minimum(opt)) ≈ F(0) atol=1E-7
    end
end
