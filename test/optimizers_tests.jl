
using LinearAlgebra
using SimpleSolvers
using SimpleSolvers: gradient, hessian, linesearch, objective, initialize!, update!, solver_step!
using Test

include("optimizers_problems.jl")


struct OptimizerTest{T} <: OptimizationAlgorithm end

test_optim = OptimizerTest{Float64}()
test_x = zeros(3)

@test_throws ErrorException gradient(test_optim)
@test_throws ErrorException hessian(test_optim)
@test_throws ErrorException linesearch(test_optim)
@test_throws ErrorException objective(test_optim)

@test_throws ErrorException initialize!(test_optim, test_x)
@test_throws ErrorException update!(test_optim, test_x)
@test_throws ErrorException solver_step!(test_x, test_optim)


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
