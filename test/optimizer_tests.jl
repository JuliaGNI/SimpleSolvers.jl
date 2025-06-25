using LinearAlgebra
using SimpleSolvers
using SimpleSolvers: gradient, hessian, linesearch, objective, initialize!, update!, solver_step!
using Test
using Random
Random.seed!(123)

include("optimizers_problems.jl")

struct OptimizerTest{T} <: OptimizationAlgorithm end

test_optim = OptimizerTest{Float64}()
test_x = zeros(3)
test_obj = MultivariateObjective(F, test_x)

@test_throws MethodError gradient(test_optim)
@test_throws MethodError hessian(test_optim)
@test_throws MethodError linesearch(test_optim)
@test_throws MethodError objective(test_optim)

# test if the correct error is thrown when calling `initialize!` on an `OptimizationAlgorithm`.
@test_throws ErrorException initialize!(test_optim, test_x)
@test_throws MethodError update!(test_optim, test_x)
@test_throws MethodError solver_step!(test_x, test_optim)

for method in (Newton(), BFGS(), DFP())
    for _linesearch in (Static(0.8), Backtracking()) # , Quadratic2(), BierlaireQuadratic(), Bisection())
        for T in (Float64, Float32)
            n = 1
            x = ones(T, n)
            opt = Optimizer(x, F; algorithm = method, linesearch = _linesearch)

            @test config(opt) == opt.config
            @test status(opt) == opt.result.status

            solve!(opt, x)
            @test norm(minimizer(opt)) ≈ 0 atol=∛(2000eps(T))
            @test norm(minimum(opt)) ≈ F(0) atol=∛(2000eps(T))

            x = ones(T, n)
            opt = Optimizer(x, F; ∇F! = ∇F!, algorithm = method, linesearch = _linesearch)

            solve!(opt, x)
            @test norm(minimizer(opt)) ≈ 0 atol=∛(2000eps(T))
            @test norm(minimum(opt)) ≈ F(0) atol=∛(2000eps(T))
        end
    end
end