using LinearAlgebra
using SimpleSolvers
using SimpleSolvers: gradient, hessian, linesearch, objective, initialize!, update!, solver_step!
using Test

include("optimizers_problems.jl")

struct OptimizerTest{T} <: OptimizationAlgorithm end

test_optim = OptimizerTest{Float64}()
test_x = zeros(3)
test_obj = MultivariateObjective(F, test_x)

@test_throws MethodError gradient(test_optim)
@test_throws MethodError hessian(test_optim)
@test_throws MethodError linesearch(test_optim)
@test_throws MethodError objective(test_optim)

@test_throws MethodError initialize!(test_optim, test_x)
@test_throws MethodError update!(test_optim, test_x)
@test_throws MethodError solver_step!(test_x, test_optim)

@test isaOptimizationAlgorithm(test_optim) == false
@test isaOptimizationAlgorithm(NewtonOptimizer(test_x, test_obj)) == true
@test isaOptimizationAlgorithm(BFGSOptimizer(test_x, test_obj)) == true
@test isaOptimizationAlgorithm(DFPOptimizer(test_x, test_obj)) == true

for method in (Newton(), BFGS(), DFP())
    for _linesearch in (Static(0.8), Backtracking(), Quadratic(), Bisection())
        for T in (Float64, Float32)
            n = 1
            x = ones(T, n)
            opt = Optimizer(x, F; algorithm = method, linesearch = _linesearch)

            @test config(opt) == opt.config
            @test status(opt) == opt.result.status

            if !(method == BFGS() && _linesearch == Quadratic() && T == Float32)
                # TODO: Investigate why this combination always fails.

                solve!(x, opt)
                # println(opt)
                @test norm(minimizer(opt)) ≈ 0 atol=1E-7
                @test norm(minimum(opt)) ≈ F(0) atol=1E-7

                x = ones(T, n)
                opt = Optimizer(x, F; ∇F! = ∇F!, algorithm = method, linesearch = _linesearch)
                solve!(x, opt)
                # println(opt)
                @test norm(minimizer(opt)) ≈ 0 atol=1E-7
                @test norm(minimum(opt)) ≈ F(0) atol=1E-7
            end
        end
    end
end