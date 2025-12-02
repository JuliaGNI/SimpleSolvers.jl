using LinearAlgebra
using SimpleSolvers
using SimpleSolvers: gradient, hessian, linesearch, problem, initialize!, update!, solver_step!
using Test
using Random
Random.seed!(123)

include("optimizers_problems.jl")

struct OptimizerTest{T} <: OptimizerState end

test_optim = OptimizerTest{Float64}()
test_x = zeros(3)
test_obj = OptimizerProblem(F, test_x)

@test_throws MethodError gradient(test_optim)
@test_throws MethodError hessian(test_optim)
@test_throws MethodError linesearch(test_optim)
@test_throws MethodError problem(test_optim)

# test if the correct error is thrown when calling `initialize!` on an `OptimizerState`.
@test_throws ErrorException initialize!(test_optim, test_x)
@test_throws MethodError update!(test_optim, test_x)
@test_throws MethodError solver_step!(test_x, test_optim)

for method in (Newton(), BFGS(), DFP())
    for _linesearch in (Static(0.1), Backtracking(), Bisection(), BierlaireQuadratic(), Quadratic2())
        for T in (Float64, Float32)
            # for T = Float32 some optimizers seem to have problems converging. TODO: investigate!!
            T == Float32 && typeof(_linesearch) <: Union{BierlaireQuadratic, Quadratic2} && continue
            n = 1
            x = ones(T, n)
            opt = Optimizer(x, F; algorithm = method, linesearch = _linesearch)
            state = NewtonOptimizerState(x)

            @test config(opt) == opt.config

            solve!(opt, state, x)
            @test norm(x) ≈ zero(T) atol=∛(2000eps(T))
            @test F(x) ≈ F(zero(T)) atol=∛(2000eps(T))

            x = ones(T, n)
            opt = Optimizer(x, F; ∇F! = ∇F!, algorithm = method, linesearch = _linesearch)
            state = NewtonOptimizerState(x)

            solve!(opt, state, x)
            @test norm(x) ≈ zero(T) atol=∛(2000eps(T))
            @test F(x) ≈ F(0) atol=∛(2000eps(T))
        end
    end
end