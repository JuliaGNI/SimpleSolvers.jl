using LinearAlgebra
using SimpleSolvers
using SimpleSolvers: gradient, hessian, linesearch, problem, initialize!, update!, solver_step!
using Test
using Random
Random.seed!(123)

include("optimizers_problems.jl")

struct OptimizerTest{T} <: OptimizerState{T} end

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

for T in (Float64, Float32)
    for method in (Newton(), DFP(), BFGS())
        for _linesearch in (Static(T(0.1)), Backtracking(T), BierlaireQuadratic(T), Quadratic(T), Bisection(T))
            @testset "$(method) & $(_linesearch) & $(T)" begin
            n = 1
            x = ones(T, n)
            opt = Optimizer(x, F; algorithm = method, linesearch = _linesearch)
            state = OptimizerState(method, x)

            method == DFP() && _linesearch == Quadratic() && T == Float32 && continue # for some reason quadratic linesearch for DFP fails in single precision!

            @test typeof(gradient(opt)) <: GradientAutodiff

            solve!(x, state, opt)
            @test norm(x) ≈ zero(T) atol=∛(2000eps(T))
            @test F(x) ≈ F(zero(T)) atol=∛(2000eps(T))

            x = ones(T, n)
            opt = Optimizer(x, F; ∇F! = ∇F!, algorithm = method, linesearch = _linesearch)

            @test typeof(gradient(opt)) <: GradientFunction

            state = OptimizerState(method, x)

            solve!(x, state, opt)
            @test norm(x) ≈ zero(T) atol=∛(2000eps(T))
            @test F(x) ≈ F(0) atol=∛(2000eps(T))
            end
        end
    end
end
