
using LinearAlgebra
using SimpleSolvers
using SimpleSolvers: gradient, hessian, linesearch, objective, initialize!, update!, solver_step!
using Test

include("optimizers_problems.jl")


struct OptimizerTest{T} <: OptimizationAlgorithm end


for DT in (Float32, Float64)

    test_optim = OptimizerTest{DT}()
    test_x = zeros(DT, 3)
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
        for linesearch in (Static(0.8), Backtracking(), Quadratic(), Bisection())
            println(method)
            println(linesearch)

            n = 1
            x = ones(DT, n)
            opt = Optimizer(x, F; algorithm = method, linesearch = linesearch)

            @test config(opt) == opt.config
            @test status(opt) == opt.result.status

            solve!(x, opt)
            # println(opt)
            @test norm(minimizer(opt)) ≈ 0 atol=sqrt(eps(DT))
            @test norm(minimum(opt)) ≈ F(0) atol=sqrt(eps(DT))

            x = ones(DT, n)
            opt = Optimizer(x, F; ∇F! = ∇F!, algorithm = method, linesearch = linesearch)
            solve!(x, opt)
            # println(opt)
            @test norm(minimizer(opt)) ≈ 0 atol=sqrt(eps(DT))
            @test norm(minimum(opt)) ≈ F(0) atol=sqrt(eps(DT))
        end
    end

end
