
using LinearAlgebra
using SimpleSolvers
using SimpleSolvers: initialize!, objective, solver_step!
using Test


struct OptimizerTest{T} <: Optimizer{T} end

test_optim = OptimizerTest{Float64}()

@test_throws ErrorException config(test_optim)
@test_throws ErrorException status(test_optim)
@test_throws ErrorException objective(test_optim)
@test_throws ErrorException initialize!(test_optim)
@test_throws ErrorException solver_step!(test_optim)


function F(x)
    # 1 + (x.^2)
    y = one(eltype(x))
    for _x in x
        y += _x^2
    end
    return y
end

function ∇F!(g, x)
    g .= 2 .* x
end

function H!(g, x)
    g .= 0
    for i in eachindex(x)
        g[i,i] = 2
    end
end


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
