
using SimpleSolvers
using Test


struct OptimizerTest{T} <: Optimizer{T} end

test_optim = OptimizerTest{Float64}()

@test_throws ErrorException solve!(test_optim)
@test_throws ErrorException status(test_optim)
@test_throws ErrorException params(test_optim)


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


for Optim in (BFGSOptimizer,)
    n = 1
    x = ones(n)
    y = zero(eltype(x))
    nl = Optim(x, F)

    @test params(nl) == nl.params
    @test status(nl) == nl.status

    setInitialConditions!(nl, x)
    solve!(nl)
    # println(nl.status.i, ", ", nl.status.rₐ,", ",  nl.status.rᵣ,", ",  nl.status.rₛ)
    for x in nl.x̄
        @test x ≈ 0 atol=1E-7
    end

    x = ones(n)
    nl = Optim(x, F; ∇F! = ∇F!)
    setInitialConditions!(nl, x)
    solve!(nl)
    # println(nl.status.i, ", ", nl.status.rₐ,", ",  nl.status.rᵣ,", ",  nl.status.rₛ)
    for x in nl.x̄
        @test x ≈ 0 atol=1E-7
    end
end
