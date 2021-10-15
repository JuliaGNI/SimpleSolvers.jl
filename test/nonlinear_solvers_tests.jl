
using SimpleSolvers
using Test


struct NonlinearSolverTest{T} <: NonlinearSolver{T} end

test_solver = NonlinearSolverTest{Float64}()

@test_throws ErrorException solve!(test_solver)
@test_throws ErrorException status(test_solver)
@test_throws ErrorException params(test_solver)


function F!(f, x)
    f .= x.^2
end

function J!(g, x)
    g .= 0
    for i in eachindex(x)
        g[i,i] = 2x[i]
    end
end


for Solver in (NewtonSolver, QuasiNewtonSolver, NLsolveNewton)
    n = 1
    x = ones(n)
    y = zero(x)
    nl = Solver(x, y, F!)

    @test params(nl) == nl.params
    @test status(nl) == nl.status

    solve!(nl)
    # println(nl.status.i, ", ", nl.status.rₐ,", ",  nl.status.rᵣ,", ",  nl.status.rₛ)
    for x in nl.x
        @test x ≈ 0 atol=1E-7
    end

    x = ones(n)
    nl = Solver(x, y, F!; J! = J!)
    solve!(nl)
    # println(nl.status.i, ", ", nl.status.rₐ,", ",  nl.status.rᵣ,", ",  nl.status.rₛ)
    for x in nl.x
        @test x ≈ 0 atol=1E-7
    end
end
