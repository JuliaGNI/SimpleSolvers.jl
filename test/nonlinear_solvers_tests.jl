
using SimpleSolvers
using Test


struct NonlinearSolverTest{T} <: NonlinearSolver{T} end

test_solver = NonlinearSolverTest{Float64}()

@test_throws ErrorException solve!(test_solver)
@test_throws ErrorException status(test_solver)
@test_throws ErrorException params(test_solver)


n = 1
T = Float64

function F(x::Vector, b::Vector)
    b .= x.^2
end

function J(x::Vector, A::Matrix)
    for i in eachindex(x)
        A[i,i] = 2x[i]
    end
end


for Solver in (NewtonSolver, QuasiNewtonSolver, NLsolveNewton)
    x = ones(T, n)
    nl = Solver(x, F)

    @test params(nl) == nl.params
    @test status(nl) == nl.status

    solve!(nl)
    # println(nl.status.i, ", ", nl.status.rₐ,", ",  nl.status.rᵣ,", ",  nl.status.rₛ)
    for x in nl.x
        @test x ≈ 0 atol=1E-7
    end

    x = ones(T, n)
    nl = Solver(x, F; J! = J)
    solve!(nl)
    # println(nl.status.i, ", ", nl.status.rₐ,", ",  nl.status.rᵣ,", ",  nl.status.rₛ)
    for x in nl.x
        @test x ≈ 0 atol=1E-7
    end
end
