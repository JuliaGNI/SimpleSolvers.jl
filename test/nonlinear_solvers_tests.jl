
using SimpleSolvers
using SimpleSolvers: initialize!, solver_step!
using Test


struct NonlinearSolverTest{T} <: NonlinearSolver{T} end

test_solver = NonlinearSolverTest{Float64}()

@test_throws ErrorException config(test_solver)
@test_throws ErrorException status(test_solver)
@test_throws ErrorException initialize!(test_solver, rand(3))
@test_throws ErrorException solver_step!(test_solver)


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

    @test config(nl) == nl.config
    @test status(nl) == nl.status

    solve!(x, nl)
    # println(nl.status.i, ", ", nl.status.rₐ,", ",  nl.status.rᵣ,", ",  nl.status.rₛ)
    for x in nl.x
        @test x ≈ 0 atol=1E-7
    end

    x = ones(n)
    nl = Solver(x, y, F!; J! = J!)
    solve!(x, nl)
    # println(nl.status.i, ", ", nl.status.rₐ,", ",  nl.status.rᵣ,", ",  nl.status.rₛ)
    for x in nl.x
        @test x ≈ 0 atol=1E-7
    end
end
