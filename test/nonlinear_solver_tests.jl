using SimpleSolvers
using SimpleSolvers: initialize!, solver_step!
using Test

struct NonlinearSolverTest{T} <: NonlinearSolver end

test_solver = NonlinearSolverTest{Float64}()

@test_throws ErrorException config(test_solver)
@test_throws ErrorException status(test_solver)
@test_throws ErrorException initialize!(test_solver, rand(3))
@test_throws ErrorException solver_step!(test_solver)

function F!(f, x)
    f .= tan.(x)
end

function J!(g, x)
    g .= 0
    for i in eachindex(x)
        g[i,i] = sec(x[i])^2
    end
end

for T ∈ (Float64, Float32)
    for (Solver, kwarguments) in (
                (NewtonSolver, (linesearch = Static(),)),
                (NewtonSolver, (linesearch = Backtracking(),)),
                (NewtonSolver, (linesearch = Quadratic(),)),
                (NewtonSolver, (linesearch = Bisection(),)),
                (QuasiNewtonSolver, (linesearch = Static(),)),
                (QuasiNewtonSolver, (linesearch = Backtracking(),)),
                (QuasiNewtonSolver, (linesearch = Quadratic(),)),
                (QuasiNewtonSolver, (linesearch = Bisection(),)),
                # (NLsolveNewton, NamedTuple()),
            )

        n = 1
        x = zeros(T, n)
        y = zero(x)
        nl = Solver(x, y; F = F!, kwarguments...)

        @test config(nl) == nl.config
        @test status(nl) == nl.status

        solve!(x, F!, nl)
        # println(status(nl))
        for _x in x
            @test _x ≈ zero(T) atol = eps(T)
        end

        x = zeros(T, n)
        nl = Solver(x, y; DF! = J!, kwarguments...)
        solve!(x, F!, nl)
        println(Solver, kwarguments)
        for _x in x
            @test _x ≈ zero(T) atol = eps(T)
        end
    end
end