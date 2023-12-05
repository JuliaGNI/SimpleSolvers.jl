
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
    f .= x.^2
end

function J!(g, x)
    g .= 0
    for i in eachindex(x)
        g[i,i] = 2x[i]
    end
end


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

    for T in (Float64, Float32)
        n = 1
        x = ones(T, n)
        y = zero(x)
        nl = Solver(x, y; kwarguments...)

        @test config(nl) == nl.config
        @test status(nl) == nl.status

        solve!(x, F!, nl)
        # println(status(nl))
        for _x in x
            @test _x ≈ 0 atol=1E-7
        end

        x = ones(T, n)
        nl = Solver(x, y; J! = J!, kwarguments...)
        solve!(x, F!, J!, nl)
        # println(status(nl))
        for _x in x
            @test _x ≈ 0 atol=1E-7
        end
    end
end
