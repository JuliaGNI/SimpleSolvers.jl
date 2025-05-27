using SimpleSolvers
using SimpleSolvers: initialize!, solver_step!, BierlaireQuadratic
using Test
using Random
using ForwardDiff
Random.seed!(1234)

struct NonlinearSolverTest{T} <: NonlinearSolver end

test_solver = NonlinearSolverTest{Float64}()

@test_throws ErrorException config(test_solver)
@test_throws ErrorException status(test_solver)
@test_throws ErrorException initialize!(test_solver, rand(3))
@test_throws ErrorException solver_step!(test_solver)

f(x::T) where {T<:Number} = exp(x) * (x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
F(x) = f.(x)
F!(y, x) = y .= F(x)

n = 1
x₀ = rand(n)
root₁ = 0.7613128434711648
root₂ = -4.7350357537069865

function J!(g, x)
    g .= 0
    for i in eachindex(x)
        g[i, i] = ForwardDiff.derivative(f, x[i])
    end
end

for T ∈ (Float64, ) # Float32)
    for (Solver, kwarguments) in (
                (NewtonSolver, (linesearch = Static(),)),
                (NewtonSolver, (linesearch = Backtracking(),)),
                # (NewtonSolver, (linesearch = Quadratic2(),)),
                # (NewtonSolver, (linesearch = BierlaireQuadratic(),)),
                # (NewtonSolver, (linesearch = Bisection(),)),
                # (QuasiNewtonSolver, (linesearch = Static(),)),
                # (QuasiNewtonSolver, (linesearch = Backtracking(),)),
                # (QuasiNewtonSolver, (linesearch = BierlaireQuadratic(),)),
                # (QuasiNewtonSolver, (linesearch = Quadratic(),)),
                # (QuasiNewtonSolver, (linesearch = Bisection(),)),
            )

        x = T.(copy(x₀))
        y = F(x)
        nl = Solver(x, y; F = F, kwarguments...)

        @test config(nl) == nl.config
        @test status(nl) == nl.status

        solve!(nl, x)
        for _x in x
            @test ≈(_x, T(root₁)) || ≈(_x, T(root₂))
        end

        x .= T.(x₀)
        # use custom Jacobian
        nl = Solver(x, y; F = F, DF! = J!, kwarguments...)
        solve!(nl, x)
        println(Solver, kwarguments)
        for _x in x
            @test ≈(_x, T(root₁)) || ≈(_x, T(root₂))
        end
    end
end