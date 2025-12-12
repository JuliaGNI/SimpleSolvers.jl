using SimpleSolvers
using SimpleSolvers: initialize!, solver_step!, BierlaireQuadratic
using Test
using Random
using ForwardDiff
Random.seed!(1234)

# struct NonlinearSolverTestMethod <: NonlinearSolverMethod end
#
# test_solver = NonlinearSolverTest{Float64}()
#
# @test_throws ErrorException config(test_solver)
# @test_throws ErrorException status(test_solver)
# @test_throws ErrorException initialize!(test_solver, rand(3))
# @test_throws ErrorException solver_step!(test_solver)

f(x::T) where {T<:Number} = exp(x) * (x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
F(x) = f.(x)
F!(y, x, params) = y .= F(x)

n = 1
x₀ = rand(n)
root₁ = 0.76131284
root₂ = -4.7350357537069865

function J!(g, x, params)
    g .= 0
    for i in eachindex(x)
        g[i, i] = ForwardDiff.derivative(f, x[i])
    end
    g
end

for T ∈ (Float64, Float32)
    for (Solver, kwarguments) in (
                (NewtonSolver, (linesearch = Static(),)),
                (NewtonSolver, (linesearch = Backtracking(),)),
                (NewtonSolver, (linesearch = Quadratic(),)),
                (NewtonSolver, (linesearch = BierlaireQuadratic(),)),
                (NewtonSolver, (linesearch = Bisection(),)),
                # (QuasiNewtonSolver, (linesearch = Static(),)),
                (QuasiNewtonSolver, (linesearch = Backtracking(),)),
                (QuasiNewtonSolver, (linesearch = Quadratic(),)),
                (QuasiNewtonSolver, (linesearch = BierlaireQuadratic(),)),
                (QuasiNewtonSolver, (linesearch = Bisection(),)),
            )

        x = T.(copy(x₀))
        y = F(x)
        nl = Solver(x, y; F = F!, kwarguments...)

        @test config(nl) == nl.config
        @test status(nl) == nl.status

        solve!(x, nl)
        for _x in x
            @test ≈(_x, T(root₁); atol=∛(2eps(T))) || ≈(_x, T(root₂); atol=∛(2eps(T)))
        end

        x .= T.(x₀)
        # use custom Jacobian
        nl = Solver(x, y; F = F!, DF! = J!, kwarguments...)
        solve!(x, nl)
        # println(Solver, kwarguments)
        for _x in x
            @test ≈(_x, T(root₁); atol=∛(2eps(T))) || ≈(_x, T(root₂); atol=∛(2eps(T)))
        end
    end
end
