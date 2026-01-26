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

const root₁ = -4.735035753706987262178160540350200552633
const root₂ = -0.6737697823920028217727631890832279199433
const root₃ =  0.7613128434711647120463439168731683731732
const root₄ =  4.560440205363600153577140702025401006278

function J!(g, x, params)
    g .= 0
    for i in eachindex(x)
        g[i, i] = ForwardDiff.derivative(f, x[i])
    end
    g
end

for T ∈ (Float64, Float32)
    # tolfac is a scaling factor for the tolerance s.th. atol = tolfac * eps(T)
    for (Solver, kwarguments, tolfac) in (
                (NewtonSolver, (linesearch = Static(),), 2),#
                (NewtonSolver, (linesearch = Backtracking(),), 2),#
                #(NewtonSolver, (linesearch = Quadratic(),), 1e6), ### this combination fails!!!
                (NewtonSolver, (linesearch = BierlaireQuadratic(),), 2),#
                (NewtonSolver, (linesearch = Bisection(),), 8),#
                #(QuasiNewtonSolver, (linesearch = Static(),), 1e6), ### this combination fails!!!
                (QuasiNewtonSolver, (linesearch = Backtracking(),), 2),#
                #(QuasiNewtonSolver, (linesearch = Quadratic(),), 1e6), ### this combination fails!!!
                (QuasiNewtonSolver, (linesearch = BierlaireQuadratic(),), 8),#
                (QuasiNewtonSolver, (linesearch = Bisection(),), 2),#
            )

        @testset "$(Solver) & $(kwarguments) & $(T)" begin
            x = T.(copy(x₀))
            y = F(x)
            nl = Solver(x, y; F = F!, kwarguments...)

#        println(Solver, ", ", kwarguments, ", ", T, ", ", tolfac, "\n")

            solve!(x, nl)

            for _x in x
                @test ≈(_x, T(root₁); atol=tolfac*eps(T)) || ≈(_x, T(root₂); atol=tolfac*eps(T)) || ≈(_x, T(root₃); atol=tolfac*eps(T)) || ≈(_x, T(root₄); atol=tolfac*eps(T))
            end

            x .= T.(x₀)
            # use custom Jacobian
            nl = Solver(x, y; F = F!, DF! = J!, kwarguments...)
            solve!(x, nl)
                for _x in x
                @test ≈(_x, T(root₁); atol=tolfac*eps(T)) || ≈(_x, T(root₂); atol=tolfac*eps(T)) || ≈(_x, T(root₃); atol=tolfac*eps(T)) || ≈(_x, T(root₄); atol=tolfac*eps(T))
            end
        end
    end
end
