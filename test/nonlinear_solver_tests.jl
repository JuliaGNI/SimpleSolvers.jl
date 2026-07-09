using SimpleSolvers
using SimpleSolvers: initialize!, solver_step!, BierlaireQuadratic
using SimpleSolvers: NonlinearSolverState, assess_convergence, residuals, update!, iteration_number
using Test
using Random
using ForwardDiff
using LinearAlgebra: SingularException
Random.seed!(1234)

@testset "Stagnation is not reported as convergence (§3 / 2.2)" begin
    config = Options()

    # A stalled step: x and y do not change between iterations (rxₛ = rfₛ = 0)
    # but the absolute residual rfₐ = ‖y‖ is large.  This used to be reported as
    # converged because the default criteria were successive-change based and
    # f_abstol defaulted to 0.
    state = NonlinearSolverState([1.0, 1.0])
    update!(state, [1.0, 1.0], [5.0, 5.0])
    update!(state, [1.0, 1.0], [5.0, 5.0])
    rxₛ, rfₐ, rfₛ = residuals(state)
    @test rxₛ == 0 && rfₛ == 0            # the step has stalled
    @test rfₐ > config.g_restol          # but the residual is large
    x_converged, f_converged, _ = assess_convergence(rxₛ, rfₐ, rfₛ, config, state)
    @test !x_converged && !f_converged   # ⇒ NOT converged

    # A genuinely converged iterate (settled AND small residual) is reported as
    # converged.
    state2 = NonlinearSolverState([1.0, 1.0])
    update!(state2, [1.0, 1.0], [0.0, 0.0])
    update!(state2, [1.0, 1.0], [0.0, 0.0])
    rxₛ2, rfₐ2, rfₛ2 = residuals(state2)
    xc2, fc2, _ = assess_convergence(rxₛ2, rfₐ2, rfₛ2, config, state2)
    @test xc2 || fc2

    # The successive-change criteria are gated by the (nonzero) residual tolerance
    # g_restol, which is what rejects the stalled step above.
    @test config.g_restol > 0
end

# struct NonlinearSolverTestMethod <: NonlinearSolverMethod end
#
# test_solver = NonlinearSolverTest{Float64}()
#
# @test_throws ErrorException config(test_solver)
# @test_throws ErrorException status(test_solver)
# @test_throws ErrorException initialize!(test_solver, rand(3))
# @test_throws ErrorException solver_step!(test_solver)

# f(x::T) where {T<:Number} = abs(tanh(x - T(0.1)))
# const root₁ = 0.1

f(x::T) where {T<:Number} = exp(x) * (x^3 - 5x^2 + 2x) + 2one(T)
const root₁ = -4.735035753706987262178160540350200552633
const root₂ = -0.6737697823920028217727631890832279199433
const root₃ = 0.7613128434711647120463439168731683731732
const root₄ = 4.560440205363600153577140702025401006278

F(x) = f.(x)
F!(y, x, params) = y .= F(x)

n = 1
x₀ = rand(n)

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
        (NewtonSolver, (linesearch=Static(T),), 2),
        (NewtonSolver, (linesearch=Backtracking(T),), 2),
        (NewtonSolver, (linesearch=Bisection(T),), 2),
        # §2.4 / 2.6: `Quadratic(T, ::SolverMethod)` no longer squares its defaults.
        # The former ε² was below machine epsilon, so the line search never met its
        # internal convergence test and over-refined to artificial (≈0 eps)
        # precision.  With the corrected ε = default_precision(T) the line search
        # converges to its designed precision, which caps the attainable accuracy
        # (≈2.5 eps in Float64, ≈17 eps in Float32); hence the looser tolfac here.
        (NewtonSolver, (linesearch=Quadratic(T, NewtonMethod()),), 32),
        (NewtonSolver, (linesearch=BierlaireQuadratic(T),), 2),
        (QuasiNewtonSolver, (linesearch=Static(T),), 2),
        (QuasiNewtonSolver, (linesearch=Backtracking(T),), 2),
        (QuasiNewtonSolver, (linesearch=Bisection(T),), 2),
        (QuasiNewtonSolver, (linesearch=Quadratic(T, NewtonMethod()),), 32),
        (QuasiNewtonSolver, (linesearch=BierlaireQuadratic(T),), 8),
        (PicardSolver, (linesearch=Bisection(T),), 8),
        (DogLegSolver, (), 1),
    )

        @testset "$(Solver) & $(kwarguments) & $(T)" begin
            x = T.(copy(x₀))
            y = F(x)
            nl = Solver(x, y; F=F!, verbosity=0, kwarguments...)
            # nl = Solver(x, y; F = F!, verbosity=2, kwarguments...)
            ss = SolverState(nl)

            solve!(x, nl, ss)

            for _x in x
                @test ≈(_x, T(root₁); atol=tolfac * eps(T)) || ≈(_x, T(root₂); atol=tolfac * eps(T)) || ≈(_x, T(root₃); atol=tolfac * eps(T)) || ≈(_x, T(root₄); atol=tolfac * eps(T))
            end

            x .= T.(x₀)
            # use custom Jacobian
            nl = Solver(x, y; F=F!, (DF!)=J!, verbosity=0, kwarguments...)
            ss = SolverState(nl)

            solve!(x, nl, ss)

            for _x in x
                @test ≈(_x, T(root₁); atol=tolfac * eps(T)) || ≈(_x, T(root₂); atol=tolfac * eps(T)) || ≈(_x, T(root₃); atol=tolfac * eps(T)) || ≈(_x, T(root₄); atol=tolfac * eps(T))
            end
        end
    end
end

# test alternative constructors
for T ∈ (Float64, Float32)
    # tolfac is a scaling factor for the tolerance s.th. atol = tolfac * eps(T)
    for (solver_method, kwarguments, tolfac) in (
        (NewtonMethod(), (linesearch=Static(T),), 2),
        (QuasiNewtonMethod(), (linesearch=Static(T),), 2),
        (PicardMethod(), (linesearch=Bisection(T),), 8),
        (DogLeg(), (), 1)
    )

        @testset "Testing alternative constructor with method = $(solver_method) & $(kwarguments) & $(T)" begin
            x = T.(copy(x₀))
            y = F(x)
            nl = NonlinearSolver(solver_method, x, y; F=F!, verbosity=0, kwarguments...)
            # nl = Solver(x, y; F = F!, verbosity=2, kwarguments...)

            # println(Solver, ", ", kwarguments, ", ", T, ", ", tolfac, "\n")

            @test config(nl) == nl.config

            solve!(x, nl)

            for _x in x
                @test ≈(_x, T(root₁); atol=tolfac * eps(T)) || ≈(_x, T(root₂); atol=tolfac * eps(T)) || ≈(_x, T(root₃); atol=tolfac * eps(T)) || ≈(_x, T(root₄); atol=tolfac * eps(T))
            end

        end
    end
end


# test regularization
for T ∈ (Float64, Float32)
    for (solver_method, kwarguments) in (
        (NewtonMethod(), (linesearch=Static(T),)),
        (QuasiNewtonMethod(5), (linesearch=Static(T),)),
        (DogLeg(), ())
    )

        @testset "Testing regularization with method = $(solver_method) & $(kwarguments) & $(T)" begin
            x = T.(copy(x₀))
            y = F(x)
            nl = NonlinearSolver(solver_method, x, y; F=F!, verbosity=0, regularization_factor=1E-3, kwarguments...)

            solve!(x, nl)

            for _x in x
                @test ≈(_x, T(root₁); atol=2eps(T)) || ≈(_x, T(root₂); atol=2eps(T)) || ≈(_x, T(root₃); atol=2eps(T)) || ≈(_x, T(root₄); atol=2eps(T))
            end

        end
    end
end


@testset "DogLeg at the exact root (§2.1)" begin
    # Starting exactly at the root, the steepest-descent (Cauchy) scaling divides
    # by ‖J·JᵀF‖² = 0, which used to produce NaN and throw.  The guard now sets
    # the direction to zero so the convergence check reports convergence in 0–1
    # steps instead.
    Flin(y, x, p) = (y .= x)
    for T in (Float64, Float32)
        x = zeros(T, 2)
        s = DogLegSolver(x, Flin, similar(x))
        ss = SolverState(s)
        solve!(x, s, ss)
        @test all(iszero, x)
        @test iteration_number(ss) ≤ 2
    end
end

@testset "Check whether direction NaN test works" begin

    function Fnan(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T}
        y .= exp.(-one(T) ./ (x .^ 2))
    end

    n = 10
    T = Float32

    J₁ = JacobianFiniteDifferences{T}(Fnan, n, n) # the finite difference Jacobian doesn't return NaNs in the first iteration.
    J₂ = JacobianAutodiff{T}(Fnan, n)
    x = zeros(T, n)
    y = zeros(T, n)

    nl₁ = NonlinearSolver(NewtonMethod(), x, y; F=Fnan, jacobian=J₁, verbosity=2)
    nl₂ = NonlinearSolver(NewtonMethod(), x, y; F=Fnan, jacobian=J₂, verbosity=2)

    x₁ = zeros(T, n)
    x₂ = zeros(T, n)

    # The solver must refuse to proceed on this pathological problem.  The finite
    # difference Jacobian at x = 0 is the zero matrix, which is singular: the LU
    # solver now throws a `SingularException` (§2.5) instead of silently returning
    # NaN.  The autodiff Jacobian produces NaN entries, which is caught as a
    # `NonlinearSolverException` (NaN in the direction vector).
    @test_throws SingularException solve!(x₁, nl₁)
    @test_throws NonlinearSolverException solve!(x₂, nl₂)

end

@testset "Phase 4.2 error-swallowing fallbacks removed" begin
    # The catch-all `initialize!(x...)` (which swallowed MethodErrors behind a
    # generic "not defined" message) was deleted.
    @test !hasmethod(initialize!, Tuple{Int})
    @test !hasmethod(initialize!, Tuple{Int,Int,Int})

    # The 1-argument `solver_step!(s::NonlinearSolver)` stub (which only errored)
    # was deleted, so an unsupported call is a proper `MethodError`.
    f!(y, x, params) = y .= x .^ 2 .- 1
    s = NewtonSolver([2.0], [3.0]; F=f!)
    @test !hasmethod(solver_step!, Tuple{typeof(s)})
    @test_throws MethodError solver_step!(s)
end
