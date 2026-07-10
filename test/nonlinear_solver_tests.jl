using SimpleSolvers
using SimpleSolvers: initialize!, solver_step!, BierlaireQuadratic
using SimpleSolvers: NonlinearSolverState, assess_convergence, residuals, update!, iteration_number
using SimpleSolvers: linesearch_problem, cache, jacobianmatrix, solution, value, direction, direction!, NullParameters
using SimpleSolvers: trust_radius, INITIAL_Δ
using Test
using Random
using ForwardDiff
using LinearAlgebra: SingularException
Random.seed!(1234)

@testset "Stagnation is not reported as convergence" begin
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
        # `Quadratic(T, ::SolverMethod)` no longer squares its defaults.
        # The former ε² was below machine epsilon, so the line search never met its
        # internal convergence test and over-refined to artificial (≈0 eps)
        # precision.  With the corrected ε = default_precision(T) the line search
        # converges to its designed precision, which caps the attainable accuracy
        # (≈2.5 eps in Float64, ≈17 eps in Float32); hence the looser tolfac here.
        (NewtonSolver, (linesearch=Quadratic(T, Newton()),), 32),
        (NewtonSolver, (linesearch=BierlaireQuadratic(T),), 2),
        # The strong-Wolfe (bracket + zoom) line search.
        (NewtonSolver, (linesearch=StrongWolfe(T),), 2),
        (QuasiNewtonSolver, (linesearch=Static(T),), 2),
        (QuasiNewtonSolver, (linesearch=Backtracking(T),), 2),
        (QuasiNewtonSolver, (linesearch=Bisection(T),), 2),
        (QuasiNewtonSolver, (linesearch=Quadratic(T, Newton()),), 32),
        (QuasiNewtonSolver, (linesearch=BierlaireQuadratic(T),), 8),
        # PicardSolver is now a (residual-safeguarded) fixed-point
        # iteration and no longer runs a derivative-based line search, so it takes
        # no `linesearch` keyword here.
        (PicardSolver, (), 8),
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
        (Newton(), (linesearch=Static(T),), 2),
        (QuasiNewton(), (linesearch=Static(T),), 2),
        (Picard(), (), 8),
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
        (Newton(), (linesearch=Static(T),)),
        (QuasiNewton(5), (linesearch=Static(T),)),
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


@testset "DogLeg at the exact root" begin
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

@testset "PicardSolver is a residual-safeguarded fixed-point iteration" begin
    # A genuine contraction: F(x) = x - cos(x) has the fixed point x = cos(x) (the
    # Dottie number ≈ 0.7390851332151607); the fixed-point iteration x ← cos(x)
    # converges.  Picard takes no derivative-based line search (the residual
    # direction d = -F is not a descent direction for ‖F‖² in general).
    Fcos(y, x, p) = (y .= x .- cos.(x))
    dottie = 0.7390851332151607
    for T in (Float64, Float32)
        # both the direct constructor and the method-based constructor
        x1 = T[0.5]
        solve!(x1, PicardSolver(x1, Fcos, similar(x1)))
        @test isapprox(x1[1], T(dottie); atol=sqrt(eps(T)))
        @test abs(x1[1] - cos(x1[1])) ≤ sqrt(eps(T))

        x2 = T[0.5]
        solve!(x2, NonlinearSolver(Picard(), x2, similar(x2); F=Fcos))
        @test isapprox(x2[1], T(dottie); atol=sqrt(eps(T)))
    end

    # The residual-monotonicity safeguard never lets the accepted step increase
    # the residual: a full (undamped) fixed-point step here would overshoot, but
    # Picard still converges (it damps α as needed) rather than diverging.
    Fover(y, x, p) = (y .= 3 .* (x .- 1.0))   # full step x ← x - 3(x-1) overshoots
    x = [2.0]
    solve!(x, PicardSolver(x, Fover, similar(x)))
    @test isapprox(x[1], 1.0; atol=1e-6)
end

@testset "DogLeg ρ-based trust region grows on good steps and carries Δ" begin
    # With the full ρ-based radius update (N&W Alg. 4.1) the trust radius is carried
    # across outer steps and *expanded* on good steps that sit on the boundary —
    # the old code reset Δ to INITIAL_Δ every step and could only shrink it.
    # For a linear residual F(x) = x the Gauss-Newton model is exact (ρ ≈ 1), and
    # starting far from the root (‖Newton step‖ = 5 > INITIAL_Δ = 1) forces several
    # boundary steps that grow Δ before the full Newton step converges.
    Flin(y, x, p) = (y .= x)
    for T in (Float64, Float32)
        x = T[5.0, 5.0]
        s = DogLegSolver(x, Flin, similar(x))
        ss = SolverState(s)
        @test trust_radius(cache(s)) == T(INITIAL_Δ)   # reset before solving
        solve!(x, s, ss)
        @test all(v -> isapprox(v, zero(T); atol=10eps(T)), x)  # converged
        @test trust_radius(cache(s)) > T(INITIAL_Δ)             # radius expanded & carried
    end
end

@testset "DogLeg trust radius resets on solver reuse (verification 2026-07-10)" begin
    # `initialize!` used to reset every DogLegCache buffer *except* the carried
    # trust radius, so a reused solver started its next solve with the radius the
    # previous solve ended with (up to DOGLEG_Δ_MAX = 1e2) instead of INITIAL_Δ.
    Flin(y, x, p) = (y .= x)
    x = [5.0, 5.0]
    s = DogLegSolver(x, Flin, similar(x))
    solve!(x, s)
    @test trust_radius(cache(s)) > INITIAL_Δ    # the first solve expanded Δ ...
    initialize!(s, [5.0, 5.0])
    @test trust_radius(cache(s)) == INITIAL_Δ   # ... but a fresh solve starts over
    x2 = [5.0, 5.0]
    solve!(x2, s)                               # and solver reuse still converges
    @test all(v -> isapprox(v, 0.0; atol=1e-10), x2)
end

@testset "DogLeg treats an undefined (NaN) trial merit as a rejected step" begin
    # F(x) = log(x) + 2 has its root at exp(-2) ≈ 0.135; from x₀ = 1 the full
    # Newton step lands at x = -1, outside the domain (the NaN-returning log
    # mimics e.g. NaNMath.log or a table lookup).  The former NaN recovery
    # rescaled d₁ and d₂ *independently*, destroying the ‖d₁‖ ≤ ‖d₂‖ relation
    # the dogleg interpolation assumes; a NaN trial merit is now
    # rejected by shrinking Δ, keeping the dogleg path intact (and never
    # reaching the ρ update, where NaN comparisons would spin the loop forever
    # at constant Δ).
    nanlog(v) = v > 0 ? log(v) : oftype(v, NaN)
    Flog(y, x, p) = (y .= nanlog.(x) .+ 2)
    x = [1.0]
    s = DogLegSolver(x, Flog, similar(x))
    initialize!(s, x)
    y = similar(x)
    Flog(y, x, NullParameters())
    state = NonlinearSolverState(x, y)
    initialize!(state, x, y)
    SimpleSolvers.trust_radius!(cache(s), 4.0)  # puts the NaN trial x = -1 inside the region
    solver_step!(x, s, state, NullParameters())
    @test all(isfinite, x)                      # the NaN trial was never accepted
    @test x[1] > 0
    @test abs(log(x[1]) + 2) < 2.0              # the accepted step reduced the residual

    # ... and a full solve on the same domain-restricted problem converges.
    x2 = [1.0]
    s2 = DogLegSolver(x2, Flog, similar(x2))
    solve!(x2, s2)
    @test isapprox(x2[1], exp(-2.0); atol=1e-8)
end

@testset "PicardSolver rejects a linesearch keyword (verification 2026-07-10)" begin
    # The Picard solver_step! is a fixed-point iteration and consults no line
    # search; a `linesearch` keyword used to be accepted and silently ignored.
    Fcos(y, x, p) = (y .= x .- cos.(x))
    x = [0.5]
    @test_throws MethodError PicardSolver(x, Fcos, similar(x); linesearch=Bisection())
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

    nl₁ = NonlinearSolver(Newton(), x, y; F=Fnan, jacobian=J₁, verbosity=2)
    nl₂ = NonlinearSolver(Newton(), x, y; F=Fnan, jacobian=J₂, verbosity=2)

    x₁ = zeros(T, n)
    x₂ = zeros(T, n)

    # The solver must refuse to proceed on this pathological problem.  The finite
    # difference Jacobian at x = 0 is the zero matrix, which is singular: the LU
    # solver now throws a `SingularException` instead of silently returning
    # NaN.  The autodiff Jacobian produces NaN entries, which is caught as a
    # `NonlinearSolverException` (NaN in the direction vector).
    @test_throws SingularException solve!(x₁, nl₁)
    @test_throws NonlinearSolverException solve!(x₂, nl₂)

end

@testset "line search does not overwrite the shared solver cache" begin
    # The line search closures must use private scratch buffers, so evaluating the
    # line search problem at a trial α ≠ 0 leaves the solver's shared cache
    # (`solution`/`value`/`jacobianmatrix`) untouched — these are read by the solver
    # after the line search returns.
    G(y, x, params) = y .= (x .- 1.0) .^ 2
    x = ones(3) / 2
    y = similar(x)
    nl = NewtonSolver(x, y; F=G)
    _params = NullParameters()

    direction!(nl, x, _params, 1)
    state = NonlinearSolverState(x)
    update!(state, x, G(y, x, _params))

    # Write recognizable sentinels into the shared cache buffers.  If the line
    # search still wrote through them, these would be clobbered by the trial-α
    # evaluations below.
    fill!(value(cache(nl)), 7.0)
    fill!(solution(cache(nl)), 3.0)
    fill!(jacobianmatrix(cache(nl)), 5.0)
    j_before = copy(jacobianmatrix(cache(nl)))
    y_before = copy(value(cache(nl)))
    x_before = copy(solution(cache(nl)))

    lsp = linesearch_problem(nl)
    params = (parameters=_params, x=state.x)
    # Evaluate at a nonzero step so the trial iterate genuinely differs from x.
    lsp.F(0.7, params)
    lsp.D(0.7, params)

    @test jacobianmatrix(cache(nl)) == j_before
    @test value(cache(nl)) == y_before
    @test solution(cache(nl)) == x_before
end

@testset "error-swallowing fallbacks removed" begin
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

# JET Group C regression (2026-07-10): `check_jacobian(s)` / `print_jacobian(s)`
# forwarded to `jacobian(s)` (the `Jacobian` functor *object*) instead of the
# Jacobian *matrix*, so both exported convenience methods threw a `MethodError`
# on every call (`check_jacobian` has only an `::AbstractMatrix` method, and
# `print_jacobian` had no base method at all after the Jacobian-object refactor).
@testset "check_jacobian / print_jacobian operate on the Jacobian matrix" begin
    f!(y, x, params) = y .= x .^ 2 .- 1
    x = [2.0]
    s = NewtonSolver(x, [3.0]; F=f!)
    solve!(x, s)   # populate the cached Jacobian matrix
    # both accept a solver and dispatch to the matrix methods without throwing
    @test check_jacobian(s) === nothing
    @test print_jacobian(s) === nothing
    # the base matrix methods exist and are what the solver forms delegate to
    @test hasmethod(check_jacobian, Tuple{AbstractMatrix})
    @test hasmethod(print_jacobian, Tuple{AbstractMatrix})
end

# Interface-consistency fixes (verification 2026-07-10):
# (a) the method-dispatch constructor `NonlinearSolver(method, …)` used to
#     discard the method's `refactorize` field, so `QuasiNewton(7)`
#     silently built a solver with the default `refactorize = 5`;
# (b) `DogLegSolver(x, y; F)` follows the same `F=missing` + friendly-error
#     pattern as NewtonSolver/PicardSolver (it used to raise a bare
#     `UndefKeywordError`).
@testset "NonlinearSolver(method, ...) honors refactorize" begin
    Flin(y, x, p) = (y .= x)
    x, y = ones(2), zeros(2)
    s5 = NonlinearSolver(QuasiNewton(), x, y; F=Flin)
    @test SimpleSolvers.method(s5).refactorize == 5
    s7 = NonlinearSolver(QuasiNewton(7), x, y; F=Flin)
    @test SimpleSolvers.method(s7).refactorize == 7
    s1 = NonlinearSolver(Newton(), x, y; F=Flin)
    @test SimpleSolvers.method(s1).refactorize == 1
    # an explicit keyword still wins over the method's field
    s9 = NonlinearSolver(QuasiNewton(7), x, y; F=Flin, refactorize=9)
    @test SimpleSolvers.method(s9).refactorize == 9

    # DogLeg carries a `refactorize` field too
    d1 = NonlinearSolver(DogLeg(), x, y; F=Flin)
    @test SimpleSolvers.method(d1).refactorize == 1
    d4 = NonlinearSolver(DogLeg(4), x, y; F=Flin)
    @test SimpleSolvers.method(d4).refactorize == 4
    d9 = NonlinearSolver(DogLeg(4), x, y; F=Flin, refactorize=9)
    @test SimpleSolvers.method(d9).refactorize == 9
end

@testset "DogLegSolver(x, y; F) convenience form" begin
    Flin(y, x, p) = (y .= x)
    s = DogLegSolver(ones(2), zeros(2); F=Flin)
    @test s isa DogLegSolver
    err = try; DogLegSolver(ones(2), zeros(2)); catch e; e; end
    @test err isa ErrorException && occursin("provide an F", err.msg)
end

@testset "Check whether standard Newton fails and Dogleg works" begin
    function dogleg_test(T::DataType)
        # This example is taken from (Powell, 1970) (the dogleg paper)

        function F(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T}
            @assert length(y) == length(x) == 2
            y[1] = x[1]
            y[2] = 10x[1] / (x[1] + one(T) / 10) + 2(x[2]^2)
        end

        ics(::Type{T}) where {T} = T[3one(T), one(T)]
        root(::Type{T}) where {T} = zeros(T, 2)
        tol(::Type{T}) where {T} = T == Float64 ? eps(T) : eps(T)

        # NewtonSolver: this now converges on the Powell
        # problem.  Previously it *stagnated* at x ≈ [1.108, 0] and that stalled
        # iterate was falsely reported as converged: the backtracking
        # line search shrank α to a denormal and the successive-change
        # convergence criteria treated the resulting zero step as convergence.
        # Fixing the backtracking stall and requiring a small residual for
        # convergence lets Newton escape the stagnation point and reach the
        # true root.

        x0 = ics(T)
        _root = root(T)
        solver = NewtonSolver(x0, F, copy(x0))

        solve!(x0, solver)
        @test ≈(x0, _root; atol=tol(T))

        # PicardSolver cannot solve this problem, but for a principled reason: it is a
        # proper (residual-safeguarded) fixed-point iteration
        # x ← x + α(-F(x)), and the Powell map is not a contraction here, so it stalls
        # at a non-root instead of converging.  Crucially it does *not* diverge to NaN
        # or falsely report convergence (residual gate) — it simply runs out of
        # iterations, so the equality assertion below fails as expected.
        x0 = ics(T)
        solver = PicardSolver(x0, F, copy(x0))

        # Running out of iterations here is deliberate (see above), so we assert
        # the expected "Solver took … iterations." warning rather than letting it
        # leak to the test log.
        @test_logs (:warn, r"Solver took \d+ iterations\.") match_mode = :any solve!(x0, solver)
        @test_throws AssertionError @assert ≈(x0, _root; atol=tol(T))

        x0 = ics(T)
        solver = DogLegSolver(x0, F, copy(x0))#; verbosity=2

        solve!(x0, solver)
        @test ≈(x0, _root; atol=tol(T))
    end

    dogleg_test(Float64)
    dogleg_test(Float32)

end
