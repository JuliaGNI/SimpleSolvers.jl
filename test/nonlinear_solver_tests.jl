using SimpleSolvers
using SimpleSolvers: initialize!, solver_step!, BierlaireQuadratic
using SimpleSolvers: NonlinearSolverState, assess_convergence, residuals, update!, iteration_number
using SimpleSolvers: meets_stopping_criteria, nonlinear_solver_warnings, NonlinearSolverStatus
using SimpleSolvers: linesearch_problem, cache, jacobianmatrix, solution, value, direction, direction!, NullParameters
using SimpleSolvers: trust_radius, DOGLEG_Δ_INITIAL
using SimpleSolvers: isconverged, isstalled, isnotprogressing, spent_without_progress, status
using SimpleSolvers: config, linesearch, stall_number, record_stall!, flag_stall!,
    stalled_step, residual_small, iterate_settled, initial_residual
using SimpleSolvers: iterations_since_progress, record_progress!, no_progress
using SimpleSolvers: compute_new_iterate!, increase_iteration_number!, Bisection, Quadratic,
    StrongWolfe, Backtracking, steplength, solve_with_status
using SimpleSolvers: should_report!, reset_warning_counts!, linesearch_outcomes,
    linesearch_failures, linesearch_index, dominant_linesearch_outcome, record_linesearch!
using Test
using Random
using ForwardDiff
using LinearAlgebra: SingularException

include("lowered_code.jl")

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
    @test rfₐ > config.f_reltol          # but the residual is large
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

    # This bare state was never initialized (`initial_residual` is `NaN`), so the residual
    # gate reduces to the pure absolute test `rfₐ ≤ f_abstol`; the default `f_abstol = 0`
    # rejects the large stalled residual. (The relative `f_reltol·‖F₀‖` path is covered next.)
    @test config.f_abstol == 0
end

@testset "Convergence gate scales with the initial residual" begin
    config = Options()
    g = config.f_reltol

    # Large-magnitude problem: the initial residual ‖F(x₀)‖ = 1e10, and the iterate has
    # settled at an absolute residual of 4e-6 — far above the absolute floor (≈ √eps) but
    # ~15 orders below the initial residual.  The old absolute-only gate rejected this
    # forever; the relative gate f_reltol·‖F(x₀)‖ accepts it.
    conv = NonlinearSolverState([sqrt(2.0)])
    initialize!(conv, [sqrt(2.0)], [1e10])       # records r₀ = 1e10
    update!(conv, [sqrt(2.0)], [4e-6])           # settled iterate, residual ≪ r₀
    update!(conv, [sqrt(2.0)], [4e-6])
    rx, rfa, rfs = residuals(conv)
    @test rfa > g                                # absolute gate alone would reject
    xc, fc, _ = assess_convergence(rx, rfa, rfs, config, conv)
    @test xc || fc                               # ...but the scale-relative gate accepts

    # A step that stalls near its initial residual (rfₐ ≈ r₀) is still NOT converged.
    stall = NonlinearSolverState([1.0])
    initialize!(stall, [1.0], [1e10])            # r₀ = 1e10
    update!(stall, [1.0], [1e10])                # residual did not decrease
    update!(stall, [1.0], [1e10])
    rx2, rfa2, rfs2 = residuals(stall)
    xc2, fc2, _ = assess_convergence(rx2, rfa2, rfs2, config, stall)
    @test !xc2 && !fc2

    # End to end: a Newton solve of the large-magnitude problem now converges to the
    # root instead of running to max_iterations at a "large" (but scale-appropriate)
    # residual.
    Fbig(y, x, p) = (y .= 1e10 .* (x .^ 2 .- 2); y)
    x = [1.0]
    s = NewtonSolver(x, similar(x); F=Fbig, verbosity=0)
    state = NonlinearSolverState(x, similar(x))
    solve!(x, s, state)
    @test isapprox(x[1], sqrt(2.0); atol=1e-7)
    @test iteration_number(state) < config.max_iterations   # stopped by convergence, not the iteration cap
    yb = similar(x)
    Fbig(yb, x, NullParameters())
    @test SimpleSolvers.l2norm(yb) > g                       # ...at a residual above the absolute gate
end

@testset "f_abstol_break stops a diverging residual" begin
    # `f_abstol_break` is the only divergence ("break") tolerance the solver
    # consults: once the absolute residual rfₐ = ‖y‖ exceeds it, the iteration halts
    # even though it has not converged (`meets_stopping_criteria` returns true and a
    # warning is emitted).  It defaults to Inf, so it never fires unless set.
    # (The former unused siblings `x_abstol_break`, `x_reltol_break`,
    # `f_reltol_break` and `g_restol_break` were removed from `Options`.)

    # A settled iterate (rxₛ = rfₛ = 0) with a large, non-converged residual rfₐ = 5.
    state = NonlinearSolverState([1.0, 1.0])
    update!(state, [1.0, 1.0], [3.0, 4.0])
    update!(state, [1.0, 1.0], [3.0, 4.0])
    _, rfₐ, _ = residuals(state)
    @test rfₐ ≈ 5.0

    # A loose break tolerance is not exceeded ⇒ this criterion does not stop the solve.
    loose = Options(; f_abstol_break=10.0, verbosity=0)
    @test !meets_stopping_criteria(state, loose)

    # A tight break tolerance is exceeded ⇒ the solve stops (without convergence) ...
    tight = Options(; f_abstol_break=1.0, verbosity=0)
    @test meets_stopping_criteria(state, tight)
    status = NonlinearSolverStatus(state, tight)
    @test !SimpleSolvers.isconverged(status)

    # ... and the "residual reached the maximally allowed value" warning is emitted.
    @test_logs (:warn, r"residual rfₐ has reached the maximally allowed value") match_mode = :any nonlinear_solver_warnings(status, tight)
end

@testset "allow_f_increases toggles stopping when the residual grows" begin
    # `f_increased` is set when ‖value‖ exceeds ‖previousvalue‖.  With
    # `allow_f_increases=false` a step that increases the residual halts the
    # iteration (and warns); the default (`true`) tolerates it.
    state = NonlinearSolverState([1.0, 1.0])
    update!(state, [1.0, 1.0], [1.0, 0.0])   # value = [1,0]  (‖·‖ = 1)
    update!(state, [1.0, 1.0], [3.0, 4.0])   # value = [3,4]  (‖·‖ = 5) > previous ⇒ increased

    status = NonlinearSolverStatus(state, Options(; verbosity=0))
    @test status.f_increased
    @test !SimpleSolvers.isconverged(status)   # a genuine (non-converged) increase

    # default tolerates the increase ⇒ this criterion alone does not stop
    allow = Options(; allow_f_increases=true, verbosity=0)
    @test !meets_stopping_criteria(state, allow)

    # disallowing increases ⇒ stop (without convergence) ...
    disallow = Options(; allow_f_increases=false, verbosity=0)
    @test meets_stopping_criteria(state, disallow)

    # ... and the "function increased and the solver stopped" warning is emitted.
    st = NonlinearSolverStatus(state, disallow)
    @test_logs (:warn, r"function increased and the solver stopped") match_mode = :any nonlinear_solver_warnings(st, disallow)
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
        # The expansion phase of issue #174 lengthens a step rather than only shrinking it, so it
        # has to be pinned end-to-end and not just at the line search: a `NewtonSolver` asked for
        # it must still land on a root, to the same tolerance as the shrink-only default.
        (NewtonSolver, (linesearch=Backtracking(T; expand=true),), 2),
        (NewtonSolver, (linesearch=Bisection(T),), 2),
        (NewtonSolver, (linesearch=Quadratic(T),), 32),
        # These rows briefly carried `tolfac = 64`, when folding the quadratic searches into
        # `linesearch_max_iterations` cost this seeded start (x₀ = 0.32597672886359486) accuracy.
        # Fixing the single-point stall in the fit removed the cause — those large errors were
        # solves hitting the iteration cap, not converging poorly — so the tolerance is back to
        # the 2 eps every other method meets.  Measured over 300 random starts, the Float64
        # 95th-percentile error is now 0.5 eps with a fresh Jacobian and 1.6 eps with
        # `refactorize = 5` (median 0.50), against 9.7e3 and 8.5e4 eps before.
        (NewtonSolver, (linesearch=BierlaireQuadratic(T),), 2),
        (NewtonSolver, (linesearch=StrongWolfe(T),), 2),
        (NewtonSolver, (linesearch=Static(T), refactorize=5), 2),
        (NewtonSolver, (linesearch=Backtracking(T), refactorize=5), 2),
        (NewtonSolver, (linesearch=Bisection(T), refactorize=5), 2),
        (NewtonSolver, (linesearch=Quadratic(T), refactorize=5), 32),
        (NewtonSolver, (linesearch=BierlaireQuadratic(T), refactorize=5), 2),  # see above
        (NewtonSolver, (linesearch=StrongWolfe(T), refactorize=5), 2),
        # DogLegSolver is a trust-region method: it sets the step length via the
        # trust-region radius, not a line search, so it takes no `linesearch`
        # keyword here.
        (DogLegSolver, (), 2),
        # PicardSolver is a (residual-safeguarded) fixed-point iteration
        # and does not run a derivative-based line search, so it takes
        # no `linesearch` keyword here.
        (PicardSolver, (), 8),
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


# test the convenience wrappers of issue #159: the same solves as the loop above, but with the
# solver assembled by `solve!`/`solve`/`solve_with_status!` from a `NonlinearProblem` instead of
# by the caller.
for T ∈ (Float64, Float32)
    for (solver_method, kwarguments, tolfac) in (
        (Newton(), (linesearch=Static(T),), 2),
        (QuasiNewton(), (linesearch=Static(T),), 2),
        (Picard(), (), 8),
        (DogLeg(), (), 1)
    )

        @testset "Testing solve!/solve/solve_with_status! with method = $(solver_method) & $(kwarguments) & $(T)" begin
            isroot(_x) = ≈(_x, T(root₁); atol=tolfac * eps(T)) || ≈(_x, T(root₂); atol=tolfac * eps(T)) || ≈(_x, T(root₃); atol=tolfac * eps(T)) || ≈(_x, T(root₄); atol=tolfac * eps(T))

            x = T.(copy(x₀))
            y = F(x)
            prob = NonlinearProblem(F!, x, y)

            # `solve!` overwrites its first argument and returns it
            xs = T.(copy(x₀))
            @test solve!(xs, prob, solver_method; verbosity=0, kwarguments...) === xs
            @test all(isroot, xs)

            # `solve` returns a new array and leaves the initial guess untouched
            xi = T.(copy(x₀))
            xn = solve(xi, prob, solver_method; verbosity=0, kwarguments...)
            @test xn !== xi
            @test xi == T.(x₀)
            @test all(isroot, xn)

            # `solve_with_status!` reports the outcome of the solve
            xt = T.(copy(x₀))
            st = solve_with_status!(xt, prob, solver_method; verbosity=0, kwarguments...)
            @test st isa NonlinearSolverStatus
            @test isconverged(st)
            @test all(isroot, xt)

            # a problem carrying its own Jacobian takes the `JacobianFunction` path — asserted on
            # the constructed solver, since the root alone would not distinguish it from autodiff
            probJ = NonlinearProblem(F!, J!, x, y)
            @test SimpleSolvers.jacobian(NonlinearSolver(solver_method, T.(copy(x₀)), probJ; verbosity=0, kwarguments...)) isa JacobianFunction
            xj = T.(copy(x₀))
            solve!(xj, probJ, solver_method; verbosity=0, kwarguments...)
            @test all(isroot, xj)

            # the solver-taking form of `solve_with_status!`, which the problem-taking one calls
            xw = T.(copy(x₀))
            sw = NonlinearSolver(solver_method, xw, prob; verbosity=0, kwarguments...)
            @test isconverged(solve_with_status!(xw, sw))
            @test all(isroot, xw)

            # the state-taking form: same solve, but the caller's state is reused rather than
            # allocated per call, and it is left holding the outcome afterwards — the point of
            # the form, since `status(s, state)` is otherwise the only way to read a solve that
            # keeps its state, and the two must not be able to disagree.
            xv = T.(copy(x₀))
            sv = NonlinearSolver(solver_method, xv, prob; verbosity=0, kwarguments...)
            statev = SolverState(sv)
            stv = solve_with_status!(xv, sv, statev)
            @test stv isa NonlinearSolverStatus
            @test isconverged(stv)
            @test all(isroot, xv)
            @test status(sv, statev) == stv

            # the state is genuinely reused: a second solve from the same guess runs through the
            # same object and reports the same outcome, so nothing carried over from the first.
            xv .= T.(x₀)
            @test solve_with_status!(xv, sv, statev) == stv
        end
    end
end

@testset "The solve! wrapper forwards params, a state and solver keywords" begin
    Fp!(y, x, params) = y .= x .^ 2 .- params.a
    prob = NonlinearProblem(Fp!, zeros(2))

    # `params` is forwarded to the problem
    x = ones(2)
    solve!(x, prob, Newton(), (a=3.0,); verbosity=0)
    @test x ≈ [sqrt(3.0), sqrt(3.0)]

    # so is a state supplied by the caller, which is how the iteration count is obtained
    state = NonlinearSolverState(zeros(2))
    x .= 1
    solve!(x, prob, Newton(), state, (a=3.0,); verbosity=0)
    @test iteration_number(state) > 0
    @test x ≈ [sqrt(3.0), sqrt(3.0)]

    # solver keywords reach the constructor: a single iteration cannot converge, and the
    # methods that consult no line search still reject a `linesearch` keyword (as their
    # constructors do — see "PicardSolver rejects a linesearch keyword" below)
    x .= 1
    st = solve_with_status!(x, prob, Newton(), (a=3.0,); max_iterations=1, verbosity=0)
    @test !isconverged(st)

    # `solve_with_status!` forwards `params` past a state the caller supplies, which is the only
    # thing its state-taking form does beyond the two calls it wraps, and which the solves above
    # cannot see because they take the three-argument form. A dropped `params` reaches `Fp!` as
    # `NullParameters`, which has no `.a`.
    x .= 1
    sp = NonlinearSolver(Newton(), x, prob; verbosity=0)
    statep = SolverState(sp)
    @test isconverged(solve_with_status!(x, sp, statep, (a=3.0,)))
    @test x ≈ [sqrt(3.0), sqrt(3.0)]

    # …and the problem-taking form takes what the `solve!` one takes: a state followed by `params`
    x .= 1
    stw = solve_with_status!(x, prob, Newton(), statep, (a=3.0,); verbosity=0)
    @test isconverged(stw)
    @test iteration_number(statep) > 0
    @test x ≈ [sqrt(3.0), sqrt(3.0)]

    @test_throws MethodError solve!(ones(2), prob, Picard(), (a=3.0,); linesearch=Static(1.0), verbosity=0)
    @test_throws MethodError solve!(ones(2), prob, DogLeg(), (a=3.0,); linesearch=Static(1.0), verbosity=0)
end

# The default residual prototype is `zero(x)` and not `similar(x)`: `alloc_j` broadcasts over it
# (`_nan(T) .* y * x'`), so an uninitialized prototype throws an `UndefRefError` for any element
# type whose `similar` leaves undefined references.  `BigFloat` is one.
@testset "The default residual prototype works for element types with undef-leaving similar" begin
    Fb!(y, x, params) = y .= x .^ 2 .- 2
    xb = BigFloat.(ones(2))
    prob = NonlinearProblem(Fb!, xb)

    @test NewtonSolver(xb, prob) isa NonlinearSolver
    @test PicardSolver(xb, prob) isa NonlinearSolver
    @test DogLegSolver(xb, prob) isa NonlinearSolver

    # …and a solve goes through.  `static=false` is needed only because the default `LU()` caches
    # a small matrix as an `MMatrix`, and StaticArrays cannot `setindex!` a non-isbits eltype —
    # a pre-existing limitation of the linear solver, unrelated to the residual prototype.
    xs = BigFloat.(ones(2))
    solve!(xs, prob, Newton(); linear_solver_method=LU(static=false), verbosity=0)
    @test xs ≈ [sqrt(BigFloat(2)), sqrt(BigFloat(2))]
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
    # by ‖J·JᵀF‖² = 0, which used to produce NaN and throw.  The guard sets
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

@testset "Picard damping commits a residual-checked step (independent of max_iterations)" begin
    # The damping loop is bounded by the step underflow, not by `max_iterations`. Even with
    # `max_iterations = 1` (which used to bound the loop and let it commit an *unchecked*,
    # extra-shrunk α — or here the full residual-increasing step), the committed iterate is
    # one that was actually evaluated and does not increase the residual.
    Fover(y, x, p) = (y .= 3 .* (x .- 1.0))   # full step from x₀ = 2 overshoots to x = -1
    x = [2.0]
    s = PicardSolver(x, Fover, similar(x); max_iterations=1)
    initialize!(s, x)
    y = similar(x)
    Fover(y, x, NullParameters())
    state = NonlinearSolverState(x, y)
    initialize!(state, x, y)
    r₀ = SimpleSolvers.l2norm(value(state))

    solver_step!(x, s, state, NullParameters())
    yₙ = similar(x)
    Fover(yₙ, x, NullParameters())
    @test SimpleSolvers.l2norm(yₙ) ≤ r₀       # committed step did not increase the residual
    @test x[1] ≈ 0.5                          # α damped to 0.5 (the full step to -1 was rejected)
end

@testset "Picard reuses the full-step residual (no redundant F evaluation)" begin
    # On a contraction the full fixed-point step already reduces the residual, so the
    # safeguard accepts α = 1 without re-evaluating F — the α = 1 residual is reused from the
    # NaN-safeguard evaluation. One solver_step! then evaluates F exactly twice: once in
    # `direction!` (F(x)) and once at the full step (F(x + d)); it used to be three (a
    # redundant F(x + d) in the damping loop). Canary against a reintroduced re-evaluation.
    evals = Ref(0)
    Fcos(y, x, p) = (evals[] += 1; y .= x .- cos.(x))
    x = [0.5]
    s = PicardSolver(x, Fcos, similar(x))
    initialize!(s, x)
    y = similar(x)
    Fcos(y, x, NullParameters())
    state = NonlinearSolverState(x, y)
    initialize!(state, x, y)

    evals[] = 0
    solver_step!(x, s, state, NullParameters())
    @test evals[] == 2
end

@testset "DogLeg ρ-based trust region grows on good steps and carries Δ" begin
    # With the full ρ-based radius update (N&W Alg. 4.1) the trust radius is carried
    # across outer steps and *expanded* on good steps that sit on the boundary —
    # the old code reset Δ to DOGLEG_Δ_INITIAL every step and could only shrink it.
    # For a linear residual F(x) = x the Gauss-Newton model is exact (ρ ≈ 1), and
    # starting far from the root (‖Newton step‖ = 5 > DOGLEG_Δ_INITIAL = 1) forces several
    # boundary steps that grow Δ before the full Newton step converges.
    Flin(y, x, p) = (y .= x)
    for T in (Float64, Float32)
        x = T[5.0, 5.0]
        s = DogLegSolver(x, Flin, similar(x))
        ss = SolverState(s)
        @test trust_radius(cache(s)) == T(DOGLEG_Δ_INITIAL)   # reset before solving
        solve!(x, s, ss)
        @test all(v -> isapprox(v, zero(T); atol=10eps(T)), x)  # converged
        @test trust_radius(cache(s)) > T(DOGLEG_Δ_INITIAL)             # radius expanded & carried
    end
end

@testset "DogLeg trust radius resets on solver reuse" begin
    # `initialize!` used to reset every DogLegCache buffer *except* the carried
    # trust radius, so a reused solver started its next solve with the radius the
    # previous solve ended with (up to DOGLEG_Δ_MAX = 1e2) instead of DOGLEG_Δ_INITIAL.
    Flin(y, x, p) = (y .= x)
    x = [5.0, 5.0]
    s = DogLegSolver(x, Flin, similar(x))
    solve!(x, s)
    @test trust_radius(cache(s)) > DOGLEG_Δ_INITIAL    # the first solve expanded Δ ...
    initialize!(s, [5.0, 5.0])
    @test trust_radius(cache(s)) == DOGLEG_Δ_INITIAL   # ... but a fresh solve starts over
    x2 = [5.0, 5.0]
    solve!(x2, s)                               # and solver reuse still converges
    @test all(v -> isapprox(v, 0.0; atol=1e-10), x2)
end

@testset "DogLeg trust-region parameters are configurable via Options" begin
    # The trust-region radius bounds and its shrink/expand factors are `Options`
    # fields (`dogleg_radius_initial`, `dogleg_radius_max`, `dogleg_radius_shrink`,
    # `dogleg_radius_expand`), so problems whose natural scale differs from 1 can tune the region.
    Flin(y, x, p) = (y .= x)
    s = DogLegSolver([5.0, 5.0], Flin, similar([5.0, 5.0]);
                     dogleg_radius_initial=0.5, dogleg_radius_max=4.0,
                     dogleg_radius_shrink=0.1, dogleg_radius_expand=3.0)
    @test config(s).dogleg_radius_initial == 0.5
    @test config(s).dogleg_radius_max == 4.0
    @test config(s).dogleg_radius_shrink == 0.1
    @test config(s).dogleg_radius_expand == 3.0
    initialize!(s, [5.0, 5.0])
    @test trust_radius(cache(s)) == 0.5              # reset uses the configured radius

    x = [5.0, 5.0]
    solve!(x, s)
    @test all(v -> isapprox(v, 0.0; atol=1e-10), x)  # still converges
    @test trust_radius(cache(s)) ≤ 4.0               # never expanded past the configured max

    # The expand factor is genuinely consumed: with `dogleg_radius_expand = 1.0` the
    # radius can never grow, so after solving from far out it stays at the initial
    # radius (a good boundary step multiplies by 1.0).  For linear F the model is
    # exact (ρ ≈ 1), so no shrink fires and the radius is pinned at the initial value.
    s2 = DogLegSolver([5.0, 5.0], Flin, similar([5.0, 5.0]);
                      dogleg_radius_initial=1.0, dogleg_radius_expand=1.0)
    x2 = [5.0, 5.0]
    solve!(x2, s2)
    @test all(v -> isapprox(v, 0.0; atol=1e-10), x2)
    @test trust_radius(cache(s2)) == 1.0             # radius never grew ⇒ expand factor was used
end

@testset "DogLeg treats an undefined (NaN) trial merit as a rejected step" begin
    # F(x) = log(x) + 2 has its root at exp(-2) ≈ 0.135; from x₀ = 1 the full
    # Newton step lands at x = -1, outside the domain (the NaN-returning log
    # mimics e.g. NaNMath.log or a table lookup).  The former NaN recovery
    # rescaled d₁ and d₂ *independently*, destroying the ‖d₁‖ ≤ ‖d₂‖ relation
    # the dogleg interpolation assumes; a NaN trial merit is
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

@testset "DogLeg falls back to steepest descent on a singular Jacobian" begin
    # F(x) = [x₁²; x₂] has Jacobian [2x₁ 0; 0 1], which is singular whenever x₁ = 0.
    # Newton cannot factorize it (SingularException); DogLeg must fall back to a
    # steepest-descent (Cauchy) step instead of letting `ldiv!` throw.  From x₀ = [0,1]
    # the Cauchy step [0,-1] lands exactly on the root [0,0], so the solve succeeds.
    Fsing(y, x, p) = (y[1] = x[1]^2; y[2] = x[2]; y)
    x0 = [0.0, 1.0]                          # Jacobian singular at the start

    nlN = NonlinearSolver(Newton(), copy(x0), zero(x0); F=Fsing, verbosity=0)
    @test_throws SingularException solve!(copy(x0), nlN)

    xD = copy(x0)
    sD = DogLegSolver(xD, Fsing, zero(x0); verbosity=0)
    solve!(xD, sD)                           # must not throw
    yD = zero(x0)
    Fsing(yD, xD, NullParameters())
    @test SimpleSolvers.l2norm(yD) < 1.0     # residual reduced from ‖F(x₀)‖ = 1
    @test isapprox(xD, [0.0, 0.0]; atol=1e-8)
end

@testset "Newton solver_step! damps the direction when the trial value is NaN" begin
    # From x₀ = 1 the full Newton step for F(x) = log(x) + 2 is d = -2, landing at
    # x = -1 where F is NaN (a domain-restricted log, as from NaNMath.log or a table
    # lookup).  The RHS-NaN safeguard in `solver_step!` halves the direction until
    # the trial value is finite *before* the line search runs: d = -2 → -1 (x = 0,
    # still NaN) → -0.5 (x = 0.5, finite).  With a Static line search (α = 1) the
    # accepted iterate is therefore exactly x = 0.5 — proving the loop ran (an
    # undamped step would have been rejected as NaN / left x at -1).
    nanlog(v) = v > 0 ? log(v) : oftype(v, NaN)
    Flog(y, x, p) = (y .= nanlog.(x) .+ 2)
    x = [1.0]
    s = NewtonSolver(x, similar(x); F=Flog, linesearch=Static(), verbosity=0)
    initialize!(s, x)
    y = similar(x)
    Flog(y, x, NullParameters())
    state = NonlinearSolverState(x, y)
    initialize!(state, x, y)

    solver_step!(x, s, state, NullParameters())
    @test x[1] ≈ 0.5                        # two halvings landed the step in-domain
    yₜ = similar(x)
    Flog(yₜ, x, NullParameters())
    @test all(isfinite, yₜ)                 # the accepted trial value is finite

    # ... and a full solve on the same domain-restricted problem converges to the
    # root exp(-2) ≈ 0.135 (each overshoot is caught by the same safeguard).
    x2 = [1.0]
    s2 = NewtonSolver(x2, similar(x2); F=Flog, linesearch=Static(), verbosity=0)
    solve!(x2, s2)
    @test isapprox(x2[1], exp(-2.0); atol=1e-8)
end

@testset "Newton solver_step! damps the direction when the trial value is Inf" begin
    # The companion of the testset above for the *other* way a trial iterate leaves the region
    # where F is representable, and the one issue #130 actually asks about: its fixture
    # 5|x-2|³ - 1/|x| is singular at x = 0, where 1/x overflows to `Inf` rather than going `NaN`.
    # The guard used to test `isnan` only, so an infinite residual broke out of the recovery loop
    # on the first pass and the undamped step was taken.
    #
    # F(x) = 1/x - 1 has its root at x = 1.  From x₀ = 2 the Newton step is exactly d = -2
    # (F = -0.5, J = -1/x² = -0.25), landing exactly on x = 0 in floating point.  One halving
    # gives d = -1, i.e. the root itself.  `Static` evaluates no merit at all, so `nan_recovery!`
    # is the only thing standing between the solver and x = 0 here.
    Finv(y, x, p) = (y .= 1 ./ x .- 1)
    for T in (Float64, Float32)
        x = T[2]
        s = NewtonSolver(x, similar(x); F=Finv, linesearch=Static(), verbosity=0)
        initialize!(s, x)
        y = similar(x)
        Finv(y, x, NullParameters())
        state = NonlinearSolverState(x, y)
        initialize!(state, x, y)

        solver_step!(x, s, state, NullParameters())
        @test x[1] ≈ one(T)                     # one halving landed the step on the root
        @test all(isfinite, value(cache(s)))    # the committed trial value is finite

        # ... and a full solve converges rather than dying: undamped, the next step divides the
        # -Inf residual at x = 0 by the -Inf Jacobian there and throws on the NaN direction.
        x2 = T[2]
        s2 = NewtonSolver(x2, similar(x2); F=Finv, linesearch=Static(), verbosity=0)
        solve!(x2, s2)
        @test isapprox(x2[1], one(T); atol=∛(eps(T)))
    end
end

@testset "a non-finite direction is rejected rather than damped" begin
    # `nan_recovery!` damps by a *factor*, and `Inf * nan_factor` is `Inf`, so a non-finite
    # direction is not something the recovery can shorten — it would spend its whole
    # `nan_max_iterations` budget reproducing the same trial iterate.  The guard in
    # `solver_step!` therefore rejects it outright, as it always has for a `NaN` direction.
    #
    # A Jacobian pinned at the smallest positive subnormal makes d = -F/J overflow exactly:
    # from x₀ = 2 the residual is 1 and the step is -1/5e-324 = -Inf.  Before, this slipped
    # through the `isnan` guard; the line search then saw an infinite directional derivative,
    # reported `LINESEARCH_NO_DESCENT` and left the iterate where it was — a solve that silently
    # made no progress instead of one that said what was wrong.
    Flin(y, x, p) = (y .= x .- 1)
    DFtiny(J, x, p) = (J .= 5e-324)
    x = [2.0]
    s = NewtonSolver(x, similar(x); F=Flin, DF! = DFtiny, verbosity=0)
    initialize!(s, x)
    y = similar(x)
    Flin(y, x, NullParameters())
    state = NonlinearSolverState(x, y)
    initialize!(state, x, y)

    @test_throws NonlinearSolverException solver_step!(x, s, state, NullParameters())

    x2 = [2.0]
    s2 = NewtonSolver(x2, similar(x2); F=Flin, DF! = DFtiny, verbosity=0)
    @test_throws NonlinearSolverException solve!(x2, s2)
end

@testset "nan_factor and nan_max_iterations are honoured" begin
    # The two `Options` that configure `nan_recovery!` had no test at all — neither their
    # defaults nor a non-default value.  The damped iterate encodes both exactly, so this pins
    # them without reaching into the loop.
    Finv(y, x, p) = (y .= 1 ./ x .- 1)

    # `nan_factor` — from x₀ = 2 the Newton step d = -2 lands on the singularity at x = 0.  The
    # default 0.5 halves it to d = -1 (x = 1); 0.25 shortens it to d = -0.5 (x = 1.5) instead.
    for (factor, expected) in ((0.5, 1.0), (0.25, 1.5))
        x = [2.0]
        s = NewtonSolver(x, similar(x); F=Finv, linesearch=Static(), verbosity=0, nan_factor=factor)
        initialize!(s, x)
        y = similar(x)
        Finv(y, x, NullParameters())
        state = NonlinearSolverState(x, y)
        initialize!(state, x, y)
        solver_step!(x, s, state, NullParameters())
        @test x[1] ≈ expected
    end

    # `nan_max_iterations` — the domain-restricted log of the testset above needs *two* halvings
    # (d = -2 → -1, x = 0 still NaN → -0.5, x = 0.5 finite).  A budget of one therefore leaves
    # the loop with a NaN residual, which is the exhaustion path: nothing reports it there, so
    # what has to hold is that the *outer* iteration notices.  It stops on the non-finite
    # residual instead of spending `max_iterations`, does not claim convergence, and says so.
    nanlog(v) = v > 0 ? log(v) : oftype(v, NaN)
    Flog(y, x, p) = (y .= nanlog.(x) .+ 2)
    x = [1.0]
    s = NewtonSolver(x, similar(x); F=Flog, linesearch=Static(), verbosity=0, nan_max_iterations=1)
    state = SolverState(s)
    solve!(x, s, state)
    st = status(s, state)
    @test SimpleSolvers.havenonfinite(st)
    @test !isconverged(st)
    @test iteration_number(state) < config(s).max_iterations

    # ... and the full budget rescues the same solve.
    x2 = [1.0]
    s2 = NewtonSolver(x2, similar(x2); F=Flog, linesearch=Static(), verbosity=0)
    solve!(x2, s2)
    @test isapprox(x2[1], exp(-2.0); atol=1e-8)

    # A budget of zero used to skip the loop body altogether, so the cache still held the trial
    # of the *previous* solver step.  The Picard step documents that it reuses that cache rather
    # than re-evaluating F (`picard_solver.jl`), so it read a stale residual and committed a
    # stale iterate — the sentinel below, from nowhere near the current one.  At least one trial
    # is now always evaluated, so the post-condition `nan_recovery!` advertises always holds.
    x3 = [1.0]
    s3 = PicardSolver(x3, Flog, similar(x3); verbosity=0, nan_max_iterations=0)
    initialize!(s3, x3)
    solution(cache(s3)) .= -99.0
    value(cache(s3)) .= 0.0
    y3 = similar(x3)
    Flog(y3, x3, NullParameters())
    state3 = NonlinearSolverState(x3, y3)
    initialize!(state3, x3, y3)
    solver_step!(x3, s3, state3, NullParameters())
    @test x3[1] ≠ -99.0
    @test all(isfinite, x3)

    # ... and a budget of zero must not damp *either*: refreshing the cache with one trial while
    # still halving the direction on the way out made a budget of zero behave exactly like a
    # budget of one, so this solve landed on the root and reported convergence.  With no damping
    # the full step onto the singularity is taken, and the solve has to fail and say so.
    x4 = [2.0]
    s4 = NewtonSolver(x4, similar(x4); F=Finv, linesearch=Static(), verbosity=0, nan_max_iterations=0)
    state4 = SolverState(s4)
    solve!(x4, s4, state4)
    @test !isconverged(status(s4, state4))
    @test SimpleSolvers.havenonfinite(status(s4, state4))

    # The pairing `nan_recovery!` leaves behind: `value(cache)` is the residual of the trial built
    # from the direction `cache` currently holds.  Damping *after* the failed trial rather than
    # before the next one left the two a factor apart on every exhaustion path, and the Picard
    # step reads them as a pair.
    x5 = [2.0]
    s5 = NewtonSolver(x5, similar(x5); F=Finv, linesearch=Static(), verbosity=0, nan_max_iterations=1)
    initialize!(s5, x5)
    direction!(s5, x5, NullParameters(), 0)
    SimpleSolvers.nan_recovery!(s5, x5, NullParameters())
    @test solution(cache(s5)) ≈ x5 .+ direction(cache(s5))
    @test value(cache(s5)) ≈ [1 / solution(cache(s5))[1] - 1]
end

@testset "a non-finite status stops the solve and is reported" begin
    # `havenonfinite`, the branch of `meets_stopping_criteria` that consults it and the warning
    # that reports it had no test between them.  The `Inf` rows are the ones that used to fail:
    # an overflowed residual passed the `isnan` test, and `rfₐ > f_abstol_break` does not catch
    # it either, since `f_abstol_break` defaults to `Inf` and `Inf > Inf` is false — so such a
    # solve ran its whole `max_iterations` budget with no diagnosis at all.
    for bad in (NaN, Inf)
        config = Options(verbosity=1)
        state = NonlinearSolverState([1.0])
        # Primed with a finite step first: a single `update!` leaves the *previous* iterate and
        # value at the `NaN` the state is allocated with, so `rxₛ` and `rfₛ` would be `NaN`
        # whatever `bad` is and the `Inf` row would pass under the old `isnan`-only predicate too.
        update!(state, [1.0], [1.0])
        update!(state, [2.0], [bad])
        increase_iteration_number!(state)
        st = NonlinearSolverStatus(state, config)

        @test SimpleSolvers.havenonfinite(st)
        @test !isconverged(st)
        @test meets_stopping_criteria(state, config)
        @test logged_any(() -> nonlinear_solver_warnings(st, config), "NaNs or Infs")
        @test !logged_any(() -> nonlinear_solver_warnings(st, Options(verbosity=0)), "NaNs or Infs")
    end

    # An overflowed initial residual is not a usable reference scale.  `residual_small` guarded
    # only `isnan(r₀)`, so `f_reltol * Inf = Inf` made its gate vacuously true and *any* finite
    # residual counted as small — the solve then reported convergence wherever it landed, which
    # is a far worse failure than not converging.  Likewise an infinite step is not a settled
    # one, even though `Inf ≤ ‖x‖·x_suctol` holds once ‖x‖ has overflowed as well.
    state = NonlinearSolverState([1.0])
    SimpleSolvers.initialize!(state, [1.0], [Inf])
    @test !residual_small(1.0e10, Options(), state)

    # `iterate_settled` needs ‖x‖ itself to have overflowed, which is the only fixture that
    # separates the new gate from the old one: with a finite iterate `Inf ≤ ‖x‖·x_suctol` was
    # already false.
    settled_state = NonlinearSolverState([1.0])
    SimpleSolvers.initialize!(settled_state, [Inf], [Inf])
    @test !iterate_settled(Inf, Options(), settled_state)

    # A finite status is untouched by the widening.  Two `update!`s, because a single one leaves
    # the *previous* iterate at the `NaN` the state is allocated with, so `rxₛ` and `rfₛ` are
    # `NaN` until the second — which is exactly why the two consumers above require
    # `iterations ≥ 1` (see the `initialize!` note in the first testset of this file).
    state = NonlinearSolverState([1.0])
    update!(state, [1.0], [1.0])
    update!(state, [1.0], [1.0])
    @test !SimpleSolvers.havenonfinite(NonlinearSolverStatus(state, Options()))
end

@testset "Picard rejects a non-finite direction and commits only finite iterates" begin
    # The Picard direction *is* the residual (d = -F(x)), so a non-finite `F` at the current
    # iterate is a non-finite direction and must be rejected for the same reason the Newton one
    # is: damping cannot shorten it.  `1/x - 1` at x₀ = 0 gives exactly that.
    Finv(y, x, p) = (y .= 1 ./ x .- 1)
    x = [0.0]
    s = PicardSolver(x, Finv, similar(x); verbosity=0)
    @test_throws NonlinearSolverException solve!(x, s)

    # And the commit guard.  `F` is finite on x ≤ 1 and `Inf` beyond it, with a residual large
    # enough that even the underflowed backtracking step α‖d‖ still crosses that wall — so no
    # trial the step evaluates has a finite residual, and the last one would be committed.  The
    # trial *iterate* is finite throughout (x is, and the direction is past the guard above),
    # which is why the guard has to test `value(cache)` rather than `solution(cache)`.  A frozen
    # iterate is the honest outcome — `stalled_step` diagnoses it — and it is what lets the solve
    # be reported rather than handing the caller an x whose residual is `Inf`.
    Fwall(y, x, p) = (y .= [xi ≤ 1 ? -1.0e10 : oftype(xi, Inf) for xi in x])
    x2 = [1.0]
    s2 = PicardSolver(x2, Fwall, similar(x2); verbosity=0)
    state2 = SolverState(s2)
    solve!(x2, s2, state2)
    @test x2 == [1.0]                                            # the poisoned trial was not committed
    @test !SimpleSolvers.havenonfinite(status(s2, state2))
    @test isstalled(status(s2, state2), config(s2))              # ... and the solve says why
end

@testset "DogLeg recovers from a collapsed trust-region radius" begin
    # Once the carried trust radius underflowed (Δ ≤ eps), the next
    # solver_step!'s `while Δ > eps(T)` never ran, so the iterate froze and the
    # solve spun to max_iterations with no progress and no failure signal.  This is
    # reachable in quasi-Newton mode (refactorize > 1), where a *stale* Jacobian's
    # steepest-descent direction need not reduce ‖F‖².  A step that enters with a
    # collapsed radius resets Δ and forces a fresh Jacobian, so it makes
    # progress instead of freezing.
    Fnl(y, x, p) = (y .= [x[1]^2 - 2, x[2]^2 - 3])   # root (√2, √3)
    for T in (Float64, Float32)
        x = T[3.0, 3.0]
        s = DogLegSolver(x, Fnl, similar(x); refactorize=3)
        y = similar(x)
        Fnl(y, x, NullParameters())
        state = NonlinearSolverState(x, y)
        initialize!(s, x)
        initialize!(state, x, y)

        # Collapse the carried radius, as a poor (stale-Jacobian) step would have.
        SimpleSolvers.trust_radius!(cache(s), eps(T) / 2)
        r₀ = SimpleSolvers.l2norm(value(state))
        x_before = copy(x)

        solver_step!(x, s, state, NullParameters())

        @test trust_radius(cache(s)) > eps(T)   # radius recovered to a workable value
        @test x != x_before                     # the iterate moved (did not freeze)
        yₜ = similar(x)
        Fnl(yₜ, x, NullParameters())
        @test SimpleSolvers.l2norm(yₜ) < r₀     # the recovered step reduced the residual

        # ... and a full quasi-Newton (refactorize > 1) solve converges end to end.
        x2 = T[3.0, 3.0]
        s2 = DogLegSolver(x2, Fnl, similar(x2); refactorize=3)
        solve!(x2, s2)
        @test isapprox(x2[1], sqrt(T(2)); atol=∛(eps(T))) &&
              isapprox(x2[2], sqrt(T(3)); atol=∛(eps(T)))
    end
end

@testset "DogLeg does not commit a merit-increasing step on radius underflow" begin
    # Reachable in quasi-Newton mode: a stale Jacobian can make both dogleg legs ascend the
    # merit ‖F‖², so every trust-region trial is rejected and Δ underflows without an
    # acceptable step. The last (smallest-Δ) trial then *increases* the merit; committing it
    # would violate monotonicity. The step must instead leave the iterate unchanged.
    Ftrue(y, x, p) = (y .= [x[1]^2 - 2, x[2]^2 - 3])
    x0 = [3.0, 3.0]
    s = DogLegSolver(copy(x0), Ftrue, similar(x0); refactorize=100, verbosity=0)  # won't refactor after step 1
    initialize!(s, x0)

    # Prime a stale, sign-flipped Jacobian J = -J_true(x0): then d₁ = -Jᵀ F and d₂ = J⁻¹(-F)
    # both point *up* the merit, so the model (built from the stale J) predicts a decrease
    # while the actual merit rises — every trial is rejected (ρ < 0).
    Jstale = [-2*x0[1] 0.0; 0.0 -2*x0[2]]
    jacobianmatrix(cache(s)) .= Jstale
    SimpleSolvers.factorize!(SimpleSolvers.linearsolver(s), Jstale)

    y = similar(x0)
    Ftrue(y, x0, NullParameters())
    state = NonlinearSolverState(x0, y)
    initialize!(state, x0, y)
    state.iterations = 5                          # mod(5,100)≠0 & >1 ⇒ directions! keeps the stale J
    SimpleSolvers.trust_radius!(cache(s), 1.0)    # > eps ⇒ the shrink loop runs to underflow

    r₀ = SimpleSolvers.l2norm(value(state))
    x = copy(x0)
    solver_step!(x, s, state, NullParameters())

    yₙ = similar(x)
    Ftrue(yₙ, x, NullParameters())
    @test x == x0                            # rejected underflow step leaves the iterate unchanged
    @test SimpleSolvers.l2norm(yₙ) ≤ r₀      # ...so the merit is not increased
end

@testset "PicardSolver rejects a linesearch keyword" begin
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

    # `min_iterations = 1` is needed to reach the pathology at all: `Fnan(0) = exp(-Inf) = 0`
    # exactly, so x = 0 *is* a root and `solve!` now tests the stopping criteria before the
    # first step (see the `while !meets_stopping_criteria(…)` loop) and returns without
    # computing a direction.  Forcing one iteration restores the path under test.
    nl₁ = NonlinearSolver(Newton(), x, y; F=Fnan, jacobian=J₁, verbosity=2, min_iterations=1)
    nl₂ = NonlinearSolver(Newton(), x, y; F=Fnan, jacobian=J₂, verbosity=2, min_iterations=1)

    x₁ = zeros(T, n)
    x₂ = zeros(T, n)

    # The solver must refuse to proceed on this pathological problem.  The finite
    # difference Jacobian at x = 0 is the zero matrix, which is singular: the LU
    # solver throws a `SingularException` instead of silently returning
    # NaN.  The autodiff Jacobian produces NaN entries, which is caught as a
    # `NonlinearSolverException` (NaN in the direction vector).
    @test_throws SingularException solve!(x₁, nl₁)
    @test_throws NonlinearSolverException solve!(x₂, nl₂)

    # Without the forced iteration the initial guess is recognised as a root and returned
    # untouched — no Jacobian, no direction, no line search.
    nl₃ = NonlinearSolver(Newton(), x, y; F=Fnan, jacobian=J₂, verbosity=0)
    x₃ = zeros(T, n)
    state₃ = SolverState(nl₃)
    @test solve!(x₃, nl₃, state₃) == zeros(T, n)
    @test iteration_number(state₃) == 0
    @test isconverged(status(nl₃, state₃))

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

# check that `check_jacobian(s)` / `print_jacobian(s)` are
# forwarded to `jacobian(s)` (the `Jacobian` functor *object*) instead of the
# Jacobian *matrix*, so both exported convenience methods threw a `MethodError`
# on every call (`check_jacobian` has only an `::AbstractMatrix` method, and
# `print_jacobian` had no base method at all after the Jacobian-object refactor).
@testset "check_jacobian / print_jacobian operate on the Jacobian matrix from the Newton solver" begin
    f!(y, x, params) = y .= x .^ 2 .- 1
    x = [2.0]
    y = [3.0]
    s = NewtonSolver(x, y; F=f!)
    solve!(x, s)   # populate the cached Jacobian matrix
    Jm = jacobianmatrix(s)
    # the solver forms forward to the `::AbstractMatrix` methods on the cached
    # Jacobian matrix, so their captured output matches the matrix form exactly
    # (this is the bug that was fixed: they used to hit the `Jacobian` functor).
    @test sprint(check_jacobian, s) == sprint(check_jacobian, Jm)
    @test sprint(print_jacobian, s) == sprint(print_jacobian, Jm)
    # and that output is the genuine diagnostic / table (not empty, not an error)
    @test occursin("Condition Number of Jacobian:", sprint(check_jacobian, s))
    @test sprint(print_jacobian, s) == repr("text/plain", Jm) * "\n"
    # the convenience solver forms without `io` write to stdout (called silently)
    @test redirect_stdout(() -> check_jacobian(s), devnull) === nothing
    @test redirect_stdout(() -> print_jacobian(s), devnull) === nothing
    # dispatch sanity
    @test hasmethod(check_jacobian, Tuple{typeof(s)})
    @test hasmethod(print_jacobian, Tuple{typeof(s)})
    # the base matrix methods exist and are what the solver forms delegate to
    @test hasmethod(check_jacobian, Tuple{AbstractMatrix})
    @test hasmethod(print_jacobian, Tuple{AbstractMatrix})
end

# Interface-consistency fixes:
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

        x0 = ics(T)
        _root = root(T)
        solver = NewtonSolver(x0, F, copy(x0))

        solve!(x0, solver)
        @test ≈(x0, _root; atol=tol(T))

        x0 = ics(T)
        solver = PicardSolver(x0, F, copy(x0))

        # PicardSolver runs out of iterations on this problem (expected); assert the warning is
        # emitted rather than letting it leak to the test log. The fixed-point map is locally
        # expanding here, so the residual safeguard damps α to nothing and the residual sits at
        # 10.4 for all 1000 iterations — a solve that makes no progress, which is what is now
        # reported instead of the bare "Solver took … iterations." (see `spent_without_progress`).
        # The report is rate limited by `should_report!` (1st, 2nd, 4th … occurrence), and unlike
        # `maxlog` — which suppresses inside the logger, where `TestLogger` still sees it — a
        # suppressed message is never emitted, so the count has to start from a known point.
        reset_warning_counts!()
        @test_logs (:warn, r"spent its full budget .* did not improve") match_mode = :any solve!(x0, solver)
        @test_throws AssertionError @assert ≈(x0, _root; atol=tol(T))

        x0 = ics(T)
        solver = DogLegSolver(x0, F, copy(x0))#; verbosity=2

        solve!(x0, solver)
        @test ≈(x0, _root; atol=tol(T))
    end

    dogleg_test(Float64)
    dogleg_test(Float32)

end

@testset "$(rpad("the line search shares the solver's Options", 80))" begin
    # Before, every solver built its `Linesearch` with an `Options` of its own, constructed
    # from nothing but defaults.  So `verbosity = 0` on the solver did not silence the line
    # search (downstream packages had to swallow the messages with a `NullLogger`), and a
    # user-supplied iteration budget never reached the inner ladder.
    F(y, x, params) = y .= x .^ 2 .- 2.0
    x = [1.0]
    y = zero(x)

    for make in (NewtonSolver, PicardSolver, DogLegSolver)
        s = make(x, F, y; verbosity=0, max_iterations=17, linesearch_max_iterations=9)
        @test config(linesearch(s)) === config(s)
        @test config(linesearch(s)).verbosity == 0
        @test config(linesearch(s)).linesearch_max_iterations == 9
    end

    # ... and the same holds for the constructor that is handed a ready-made `Linesearch`
    # carrying its own (default) `Options`: it is rebuilt on the solver's config.
    s = NewtonSolver(x, F, y; linesearch=Backtracking(), verbosity=0)
    @test config(linesearch(s)) === config(s)
    @test config(linesearch(s)).verbosity == 0
end

@testset "$(rpad("verbosity = 0 silences a stagnating solve completely", 80))" begin
    # The regression test for the reported warning flood: a residual whose round-off floor
    # (≈ 1e-8 here, from the cancellation in `(1e8 + x) - 1e8`) lies far above the requested
    # `f_abstol` cannot be driven to the tolerance.  Before, this produced one line-search
    # warning per iteration for `max_iterations` iterations plus a "Solver took 1000
    # iterations." message, and none of it could be silenced through the public API.
    Ffloor(y, x, params) = y .= ((1e8 .+ x) .- 1e8) .- 1e-9

    x = [1.0]
    s = NewtonSolver(x, Ffloor, zero(x); f_abstol=1e-20, f_reltol=0.0, verbosity=0)
    state = SolverState(s)
    @test_logs solve!(x, s, state)          # no messages at all

    st = status(s, state)
    @test isstalled(st, config(s))
    @test !isconverged(st)
    @test iteration_number(state) ≤ 4       # was `max_iterations` (1000)
    @test iteration_number(state) < config(s).max_iterations
    @test st.rfₐ ≈ 1e-9 rtol = 1e-6         # the achievable floor is reported
    @test st.stalls == config(s).max_stalls

    # stagnation and convergence are mutually exclusive
    @test !(isconverged(st) && isstalled(st, config(s)))

    # at the default verbosity the stagnation warning names the achieved residual and the
    # requested tolerance
    x2 = [1.0]
    s2 = NewtonSolver(x2, Ffloor, zero(x2); f_abstol=1e-20, f_reltol=0.0)
    reset_warning_counts!()
    @test_logs (:warn, r"stagnated") match_mode = :any solve!(x2, s2, SolverState(s2))

    # ... and it names what the line search reported, which is the only trace the line search
    # leaves: it does not log from inside the iteration, it reports to the solver (see
    # `record_linesearch!`). Without this clause the message would name the symptom and no cause.
    x2b = [1.0]
    s2b = NewtonSolver(x2b, Ffloor, zero(x2b); f_abstol=1e-20, f_reltol=0.0)
    state2b = SolverState(s2b)
    reset_warning_counts!()
    @test_logs (:warn, r"The line search reported LINESEARCH_FLOOR on \d+ of the \d+ step\(s\)") match_mode = :any solve!(x2b, s2b, state2b)

    # The same information, programmatically — this is what a caller acts on, rather than the log.
    stb = status(s2b, state2b)
    @test linesearch_failures(stb) > 0
    @test dominant_linesearch_outcome(stb) === LINESEARCH_FLOOR
    @test linesearch_outcomes(stb)[linesearch_index(LINESEARCH_FLOOR)] > 0
    @test sum(linesearch_outcomes(stb)) == iteration_number(state2b)

    # And no line-search message escapes the solve itself: the flood the caps used to hold back
    # is gone at its source, not rate limited.
    x2c = [1.0]
    s2c = NewtonSolver(x2c, Ffloor, zero(x2c); f_abstol=1e-20, f_reltol=0.0, verbosity=2)
    reset_warning_counts!()
    logs2c, _ = Test.collect_test_logs() do
        solve!(x2c, s2c, SolverState(s2c))
    end
    @test !any(r -> occursin("line search: ", string(r.message)), logs2c)

    # ... and it *replaces* the bare iteration count rather than joining it. This solve stagnates
    # after three iterations, so it only reaches `warn_iterations` when that is set below them;
    # the two messages would otherwise never be seen together and the mutual exclusivity
    # `nonlinear_solver_warnings` promises would go untested.
    x4 = [1.0]
    s4 = NewtonSolver(x4, Ffloor, zero(x4); f_abstol=1e-20, f_reltol=0.0, warn_iterations=1)
    reset_warning_counts!()
    logs, _ = Test.collect_test_logs() do
        solve!(x4, s4, SolverState(s4))
    end
    messages = [string(record.message) for record in logs]
    @test any(m -> occursin("stagnated", m), messages)
    @test !any(m -> occursin("Solver took", m), messages)

    # raising `f_abstol` above the floor makes the very same problem converge, quietly
    x3 = [1.0]
    s3 = NewtonSolver(x3, Ffloor, zero(x3); f_abstol=1e-6, f_reltol=0.0)
    state3 = SolverState(s3)
    @test_logs solve!(x3, s3, state3)
    @test isconverged(status(s3, state3))
    @test !isstalled(status(s3, state3), config(s3))
    @test stall_number(state3) == 0
end

@testset "$(rpad("stall predicates and the consecutive-stall counter", 80))" begin
    config₀ = Options(Float64; f_abstol=1e-10, f_reltol=0.0, x_suctol=1e-12)

    # a frozen iterate whose residual is *large*: stagnation
    state = NonlinearSolverState([1.0])
    initialize!(state, [1.0], [1.0])          # r₀ = 1
    update!(state, [1.0], [1.0])              # x̄ = x, ȳ = y ⇒ rxₛ = 0, rfₐ = 1
    rxₛ, rfₐ, _ = residuals(state)
    @test iterate_settled(rxₛ, config₀, state)
    @test !residual_small(rfₐ, config₀, state)
    @test stalled_step(rxₛ, rfₐ, config₀, state)

    # ... and the same frozen iterate with a *small* residual is convergence, not stagnation
    update!(state, [1.0], [0.0])
    rxₛ₂, rfₐ₂, _ = residuals(state)
    @test iterate_settled(rxₛ₂, config₀, state)
    @test residual_small(rfₐ₂, config₀, state)
    @test !stalled_step(rxₛ₂, rfₐ₂, config₀, state)

    # the counter increments on consecutive stalls and resets on progress
    state2 = NonlinearSolverState([1.0])
    initialize!(state2, [1.0], [1.0])
    @test stall_number(state2) == 0
    update!(state2, [1.0], [1.0])
    @test record_stall!(state2, config₀) == 1
    @test record_stall!(state2, config₀) == 2
    update!(state2, [5.0], [1.0])             # the iterate moved
    @test record_stall!(state2, config₀) == 0

    # a line-search floor flag counts as a stall, but only while the residual is not small
    update!(state2, [5.0], [1.0])
    flag_stall!(state2)
    @test record_stall!(state2, config₀) == 1
    update!(state2, [9.0], [0.0])             # residual small ⇒ success, not stagnation
    flag_stall!(state2)
    @test record_stall!(state2, config₀) == 0
end

@testset "$(rpad("progress predicates and the no-progress counter", 80))" begin
    config₀ = Options(Float64; f_abstol=1e-10, f_reltol=0.0, f_stall_window=3)

    state = NonlinearSolverState([1.0])
    initialize!(state, [1.0], [1.0])          # r₀ = 1 is the first progress reference
    @test iterations_since_progress(state) == 0

    # an iteration that does not reduce the residual by `f_stall_factor` does not reset the clock
    for i in 1:3
        state.iterations = i
        update!(state, [Float64(i)], [0.9])   # a 10 % improvement is not progress at 0.5
        @test record_progress!(state, config₀) == i
    end

    # ... and one that does resets it, from where it happened
    state.iterations = 4
    update!(state, [4.0], [0.4])
    @test record_progress!(state, config₀) == 0
    state.iterations = 5
    update!(state, [5.0], [0.4])
    @test record_progress!(state, config₀) == 1

    # the reference is the *best* residual so far, so undoing progress does not reset the clock
    state.iterations = 6
    update!(state, [6.0], [0.8])              # worse than the 0.4 reference
    @test record_progress!(state, config₀) == 2
    state.iterations = 7
    update!(state, [7.0], [0.8])
    @test record_progress!(state, config₀) == 3

    # the criterion fires once the window is full, and only while the residual is not small
    _, rfₐ, _ = residuals(state)
    @test iterations_since_progress(state) == config₀.f_stall_window
    @test !residual_small(rfₐ, config₀, state)
    @test no_progress(rfₐ, config₀, state)
    @test !no_progress(rfₐ, Options(Float64; f_abstol=1e10, f_stall_window=3), state)  # residual small
    @test !no_progress(rfₐ, Options(Float64; f_abstol=1e-10, f_reltol=0.0), state)     # window disabled

    # the report threshold is independent of the window: half the iterations without progress,
    # and at least `F_STALL_REPORT_MINIMUM` of them
    status₁ = NonlinearSolverStatus(state, config₀)
    @test status₁.iterations_since_progress == 3
    @test status₁.iterations == 7
    @test !spent_without_progress(status₁)    # 3 of 7: neither half nor enough

    # a fourth unproductive iteration makes it half — and still too few to mean anything
    state.iterations = 8
    update!(state, [8.0], [0.8])
    @test record_progress!(state, config₀) == 4
    @test 2 * iterations_since_progress(state) ≥ iteration_number(state)
    @test !spent_without_progress(NonlinearSolverStatus(state, config₀))

    # ... and enough of them satisfies both
    while iterations_since_progress(state) < SimpleSolvers.F_STALL_REPORT_MINIMUM
        state.iterations += 1
        update!(state, [Float64(state.iterations)], [0.8])
        @test record_progress!(state, config₀) == iterations_since_progress(state)
    end
    @test 2 * iterations_since_progress(state) ≥ iteration_number(state)
    @test spent_without_progress(NonlinearSolverStatus(state, config₀))

    # the proportion is the independent half of the threshold: the *same* run of unproductive
    # iterations, after a long productive stretch, is a solve that got somewhere and is not
    # reported for the tail it ended on
    long = NonlinearSolverState([1.0])
    initialize!(long, [1.0], [1.0])
    for i in 1:30                             # thirty iterations, every one of them progress
        long.iterations = i
        update!(long, [Float64(i)], [0.5^i])
        @test record_progress!(long, config₀) == 0
    end
    for i in 31:(30+SimpleSolvers.F_STALL_REPORT_MINIMUM)
        long.iterations = i
        update!(long, [Float64(i)], [0.5^30])
        record_progress!(long, config₀)
    end
    @test iterations_since_progress(long) == SimpleSolvers.F_STALL_REPORT_MINIMUM
    @test !isconverged(NonlinearSolverStatus(long, config₀))
    @test 2 * iterations_since_progress(long) < iteration_number(long)
    @test !spent_without_progress(NonlinearSolverStatus(long, config₀))

    # ... and a *converged* solve is never reported however lopsided the proportion. That is the
    # guard `show` needs, having no `Options` with which to check the budget the warning checks:
    # a plateau the solve sits on for most of its iterations is stagnation only if it is above
    # the tolerance. Here the same 0.8 plateau is read against a tolerance that accepts it, and
    # two identical iterates make both successive residuals vanish, so the solve has converged.
    loose = Options(Float64; f_abstol=1.0, f_reltol=0.0, f_stall_window=3)
    for _ in 1:2
        state.iterations += 1
        update!(state, [9.0], [0.8])
        record_progress!(state, loose)        # still not progress: the plateau does not move
    end
    converged = NonlinearSolverStatus(state, loose)
    @test isconverged(converged)
    @test converged.iterations_since_progress ≥ SimpleSolvers.F_STALL_REPORT_MINIMUM
    @test 2 * converged.iterations_since_progress ≥ converged.iterations
    @test !spent_without_progress(converged)
    @test !occursin("no progress", sprint(show, converged))

    # the absolute minimum is the backstop for the solve that is short rather than converged: two
    # iterations of which the last did not halve the residual satisfies the proportion on its own
    short = NonlinearSolverState([1.0])
    initialize!(short, [1.0], [1.0])
    short.iterations = 1
    update!(short, [2.0], [0.9])
    @test record_progress!(short, config₀) == 1
    short.iterations = 2
    @test !isconverged(NonlinearSolverStatus(short, config₀))
    @test 2 * iterations_since_progress(short) ≥ iteration_number(short)
    @test !spent_without_progress(NonlinearSolverStatus(short, config₀))

    # a freshly initialized state is neither, so a fresh status prints unchanged
    fresh = NonlinearSolverState([1.0])
    @test !spent_without_progress(NonlinearSolverStatus(fresh, config₀))
end

@testset "$(rpad("an iteration that never records keeps both counters at zero", 80))" begin
    # `record_stall!` and `record_progress!` are increments, not predicates: the counters are
    # fields of the state rather than differences of iteration numbers, so a hand-rolled loop
    # that drives `solver_step!` directly and never calls `record_iteration!` behaves exactly as
    # it did before either counter existed. Were `iterations_since_progress` derived from
    # `iteration_number`, such a loop would report every one of its iterations as unproductive —
    # and with `f_stall_window` set, `meets_stopping_criteria` would cut it short for it.
    config₀ = Options(Float64; f_abstol=1e-10, f_reltol=0.0, f_stall_window=5)

    state = NonlinearSolverState([1.0])
    initialize!(state, [1.0], [1.0])
    for i in 1:15
        increase_iteration_number!(state)
        update!(state, [Float64(i)], [1.0])   # the residual never moves, and nobody records it
    end

    @test iteration_number(state) == 15
    @test iterations_since_progress(state) == 0
    @test stall_number(state) == 0

    st = NonlinearSolverStatus(state, config₀)
    @test !spent_without_progress(st)
    @test !isnotprogressing(st)
    @test !occursin("no progress", sprint(show, st))
    @test !meets_stopping_criteria(state, config₀)

    # ... whereas recording those same iterations does report them
    for _ in 1:15
        record_progress!(state, config₀)
    end
    @test iterations_since_progress(state) == 15
    @test spent_without_progress(NonlinearSolverStatus(state, config₀))
end

@testset "$(rpad("a solve that moves but gets nowhere is reported, and can be stopped", 80))" begin
    # The gap in the stall machinery reported in issue #173: the iterate keeps moving normally
    # while the residual sits on a floor far above the requested tolerance. Here the floor is the
    # whole residual — `F` is constant, so no step can reduce it — while the Picard step
    # `d = -F` moves the iterate by 1e-3 every iteration, which is twelve orders of magnitude
    # above the round-off level of `x`. By construction neither convergence branch nor
    # `stalled_step` can fire, so nothing used to end this solve or explain it.
    Ffloor(y, x, params) = y .= 1e-3

    x = [0.0]
    s = PicardSolver(x, Ffloor, zero(x); verbosity=0)
    state = SolverState(s)
    @test_logs solve!(x, s, state)                      # verbosity = 0 silences it completely
    st = status(s, state)

    @test !isconverged(st)
    @test !isstalled(st, config(s))                     # the iterate never froze ...
    @test stall_number(state) == 0                      # ... so nothing was ever counted
    @test iteration_number(state) == config(s).max_iterations
    @test iterations_since_progress(state) == config(s).max_iterations
    @test spent_without_progress(st)
    @test !isnotprogressing(st)                         # the criterion is off by default
    @test st.rfₐ ≈ 1e-3

    # at the default verbosity the report names the residual and the missing progress, in place
    # of the bare "Solver took … iterations." that used to be the only signal
    x2 = [0.0]
    s2 = PicardSolver(x2, Ffloor, zero(x2))
    reset_warning_counts!()
    @test_logs (:warn, r"spent its full budget .* did not improve by the factor") match_mode = :any solve!(x2, s2, SolverState(s2))

    # `f_stall_window` turns the report into a stopping criterion
    x3 = [0.0]
    s3 = PicardSolver(x3, Ffloor, zero(x3); f_stall_window=20, verbosity=0)
    state3 = SolverState(s3)
    @test_logs solve!(x3, s3, state3)                   # ... including when it stops the solve
    st3 = status(s3, state3)

    @test isnotprogressing(st3)
    @test !isconverged(st3)                             # giving up is not converging
    @test iteration_number(state3) == 20
    @test iteration_number(state3) < config(s3).max_iterations

    x4 = [0.0]
    s4 = PicardSolver(x4, Ffloor, zero(x4); f_stall_window=20)
    reset_warning_counts!()
    @test_logs (:warn, r"gave up after 20 iterations") match_mode = :any solve!(x4, s4, SolverState(s4))

    # A slow but healthy solve must survive the same window: this one contracts by 0.9 per
    # iteration and needs 316 of them, far more than the window, but it halves its residual every
    # seven — which is what `f_stall_factor` measures and `max_iterations` alone cannot tell
    # apart. This is why the window is opt-in; see `F_STALL_WINDOW`.
    Fslow(y, x, params) = y .= (1 - 0.9) .* (x .- 1.0)

    x5 = [0.0]
    s5 = PicardSolver(x5, Fslow, zero(x5); f_stall_window=20)
    state5 = SolverState(s5)
    @test_logs solve!(x5, s5, state5)                   # converges, quietly
    st5 = status(s5, state5)

    @test isconverged(st5)
    @test !isnotprogressing(st5)
    @test !spent_without_progress(st5)
    @test iteration_number(state5) > 20                 # ... despite taking far longer than the window
    @test x5[1] ≈ 1.0
end

@testset "$(rpad("pre-step convergence check and the merit-evaluation canary", 80))" begin
    # An initial guess that already satisfies the stopping criteria must not be perturbed by a
    # full solver step — including a line search asked to improve an already-exact residual.
    n = Ref(0)
    Fexact(y, x, params) = (n[] += 1; y .= x .- 1.0)

    x = [1.0]
    s = NewtonSolver(x, Fexact, zero(x); verbosity=0)
    state = SolverState(s)
    n[] = 0
    solve!(x, s, state)
    @test iteration_number(state) == 0
    @test n[] == 1                 # the single residual evaluation of `initialize!`
    @test isconverged(status(s, state))
    @test x == [1.0]

    # `min_iterations` still forces a step
    x2 = [1.0]
    s2 = NewtonSolver(x2, Fexact, zero(x2); verbosity=0, min_iterations=1)
    state2 = SolverState(s2)
    solve!(x2, s2, state2)
    @test iteration_number(state2) == 1

    # The α = 0 anchor — which every line search evaluates first — costs no residual
    # evaluation when the caller supplies the merit it has already computed as `params.φ₀`
    # (`solver_step!` does).  Only α = 0 is short-circuited; every other trial step is
    # evaluated as before.
    Fcount(y, x, params) = (n[] += 1; y .= x .^ 2 .- 2.0)
    x3 = [1.0]
    s3 = NewtonSolver(x3, Fcount, zero(x3); verbosity=0)
    state3 = NonlinearSolverState(x3)
    initialize!(s3, x3)
    initialize!(state3, x3, [-1.0])
    direction!(s3, x3, NullParameters(), 1)
    prob = SimpleSolvers.problem(linesearch(s3))

    shared = (x=x3, parameters=NullParameters(), φ₀=1.0)
    plain = (x=x3, parameters=NullParameters())

    n[] = 0
    @test value(prob, 0.0, shared) == 1.0
    @test n[] == 0                       # the anchor is taken from `params.φ₀`

    n[] = 0
    @test value(prob, 0.0, plain) == 1.0  # ... and is otherwise computed, to the same value
    @test n[] == 1

    n[] = 0
    value(prob, 0.5, shared)
    @test n[] == 1                       # a trial step α ≠ 0 is always evaluated

    # a whole solver step evaluates the residual a bounded number of times (the residual is
    # also evaluated through the autodiff Jacobian, twice here)
    n[] = 0
    solver_step!(x3, s3, state3, NullParameters())
    @test n[] == 6
end

@testset "$(rpad("no line search returns α ≤ 0 during a Newton solve", 80))" begin
    # `Bisection` and `Quadratic` inherit a direction flip from `bracket_minimum`, which used to
    # let them return a *negative* step inside a Newton solve — measured at up to α = -3 for
    # Bisection (49 of ~3750 line-search calls with refactorize = 5).  A negative α steps against
    # a direction that has already been chosen, so the α > 0 contract now forbids it.  Driving
    # the solver loop by hand is the only way to observe every α the line search returns.
    fscalar(x::T) where {T<:Number} = exp(x) * (x^3 - 5x^2 + 2x) + 2one(T)
    Fls!(y, x, params) = y .= fscalar.(x)

    Random.seed!(4321)
    for lsmethod in (Bisection, Quadratic, BierlaireQuadratic, Backtracking, StrongWolfe)
        for refac in (1, 5)
            for _ in 1:20
                x = rand(1)
                nl = NewtonSolver(x, similar(x); F=Fls!, linesearch=lsmethod(Float64),
                    verbosity=0, refactorize=refac)
                SimpleSolvers.initialize!(nl, x)
                state = NonlinearSolverState(x)
                SimpleSolvers.initialize!(state, x, [fscalar(x[1])])
                for it in 1:12
                    SimpleSolvers.increase_iteration_number!(state)
                    direction!(nl, x, NullParameters(), it)
                    any(isnan, direction(cache(nl))) && break
                    α = solve(SimpleSolvers.linesearch(nl), 1.0, (x=x, parameters=NullParameters()))
                    @test α > 0.0
                    compute_new_iterate!(x, α, direction(cache(nl)))
                end
            end
        end
    end
end

@testset "$(rpad("BierlaireQuadratic no longer aborts a solve", 80))" begin
    # The exact starting points measured to throw
    # `ERROR: The function f must be decreasing at 0.0` out of `triple_point_finder`, which
    # aborted the whole solve.  Two distinct causes: an ascent anchor from a stale Jacobian
    # (refactorize = 5) and a merit flat to round-off (Float32, refactorize = 1).
    #
    # What must hold for *every* one of them is that the bracketing failure is reported rather
    # than raised.  Two of the Float32 quasi-Newton starts still fail — but through the
    # pre-existing, legitimate channel of a direction vector that goes `NaN`/`Inf` under a stale
    # Jacobian, which `solver_step!` raises deliberately and which this change does not touch.
    fscalar(x::T) where {T<:Number} = exp(x) * (x^3 - 5x^2 + 2x) + 2one(T)
    Fb!(y, x, params) = y .= fscalar.(x)
    roots = (-4.735035753706987262178160540350200552633, -0.6737697823920028217727631890832279199433,
        0.7613128434711647120463439168731683731732, 4.560440205363600153577140702025401006278)

    fixtures = ((Float64, 5, 0.1440401297), (Float64, 5, 0.2834847806), (Float64, 5, 0.2831226663),
                (Float32, 1, 0.2834847867), (Float32, 1, 0.2831226587),
                (Float32, 5, 0.2834847867), (Float32, 5, 0.2831226587))

    for (T, refac, x₀) in fixtures
        x = T[x₀]
        nl = NewtonSolver(x, similar(x); F=Fb!, linesearch=BierlaireQuadratic(T),
            verbosity=0, refactorize=refac)
        state = SolverState(nl)
        # The bracketing error must be gone.  A `NonlinearSolverException` (NaN direction) is a
        # different, legitimate failure and is allowed through.
        try
            solve!(x, nl, state)
        catch e
            @test e isa NonlinearSolverException
            @test !(e isa ErrorException && occursin("must be decreasing", e.msg))
            continue
        end
        @test iteration_number(state) ≤ config(nl).max_iterations
    end

    # The five starts that are genuinely solvable now converge — including
    # x₀ = 0.1440401297, which was expected to be unreachable.
    for (T, refac, x₀) in ((Float64, 5, 0.1440401297), (Float64, 5, 0.2834847806),
                           (Float64, 5, 0.2831226663), (Float32, 1, 0.2834847867),
                           (Float32, 1, 0.2831226587))
        x = T[x₀]
        nl = NewtonSolver(x, similar(x); F=Fb!, linesearch=BierlaireQuadratic(T),
            verbosity=0, refactorize=refac)
        state = SolverState(nl)
        solve!(x, nl, state)
        @test isconverged(status(nl, state))
        @test minimum(abs(Float64(x[1]) - r) for r in roots) < 1e-5
    end

    # And `triple_point_finder` itself never raises for any of them: the situation is reported
    # through the status instead.
    for (T, refac, x₀) in fixtures
        x = T[x₀]
        nl = NewtonSolver(x, similar(x); F=Fb!, linesearch=BierlaireQuadratic(T),
            verbosity=0, refactorize=refac)
        SimpleSolvers.initialize!(nl, x)
        state = NonlinearSolverState(x)
        SimpleSolvers.initialize!(state, x, T[fscalar(x₀)])
        for it in 1:8
            increase_iteration_number!(state)
            direction!(nl, x, NullParameters(), it)
            any(!isfinite, direction(cache(nl))) && break
            st = SimpleSolvers.solve_with_status(linesearch(nl), one(T), (x=x, parameters=NullParameters()))
            @test steplength(st) > zero(T)
            compute_new_iterate!(x, steplength(st), direction(cache(nl)))
            any(!isfinite, x) && break
        end
    end
end

@testset "$(rpad("an ascent direction freezes the iterate and forces a fresh Jacobian", 80))" begin
    # A deterministic ascent anchor. The direction is computed from the *regularized* Jacobian
    # `J + λI` while the line search's φ'(0) = 2F·(J·d) uses the raw `J`, so for J = -1 and λ = 2
    # the two disagree in sign: d = -(J+λ)⁻¹F = (x - 1) points *away* from the root at x = 1 and
    # φ'(0) = -2F²J/(J+λ) = +2F² > 0.  (`check_anchor` names exactly this cause: a direction that
    # did not come from an exact, freshly factorized Newton solve.)
    Fneg(y, x, params) = y .= -(x .- 1.0)
    DFneg!(J, x, params) = (J .= -1.0)

    x = [0.0]
    s = NewtonSolver(x, Fneg, zero(x); DF! = DFneg!, regularization_factor=2.0, verbosity=0)
    state = SolverState(s)
    solve!(x, s, state)

    # The step is not taken: moving along a direction the line search rejected outright would
    # only make the retry start from a worse point.  Before, the full step was taken, the iterate
    # moved away from the root, nothing counted it as a stall, and the solve ran to
    # `max_iterations` while diverging.
    @test x == [0.0]
    @test iteration_number(state) ≤ 4
    @test iteration_number(state) < config(s).max_iterations
    st = status(s, state)
    @test isstalled(st, config(s))
    @test !isconverged(st)

    # the line search really is reporting a non-descent anchor here
    SimpleSolvers.initialize!(s, x)
    st0 = SolverState(s)
    SimpleSolvers.initialize!(st0, x, SimpleSolvers.value!(value(cache(s)), SimpleSolvers.nonlinearproblem(s), x, NullParameters()))
    direction!(s, x, NullParameters(), 1)
    lsst = solve_with_status(SimpleSolvers.linesearch(s), 1.0,
        (x=x, parameters=NullParameters(), φ₀=SimpleSolvers.L2norm(value(st0))))
    @test SimpleSolvers.outcome(lsst) == LINESEARCH_NO_DESCENT
    @test lsst.d₀ > 0
end

@testset "$(rpad("a stalled step forces a refactorization whatever refactorize is", 80))" begin
    # `maybe_refactorize!` used to refresh only on `mod(iteration, refactorize) == 0`, so with
    # `refactorize = 5` the stale Jacobian survived iterations 6–9. Two consecutive stalls in that
    # window would end the solve (`max_stalls = 2`) for a reason a fresh Jacobian could have
    # fixed. A stall now refreshes immediately, which is what makes `max_stalls = 2` conclusive
    # for every `refactorize` rather than only for `refactorize = 1`.
    njac = Ref(0)
    Fq(y, x, params) = y .= x .^ 2 .- 2.0
    DFq!(J, x, params) = (njac[] += 1; J .= 0.0; J[1, 1] = 2x[1])

    x = [1.0]
    s = NewtonSolver(x, Fq, zero(x); DF! = DFq!, refactorize=5, verbosity=0)

    njac[] = 0
    SimpleSolvers.maybe_refactorize!(s, x, NullParameters(), 7)
    @test njac[] == 0                     # mid-cycle: the stale factorization is reused

    SimpleSolvers.maybe_refactorize!(s, x, NullParameters(), 7; stalled=true)
    @test njac[] == 1                     # ... unless the previous step stalled

    SimpleSolvers.maybe_refactorize!(s, x, NullParameters(), 10)
    @test njac[] == 2                     # and the refactorize cycle still fires

    # `needs_refresh` is what `solver_step!` feeds in, from either source of the verdict
    state = NonlinearSolverState([1.0])
    initialize!(state, [1.0], [1.0])
    @test !SimpleSolvers.needs_refresh(state)
    flag_stall!(state)
    @test SimpleSolvers.needs_refresh(state)      # flagged by the line search this step
    record_stall!(state, config(s))               # consumes the flag into the counter
    @test !state.stallflag
    @test SimpleSolvers.needs_refresh(state)      # still true, now via the counter
    update!(state, [5.0], [1.0])                  # the iterate moved
    record_stall!(state, config(s))
    @test SimpleSolvers.needs_refresh(state) == false

    # end to end: a solve that stagnates with refactorize = 5 still stops on the stall counter
    # rather than running to max_iterations
    Ffloor(y, x, params) = y .= ((1e8 .+ x) .- 1e8) .- 1e-9
    x2 = [1.0]
    s2 = NewtonSolver(x2, Ffloor, zero(x2); f_abstol=1e-20, f_reltol=0.0, refactorize=5, verbosity=0)
    state2 = SolverState(s2)
    solve!(x2, s2, state2)
    @test isstalled(status(s2, state2), config(s2))
    @test iteration_number(state2) < config(s2).max_iterations
end

@testset "$(rpad("the solver messages are compiled once, not once per solver", 80))" begin
    # `directions!`, `solver_step!`, `nan_recovery!` and the `NewtonSolver` constructor are all
    # specialized on the solver, which carries the closure types of its `NonlinearProblem` and
    # `Jacobian` — so a message in any of their bodies is re-inferred and re-codegen'd once per
    # problem a solver is built for. Every one of them therefore delegates to a `@noinline` reporter
    # taking nothing but numbers and the `Options`. See `report_linesearch_status` for why, and
    # `has_logging_code` in `test/lowered_code.jl` for the check.
    for f in (SimpleSolvers.report_dogleg_singular, SimpleSolvers.report_dogleg_nan,
        SimpleSolvers.report_dogleg_underflow, SimpleSolvers.report_nan_direction,
        SimpleSolvers.report_static_refactorize,
        # `nonlinear_solver_warnings` is the outer iteration's own barrier: it takes a
        # `NonlinearSolverStatus` and an `Options`, never the solver, and so has always been
        # compiled once per element type rather than once per problem.
        nonlinear_solver_warnings)
        @test has_logging_code(f)
    end
    for f in (SimpleSolvers.directions!, solver_step!, SimpleSolvers.nan_recovery!)
        @test !has_logging_code(f)
    end

    # Now that the gates live in the reporters rather than at the site that decides to report, pin
    # them: each message fires at its documented verbosity and is silent one below. Neither of the
    # two below is rate limited, so this is repeatable within a session — unlike the three messages
    # of `nonlinear_solver_warnings`, which back off through `should_report!`.
    #
    # `report_dogleg_underflow` is not pinned this way: a collapsed trust-region radius needs a
    # merit that is finite and non-decreasing along every direction the method tries, and every
    # constructible candidate is caught first by the singular-Jacobian fallback and the stall
    # counter, which end the solve before Δ reaches `eps`. It is a defensive path; the scan above
    # still asserts that its message lives in a barrier rather than in `solver_step!`.

    # A Jacobian that is singular at x₀ — the scenario of "DogLeg falls back to steepest descent on
    # a singular Jacobian" above, rerun for its message alone.
    Fsing(y, x, p) = (y[1] = x[1]^2; y[2] = x[2]; y)
    singular(v) = function ()
        x = [0.0, 1.0]
        solve!(x, DogLegSolver(x, Fsing, zero(x); verbosity=v))
    end
    @test logged_any(singular(2), "singular Jacobian")
    @test !logged_any(singular(1), "singular Jacobian")

    # A merit that is NaN at the full trial step, so the trust-region radius shrinks: the
    # domain-restricted log used by the Newton NaN-damping testset above, whose Newton step from
    # x₀ = 1 lands at x = -1 where F is undefined.
    nanlog(v) = v > 0 ? log(v) : oftype(v, NaN)
    Flog(y, x, p) = (y .= nanlog.(x) .+ 2)
    nanmerit(v) = function ()
        x = [1.0]
        solve!(x, DogLegSolver(x, Flog, similar(x); verbosity=v))
    end
    @test logged_any(nanmerit(2), "non-finite merit")
    @test !logged_any(nanmerit(1), "non-finite merit")

    # The generic step's own damping reporter, which was the one message in this family with no
    # gate assertion at all. `Finv` overshoots from x₀ = 2 to exactly x = 0, where 1/x is `Inf`
    # — the `-1/|x|` singularity of issue #130 — so `nan_recovery!` reports and halves.
    Finv(y, x, p) = (y .= 1 ./ x .- 1)
    nonfinite(v) = function ()
        x = [2.0]
        solve!(x, NewtonSolver(x, similar(x); F=Finv, linesearch=Static(), verbosity=v))
    end
    @test logged_any(nonfinite(2), "Reducing length of direction vector")
    @test !logged_any(nonfinite(1), "Reducing length of direction vector")
end

@testset "$(rpad("a converged solve allocates nothing", 80))" begin
    # The companion of the line-search assertion in `linesearch_tests.jl`, for the two solvers that
    # take no line search. Measured inside a function, because from global scope the arguments are
    # boxed and the number says nothing about the code under test — and guarded, because under
    # `--check-bounds=yes` it says nothing either (see `AS_A_CALLER_COMPILES_IT`).
    for f in (solver_step!, SimpleSolvers.directions!, SimpleSolvers.nan_recovery!)
        @test !has_boxed_capture(f)
    end

    F(y, x, params) = y .= x .^ 2 .- 2
    function solve_allocations(S)
        x = ones(3)
        s = S(x, F, similar(x); verbosity=0)
        state = SolverState(s)
        solve!(x, s, state)
        x .= 1.0
        @allocated solve!(x, s, state)
    end
    # `PicardSolver` has no linear solver at all; `DogLegSolver` defaults to `LapackLU`, whose
    # `factorize!` and `ldiv!` are both allocation-free once the solver is built. So the same
    # assertion holds for either of them.
    @test solve_allocations(PicardSolver) == 0 skip = !AS_A_CALLER_COMPILES_IT
    @test solve_allocations(DogLegSolver) == 0 skip = !AS_A_CALLER_COMPILES_IT
end

@testset "$(rpad("the solver report backs off geometrically instead of going silent", 80))" begin
    # `maxlog` is keyed on the source location of the `@warn`, so its budget was process-global
    # *and* was never reset between solves: after three reports the message was gone for the rest
    # of the session, and a genuine failure in the tenth solve of a run was silent. `should_report!`
    # reports the 1st, 2nd, 4th, 8th … occurrence instead — O(log N) messages over N solves, and no
    # point beyond which nothing is ever said again.
    reset_warning_counts!()
    @test [should_report!(:probe) for _ in 1:16] ==
          [true, true, false, true, false, false, false, true,
        false, false, false, false, false, false, false, true]

    # A *different* diagnosis is reported at once, however late it turns up — its own counter is
    # still at zero. This is the half of the defect that mattered: a run that was healthy for ten
    # thousand steps and then was not says so.
    @test should_report!(:probe_other)

    # Which is why the keys carry the dominant line-search outcome: a solve that starts stagnating
    # for a new reason is a new diagnosis, not a continuation of the old one.
    reset_warning_counts!()
    @test should_report!(Symbol(:stagnated_, LINESEARCH_FLOOR))
    @test should_report!(Symbol(:stagnated_, LINESEARCH_NO_DESCENT))

    # End to end: the same unattainable solve repeated is not reported every time ...
    Ffloor(y, x, params) = y .= ((1e8 .+ x) .- 1e8) .- 1e-9
    function stagnation_reports(n)
        reset_warning_counts!()
        count = 0
        for _ in 1:n
            x = [1.0]
            s = NewtonSolver(x, Ffloor, zero(x); f_abstol=1e-20, f_reltol=0.0)
            logs, _ = Test.collect_test_logs() do
                solve!(x, s, SolverState(s))
            end
            count += any(r -> occursin("stagnated", string(r.message)), logs)
        end
        count
    end
    @test stagnation_reports(1) == 1
    @test stagnation_reports(16) == 5      # solves 1, 2, 4, 8, 16
    @test stagnation_reports(4) == 3       # solves 1, 2, 4
end

@testset "$(rpad("the line-search tally is per solve and covers every step", 80))" begin
    # The tally lives in the state and is reset by `initialize!`, exactly like `stalls`: what the
    # line search reported during the *previous* solve says nothing about this one.
    F(y, x, params) = y .= x .^ 2 .- 2
    x = [1.0, 1.0]
    s = NewtonSolver(x, F, zero(x); verbosity=0)
    state = SolverState(s)

    solve!(x, s, state)
    first = collect(linesearch_outcomes(status(s, state)))
    @test sum(first) == iteration_number(state)

    x .= 1.0
    solve!(x, s, state)
    @test collect(linesearch_outcomes(status(s, state))) == first   # reset, not accumulated

    # A hand-rolled iteration that never initializes keeps it at zero, the same guarantee
    # `stall_number` gives.
    fresh = NonlinearSolverState(x)
    @test all(iszero, linesearch_outcomes(fresh))
    @test linesearch_failures(fresh) == 0

    # `record_linesearch!` counts by outcome, and `linesearch_failures` counts only the outcomes
    # that are not benign.
    record_linesearch!(fresh, LINESEARCH_DECREASED)
    record_linesearch!(fresh, LINESEARCH_NO_DESCENT)
    record_linesearch!(fresh, LINESEARCH_NO_DESCENT)
    record_linesearch!(fresh, LINESEARCH_STATIONARY)
    @test linesearch_outcomes(fresh)[linesearch_index(LINESEARCH_NO_DESCENT)] == 2
    @test linesearch_failures(fresh) == 2
    @test SimpleSolvers.isbenign(LINESEARCH_DECREASED)
    @test SimpleSolvers.isbenign(LINESEARCH_STATIONARY)
    @test SimpleSolvers.isbenign(LINESEARCH_UNKNOWN)
    @test !SimpleSolvers.isbenign(LINESEARCH_FLOOR)
    @test !SimpleSolvers.isbenign(LINESEARCH_EXHAUSTED)
    @test !SimpleSolvers.isbenign(LINESEARCH_NO_DESCENT)

    # Ties go to the outcome declared first, i.e. away from the more alarming diagnosis.
    record_linesearch!(fresh, LINESEARCH_FLOOR)
    record_linesearch!(fresh, LINESEARCH_FLOOR)
    st = NonlinearSolverStatus(fresh, Options())
    @test dominant_linesearch_outcome(st) === LINESEARCH_FLOOR
end

@testset "$(rpad("a converged solve reports the line search only when it went wrong", 80))" begin
    # The last step of a converged solve reaches the merit's round-off floor as a matter of course
    # — that is *how* it converged — so `LINESEARCH_FLOOR` in the tally of a converged solve is the
    # healthy case. Reporting on the tally as a whole announced that every textbook solve "did not
    # go smoothly", which is the noise `linesearch_reason` itself defines: one failure is the last
    # step, a count is evidence. So the converged report counts the outcomes that are *not* the
    # expected floor.
    converged = NonlinearSolverState([1.0])
    initialize!(converged, [1.0], [0.0])
    update!(converged, [1.0], [0.0])
    update!(converged, [1.0], [0.0])

    for _ in 1:3
        record_linesearch!(converged, LINESEARCH_DECREASED)
    end
    record_linesearch!(converged, LINESEARCH_FLOOR)
    floored = NonlinearSolverStatus(converged, Options(Float64; verbosity=2))
    @test isconverged(floored)
    @test linesearch_failures(floored) == 1          # the floor still counts in the tally ...
    @test !logged_any(() -> nonlinear_solver_warnings(floored, Options(Float64; verbosity=2)),
        "not every step went smoothly")               # ... but is not news about a converged solve

    # A step the search could not resolve at all is news, however, and it is the only trace of the
    # line search a solve leaves.
    record_linesearch!(converged, LINESEARCH_EXHAUSTED)
    rough = NonlinearSolverStatus(converged, Options(Float64; verbosity=2))
    @test isconverged(rough)
    @test logged_any(() -> nonlinear_solver_warnings(rough, Options(Float64; verbosity=2)),
        "not every step went smoothly")
    @test logged_any(() -> nonlinear_solver_warnings(rough, Options(Float64; verbosity=2)),
        "LINESEARCH_EXHAUSTED")
    # ... at `verbosity ≥ 2` only, the same gate the floor and stationary outcomes use.
    @test !logged_any(() -> nonlinear_solver_warnings(rough, Options(Float64; verbosity=1)),
        "not every step went smoothly")

    # End to end, on the solve the package's own doctests use: nothing at all is said about the
    # line search, at any verbosity, for a solve that simply worked.
    F(y, x, params) = y .= x .^ 2 .- 2
    healthy() = function ()
        x = ones(3)
        s = NewtonSolver(x, F, zero(x); verbosity=2)
        state = SolverState(s)
        solve!(x, s, state)
        @test linesearch_outcomes(status(s, state))[linesearch_index(LINESEARCH_FLOOR)] > 0
    end
    reset_warning_counts!()
    @test !logged_any(healthy(), "not every step went smoothly")
    @test !logged_any(healthy(), "line search")
end
