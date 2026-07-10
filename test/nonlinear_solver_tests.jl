using SimpleSolvers
using SimpleSolvers: initialize!, solver_step!, BierlaireQuadratic
using SimpleSolvers: NonlinearSolverState, assess_convergence, residuals, update!, iteration_number
using SimpleSolvers: meets_stopping_criteria, nonlinear_solver_warnings, NonlinearSolverStatus
using SimpleSolvers: linesearch_problem, cache, jacobianmatrix, solution, value, direction, direction!, NullParameters
using SimpleSolvers: trust_radius, DOGLEG_Δ_INITIAL
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
        (NewtonSolver, (linesearch=Bisection(T),), 2),
        (NewtonSolver, (linesearch=Quadratic(T),), 32),
        (NewtonSolver, (linesearch=BierlaireQuadratic(T),), 2),
        (NewtonSolver, (linesearch=StrongWolfe(T),), 2),
        (QuasiNewtonSolver, (linesearch=Static(T),), 2),
        (QuasiNewtonSolver, (linesearch=Backtracking(T),), 2),
        (QuasiNewtonSolver, (linesearch=Bisection(T),), 2),
        (QuasiNewtonSolver, (linesearch=Quadratic(T),), 32),
        (QuasiNewtonSolver, (linesearch=BierlaireQuadratic(T),), 8),
        (QuasiNewtonSolver, (linesearch=StrongWolfe(T),), 2),
        (DogLegSolver, (linesearch=Static(T),), 2),
        (DogLegSolver, (linesearch=Backtracking(T),), 2),
        (DogLegSolver, (linesearch=Bisection(T),), 2),
        (DogLegSolver, (linesearch=Quadratic(T),), 2),
        (DogLegSolver, (linesearch=BierlaireQuadratic(T),), 2),
        (DogLegSolver, (linesearch=StrongWolfe(T),), 2),
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

    nl₁ = NonlinearSolver(Newton(), x, y; F=Fnan, jacobian=J₁, verbosity=2)
    nl₂ = NonlinearSolver(Newton(), x, y; F=Fnan, jacobian=J₂, verbosity=2)

    x₁ = zeros(T, n)
    x₂ = zeros(T, n)

    # The solver must refuse to proceed on this pathological problem.  The finite
    # difference Jacobian at x = 0 is the zero matrix, which is singular: the LU
    # solver throws a `SingularException` instead of silently returning
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

        # PicardSolver runs out of iterations on this problem (expected); assert the
        # "Solver took … iterations." warning is emitted rather than letting it leak
        # to the test log.
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
