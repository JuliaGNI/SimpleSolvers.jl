using Random
using SimpleSolvers
using Test

using LinearAlgebra: rmul!, ldiv!
using SimpleSolvers: BierlaireQuadratic, Quadratic, NullParameters
using SimpleSolvers: factorize!, linearsolver, jacobian, jacobian!, cache, linesearch_problem, direction, compute_new_iterate, compute_new_iterate!, direction!, nonlinearproblem, iteration_number
using SimpleSolvers: change_precision, bisection, bracket_root, triple_point_finder
using SimpleSolvers: CurvatureCondition, SufficientDecreaseCondition
using SimpleSolvers: issufficient, isfloor
using SimpleSolvers: steplength, outcome, trials, armijo_tolerance, armijo_ulps, backtracking_αmin,
    backtracking_interpolation, with_config, problem, method, config

include("logging_code.jl")

f(x) = x^2 - 1
g(x) = 2x
δx(x) = -g(x) / 2

function compute_next_iterate(ls::Linesearch, x₀::T) where {T}
    α = solve(ls, 1.0, (x = x₀,))
    compute_new_iterate(x₀, α, δx(x₀))
end

function compute_next_iterate(ls::Linesearch, x₀::T, n::Integer) where {T}
    x = x₀
    for _ in 1:n
        x = compute_next_iterate(ls, x)
    end
    x
end

function make_linesearch_problem(x₀::Number)
    # `_d` is the derivative of the merit φ(α) = f(x₀ + α·δx) with respect to the
    # step α, i.e. the *directional* derivative g(x₀ + α·δx)·δx (chain rule) — the
    # same convention the real `linesearch_problem` uses (its `D` is 2·F·(J·d)).
    _f(α, _) = f(compute_new_iterate(x₀, α, δx(x₀)))
    _d(α, _) = g(compute_new_iterate(x₀, α, δx(x₀))) * δx(x₀)
    LinesearchProblem{typeof(x₀)}(_f, _d)
end

function test_linesearch(method::LinesearchMethod, n::Integer=1)
    x₀ = -3.0
    x₁ = +3.0
    xₛ = 0.0

    ls = Linesearch(make_linesearch_problem(x₀), method; x_abstol=zero(x₀))

    @test compute_next_iterate(ls, x₀, n) ≈ xₛ atol = ∛(2eps())
    @test compute_next_iterate(ls, x₀, n) ≈ xₛ atol = ∛(2eps())
end


@testset "$(rpad("Bracketing",80))" begin
    @test bracket_minimum(x -> x^2, 0.0) == (-SimpleSolvers.DEFAULT_BRACKETING_s, +SimpleSolvers.DEFAULT_BRACKETING_s)
    @test bracket_minimum(x -> (x - 1)^2, 0.0) == (0.64, 2.56)

    # bracket_minimum must return an interval that actually contains
    # the minimum (the early-exit path must not bracket a maximum).
    lo, hi = bracket_minimum(x -> (x - 1)^2, 0.0)
    @test lo < 1.0 < hi
end

@testset "$(rpad("triple_point_finder", 80))" begin
    # An immediate return (f(x₁ + 2δ) > f(x₁) on the first iteration) must not
    # produce a degenerate triple with two identical points, which would make the
    # downstream quadratic fit singular.  Here f has its minimum exactly at
    # x₁ = x₀ + δ.
    x₀ = 0.0
    δ = 0.01
    a, b, c = triple_point_finder(x -> (x - (x₀ + δ))^2, x₀; δ=δ)
    @test a < b < c
    @test length(unique((a, b, c))) == 3

    # sanity: the standard example still brackets the minimum of x²
    a2, b2, c2 = triple_point_finder(x -> x^2, -1.0)
    @test a2 < b2 < c2
    @test a2^2 ≥ b2^2 && b2^2 ≤ c2^2   # middle point has the smallest value
end

@testset "$(rpad("Static",80))" begin
    x₀ = -3.0
    x₁ = +3.0
    δx = x₁ - x₀
    x = copy(x₀)

    ls_problem = make_linesearch_problem(x₀)

    @test Linesearch(ls_problem, Static()) == Linesearch(ls_problem, Static(1.0))

    ls1 = Linesearch(ls_problem, Static())
    ls2 = Linesearch(ls_problem, Static(1.0))
    ls3 = Linesearch(ls_problem, Static(0.8))

    @test solve(ls1, 0.0) == 1.0
    @test solve(ls2, 0.0) == 1.0
    @test solve(ls3, 0.0) == 0.8

end

@testset "$(rpad("Bisection", 80))" begin

    test_linesearch(Bisection(), 1)

end

@testset "$(rpad("Backtracking", 80))" begin

    test_linesearch(Backtracking(), 20)

end

@testset "$(rpad("Backtracking stall", 80))" begin
    # f(α) = (α - 100)² starting at α = 1: shrinking α makes the curvature
    # condition impossible to satisfy, so the old shrink-only loop (which
    # required both Wolfe conditions) ran all iterations and silently returned a
    # denormal α ≈ 9.3e-302.  The loop terminates on sufficient decrease
    # alone, so the step at α = 1 (which already decreases f) is accepted.
    prob = LinesearchProblem{Float64}((α, _) -> (α - 100.0)^2, (α, _) -> 2.0 * (α - 100.0))
    ls = Linesearch(prob, Backtracking(); verbosity=0)
    α = solve(ls, 1.0)
    @test α ≥ 1e-10
    @test α ≈ 1.0

    # constructor input validation: 0 < p < 1 and 0 < c₁ < c₂ < 1
    @test_throws AssertionError Backtracking(; p=1.5)
    @test_throws AssertionError Backtracking(; p=0.0)
    @test_throws AssertionError Backtracking(; c₁=0.5, c₂=0.1)  # c₁ < c₂ violated
    @test_throws AssertionError Backtracking(; c₁=-1e-4)        # c₁ > 0 violated
    @test Backtracking() isa Backtracking                       # defaults are valid
end

@testset "$(rpad("Backtracking: non-descent and stationary anchors are reported, not searched", 80))" begin
    # A merit with positive slope at 0 can never satisfy the sufficient-decrease condition, so
    # the former 53 shrinking trials (which returned α ≈ eps) were 53 wasted merit
    # evaluations.  The anchor is now tested up front and, consistently with StrongWolfe, the
    # caller's trial step is returned — never the α₀ = 0 anchor, which would freeze the outer
    # solver iterate (x .+= 0 .* d) and spin it to max_iterations.
    prob = LinesearchProblem{Float64}((α, _) -> α + 1.0, (α, _) -> 1.0)  # φ'(0) = 1 > 0
    ls = Linesearch(prob, Backtracking(); verbosity=0)
    st = solve_with_status(ls, 0.7)
    @test outcome(st) == LINESEARCH_NO_DESCENT
    @test steplength(st) == 0.7   # the caller's step, not the α₀ = 0 anchor
    @test steplength(st) > zero(steplength(st))
    @test trials(st) == 0         # nothing beyond the anchor was evaluated
    @test solve(ls, 0.7) == steplength(st)
    # ... and it agrees with StrongWolfe, which has the same up-front descent check
    @test solve(ls, 0.7) == solve(Linesearch(prob, StrongWolfe(); verbosity=0), 0.7)

    # A stationary anchor (φ'(0) = 0, e.g. a vanishing Newton direction at an exact root) is
    # benign: it must not warn at the default verbosity, since it happens on the last
    # iteration of a converged solve.
    flat = LinesearchProblem{Float64}((α, _) -> -1.0, (α, _) -> 0.0)
    lsflat = Linesearch(flat, Backtracking())     # verbosity = 1, the default
    @test outcome(solve_with_status(lsflat, 1.0)) == LINESEARCH_STATIONARY
    @test (@test_logs solve(lsflat, 1.0)) == 1.0

    # A merit whose slope contradicts its values (a stale or regularized Jacobian, an inexact
    # linear solve) is a genuine failure and must be told apart from the round-off floor.
    lying = LinesearchProblem{Float64}((α, _) -> 1.0 + α, (α, _) -> -2.0)
    @test outcome(solve_with_status(Linesearch(lying, Backtracking(); verbosity=0), 1.0)) == LINESEARCH_EXHAUSTED
end

@testset "$(rpad("Backtracking: stagnation at the merit's round-off floor", 80))" begin
    φ₀ = 1.0

    # Fixture A — the *silent* variant.  The merit is frozen below α_move = 1e-6 (there the
    # trial iterate x + α·δx no longer differs from x in floating point) and increases above
    # it, so no α decreases it; d₀ = -2φ₀ is the exact-Newton slope of ‖F‖².
    # Before: the Armijo right-hand side φ₀ + c₁·α·d₀ rounds back up to φ₀ for
    # α < α* = eps/(4c₁) = 5.55e-13, the frozen merit tied it, and the search reported
    # *success* at α = 2.2737367544323206e-13 after 43 trials — with no warning whatsoever.
    frozen = LinesearchProblem{Float64}((α, _) -> α ≤ 1e-6 ? φ₀ : φ₀ * (1 + α), (α, _) -> -2φ₀)
    ls = Linesearch(frozen, Backtracking(); verbosity=0)
    st = solve_with_status(ls, 1.0)
    @test outcome(st) == LINESEARCH_FLOOR   # the tie is *not* reported as a decrease
    @test !issufficient(st)
    @test isfloor(st)
    @test trials(st) < 43                   # 14 (was 43)
    @test steplength(st) > 1e-8             # stops at the frozen scale, not at 2.3e-13
    @test solve(ls, 1.0) == steplength(st)  # `solve` still returns the step length

    # Fixture B — the *reported* variant: every α > 0 lands one ulp above φ₀, i.e. ‖F‖² is
    # pure round-off noise.  Before: 53 halvings down to α = eps(1.0) = 2.220446049250313e-16
    # and a warning claiming the sufficient decrease condition failed "within 1000
    # iterations" — neither the count nor the reason was true.
    noise = LinesearchProblem{Float64}((α, _) -> α > zero(α) ? nextfloat(φ₀) : φ₀, (α, _) -> -2φ₀)
    lsn = Linesearch(noise, Backtracking(); verbosity=0)
    stn = solve_with_status(lsn, 1.0)
    @test outcome(stn) == LINESEARCH_FLOOR
    @test isfloor(stn)
    @test trials(stn) < 53                          # 33 (was 53)
    @test steplength(stn) > eps(1.0)                # the αmin floor, not eps(1.0)
    @test steplength(stn) == stn.αmin
    @test solve(lsn, 1.0) == steplength(stn)

    # αmin sits a factor 2·τ_ulps above the α* at which fl(φ₀ + c₁αd₀) rounds back to φ₀, so
    # the search never enters the region where the test is decided by rounding.
    αstar = eps(1.0) / (4 * SimpleSolvers.DEFAULT_WOLFE_c₁)
    @test stn.τ == 4eps(φ₀)
    @test stn.αmin ≈ 2 * SimpleSolvers.DEFAULT_ARMIJO_τ_ULPS * αstar
    @test stn.αmin == 4.440892098500626e-12
end

@testset "$(rpad("armijo_tolerance / backtracking_αmin / interpolation", 80))" begin
    @test armijo_tolerance(1.0, 4.0) == 4eps(1.0)
    @test armijo_tolerance(1e-20, 4.0) == 4eps(1e-20)

    τ = armijo_tolerance(1.0, 4.0)
    @test backtracking_αmin(1e-4, -2.0, τ) == 4.440892098500626e-12
    @test backtracking_αmin(1e-4, -0.0, τ) == sqrt(eps(1.0))   # no division by zero
    @test backtracking_αmin(1e-4, -1e30, τ) == eps(1.0)        # clamped from below
    @test backtracking_αmin(1e-4, -1e-30, τ) == sqrt(eps(1.0)) # clamped from above
    @test backtracking_αmin(1e-4, -2.0, 0.0) == eps(1.0)       # τ = 0 keeps the old floor

    # the interpolated step always stays inside [BACKTRACKING_SHRINK_MIN·α, p·α]
    for (φα, αp, φp) in [(3.0, NaN, NaN), (3.0, 2.0, 5.0), (NaN, NaN, NaN), (Inf, 1.5, 2.0)]
        αₙ = backtracking_interpolation(1.0, -2.0, 1.0, φα, αp, φp, 0.5)
        @test SimpleSolvers.BACKTRACKING_SHRINK_MIN ≤ αₙ ≤ 0.5
    end

    # φ(α) = 1 - 2α + 1000α²: α = 1 overshoots badly.  Plain halving needs 10 trials,
    # safeguarded interpolation fewer — and one merit evaluation per trial plus the anchor
    # (the two-argument SufficientDecreaseCondition must not re-evaluate the merit).
    n = Ref(0)
    prob = LinesearchProblem{Float64}((α, _) -> (n[] += 1; 1.0 - 2α + 1000α^2), (α, _) -> -2.0 + 2000α)
    st = solve_with_status(Linesearch(prob, Backtracking(); verbosity=0), 1.0)
    @test issufficient(st)
    @test trials(st) < 10
    @test n[] == trials(st) + 1
    @test steplength(st) ≤ 0.5
end

@testset "$(rpad("SufficientDecreaseCondition round-off allowance", 80))" begin
    # τ = 0 (the default) is the exact former condition: a merit one ulp above φ₀ is rejected
    # at every α — including the α where the right-hand side rounds back to φ₀.
    sdc = SufficientDecreaseCondition(1e-4, 1.0, -2.0, α -> nextfloat(1.0))
    @test !sdc(1.0)
    @test !sdc(1e-13)

    # τ slackens the decrease that is *demanded* — a merit that misses the Armijo target by
    # less than τ is accepted where the exact condition rejects it ...
    target = 1.0 + 1e-4 * 0.1 * -2.0      # the right-hand side at α = 0.1, ≈ 1 - 2e-5
    φ = α -> nextfloat(target, 2)         # misses it by 2 ulps, well inside τ = 4 ulps of φ₀
    @test !SufficientDecreaseCondition(1e-4, 1.0, -2.0, φ)(0.1)
    @test SufficientDecreaseCondition(1e-4, 1.0, -2.0, φ; τ=4eps(1.0))(0.1)

    # ... but it never licenses a step where the demanded decrease is meaningful ...
    sdcτ = SufficientDecreaseCondition(1e-4, 1.0, -2.0, α -> nextfloat(1.0); τ=4eps(1.0))
    @test !sdcτ(1.0)
    # ... and, thanks to the min against φ₀, it never licenses an *increase* either, not even at
    # the degenerate α where fl(φ₀ + c₁αd₀) has rounded back up to φ₀. The unbounded form
    # accepted here, which is a 0.4% uphill step in Float16 (4 ulps of φ₀ = 1).
    @test !sdcτ(1e-13)
    @test 1.0 + 1e-4 * 1e-13 * -2.0 == 1.0     # the right-hand side really has degenerated
    # a *non-increase* at the same α is accepted, which is what keeps the ladder from shrinking
    # to eps once the merit has reached its round-off floor
    @test SufficientDecreaseCondition(1e-4, 1.0, -2.0, α -> 1.0; τ=4eps(1.0))(1e-13)

    # the bound holds in the precision where it bites: 4 ulps of Float16 is 3.9e-3, twenty times
    # the 2c₁ = 2e-4 demanded at α = 1, so an unbounded τ would accept a visible increase
    τ16 = 4eps(Float16(1))
    sdc16 = SufficientDecreaseCondition(Float16(1e-4), Float16(1), Float16(-2), α -> Float16(1) + τ16 / 2; τ=τ16)
    @test !sdc16(Float16(1))
    @test !sdc16(Float16(1e-3))

    @test_throws AssertionError SufficientDecreaseCondition(1e-4, 1.0, -2.0, sin; τ=-1.0)

    # the two-argument form uses the merit value supplied by the caller
    n = Ref(0)
    sdc2 = SufficientDecreaseCondition(1e-4, 1.0, -2.0, α -> (n[] += 1; 0.0))
    @test sdc2(1.0, 0.0)
    @test n[] == 0
    @test sdc2(1.0)
    @test n[] == 1
end

@testset "$(rpad("Backtracking: τ_ulps validation and generic solve_with_status fallback", 80))" begin
    @test_throws AssertionError Backtracking(; τ_ulps=-1.0)
    @test Backtracking(; τ_ulps=0.0) isa Backtracking
    @test Backtracking().τ_ulps == SimpleSolvers.DEFAULT_ARMIJO_τ_ULPS

    # τ_ulps = 0 recovers the exact condition, i.e. the old ladder down to the eps floor
    noise = LinesearchProblem{Float64}((α, _) -> α > zero(α) ? nextfloat(1.0) : 1.0, (α, _) -> -2.0)
    st = solve_with_status(Linesearch(noise, Backtracking(; τ_ulps=0.0); verbosity=0), 1.0)
    @test steplength(st) ≤ eps(1.0)

    # Every built-in method reports a real outcome now, so none of them relies on the generic
    # `LINESEARCH_UNKNOWN` fallback any more.  `Static` is the exception by nature: it ignores
    # the caller's step and never evaluates the merit, so it has established nothing.
    prob = LinesearchProblem{Float64}((α, _) -> (α - 0.7)^2, (α, _) -> 2 * (α - 0.7))
    for m in (Bisection(), Quadratic(), BierlaireQuadratic(), StrongWolfe(), Backtracking())
        ls = Linesearch(prob, m; verbosity=0)
        st = solve_with_status(ls, 1.0)
        @test outcome(st) == LINESEARCH_DECREASED
        @test issufficient(st)
        @test !isfloor(st)
        @test steplength(st) == solve(ls, 1.0)
    end

    lstatic = Linesearch(prob, Static(); verbosity=0)
    ststatic = solve_with_status(lstatic, 1.0)
    @test outcome(ststatic) == LINESEARCH_UNKNOWN
    @test !issufficient(ststatic)
    @test !isfloor(ststatic)
    @test steplength(ststatic) == solve(lstatic, 1.0)

    # The generic `solve_with_status` fallback is kept for user-defined `LinesearchMethod`s
    # (it reports `LINESEARCH_UNKNOWN`), but no built-in method dispatches to it any more.
end

@testset "$(rpad("with_config replaces the Options and keeps problem and method", 80))" begin
    prob = LinesearchProblem{Float64}((α, _) -> (α - 0.7)^2, (α, _) -> 2 * (α - 0.7))
    ls = Linesearch(prob, Backtracking())
    @test config(ls).verbosity == 1

    ls2 = with_config(ls, Options(Float64; verbosity=0, linesearch_max_iterations=7))
    @test problem(ls2) === problem(ls)
    @test method(ls2) === method(ls)
    @test config(ls2).verbosity == 0
    @test config(ls2).linesearch_max_iterations == 7
    @test config(ls).verbosity == 1   # the original is untouched

    # a mismatched element type is rejected rather than silently accepted
    @test_throws MethodError with_config(ls, Options(Float32))
end

@testset "$(rpad("linesearch_max_iterations bounds the ladder, max_iterations does not", 80))" begin
    noise = LinesearchProblem{Float64}((α, _) -> α > zero(α) ? nextfloat(1.0) : 1.0, (α, _) -> -2.0)

    unbounded = solve_with_status(Linesearch(noise, Backtracking(), Options(Float64; verbosity=0)), 1.0)
    @test outcome(unbounded) == LINESEARCH_FLOOR

    # the inner budget reaches the ladder, and a spent budget is *not* reported as the floor
    capped = solve_with_status(Linesearch(noise, Backtracking(), Options(Float64; verbosity=0, linesearch_max_iterations=3)), 1.0)
    @test trials(capped) == 3
    @test outcome(capped) == LINESEARCH_EXHAUSTED

    # the outer budget does not
    outer = solve_with_status(Linesearch(noise, Backtracking(), Options(Float64; verbosity=0, max_iterations=1)), 1.0)
    @test trials(outer) == trials(unbounded)
    @test outcome(outer) == LINESEARCH_FLOOR

    @test SimpleSolvers.linesearch_iterations(Float64) == 60
    @test SimpleSolvers.linesearch_iterations(Float32) == 31
    @test SimpleSolvers.linesearch_iterations(Float16) == 18

    # The quadratic searches are bounded by the same field (they used to carry their own
    # `max_number_of_quadratic_linesearch_iterations`).  They converge on their `ε` tolerance
    # long before the budget, so it acts purely as a backstop — but it is reachable, and it is
    # now settable by the user like every other line search's.
    quad = LinesearchProblem{Float64}((α, _) -> (α - 1.0)^2, (α, _) -> 2(α - 1.0))
    for m in (Quadratic(), BierlaireQuadratic())
        generous = solve(Linesearch(quad, m, Options(Float64; verbosity=0, linesearch_max_iterations=60)), 0.5)
        @test generous ≈ 1.0 atol = 1e-8
        # the default budget reaches the same answer, i.e. it is not the binding constraint
        @test solve(Linesearch(quad, m; verbosity=0), 0.5) ≈ generous atol = 1e-12
    end
    # a budget of one iteration truncates the Bierlaire fit, proving the field reaches the loop
    @test solve(Linesearch(quad, BierlaireQuadratic(), Options(Float64; verbosity=0, linesearch_max_iterations=1)), 0.5) ≉ 1.0
end

@testset "$(rpad("Quadratic Linesearch (Bierlaire)", 80))" begin

    test_linesearch(BierlaireQuadratic(), 1)

end

@testset "$(rpad("Quadratic Linesearch (Derivative-Based)", 80))" begin

    test_linesearch(Quadratic(), 1)

end


@testset "$(rpad("Quadratic defaults", 80))" begin
    # Quadratic(T, ::SolverMethod) used to square ε, s and s_reduction (an
    # accidental `^2`), disagreeing with the keyword constructor and pushing ε
    # below machine epsilon.  It matches the keyword constructor defaults and
    # dispatches on ::SolverMethod like its siblings.
    for T in (Float32, Float64)
        q = Quadratic(T, Newton())
        @test q ≈ Quadratic(T)
        @test q.ε == SimpleSolvers.default_precision(T)
        @test q.s == T(SimpleSolvers.DEFAULT_BRACKETING_s)
        @test q.s_reduction == T(SimpleSolvers.DEFAULT_s_REDUCTION)
    end

    # `default_precision` used to error for any float type other than
    # Float32/Float64 although `8eps(T)` is generic; it is defined for all
    # `AbstractFloat`s.
    @test SimpleSolvers.default_precision(Float16) == 8eps(Float16)
    @test SimpleSolvers.default_precision(BigFloat) == 8eps(BigFloat)
    @test BierlaireQuadratic(Float16) isa BierlaireQuadratic{Float16}
end

@testset "$(rpad("Linesearch Integration Tests", 80))" begin

    Random.seed!(1234)

    x = -10 * rand(1)

    function linesearch_factory(x::AbstractVector{T}, params) where {T}
        f(x::T) where {T<:Number} = exp(x) * (T(0.5) * x^3 - 5x^2 + 2x) + 2one(T)
        f(x::AbstractArray{T}) where {T<:Number} = @. exp(x) * (T(0.5) * x^3 - 5 * x^2 + 2x) + 2one(T)
        f!(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T} = y .= f.(x)

        function j!(j::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
            f_closure!(y, x) = f!(y, x, params)
            SimpleSolvers.ForwardDiff.jacobian!(j, f_closure!, similar(x), x)
        end

        jacobian = JacobianFunction{T}(f!, j!)
        solver = NewtonSolver(x, f.(x); F=f!, (DF!)=j!, jacobian=jacobian)
        state = NonlinearSolverState(x, value(cache(solver)))

        direction!(solver, x, params, iteration_number(state))
        # update!(state, x, value(cache(solver)))

        linesearch_problem(solver)
    end

    function check_linesearch(T, ls_method)
        params = (x=T.(x), parameters=NullParameters())
        ls = Linesearch(linesearch_factory(params.x, params.parameters), ls_method)
        α = solve(ls, one(T), params)
        @test ≈(problem(ls).D(α, params), zero(T); atol=(∛(2eps(T))))
    end

    for T ∈ (Float32, Float64)
        for ls_method ∈ (Bisection(T), Quadratic(T), BierlaireQuadratic(T))
            check_linesearch(T, ls_method)
        end
    end
end


# Regression: for a non-Float64 `LinesearchProblem{T}`, the `solve(prob, method, α)`
# convenience form must default `config` to `Options(T)` (not the hard-coded
# `Options()` == `Options{Float64}`) and convert `α` to `T`.  With the former
# `Options{Float64}` default the inner `Linesearch{T}` constructor (which requires
# `config::Options{T}`) failed for `T ≠ Float64`.  The `Linesearch(prob, method,
# config)` overload is likewise pinned to `config::Options{T}`, so a mismatched
# `Options` eltype is rejected rather than silently building a broken object.
@testset "$(rpad("Linesearch T-consistency (config default / α promotion)", 80))" begin
    for T ∈ (Float32, Float64)
        prob = LinesearchProblem{T}((α, _) -> (α - 2one(T))^2, (α, _) -> 2 * (α - 2one(T)))

        for method ∈ (Static(T), Backtracking(T), StrongWolfe(T), Bisection(T))
            # no explicit config: must default to Options(T) and stay in T
            α = solve(prob, method, one(T))
            @test α isa T

            # α given in a different precision is promoted to T
            α₂ = solve(prob, method, 1)
            @test α₂ isa T
        end

        # explicit matching config is accepted and preserves T
        ls = Linesearch(prob, Backtracking(T), Options(T))
        @test ls isa Linesearch{T}
    end

    # a config whose eltype disagrees with the problem is rejected, not silently
    # coerced into a broken `Linesearch`
    prob32 = LinesearchProblem{Float32}((α, _) -> (α - 2f0)^2, (α, _) -> 2f0 * (α - 2f0))
    @test_throws MethodError Linesearch(prob32, Backtracking(Float32), Options(Float64))
end


@testset "$(rpad("Linesearch Conversion Tests", 80))" begin

    function allocate_linesearch_methods(T::DataType)
        st = Static(T; α=one(T))
        bt = Backtracking(T)
        qu = Quadratic(T; ε=T(1e-5)) # here this constant is specified manually as it otherwise depends on the DataType used
        bq = BierlaireQuadratic(T)
        bi = Bisection(T)
        st, bt, qu, bq, bi
    end

    function convert_linesearches_test(T₁::DataType, T₂::DataType; rtol=T₂(1e-3))
        st₁, bt₁, qu₁, bq₁, bi₁ = allocate_linesearch_methods(T₁)
        st₂, bt₂, qu₂, bq₂, bi₂ = allocate_linesearch_methods(T₂)

        @test ≈(st₂, change_precision(T₂, st₁); rtol=rtol)
        @test ≈(bt₂, change_precision(T₂, bt₁); rtol=rtol)
        @test ≈(qu₂, change_precision(T₂, qu₁); rtol=rtol)
        @test ≈(bq₂, change_precision(T₂, bq₁); rtol=rtol)
        @test ≈(bi₂, change_precision(T₂, bi₁); rtol=rtol)

        nothing
    end

    convert_linesearches_test(Float32, Float64)
    convert_linesearches_test(Float64, Float32)

    # the former `Base.convert(::Type, ::LinesearchMethod)`
    # catch-all was ambiguous with Base and violated the `convert` contract.
    # `convert` on a linesearch method falls back to Base's default behaviour
    # and no longer throws an ambiguity error.
    @test convert(Any, Static()) === Static()
    @test convert(Static, Static()) === Static()
    # precision changes go through `change_precision`, which returns the
    # correct element type (not a differently-typed object from `convert`).
    @test change_precision(Float32, Static()) isa Static{Float32}
    @test eltype(change_precision(Float32, Static())) == Float32

end


@testset "$(rpad("Broken convenience entry points", 80))" begin
    x₀ = -3.0
    ls_problem = make_linesearch_problem(x₀)

    @test Linesearch(ls_problem, Static(1.0), Options()) isa Linesearch
    @test solve(ls_problem, Static(1.0), 0.0) == 1.0
    @test solve(ls_problem, Static(0.8), 0.0, NullParameters(), Options()) == 0.8

    fb(α, _) = α - 1.0
    @test bisection(fb, 0.5) ≈ 1.0 atol = 1e-6

    root_problem = LinesearchProblem{Float64}((α, _) -> α - 1.0, (α, _) -> 1.0)
    lo, hi = bracket_root(root_problem, NullParameters(), 0.5)
    @test lo ≤ 1.0 ≤ hi
end


@testset "$(rpad("Mixed-precision compute_new_iterate!", 80))" begin
    x = Float32[1.0, 2.0]
    p = Float32[1.0, 1.0]
    # α is Float64 → mixed precision path, which emits a warning by design
    @test_logs (:warn,) compute_new_iterate!(x, 1.0, p)
    @test x ≈ Float32[2.0, 3.0]
end


@testset "$(rpad("Type-stability fixes", 80))" begin
    @test (@inferred Bisection()) === Bisection{Float64}()
    @test (@inferred Bisection(Float32)) === Bisection{Float32}()

    # `bisection` promotes integer endpoints to floating point on entry.
    fint(α, _) = α - 2.0
    r = bisection(fint, 0, 4)
    @test r ≈ 2.0 atol = 1e-6
    @test r isa AbstractFloat

    # `CurvatureCondition` encodes the mode in the type (via `Val`) so it is
    # inference-stable, validates `c ∈ (0, 1)`, and the strong condition uses `≤`.
    @test CurvatureCondition(0.9, -1.0, sin, Val(:Strong)) isa CurvatureCondition{Float64,typeof(sin),:Strong}
    @test CurvatureCondition(0.9, -1.0, sin) isa CurvatureCondition{Float64,typeof(sin),:Standard}
    @test_throws AssertionError CurvatureCondition(1.5, -1.0, sin)   # c ∉ (0, 1)
    @test_throws AssertionError CurvatureCondition(0.0, -1.0, sin)   # c ∉ (0, 1)
    # strong-Wolfe boundary: |D(α)| == |c·d| must pass (was strict `<`)
    ccs = CurvatureCondition(0.9, -1.0, α -> 0.9, Val(:Strong))
    @test ccs(0.0)                                                   # |0.9| ≤ |0.9·(-1)|
    # standard curvature: D(α) ≥ c·d
    ccn = CurvatureCondition(0.9, -1.0, α -> -0.5, Val(:Standard))
    @test ccn(0.0)                                                   # -0.5 ≥ -0.9

    @test Options(Float64; x_abstol=0) isa Options
    @test Options(Float64; f_abstol=1 // 100) isa Options
    @test Options().f_abstol == 0.0
end


@testset "$(rpad("Bisection hardening", 80))" begin
    # A genuine sign-changing bracket still bisects to the root.
    froot(α, _) = α - 1.0
    @test bisection(froot, 0.0, 2.0) ≈ 1.0 atol = 1e-6

    # No sign change over the bracket: rather than silently collapsing onto α₁
    # or erroring (which would break the line search once the
    # derivative has flattened at a minimum), `bisection` returns the endpoint
    # closest to a root (smallest |f|).
    fpos(α, _) = α + 1.0            # strictly positive on [0, 1] → no sign change
    @test bisection(fpos, 0.0, 1.0) == 0.0    # |f(0)| = 1 < |f(1)| = 2
    @test bisection(fpos, 1.0, 0.0) == 0.0    # endpoints get flipped internally

    # The debug `println` and hard `error("Max iteration number exceeded")` were
    # A tight tolerance forces exhaustion here.
    fslow(α, _) = α - 1 / 3
    cfg = Options(Float64; linesearch_max_iterations=2, x_suctol=0.0, f_abstol=0.0, verbosity=0)
    local αbest
    @test (αbest = bisection(fslow, 0.0, 1.0, NullParameters(), cfg)) isa Float64
    @test 0.0 ≤ αbest ≤ 1.0
end

@testset "$(rpad("bisection interval/config disambiguation", 80))" begin
    froot(α, _) = α - 1.0
    cfg = Options(Float64)
    @test bisection(froot, 0.0, 2.0, cfg) ≈ 1.0 atol = 1e-6
    @test bisection(froot, 0.0, 2.0, cfg) == bisection(froot, 0.0, 2.0, NullParameters(), cfg)
end

@testset "$(rpad("StrongWolfe line search (bracket + zoom)", 80))" begin
    # For f(x) = x² − 1 with the Newton direction δx = −g/2 the line minimiser is
    # at α = 1 (φ'(1) = 0), so the strong Wolfe conditions are met exactly there.
    prob = make_linesearch_problem(-3.0)
    c₁ = 1e-4
    c₂ = 0.9
    ls = Linesearch(prob, StrongWolfe(); x_abstol=0.0)
    φ0 = value(prob, 0.0, NullParameters())
    d0 = derivative(prob, 0.0, NullParameters())
    # The returned step must satisfy *both* strong Wolfe conditions (that is the
    # whole point of the method — Backtracking guarantees only sufficient decrease).
    # Note the conditions do not pin α = 1: with c₂ = 0.9 the strong-curvature
    # condition holds for a range of α, so a good α₀ may be accepted directly.
    for α₀ in (0.1, 0.5, 1.0, 2.0)
        α = solve(ls, α₀)
        φα = value(prob, α, NullParameters())
        dα = derivative(prob, α, NullParameters())
        @test φα ≤ φ0 + c₁ * α * d0            # sufficient decrease (Armijo)
        @test abs(dα) ≤ c₂ * abs(d0)           # strong curvature
        @test α > zero(α)                      # a genuine positive step
    end

    # A tighter c₂ forces the search onto the exact minimiser α = 1 (φ'(1) = 0),
    # exercising the bracket → zoom path from an overshooting α₀.
    ls_tight = Linesearch(prob, StrongWolfe(; c₂=1e-2); x_abstol=0.0)
    @test solve(ls_tight, 2.0) ≈ 1.0 atol = 1e-6
    @test compute_new_iterate(-3.0, solve(ls_tight, 2.0), δx(-3.0)) ≈ 0.0 atol = 1e-6

    # Constructor validation and helpers.
    @test_throws AssertionError StrongWolfe(Float64; c₁=0.9, c₂=0.1)   # need c₁ < c₂
    @test_throws AssertionError StrongWolfe(Float64; αmax=-1.0)
    @test change_precision(Float32, StrongWolfe(Float64)) isa StrongWolfe{Float32}
    @test StrongWolfe(Float64) ≈ StrongWolfe(Float64)

    # Non-descent direction: φ'(0) = 2 ≥ 0 ⇒ the method returns the trial step and
    # does not attempt a search it cannot complete.
    ascent = LinesearchProblem{Float64}((a, _) -> (a + 1.0)^2, (a, _) -> 2(a + 1.0))
    ls_asc = Linesearch(ascent, StrongWolfe(); verbosity=0)
    @test solve(ls_asc, 0.7) == 0.7
end

@testset "$(rpad("bracketing line searches use the caller's α₀", 80))" begin
    # For f(x) = x² − 1 with the Newton direction δx = −g/2 the line minimiser is at
    # α = 1 (x₀ + 1·δx = 0); every α₀ (spanning under-/over-shoot) must converge there.
    prob = make_linesearch_problem(-3.0)
    for method in (Bisection(), Quadratic(), BierlaireQuadratic())
        ls = Linesearch(prob, method; x_abstol=0.0)
        for α₀ in (0.25, 0.5, 1.0, 2.0, 4.0)
            @test compute_new_iterate(-3.0, solve(ls, α₀), δx(-3.0)) ≈ 0.0 atol = ∛(2eps())
        end
    end
end

@testset "$(rpad("Quadratic returns the tested bracket point, and never a negative step", 80))" begin
    # φ(α) = (α + 1)² is *increasing* at the α = 0 anchor (φ'(0) = 2 > 0) with its minimiser at
    # α = -1.  `bracket_minimum_with_fixed_point` handles that by flipping direction, so this
    # search used to return α ≈ -1 — a step *against* the direction.  That is meaningless as a
    # step length (α scales a direction that is already chosen), and measured on a Newton solve
    # the capability was emergent rather than designed: it arises only from a stale Jacobian, it
    # was inherited from whichever bracketer each method happened to call, and `Bisection`
    # produced steps as large as α = -3.  The α > 0 contract now applies to every method, so an
    # ascent anchor is reported instead.
    prob = LinesearchProblem{Float64}((a, _) -> (a + 1.0)^2, (a, _) -> 2.0 * (a + 1.0))
    ls = Linesearch(prob, Quadratic(); x_abstol=0.0, verbosity=0)
    st = solve_with_status(ls, 0.0)
    @test outcome(st) == LINESEARCH_NO_DESCENT
    @test steplength(st) > 0.0
    @test solve(ls, 0.0) == steplength(st)

    # The original point of this test survives on a *descent* anchor: the bracket's left
    # endpoint `a`, where the derivative is tested and the near-stationary early return fires,
    # differs from the loop's start `α`, and it is `a` that must be returned.  Here the
    # minimiser sits at α = +1.
    descent = LinesearchProblem{Float64}((a, _) -> (a - 1.0)^2, (a, _) -> 2.0 * (a - 1.0))
    lsd = Linesearch(descent, Quadratic(); x_abstol=0.0, verbosity=0)
    @test solve(lsd, 0.0) ≈ 1.0 atol = ∛(2eps())
    @test solve(lsd, 2.0) ≈ 1.0 atol = ∛(2eps())   # α₀ past the minimiser still lands on it
end

# Quadratic and BierlaireQuadratic validate their constructor parameters, like
# Backtracking and StrongWolfe.
@testset "$(rpad("Quadratic/BierlaireQuadratic constructor validation", 80))" begin
    @test_throws AssertionError Quadratic(Float64; ε=0.0)
    @test_throws AssertionError Quadratic(Float64; s=-1.0)
    @test_throws AssertionError Quadratic(Float64; s_reduction=1.5)
    @test_throws AssertionError BierlaireQuadratic(Float64; ε=0.0)
    @test_throws AssertionError BierlaireQuadratic(Float64; ξ=-1.0)
    @test Quadratic() isa Quadratic                      # defaults are valid
    @test BierlaireQuadratic() isa BierlaireQuadratic    # defaults are valid
end

# Check that `bracket_minimum_with_fixed_point` returns the
# merit values at the bracket endpoints alongside the bracket — they are computed
# during bracketing anyway, so the Quadratic line search no longer re-evaluates
# them.  Both quadratic line searches iterate instead of recursing, which for
# BierlaireQuadratic also stops fa/fb/fc from being recomputed at every level.
@testset "$(rpad("bracket_minimum_with_fixed_point returns endpoint values", 80))" begin
    f(x) = (x - 1)^2
    a, b, fa, fb = SimpleSolvers.bracket_minimum_with_fixed_point(f, 0.0, 0.01)
    @test a < 1.0 < b                      # brackets the minimum
    @test fa == f(a) && fb == f(b)         # returned values match the endpoints

    # flipped start: the merit initially increases to the right, so the search
    # flips and expands leftward — the value/endpoint pairing must survive both
    # the flip and the final reordering
    g(x) = x^2
    a2, b2, fa2, fb2 = SimpleSolvers.bracket_minimum_with_fixed_point(g, 0.1, 0.01)
    @test a2 < 0.0 < b2
    @test fa2 == g(a2) && fb2 == g(b2)
end

@testset "$(rpad("Quadratic searches: merit-evaluation canary", 80))" begin
    # Deterministic evaluation counts on an exactly quadratic merit; measured
    # 13 (Quadratic) and 16 (Bierlaire) after removing the redundant endpoint
    # re-evaluations / per-recursion recomputation.  The bounds leave headroom
    # but catch a reintroduced per-iteration re-evaluation.
    for (method, α_expected, bound) in ((Quadratic(), 1.0, 16), (BierlaireQuadratic(), 1.0, 20))
        cnt = Ref(0)
        prob = LinesearchProblem{Float64}((α, p) -> (cnt[] += 1; (α - 1)^2), (α, p) -> 2(α - 1))
        ls = Linesearch(prob, method; verbosity=0)
        α = solve(ls, 0.5)
        @test α ≈ α_expected atol = 1e-8
        @test cnt[] ≤ bound
    end
end

@testset "$(rpad("the benign outcomes do not warn at the default verbosity", 80))" begin
    # Reaching the merit's round-off floor is the *normal* final state of a converged solve, so
    # reporting it at the default verbosity would mean warning about success.  Measured on
    # GeometricProblems, gating it at `verbosity ≥ 1` newly surfaced a message on three
    # previously-silent, correctly-converging integrations.  Whether an irreducible merit
    # matters is the outer iteration's call (see `stalled_step`).
    noise = LinesearchProblem{Float64}((α, _) -> α > zero(α) ? nextfloat(1.0) : 1.0, (α, _) -> -2.0)
    lsdefault = Linesearch(noise, Backtracking())          # verbosity = 1, the default
    @test outcome(solve_with_status(lsdefault, 1.0)) == LINESEARCH_FLOOR
    @test (@test_logs solve(lsdefault, 1.0)) == solve(lsdefault, 1.0)   # silent

    # ... but it is available for diagnosis
    lsverbose = Linesearch(noise, Backtracking(); verbosity=2)
    @test_logs (:warn, r"round-off floor") match_mode = :any solve(lsverbose, 1.0)

    # a genuine inconsistency between the slope and the values *does* warn at verbosity 1
    lying = LinesearchProblem{Float64}((α, _) -> 1.0 + α, (α, _) -> -2.0)
    @test_logs (:warn, r"did not satisfy|no step satisfied") match_mode = :any solve(Linesearch(lying, Backtracking()), 1.0)

    # ... and so does a non-descent direction
    up = LinesearchProblem{Float64}((α, _) -> α + 1.0, (α, _) -> 1.0)
    @test_logs (:warn, r"not a descent direction") match_mode = :any solve(Linesearch(up, Backtracking()), 1.0)
end

@testset "$(rpad("bracketing helpers report failure instead of throwing", 80))" begin
    # A merit that is flat to round-off: `1.0 + 1e-20x` computes to exactly 1.0 for every small
    # x, so no probe can find a decrease.  This used to halve δ five times and then throw,
    # aborting the enclosing solve — and halving is precisely the wrong response, since a
    # smaller probe is strictly *less* informative than the one that already failed.
    n = Ref(0)
    flat = x -> (n[] += 1; 1.0 + 1e-20 * x)
    n[] = 0
    @test triple_point_finder(flat, 0.0) === :flat
    @test n[] == 2                       # the anchor and one probe; was 12 followed by a throw

    # A strictly increasing merit, and one with no minimum to the right, likewise report rather
    # than throw — but as `:unbracketable`, *not* `:flat`.  Both of these have a merit that
    # varies far more than round-off, so a caller must not report them as a round-off floor: the
    # second one is in fact descending without bound, which is the opposite of stagnation.
    @test triple_point_finder(x -> x + 1.0, 0.0) === :unbracketable
    @test triple_point_finder(x -> -x, 0.0) === :unbracketable

    # ... while a genuine overshoot still gets the δ-halving retry it needs
    @test triple_point_finder(x -> (x - 0.001)^2, 0.0) isa Tuple

    # the success path is unchanged
    a, b, c = triple_point_finder(x -> x^2, -1.0)
    @test a < b < c

    # the sibling bracketers report `nothing` on exhaustion rather than erroring
    @test bracket_minimum(x -> -x, 0.0; nmax=3) === nothing
    @test SimpleSolvers.bracket_minimum_with_fixed_point(x -> -x, 0.0, 0.01, 2.0, 3) === nothing
    @test bracket_minimum(x -> (x - 1)^2, 0.0) !== nothing
end

@testset "$(rpad("check_anchor", 80))" begin
    @test outcome(SimpleSolvers.check_anchor(NaN, -1.0, 0.7)) == LINESEARCH_NO_DESCENT
    @test outcome(SimpleSolvers.check_anchor(1.0, NaN, 0.7)) == LINESEARCH_NO_DESCENT
    @test outcome(SimpleSolvers.check_anchor(Inf, -1.0, 0.7)) == LINESEARCH_NO_DESCENT
    @test outcome(SimpleSolvers.check_anchor(1.0, 2.0, 0.7)) == LINESEARCH_NO_DESCENT
    @test outcome(SimpleSolvers.check_anchor(1.0, 0.0, 0.7)) == LINESEARCH_STATIONARY
    @test SimpleSolvers.check_anchor(1.0, -2.0, 0.7) === nothing   # healthy anchor: proceed

    # the caller's trial step is handed back, and a non-positive one is replaced by the unit
    # step so that the α > 0 guarantee holds regardless of what the caller passed
    @test steplength(SimpleSolvers.check_anchor(1.0, 2.0, 0.7)) == 0.7
    @test steplength(SimpleSolvers.check_anchor(1.0, 2.0, 0.0)) == 1.0
    @test steplength(SimpleSolvers.check_anchor(1.0, 2.0, -3.0)) == 1.0
end

@testset "$(rpad("the line search contract holds for every method", 80))" begin
    # One loop over every element type × every method × every pathological anchor.  This is the
    # test that keeps the contract standardised: none of these may throw, and none may return
    # α ≤ 0.  The low-precision rows matter in their own right: `Float16` is the precision where
    # τ = 4·ulp(φ(0)) is *larger* than the decrease the Armijo condition demands at α = 1, and
    # where `backtracking_αmin`'s √eps clamp is always active, so it is where the bounds on the
    # round-off allowance are load-bearing rather than decorative.
    for T in (Float64, Float32, Float16)
        one_T = one(T)
        pathological = (
            ("NaN merit", (α, _) -> T(NaN), (α, _) -> T(NaN)),
            ("NaN derivative", (α, _) -> one_T - α, (α, _) -> T(NaN)),
            ("Inf merit", (α, _) -> T(Inf), (α, _) -> -one_T),
            ("ascent anchor", (α, _) -> α + one_T, (α, _) -> one_T),
            ("stationary anchor", (α, _) -> -one_T, (α, _) -> zero(T)),
            ("flat to round-off", (α, _) -> α > zero(α) ? nextfloat(one_T) : one_T, (α, _) -> -2one_T),
            ("minimiser at α < 0", (α, _) -> (α + one_T)^2, (α, _) -> 2 * (α + one_T)),
            ("slope contradicts values", (α, _) -> one_T + α, (α, _) -> -2one_T),
        )
        for m in (Static(T), Backtracking(T), StrongWolfe(T), Bisection(T), Quadratic(T), BierlaireQuadratic(T))
            for (nm, f, d) in pathological
                ls = Linesearch(LinesearchProblem{T}(f, d), m; verbosity=0)
                st = @test_nowarn solve_with_status(ls, one_T)
                @test st isa LinesearchStatus{T}
                @test steplength(st) > zero(T)            # the α > 0 guarantee
                @test isfinite(steplength(st))
                @test solve(ls, one_T) == steplength(st)  # `solve` is a thin wrapper
            end
        end
    end
end

@testset "$(rpad("armijo_ulps caps the round-off resolution at what the precision supports", 80))" begin
    # τ = τ_ulps·ulp(φ₀) has to be at least ~an ulp to recognise a merit at its round-off floor,
    # and far below the decrease the Armijo condition demands at α = 1 (2c₁·φ₀ for the canonical
    # ‖F‖² Newton merit) to leave that condition meaningful.  Those are compatible only while
    # eps(T) ≪ 2c₁ — true by a wide margin in Float64, comfortably in Float32, and *false* in
    # Float16, where eps = 9.8e-4 already exceeds 2c₁ = 2e-4.  `armijo_ulps` caps the nominal 4
    # accordingly, which is a no-op in the two precisions that can afford it.
    @test armijo_ulps(Float64) == SimpleSolvers.DEFAULT_ARMIJO_τ_ULPS
    @test armijo_ulps(Float32) == SimpleSolvers.DEFAULT_ARMIJO_τ_ULPS
    @test armijo_ulps(Float16) < 1                       # no room for even a single ulp
    @test armijo_ulps(Float16) > 0

    for T in (Float64, Float32, Float16)
        c₁ = T(SimpleSolvers.DEFAULT_WOLFE_c₁)
        τ = armijo_tolerance(one(T), armijo_ulps(T))
        demanded = 2 * c₁                                # the decrease demanded at α = 1
        # the invariant the cap exists for; the slack absorbs the rounding of the cap itself,
        # which in Float16 is computed in Float16
        @test τ ≤ 1.1 * SimpleSolvers.ARMIJO_τ_DEMAND_FRACTION * demanded
        # ... and a tighter c₁ tightens the cap rather than being ignored
        @test armijo_ulps(T, c₁ / 1000) ≤ armijo_ulps(T, c₁)
    end

    # The cap is applied by the inner constructor, so *every* path into a `Backtracking{T}` gets
    # a resolution the element type supports — including `change_precision`, which converts a
    # method built for a different `T`, and an explicit value that is too large.
    for T in (Float64, Float32, Float16)
        @test Backtracking(T).τ_ulps == armijo_ulps(T, T(SimpleSolvers.DEFAULT_WOLFE_c₁))
        @test change_precision(T, Backtracking()).τ_ulps == Backtracking(T).τ_ulps
        @test Backtracking(T; τ_ulps=T(4)).τ_ulps ≤ armijo_ulps(T)
        @test Backtracking(T; τ_ulps=zero(T)).τ_ulps == 0   # opting out still works
    end
end

@testset "$(rpad("a small but genuine Float16 decrease is a decrease, not the floor", 80))" begin
    # With the nominal 4 ulps, τ/φ₀ = 3.9e-3 in Float16 — twenty times the 2c₁ = 2e-4 the
    # condition demands at α = 1.  A merit that really did decrease by two ulps was then
    # classified `LINESEARCH_FLOOR`, which `solver_step!` feeds to `flag_stall!`: two such steps
    # and a *converging* solve was reported as stagnated.  The cap fixes the classification.
    T = Float16
    φ = α -> one(T) - T(0.004) * α + T(0.002) * α^2       # minimum at α = 1, φ(1) = 0.998
    prob = LinesearchProblem{T}((α, _) -> φ(α), (α, _) -> -T(0.004) + T(0.004) * α)

    decrease = φ(zero(T)) - φ(one(T))
    @test decrease == 2 * eps(T)                          # two ulps: the smallest Float16 can show

    τ = armijo_tolerance(one(T), armijo_ulps(T))
    τnominal = armijo_tolerance(one(T), T(SimpleSolvers.DEFAULT_ARMIJO_τ_ULPS))
    @test φ(one(T)) ≤ φ(zero(T)) - τ                      # counts as a decrease with the cap ...
    @test !(φ(one(T)) ≤ φ(zero(T)) - τnominal)            # ... and did not without it

    # the two methods that actually run an Armijo test report it as progress
    for m in (Backtracking(T), StrongWolfe(T))
        st = solve_with_status(Linesearch(prob, m; verbosity=0), one(T))
        @test issufficient(st)
        @test outcome(st) == LINESEARCH_DECREASED
    end

    # and Float64 is untouched: the same relative decrease is far above its τ either way
    @test armijo_tolerance(1.0, armijo_ulps(Float64)) == armijo_tolerance(1.0, 4)
end

@testset "$(rpad("no method accepts a step that increases the merit", 80))" begin
    # The round-off allowance τ slackens the *demanded* decrease, so it must never license a step
    # whose merit is above φ(0). The bound is invisible in Float64 (τ/φ₀ ≈ 1e-15) and essential in
    # Float16, where τ/φ₀ = 3.9e-3 is twenty times the 2c₁ = 2e-4 demanded at α = 1: an unbounded
    # τ accepted α = 1 on this merit and reported it as a step.
    for T in (Float64, Float32, Float16)
        τ = 4 * eps(one(T))
        # rises immediately, by less than τ — inside the unbounded allowance, outside the bounded one
        creep = LinesearchProblem{T}((α, _) -> one(T) + (α > zero(α) ? τ / 2 : zero(T)), (α, _) -> -2one(T))
        for m in (Backtracking(T), StrongWolfe(T), Bisection(T), Quadratic(T), BierlaireQuadratic(T))
            st = solve_with_status(Linesearch(creep, m; verbosity=0), one(T))
            @test !issufficient(st)                 # never reported as a genuine decrease
            @test outcome(st) != LINESEARCH_DECREASED
        end
        # and the condition object itself rejects it at every α
        sdc = SufficientDecreaseCondition(T(1e-4), one(T), -2one(T), α -> one(T) + τ / 2; τ=τ)
        @test !sdc(one(T))
        @test !sdc(T(1e-3))
        @test !sdc(eps(T))
    end
end

@testset "$(rpad("StrongWolfe reports a non-finite anchor instead of asserting", 80))" begin
    # A `NaN` derivative is not `≥ zero(T)`, so it used to slip past StrongWolfe's descent check
    # and trip `SufficientDecreaseCondition`'s `@assert !isnan(d₀)`, throwing an `AssertionError`
    # out of the enclosing solve.  It now matches `Backtracking` on the same problems.
    for (f, d) in (((α, _) -> NaN, (α, _) -> NaN), ((α, _) -> 1.0 - α, (α, _) -> NaN))
        prob = LinesearchProblem{Float64}(f, d)
        sw = solve_with_status(Linesearch(prob, StrongWolfe(); verbosity=0), 0.7)
        bt = solve_with_status(Linesearch(prob, Backtracking(); verbosity=0), 0.7)
        @test outcome(sw) == outcome(bt) == LINESEARCH_NO_DESCENT
        @test steplength(sw) == steplength(bt) == 0.7
    end
end

@testset "$(rpad("cost is independent of the merit's scale", 80))" begin
    # φ(α) = c·(α-1)² has its minimiser at α = 1 for every c > 0, so neither the returned step
    # nor the number of merit evaluations may depend on c.  `BierlaireQuadratic` used to cost 15
    # / 70 / 14 evaluations at c = 1 / 1e-6 / 1e-12: at c = 1e-6 it stalled on a single point
    # (evaluating α = 0.99999999999999911 thirty-plus times in a row) and ran out its budget.
    for m in (Backtracking(), StrongWolfe(), Bisection(), Quadratic(), BierlaireQuadratic())
        αs = Float64[]
        counts = Int[]
        for c in (1e-12, 1e-6, 1.0, 1e6, 1e12)
            n = Ref(0)
            prob = LinesearchProblem{Float64}((α, _) -> (n[] += 1; c * (α - 1.0)^2), (α, _) -> c * 2 * (α - 1.0))
            push!(αs, solve(Linesearch(prob, m; verbosity=0), 0.5))
            push!(counts, n[])
        end
        @test all(≈(first(αs)), αs)                        # same step at every scale
        @test maximum(counts) - minimum(counts) ≤ 2        # and essentially the same cost
        @test maximum(counts) ≤ 25
    end
end

@testset "$(rpad("BierlaireQuadratic contracts its bracket every iteration", 80))" begin
    # The no-stall canary for the scale at which it used to spin: no single α may be evaluated
    # more than a couple of times.
    for c in (1.0, 1e-6)
        αs = Float64[]
        prob = LinesearchProblem{Float64}((α, _) -> (push!(αs, α); c * (α - 1.0)^2), (α, _) -> c * 2 * (α - 1.0))
        α = solve(Linesearch(prob, BierlaireQuadratic(); verbosity=0), 0.5)
        @test α ≈ 1.0 atol = 1e-8
        @test maximum(count(==(u), αs) for u in unique(αs)) ≤ 3
        @test length(αs) ≤ 25
    end
end

@testset "$(rpad("trials is a real evaluation count for every method", 80))" begin
    # `trials` used to be a hardcoded 0 for Bisection/Quadratic/BierlaireQuadratic — so the
    # round-off-floor message read "in 0 trial step(s)" — and StrongWolfe counted only its
    # expansion loop, not its zoom phase.  What each method counts is the problem evaluations of
    # its *own* iteration: the merit, except for `Bisection`, which drives on the derivative it
    # bisects.  See the `trials` field of `LinesearchStatus`.
    function counted(m)
        nf, nd = Ref(0), Ref(0)
        prob = LinesearchProblem{Float64}((α, _) -> (nf[] += 1; 1.0 - 2α + 1000α^2),
            (α, _) -> (nd[] += 1; -2.0 + 2000α))
        st = solve_with_status(Linesearch(prob, m; verbosity=0), 1.0)
        (st, nf[], nd[])
    end

    # Backtracking and StrongWolfe drive on the merit alone, and their count is *exact*: every
    # evaluation is either the α = 0 anchor or a counted trial.  StrongWolfe used to evaluate φ
    # twice per trial — once directly, once inside the one-argument `sdc` — and once more when
    # building its status, so this identity is the regression test for that fix.
    for m in (Backtracking(), StrongWolfe())
        st, nf, _ = counted(m)
        @test trials(st) > 0
        @test nf == trials(st) + 1
    end

    # The bracketing searches additionally spend evaluations inside `bracket_minimum` /
    # `triple_point_finder`, which are not counted, so their `trials` is a lower bound on the
    # total cost — but it is a real count of their own iteration, never zero and never inflated.
    for m in (Quadratic(), BierlaireQuadratic())
        st, nf, _ = counted(m)
        @test 0 < trials(st) ≤ nf
    end
    let (st, _, nd) = counted(Bisection())
        @test 0 < trials(st) ≤ nd
    end
end

@testset "$(rpad("αmin is reported only where it means something", 80))" begin
    # αmin is a shrinking-ladder quantity. `Backtracking` derives a real one; the bracketing and
    # minimising searches have none, report zero, and must not name it in their messages.
    noise = LinesearchProblem{Float64}((α, _) -> α > zero(α) ? nextfloat(1.0) : 1.0, (α, _) -> -2.0)
    @test solve_with_status(Linesearch(noise, Backtracking(); verbosity=0), 1.0).αmin > 0
    for m in (StrongWolfe(), Bisection(), Quadratic(), BierlaireQuadratic())
        @test iszero(solve_with_status(Linesearch(noise, m; verbosity=0), 1.0).αmin)
    end

    # the verbosity-2 floor message names αmin for Backtracking, and reports the true trial count
    msg = @test_logs (:warn, r"smallest informative step is αmin") match_mode = :any solve(
        Linesearch(noise, Backtracking(); verbosity=2), 1.0)
    @test msg isa Float64
    # ... and the count is whatever the method really spent, which for `BierlaireQuadratic` on
    # this merit is legitimately zero: `triple_point_finder` recognises the flat merit from the
    # anchor and one probe, so the fit never runs. A *genuine* zero is fine; the defect was a
    # hardcoded one, which the previous testset pins down on a merit that does iterate.
    for m in (StrongWolfe(), Bisection(), Quadratic(), BierlaireQuadratic())
        st = solve_with_status(Linesearch(noise, m; verbosity=0), 1.0)
        @test trials(st) ≥ 0
        @test iszero(st.αmin)
    end
end

@testset "$(rpad("the linesearch messages are compiled once, not once per solver", 80))" begin
    # `linesearch_warnings` is called from `solver_step!` on every iteration of every solve, and
    # it is specialized on the `Linesearch` — hence on the closure types of its
    # `LinesearchProblem`, hence once per *problem* a solver is built for. The messages therefore
    # live behind the `report_linesearch_status` barrier, whose signature mentions no closure type;
    # see its docstring.

    # Half one: the specialization set is bounded by the *signature*. Julia specializes on the
    # concrete types of the arguments, so if no parameter type can admit a `Linesearch` — which
    # includes an untyped parameter, since `Linesearch <: Any` — then the concrete argument types
    # are drawn from `{LinesearchStatus{T}} × {Symbol} × {Options{T}}` and cannot grow with the
    # number of solvers built. This must stay a test on the types: a parameter written
    # `ls::Linesearch` stringifies *without* braces, so a substring test for "Linesearch{" would
    # miss exactly the regression guarded against here, while one for "Linesearch" would match
    # `LinesearchStatus`.
    for m in methods(SimpleSolvers.report_linesearch_status)
        argtypes = collect(Base.unwrap_unionall(m.sig).parameters)[2:end]
        @test !any(p -> Linesearch <: Base.unwrap_unionall(p), argtypes)
        @test !any(p -> NamedTuple <: Base.unwrap_unionall(p), argtypes)
    end

    # Half two: the messages are in the barrier and nowhere else. `@warn` is expanded by the macro,
    # so a message in the body of `f` is visible in `f`'s lowered code as a `GlobalRef` into
    # `Base.CoreLogging` — which makes "no reporting code in a function specialized on a merit
    # closure" directly checkable, for every reporter at once, without any notion of
    # specialization. `curvature_diagnostic` is included on the closure-free side because it too
    # is called from `linesearch_warnings` and is specialized per problem.
    for f in (SimpleSolvers.report_linesearch_status, SimpleSolvers.report_curvature_violation,
        SimpleSolvers.report_bisection_nonconvergence, SimpleSolvers.report_bisection_nobracket)
        @test has_logging_code(f)
    end
    for f in (SimpleSolvers.linesearch_warnings, SimpleSolvers.curvature_diagnostic,
        SimpleSolvers.bisection, SimpleSolvers._bisection_core)
        @test !has_logging_code(f)
    end

    # The barrier really is on the path a solve takes, for every method, and stays quiet when there
    # is nothing to report.
    F(y, x, params) = y .= x .^ 2 .- 2
    for ls in (Backtracking(), StrongWolfe(), Bisection(), Quadratic(), BierlaireQuadratic())
        x = ones(3)
        s = NewtonSolver(x, similar(x); F=F, linesearch=ls, verbosity=1)
        @test_logs solve!(x, s)
        @test x ≈ fill(sqrt(2), 3)
    end

    # The two `bisection` messages moved into reporters of their own, so their verbosity gates are
    # no longer visible at the site that decides to report. Pin them: each fires at its documented
    # level and is silent one below. Neither carries `maxlog`, so this is repeatable within a
    # session, unlike the line-search messages.
    fslow(α, _) = α - 1 / 3
    nonconvergence(v) = () -> bisection(fslow, 0.0, 1.0, NullParameters(),
        Options(Float64; linesearch_max_iterations=2, x_suctol=0.0, f_abstol=0.0, verbosity=v))
    @test logged_any(nonconvergence(1), "did not converge within")
    @test !logged_any(nonconvergence(0), "did not converge within")

    fpos(α, _) = α + 1.0            # strictly positive on [0, 1] → no sign change
    nobracket(v) = () -> bisection(fpos, 0.0, 1.0, NullParameters(), Options(Float64; verbosity=v))
    @test logged_any(nobracket(2), "shows no sign change")
    @test !logged_any(nobracket(1), "shows no sign change")
end

@testset "$(rpad("every clause of a linesearch message is built only when it is shown", 80))" begin
    # The `αmin` clause of `LINESEARCH_FLOOR` and both wordings of `LINESEARCH_EXHAUSTED` are
    # interpolated inside their `@warn` rather than into a temporary before it, so that a message
    # the verbosity gate or `maxlog` discards costs nothing (see `report_linesearch_status`). The
    # exact texts are pinned here because that laziness is easy to undo by rewriting the
    # interpolation, and easy to undo *silently*. `Test.TestLogger` records every message regardless
    # of `maxlog`, which is keyed on the source location and therefore process-global.
    reported(st, v) = () -> SimpleSolvers.report_linesearch_status(st, :Backtracking,
        Options(Float64; verbosity=v, linesearch_max_iterations=7))

    # φ = 0.5 against φ₀ = 1.0, so the merit difference the second EXHAUSTED wording names is exact.
    status(oc, α, αmin) = LinesearchStatus{Float64}(α, oc, 3, 1.0, -2.0, 0.5, 1.0e-16, αmin)

    @test logged_any(reported(status(LINESEARCH_FLOOR, 0.5, 0.0), 2),
        "Backtracking line search: no trial step changed the merit by more than the round-off resolution τ = 1.0e-16 in 3 trial step(s). φ(0) = 1.0 has reached its round-off floor, so no step can decrease it. Returning α = 0.5. Check whether the requested residual tolerance is attainable in this precision.")
    @test logged_any(reported(status(LINESEARCH_FLOOR, 0.5, 1.0e-8), 2),
        "so no step can decrease it (the smallest informative step is αmin = 1.0e-8). Returning α = 0.5.")
    @test !logged_any(reported(status(LINESEARCH_FLOOR, 0.5, 1.0e-8), 1), "round-off floor")

    # α > αmin selects the budget wording, α ≤ αmin the inconsistent-derivative one.
    @test logged_any(reported(status(LINESEARCH_EXHAUSTED, 0.5, 0.0), 1),
        "Backtracking line search: no step satisfied the sufficient decrease condition in 3 trial step(s) — the budget linesearch_max_iterations = 7 was spent, or the merit could not be bracketed. Returning α = 0.5.")
    @test logged_any(reported(status(LINESEARCH_EXHAUSTED, 1.0e-8, 1.0e-8), 1),
        "— the merit changed by -0.5 at the smallest informative step αmin = 1.0e-8, which exceeds the round-off resolution τ = 1.0e-16, so φ'(0) = -2.0 is inconsistent with the merit (a stale or regularized Jacobian, an inexact linear solve, or a non-smooth problem). Returning α = 1.0e-8.")
    @test !logged_any(reported(status(LINESEARCH_EXHAUSTED, 0.5, 0.0), 0), "sufficient decrease condition")

    # And nothing is built for a message that is not shown. Measured inside a function: from global
    # scope the arguments are boxed and the numbers say nothing about the code under test. This and
    # the solve-path assertions below are the only ones in the suite that depend on codegen, so a
    # future Julia release may call for revisiting them rather than the package.
    function reporting_allocations(oc, v)
        st = status(oc, 0.5, 1.0e-8)
        cfg = Options(Float64; verbosity=v)
        SimpleSolvers.report_linesearch_status(st, :Backtracking, cfg)
        @allocated SimpleSolvers.report_linesearch_status(st, :Backtracking, cfg)
    end
    for oc in instances(LinesearchOutcome)
        @test reporting_allocations(oc, 0) == 0
    end
end

@testset "$(rpad("a converged solve allocates nothing", 80))" begin
    # Every line search is expected to run a solve without touching the heap: the merit closures
    # capture nothing they mutate, and the bracketing helpers return concrete types. `StrongWolfe`
    # is the exception by design — its one-slot memo of φ and φ′ is handed to `_wolfe_zoom` and to
    # the condition objects, so it cannot stay on the stack — and one small holder per line search
    # is the whole of it, which is what the bound below pins.
    F(y, x, params) = y .= x .^ 2 .- 2
    function solve_allocations(ls)
        x = ones(3)
        s = NewtonSolver(x, similar(x); F=F, linesearch=ls, verbosity=0)
        state = SolverState(s)
        solve!(x, s, state)
        x .= 1.0
        @allocated solve!(x, s, state)
    end
    for ls in (Static(), Backtracking(), Bisection(), Quadratic(), BierlaireQuadratic())
        @test solve_allocations(ls) == 0
    end

    function wolfe_allocations()
        ls = Linesearch(make_linesearch_problem(2.0), StrongWolfe(); verbosity=0)
        solve_with_status(ls, 1.0)
        @allocated solve_with_status(ls, 1.0)
    end
    @test wolfe_allocations() ≤ 64
end
