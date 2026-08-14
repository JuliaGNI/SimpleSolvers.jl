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
    backtracking_interpolation, backtracking_extrapolation, with_config, problem, method, config

include("lowered_code.jl")

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

# Two stand-ins for a downstream `LinesearchMethod`, used to pin the extension point: a method
# implements `solve_with_status` and gets `solve` derived from it, and one that implements neither
# is told which of the two it owes instead of recursing between the generic definitions.
struct ToyLinesearch{T} <: LinesearchMethod{T} end
ToyLinesearch(::Type{T}=Float64) where {T} = ToyLinesearch{T}()
SimpleSolvers.solve_with_status(::Linesearch{T,<:ToyLinesearch}, α::T, params=SimpleSolvers.NullParameters()) where {T} =
    LinesearchStatus{T}(T(0.25), LINESEARCH_EXHAUSTED, 1, one(T), -one(T), one(T), zero(T), zero(T))

struct MuteLinesearch{T} <: LinesearchMethod{T} end
MuteLinesearch(::Type{T}=Float64) where {T} = MuteLinesearch{T}()

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
    @test_throws AssertionError Backtracking(; q=1.0)           # q > 1 violated
    # `expand` is the only way to switch the expansion phase off, so `nexpand = 0` — a second,
    # silent encoding of "disabled" — is rejected rather than accepted as a synonym.
    @test_throws AssertionError Backtracking(; nexpand=0)
    @test Backtracking() isa Backtracking                       # defaults are valid
    @test !Backtracking().expand                                # ... and shrink-only

    # `show` says which of the two algorithms the method is, and names the expansion budget only
    # where it is read — the shrink-only default does not carry `q` and `nexpand` into its
    # description of itself.
    let shown = sprint(show, Backtracking())
        @test occursin("shrinking only", shown)
        @test !occursin("q = ", shown)
    end
    let shown = sprint(show, Backtracking(; expand=true, nexpand=2))
        @test occursin("q = $(SimpleSolvers.DEFAULT_BACKTRACKING_q)", shown)
        @test occursin("2 trial(s)", shown)
        @test !occursin("shrinking only", shown)
    end
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

@testset "$(rpad("backtracking_extrapolation", 80))" begin
    # φ(α) = (α - 11)², anchored at α = 0: φ₀ = 121, φ'(0) = -22, φ(1) = 100. The quadratic
    # model through those three values has its minimiser at the true one, α★ = 11.
    @test backtracking_extrapolation(121.0, -22.0, 1.0, 100.0, 100.0) == 11.0
    @test backtracking_extrapolation(121.0, -22.0, 1.0, 100.0, 10.0) == 10.0   # clamped to q·α

    # A well-scaled direction: φ(α) = (α - 1)² has α★ = α = 1, so the step is returned unchanged
    # and the caller stops without spending a merit evaluation. This is the zero-cost gate.
    @test backtracking_extrapolation(1.0, -2.0, 1.0, 0.0, 10.0) == 1.0

    # ... and so is a model minimiser that is longer but not by the factor BACKTRACKING_GROW_MIN.
    # φ(α) = (α - 1.5)²: α★ = 1.5 < 2·1.
    @test backtracking_extrapolation(2.25, -3.0, 1.0, 0.25, 10.0) == 1.0

    # A non-convex model — the merit fell at least as fast as its tangent, so it is still
    # dropping steeply — grows by the full factor q.
    @test backtracking_extrapolation(1.0, -2.0, 1.0, -10.0, 10.0) == 10.0

    # A merit at its round-off floor (φ(α) = φ₀) gives α★ = α/2 and is therefore *not* expanded:
    # the model declines to grow into rounding noise without needing a special case.
    @test backtracking_extrapolation(1.0, -2.0, 1.0, 1.0, 10.0) == 1.0

    # A non-finite model must not propagate a NaN step into the search.
    @test backtracking_extrapolation(1.0, -2.0, 1.0, NaN, 10.0) == 1.0

    # The gate is on the step that would actually be tried, not on the raw minimiser, so a q below
    # BACKTRACKING_GROW_MIN buys nothing whichever branch the model takes: the convex one wants
    # α★ = 11 but may only reach 1.5·α, and the non-convex one asks for exactly that.
    @test backtracking_extrapolation(121.0, -22.0, 1.0, 100.0, 1.5) == 1.0
    @test backtracking_extrapolation(1.0, -2.0, 1.0, -10.0, 1.5) == 1.0
    # ... and at q = BACKTRACKING_GROW_MIN exactly, both do expand, by that factor.
    @test backtracking_extrapolation(121.0, -22.0, 1.0, 100.0, 2.0) == 2.0
    @test backtracking_extrapolation(1.0, -2.0, 1.0, -10.0, 2.0) == 2.0
end

@testset "$(rpad("Backtracking expansion phase (issue #174)", 80))" begin
    quadratic(αmin) = LinesearchProblem{Float64}((α, _) -> (α - αmin)^2, (α, _) -> 2(α - αmin))
    search(prob, m) = solve_with_status(Linesearch(prob, m; verbosity=0), 1.0)

    shrink = Backtracking()
    grow = Backtracking(; expand=true)

    # The defect: a direction whose natural scale is larger than the trial step. A shrink-only
    # search accepts the trial step — it does satisfy sufficient decrease — and hands back the
    # ceiling it was given, at every outer iteration. That is what cost DFP two orders of
    # magnitude in the issue.
    for αmin in (11.0, 100.0)
        st = search(quadratic(αmin), shrink)
        @test steplength(st) == 1.0
        @test trials(st) == 1
    end

    # With the expansion phase the step reaches that scale instead, in at most `nexpand` further
    # merit evaluations. Landing within a factor BACKTRACKING_GROW_MIN of the minimiser is the
    # whole point: it is the *scale* that was wrong, not the last digit.
    for αmin in (11.0, 100.0)
        st = search(quadratic(αmin), grow)
        @test issufficient(st)
        @test αmin / SimpleSolvers.BACKTRACKING_GROW_MIN ≤ steplength(st) ≤ αmin
        @test 1 < trials(st) ≤ 1 + grow.nexpand
    end

    # And it costs nothing where it can gain nothing: a direction already scaled like a Newton
    # step is at its model minimum, so the phase returns after the one trial the shrink-only
    # search would have made. This is why the model is extrapolated rather than α simply grown.
    stw = search(quadratic(1.0), grow)
    @test steplength(stw) == 1.0
    @test trials(stw) == trials(search(quadratic(1.0), shrink)) == 1

    # A shrunken step is never expanded again: the longer steps have already been rejected.
    # φ(α) = 1 - 2α + 1000α² needs several backtracks from α = 1, and `expand` must not change
    # what that returns.
    overshoot = LinesearchProblem{Float64}((α, _) -> 1.0 - 2α + 1000α^2, (α, _) -> -2.0 + 2000α)
    @test steplength(search(overshoot, grow)) == steplength(search(overshoot, shrink))
    @test trials(search(overshoot, grow)) == trials(search(overshoot, shrink))

    # The merit's round-off floor is not expanded into, and stays reported as a floor.
    noise = LinesearchProblem{Float64}((α, _) -> α > 0 ? nextfloat(1.0) : 1.0, (α, _) -> -2.0)
    @test outcome(search(noise, grow)) == outcome(search(noise, shrink)) == LINESEARCH_FLOOR

    # `noise` never accepts, so it cannot exercise the one case where the two meet: a *first*
    # trial accepted at the floor, with `expand` set. A merit that is flat enough for the demanded
    # decrease c₁α|φ'(0)| to fall below τ accepts α = 1 while decreasing by less than τ, and there
    # the model gives α★ ≈ α/2 — so the phase declines to grow, the outcome stays a floor, and the
    # shrink-only search is matched trial for trial.
    let τ = armijo_tolerance(1.0, Backtracking().τ_ulps)
        flat = LinesearchProblem{Float64}((α, _) -> α > 0 ? 1.0 - τ / 2 : 1.0, (α, _) -> -1e-13)
        stf = search(flat, grow)
        @test outcome(stf) == LINESEARCH_FLOOR
        @test steplength(stf) == 1.0
        @test trials(stf) == trials(search(flat, shrink)) == 1
    end

    # Contract item 5: the cost does not depend on the scale of the merit, and neither does the
    # step — the model is built from φ₀, φ'(0) and φ(α), which all scale together.
    scaled(s) = LinesearchProblem{Float64}((α, _) -> s * (α - 11.0)^2, (α, _) -> s * 2(α - 11.0))
    for s in (1e-8, 1.0, 1e8)
        @test steplength(search(scaled(s), grow)) == steplength(search(scaled(1.0), grow))
        @test trials(search(scaled(s), grow)) == trials(search(scaled(1.0), grow))
    end

    # Every trial is counted, expansions included.
    n = Ref(0)
    counted = LinesearchProblem{Float64}((α, _) -> (n[] += 1; (α - 11.0)^2), (α, _) -> 2(α - 11.0))
    st = search(counted, grow)
    @test n[] == trials(st) + 1   # + the α = 0 anchor

    # `nexpand` is a hard cap on the extra evaluations, whatever the merit does.
    @test trials(search(quadratic(1e10), Backtracking(; expand=true, nexpand=1))) == 2

    # ... and it caps the phase from *within* `linesearch_max_iterations`, not beside it, so the
    # whole search still spends at most the budget `Options` documents. A merit whose scale is far
    # above the trial step would take every one of `nexpand`'s trials; a budget of one leaves it
    # none, and a budget of two leaves it exactly one.
    for (budget, expected) in ((1, 1), (2, 2), (3, 3))
        st = solve_with_status(Linesearch(quadratic(1e10), grow; verbosity=0,
                linesearch_max_iterations=budget), 1.0)
        @test trials(st) == expected ≤ budget
    end

    # The expansion never returns a step worse than the one the shrink-only search accepted: a
    # trial that fails sufficient decrease or does not improve the merit costs one evaluation and
    # the previous best is kept. φ(α) = (α-11)² for α ≤ 5 and a cliff above it.
    cliff = LinesearchProblem{Float64}((α, _) -> α > 5 ? 1e6 : (α - 11.0)^2, (α, _) -> 2(α - 11.0))
    stc = search(cliff, grow)
    @test steplength(stc) == 1.0
    @test stc.φ == 100.0
    @test trials(stc) == 2   # the one rejected expansion, and nothing more
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

@testset "$(rpad("Backtracking: τ_ulps validation, and solve is derived from the status", 80))" begin
    @test_throws AssertionError Backtracking(; τ_ulps=-1.0)
    @test Backtracking(; τ_ulps=0.0) isa Backtracking
    @test Backtracking().τ_ulps == SimpleSolvers.DEFAULT_ARMIJO_τ_ULPS

    # τ_ulps = 0 recovers the exact condition, i.e. the old ladder down to the eps floor
    noise = LinesearchProblem{Float64}((α, _) -> α > zero(α) ? nextfloat(1.0) : 1.0, (α, _) -> -2.0)
    st = solve_with_status(Linesearch(noise, Backtracking(; τ_ulps=0.0); verbosity=0), 1.0)
    @test steplength(st) ≤ eps(1.0)

    # Every built-in method reports a real outcome, and `solve` is *derived* from it — one
    # definition for all of them, in `linesearch.jl` — so the two agree by construction rather
    # than because six copies of the same three lines happen to.  `Static` is the exception by
    # nature: it ignores the caller's step and never evaluates the merit, so it has established
    # nothing and reports `LINESEARCH_UNKNOWN`, which `linesearch_warnings` passes over in silence.
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

    # A user-defined method implements `solve_with_status` and gets `solve` for free, with the
    # report and the α > 0 contract that come with it. This is the direction that makes the
    # contract structural: there is no path by which the package calls a method's own `solve`, so
    # a method cannot reinstate the per-iteration message a solve must not emit.
    ls_toy = Linesearch(prob, ToyLinesearch(); verbosity=1)
    @test solve(ls_toy, 1.0) == 0.25
    @test outcome(solve_with_status(ls_toy, 1.0)) == LINESEARCH_EXHAUSTED
    @test logged_any(() -> solve(ls_toy, 1.0), "no step satisfied the sufficient decrease")
    @test !logged_any(() -> solve_with_status(ls_toy, 1.0), "no step satisfied the sufficient decrease")

    # ... and a method that implements neither says which one it owes, rather than recursing
    # between the two generic definitions.
    ls_mute = Linesearch(prob, MuteLinesearch(); verbosity=0)
    @test_throws "does not implement `solve_with_status`" solve_with_status(ls_mute, 1.0)
    @test_throws "does not implement `solve_with_status`" solve(ls_mute, 1.0)

    # `α` is converted to the element type of the `Linesearch`, so a caller does not have to spell
    # out the precision of a step that has no fractional part.
    @test solve(Linesearch(prob, Backtracking(); verbosity=0), 1) == 1.0
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

    # No sign change over the bracket: rather than silently collapsing onto α₁ or erroring (which
    # would abort the enclosing solve), `bisection` returns the endpoint closest to a root
    # (smallest |f|) *and reports the failure* — silenced here, and asserted on its own below.
    fpos(α, _) = α + 1.0            # strictly positive on [0, 1] → no sign change
    quiet = Options(Float64; verbosity=0)
    @test bisection(fpos, 0.0, 1.0, NullParameters(), quiet) == 0.0    # |f(0)| = 1 < |f(1)| = 2
    @test bisection(fpos, 1.0, 0.0, NullParameters(), quiet) == 0.0    # endpoints flipped internally

    # The debug `println` and hard `error("Max iteration number exceeded")` were
    # A tight tolerance forces exhaustion here.
    fslow(α, _) = α - 1 / 3
    cfg = Options(Float64; linesearch_max_iterations=2, x_suctol=0.0, f_abstol=0.0, verbosity=0)
    local αbest
    @test (αbest = bisection(fslow, 0.0, 1.0, NullParameters(), cfg)) isa Float64
    @test 0.0 ≤ αbest ≤ 1.0
end

@testset "$(rpad("_bisection_core tells a located root from a failed bracket", 80))" begin
    # The three outcomes are distinct, and "no sign change" is not one of the successes. Folding it
    # into `converged = true` is what let an unbracketable derivative be claimed as a line
    # minimiser and then classified as `LINESEARCH_FLOOR` — see the `Bisection` testset below.
    froot(α, _) = α - 1.0
    fpos(α, _) = α + 1.0
    fslow(α, _) = α - 1 / 3

    @test SimpleSolvers._bisection_core(froot, 0.0, 2.0, NullParameters(), Options())[2] ===
          SimpleSolvers.BISECTION_CONVERGED
    @test SimpleSolvers._bisection_core(fpos, 0.0, 1.0, NullParameters(), Options())[2] ===
          SimpleSolvers.BISECTION_NOBRACKET
    @test SimpleSolvers._bisection_core(fslow, 0.0, 1.0, NullParameters(),
        Options(Float64; linesearch_max_iterations=2, x_suctol=0.0, f_abstol=0.0))[2] ===
          SimpleSolvers.BISECTION_EXHAUSTED

    # The endpoint flip does not change the verdict, and the returned value is unchanged from
    # before: the endpoint with the smallest |f|. Only the claim made about it is.
    α, oc, _ = SimpleSolvers._bisection_core(fpos, 1.0, 0.0, NullParameters(), Options())
    @test oc === SimpleSolvers.BISECTION_NOBRACKET
    @test α == 0.0

    # And the outcome is inferred, not boxed — it is on the line search's hot path.
    @test (@inferred SimpleSolvers._bisection_core(froot, 0.0, 2.0, NullParameters(), Options())) isa
          Tuple{Float64,SimpleSolvers.BisectionOutcome,Int}
end

@testset "$(rpad("a Bisection that cannot bracket never reports a floor", 80))" begin
    # `LINESEARCH_FLOOR` asserts that *no* line search can make progress along this direction, and
    # the outer iteration acts on it: `flag_stall!`, then `max_stalls`. A bisection that could not
    # bracket φ′ has established nothing of the kind, so it may not make that claim (issue: the
    # `converged = true` of the no-sign-change branch). It reports what the *merit* says instead.
    #
    # `Bisection` bisects φ′ but brackets on φ, so the case arises whenever the two disagree —
    # which is the realistic one: a stale or regularized Jacobian, an inexact linear solve, a
    # non-smooth merit. Both problems below state it outright, with a φ that has a proper minimum
    # and a "derivative" that never changes sign, so `bracket_minimum` succeeds and the bisection
    # it hands the bracket to cannot start.

    # The minimum of φ lies at α = 1, so the search still finds a step that improves the merit.
    # A genuine decrease stays reported as a decrease — the failed bracket is not held against it.
    ok = LinesearchProblem{Float64}((α, _) -> (α - 1.0)^2, (α, _) -> -1.0)
    st = solve_with_status(Linesearch(ok, Bisection(); verbosity=0), 1.0)
    @test outcome(st) === LINESEARCH_DECREASED
    @test steplength(st) > 0.0
    @test st.φ ≤ st.φ₀ - st.τ

    # The minimum of φ lies at α = -1, so no positive step improves the merit. This is the case
    # the defect turned into `LINESEARCH_FLOOR`: the endpoint of a bracket that never was got
    # claimed as the line minimiser, and the outer iteration counted the step towards `max_stalls`
    # (`flag_stall!` in `solver_step!`) on the strength of that claim. It is `LINESEARCH_EXHAUSTED`
    # — no acceptable step was *found*, which leaves open that one exists.
    bad = LinesearchProblem{Float64}((α, _) -> (α + 1.0)^2, (α, _) -> -1.0)
    stbad = solve_with_status(Linesearch(bad, Bisection(); verbosity=0), 0.01)
    @test outcome(stbad) === LINESEARCH_EXHAUSTED
    @test outcome(stbad) ≠ LINESEARCH_FLOOR
    @test !SimpleSolvers.isfloor(stbad)
    @test steplength(stbad) > 0.0      # the α > 0 contract holds either way

    # Both of these return the caller's α rather than a step of their own, and the status says what
    # the merit *is* there rather than repeating φ(0). Filling `φ` with `φ₀` used to make
    # `linesearch_exhausted_reason` report "the merit changed by 0.0" for a merit nothing had
    # measured — most visibly for one that descends forever, where `bracket_minimum` finds no
    # bracket at all and the true value is as far from φ(0) as the step is long.
    @test stbad.φ == (α -> (α + 1.0)^2)(steplength(stbad))
    forever = LinesearchProblem{Float64}((α, _) -> 1.0 - α, (α, _) -> -1.0)
    # A merit that descends forever is now answered by the ceiling rather than by a failure: the
    # bracketing stops at `αmax`, and the step it hands back really does decrease the merit — by
    # 65535 here — so it is a decrease and not an exhausted search.  Reaching `:unbracketable` at
    # all needs the ceiling switched off, and that is the case the `φ` field was fixed for.
    stf = solve_with_status(Linesearch(forever, Bisection(); verbosity=0), 1.0)
    @test outcome(stf) === LINESEARCH_DECREASED
    @test steplength(stf) == SimpleSolvers.DEFAULT_LINESEARCH_αmax
    @test stf.φ == 1.0 - steplength(stf)
    @test stf.φ < stf.φ₀

    stfu = solve_with_status(Linesearch(forever, Bisection(; αmax=Inf); verbosity=0), 1.0)
    @test outcome(stfu) === LINESEARCH_EXHAUSTED
    @test stfu.φ == 1.0 - steplength(stfu)   # was φ₀ = 1.0, i.e. "the merit did not change"
    @test stfu.φ < stfu.φ₀
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
    @test_throws AssertionError Quadratic(Float64; αmax=0.0)
    @test_throws AssertionError BierlaireQuadratic(Float64; αmax=-1.0)
    @test_throws AssertionError Bisection(Float64; αmax=0.0)
    @test Bisection() isa Bisection                      # defaults are valid
end

# The bracketing searches used to grow their bracket outward until the merit stopped falling, and
# nothing bounded how far that was: `bracket_minimum_with_fixed_point` and `_triple_point_core`
# double their probe step up to `DEFAULT_BRACKETING_nmax = 100` times, so from `s = 1e-2` the right
# endpoint can reach 1e28.  Measured downstream (GeometricOptimizers issue D6, the upstream half of
# their A1b), `Quadratic` returned α = 4.3e7 on a direction of norm 5.5.
#
# In a Euclidean problem that is self-correcting — φ grows like α², so the search's own decrease
# test throws the step out — which is why it went unreported for so long.  On a *compact* manifold
# it is not: φ is bounded there and can be genuinely lower at α = 1e9 than at α = 0, so nothing the
# search can measure calls the step too large.  The merit is not a bound on the step.
@testset "$(rpad("no search extrapolates past its ceiling", 80))" begin
    αmax = SimpleSolvers.DEFAULT_LINESEARCH_αmax
    minimising = (Bisection(), Quadratic(), BierlaireQuadratic())
    every = (Static(), Backtracking(), Backtracking(; expand=true), StrongWolfe(), minimising...)

    # A genuine minimiser, ten million steps away.  Scaled by 1e-14 so the merit stays of order one
    # over the whole range and the outcome turns on the ceiling rather than on any tolerance.
    far = LinesearchProblem{Float64}((α, _) -> (α - 1.0e7)^2 / 1.0e14, (α, _) -> 2(α - 1.0e7) / 1.0e14)

    # Switching the ceiling off recovers the old behaviour, which is what says the defect was real
    # and that this is what fixes it rather than some tolerance change alongside.
    for m in (Bisection(; αmax=Inf), Quadratic(; αmax=Inf), BierlaireQuadratic(; αmax=Inf))
        @test steplength(solve_with_status(Linesearch(far, m; verbosity=0), 1.0)) > 1.0e6
    end

    # With it, every minimising search stops exactly there — and reports a *decrease*, because the
    # merit at the ceiling really is lower.  A ceiling is not a failure.
    for m in minimising
        st = solve_with_status(Linesearch(far, m; verbosity=0), 1.0)
        @test steplength(st) == αmax
        @test outcome(st) === LINESEARCH_DECREASED
        @test st.φ == (α -> (α - 1.0e7)^2 / 1.0e14)(steplength(st))   # the merit at the step handed back
        @test st.φ ≤ st.φ₀ - st.τ
    end

    # The caller's ceiling binds for *every* method, including the two that have none of their own.
    # This is the half GeometricOptimizers needs: its bound is the 2π of a rotation divided by the
    # norm of the direction, so it changes at every solver step and cannot live in a struct field.
    for m in every, ceiling in (10.0, 0.25)
        st = solve_with_status(Linesearch(far, m; verbosity=0), 1.0, (αmax=ceiling,))
        @test steplength(st) ≤ ceiling
        @test steplength(st) > 0.0                       # the α > 0 contract still holds
    end

    # It binds below the natural minimiser too, i.e. it is a ceiling and not just a backstop.
    near = LinesearchProblem{Float64}((α, _) -> (α - 1.0)^2, (α, _) -> 2(α - 1.0))
    for m in every
        st = solve_with_status(Linesearch(near, m; verbosity=0), 1.0, (αmax=0.5,))
        @test 0.0 < steplength(st) ≤ 0.5
    end

    # A ceiling that does not bind changes nothing at all — not the step, not the outcome, not the
    # number of merit evaluations.  This is the assertion that the default path is untouched.
    for m in every
        ls = Linesearch(near, m; verbosity=0)
        plain = solve_with_status(ls, 1.0)
        roomy = solve_with_status(ls, 1.0, (αmax=1.0e6,))
        @test steplength(roomy) === steplength(plain)
        @test outcome(roomy) === outcome(plain)
        @test trials(roomy) == trials(plain)
    end

    # `check_anchor` respects it too, so the ceiling holds on the returns that never search: an
    # ascent anchor hands back the caller's trial step, and a caller that asked for less gets less.
    ascent = LinesearchProblem{Float64}((α, _) -> (α + 1.0)^2, (α, _) -> 2(α + 1.0))
    for m in every
        st = solve_with_status(Linesearch(ascent, m; verbosity=0), 4.0, (αmax=0.5,))
        @test 0.0 < steplength(st) ≤ 0.5
    end

    # The ceiling bounds where the merit is *evaluated*, not only what is returned.  A ceiling below
    # the bracketing step `DEFAULT_BRACKETING_s` used to be stepped over by the first probe, before
    # the loop tested the bound for the first time — one evaluation outside the range the caller
    # called admissible, which on a problem where such a step is meaningless is exactly the
    # evaluation the ceiling exists to avoid.
    for m in every
        probed = Float64[]
        watched = LinesearchProblem{Float64}((α, _) -> (push!(probed, α); (α - 1.0)^2),
            (α, _) -> (push!(probed, α); 2(α - 1.0)))
        st = solve_with_status(Linesearch(watched, m; verbosity=0), 1.0, (αmax=0.005,))
        @test isempty(probed) || maximum(probed) ≤ 0.005     # `Static` evaluates nothing at all
        @test 0.0 < steplength(st) ≤ 0.005
    end

    # A ceiling that is not a usable step length is a caller error, and is reported as one before a
    # single merit evaluation is spent — not silently ignored, which would hand back exactly the
    # unbounded step the caller was trying to rule out.
    for bad in (0.0, -1.0, NaN)
        n = Ref(0)
        counted = LinesearchProblem{Float64}((α, _) -> (n[] += 1; (α - 1.0)^2), (α, _) -> (n[] += 1; 2(α - 1.0)))
        for m in every
            @test_throws ArgumentError solve_with_status(Linesearch(counted, m; verbosity=0), 1.0, (αmax=bad,))
        end
        @test n[] == 0
    end
    # `Inf` is not one of them: it says the caller has no scale of its own, which is what a bound
    # derived from one (2π / ‖δ‖, say) degenerates to for a vanishing direction. The method's own
    # ceiling then stands, so it cannot produce an unbounded step either.
    @test steplength(solve_with_status(Linesearch(far, Quadratic(); verbosity=0), 1.0, (αmax=Inf,))) == αmax
end

# The bracketing helpers report a truncated bracket as such rather than as a found one: the fits
# would otherwise be handed an interval over which the merit only falls, where `Quadratic`'s
# curvature guard bisects and returns a midpoint strictly worse than the endpoint.
@testset "$(rpad("the bracketing helpers report a bracket truncated at the ceiling", 80))" begin
    descending(x) = 1.0 - x                    # never turns, so only the ceiling can stop it
    turning(x) = (x - 1.0)^2                   # turns at 1, well inside the ceilings below

    a, b, ya, yb, st = SimpleSolvers._bracket_minimum_with_fixed_point_core(descending, 0.0, 0.01, 2.0, 100, 5.0)
    @test st === :capped
    @test b == 5.0 && yb == descending(5.0)
    @test SimpleSolvers._bracket_minimum_with_fixed_point_core(turning, 0.0, 0.01, 2.0, 100, 5.0)[end] === :ok
    # …and without a ceiling the same merit is what it always was: unbracketable.
    @test SimpleSolvers._bracket_minimum_with_fixed_point_core(descending, 0.0, 0.01, 2.0, 100, Inf)[end] === :unbracketable

    lo, hi, stm = SimpleSolvers._bracket_minimum_core(descending, 0.0, 0.01, 2.0, 100, 5.0)
    @test stm === :capped && hi == 5.0
    @test SimpleSolvers._bracket_minimum_core(turning, 0.0, 0.01, 2.0, 100, 5.0)[end] === :ok

    _, _, c, stt = SimpleSolvers._triple_point_core(descending, 0.0, 0.01, 100, 1, 5.0)
    @test stt === :capped && c == 5.0
    @test SimpleSolvers._triple_point_core(turning, 0.0, 0.01, 100, 1, 5.0)[end] === :ok
    # A ceiling below the initial probe still keeps every evaluation inside the admissible range.
    @test SimpleSolvers._triple_point_core(descending, 0.0, 0.01, 100, 1, 0.005)[3] ≤ 0.005

    # The public wrappers keep the return type they document; only the cores carry the status.
    @test SimpleSolvers.bracket_minimum_with_fixed_point(descending, 0.0, 0.01, 2.0, 100, 5.0) == (0.0, 5.0, 1.0, -4.0)
    @test isnothing(SimpleSolvers.bracket_minimum_with_fixed_point(descending, 0.0, 0.01, 2.0, 100, Inf))
    # `bracket_minimum` moves its left endpoint as it walks (unlike the fixed-point variant above),
    # so only the right end is pinned to the ceiling.
    @test bracket_minimum(descending, 0.0, 0.01, 2.0, 100, 5.0)[2] == 5.0
    @test isnothing(bracket_minimum(descending, 0.0, 0.01, 2.0, 100, Inf))
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
        for m in (Static(T), Backtracking(T), Backtracking(T; expand=true), StrongWolfe(T), Bisection(T), Quadratic(T), BierlaireQuadratic(T))
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

        # The expansion keys survive a precision change too — `expand` and `nexpand` unconverted,
        # `q` in the new element type — and `isapprox` compares them.
        grow = Backtracking(; expand=true, nexpand=2)
        @test change_precision(T, grow).expand
        @test change_precision(T, grow).nexpand == 2
        @test change_precision(T, grow).q == T(SimpleSolvers.DEFAULT_BACKTRACKING_q)
        @test change_precision(T, grow) ≈ Backtracking(T; expand=true, nexpand=2)
        @test !(change_precision(T, grow) ≈ Backtracking(T; nexpand=2))
        @test !(change_precision(T, grow) ≈ Backtracking(T; expand=true, nexpand=3))

        # ... and the phase *runs* in the new precision, not merely survives conversion into it:
        # `BACKTRACKING_GROW_MIN` and `q` are converted with the method, so a merit whose scale is
        # an order of magnitude above the trial step is reached in Float16 exactly as in Float64,
        # while the shrink-only search stays pinned at the caller's ceiling.
        let prob = LinesearchProblem{T}((α, _) -> (α - T(11))^2, (α, _) -> 2 * (α - T(11))),
            expander = Backtracking(T; expand=true)

            st = solve_with_status(Linesearch(prob, expander; verbosity=0), one(T))
            @test issufficient(st)
            @test T(11) / T(SimpleSolvers.BACKTRACKING_GROW_MIN) ≤ steplength(st) ≤ T(11)
            @test 1 < trials(st) ≤ 1 + expander.nexpand
            @test steplength(solve_with_status(Linesearch(prob, Backtracking(T); verbosity=0), one(T))) == one(T)
        end
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
    # `linesearch_warnings` is called from `solve`, i.e. from every direct call to a line search,
    # and it is specialized on the `Linesearch` — hence on the closure types of its
    # `LinesearchProblem`, hence once per *problem* a line search is built for. The messages
    # therefore live behind the `report_linesearch_status` barrier, whose signature mentions no
    # closure type; see its docstring.

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

    # The barrier really is on the path a *direct* `solve(ls, α)` takes, for every method — that is
    # the only path that reaches it, since a solver consults `solve_with_status` and reports through
    # `nonlinear_solver_warnings` instead — and it stays quiet when there is nothing to report.
    for ls in (Backtracking(), StrongWolfe(), Bisection(), Quadratic(), BierlaireQuadratic())
        lsp = Linesearch(make_linesearch_problem(-3.0), ls; verbosity=1)
        @test (@test_logs solve(lsp, 1.0)) > 0.0
    end

    # ... and a solve is quiet too, for every method, now that the line search does not report from
    # inside the iteration at all.
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
    # session.
    fslow(α, _) = α - 1 / 3
    nonconvergence(v) = () -> bisection(fslow, 0.0, 1.0, NullParameters(),
        Options(Float64; linesearch_max_iterations=2, x_suctol=0.0, f_abstol=0.0, verbosity=v))
    @test logged_any(nonconvergence(1), "did not converge within")
    @test !logged_any(nonconvergence(0), "did not converge within")

    # `nobracket` sits at `verbosity ≥ 1`, not 2. It used to be gated at 2 because the *line
    # search* routed through `bisection` and a flat derivative at a minimum made the message a
    # false alarm; it no longer does — `_bisect_on` calls `_bisection_core` and reports through its
    # `LinesearchStatus` — so the only caller left is a user asking for a root, and "there is no
    # root in your interval" is a genuine failure for them.
    fpos(α, _) = α + 1.0            # strictly positive on [0, 1] → no sign change
    nobracket(v) = () -> bisection(fpos, 0.0, 1.0, NullParameters(), Options(Float64; verbosity=v))
    @test logged_any(nobracket(1), "shows no sign change")
    @test !logged_any(nobracket(0), "shows no sign change")
end

@testset "$(rpad("every clause of a linesearch message is built only when it is shown", 80))" begin
    # The `αmin` clause of `LINESEARCH_FLOOR` and both wordings of `LINESEARCH_EXHAUSTED` are
    # interpolated inside their `@warn` rather than into a temporary before it, so that a message
    # the verbosity gate discards costs nothing (see `report_linesearch_status`) — and the
    # overwhelmingly common case is a caller running at a verbosity that discards it. The exact
    # texts are pinned here because that laziness is easy to undo by rewriting the interpolation,
    # and easy to undo *silently*.
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
    # Every line search runs a solve without touching the heap: no kernel counts its merit
    # evaluations in a variable that a closure captures and the loop mutates, and the bracketing
    # helpers return concrete types. `StrongWolfe` is the one exception by design — its one-slot memo
    # of φ and φ′ is handed to `_wolfe_zoom` and to the condition objects, so it cannot stay on the
    # stack — and one small holder per line search is the whole of it.
    #
    # The two causes are asserted first, on lowered code and by inference, because those hold however
    # the session was started. The byte counts that motivated them follow, guarded: they are the
    # end-to-end statement but say nothing under `--check-bounds=yes` (see `AS_A_CALLER_COMPILES_IT`).
    for f in (solve, solve_with_status, SimpleSolvers._bierlaire_fit, SimpleSolvers._wolfe_zoom,
        SimpleSolvers.backtracking_expand,
        bisection, SimpleSolvers._bisection_core, SimpleSolvers._triple_point_core)
        @test !has_boxed_capture(f)
    end

    # A boxed counter is invisible in the result but erases the type of everything built from it, so
    # pin that too: the `trials` of a `LinesearchStatus` and the `Symbol` a bracketing attempt reports.
    probe(a) = (a - 0.5)^2 + 1.0
    @test (@inferred SimpleSolvers._triple_point_core(probe, 0.0, 0.01, 100, 1)) isa Tuple{Float64,Float64,Float64,Symbol}
    # The ceiling adds a third status but must not add a type: `:capped` travels the same slot.
    @test (@inferred SimpleSolvers._triple_point_core(probe, 0.0, 0.01, 100, 1, 0.2)) isa Tuple{Float64,Float64,Float64,Symbol}
    @test (@inferred SimpleSolvers._bracket_minimum_with_fixed_point_core(probe, 0.0, 0.01, 2.0, 100, 0.2)) isa Tuple{Float64,Float64,Float64,Float64,Symbol}
    @test (@inferred SimpleSolvers._bracket_minimum_core(probe, 0.0, 0.01, 2.0, 100, 0.2)) isa Tuple{Float64,Float64,Symbol}
    let ls = Linesearch(make_linesearch_problem(2.0), BierlaireQuadratic(); verbosity=0)
        a, b, c = SimpleSolvers._triple_point_core(problem(ls), NullParameters(), 0.0)
        @test (@inferred SimpleSolvers._bierlaire_fit(ls, a, b, c, NullParameters(), 1.0e-16)) isa Tuple{Float64,Float64,Int}
    end
    @test (@inferred solve_with_status(Linesearch(make_linesearch_problem(2.0), StrongWolfe(); verbosity=0), 1.0)) isa LinesearchStatus

    F(y, x, params) = y .= x .^ 2 .- 2
    function solve_allocations(ls)
        x = ones(3)
        s = NewtonSolver(x, similar(x); F=F, linesearch=ls, verbosity=0)
        state = SolverState(s)
        solve!(x, s, state)
        x .= 1.0
        @allocated solve!(x, s, state)
    end
    for ls in (Static(), Backtracking(), Backtracking(; expand=true), Bisection(), Quadratic(), BierlaireQuadratic())
        @test solve_allocations(ls) == 0 skip = !AS_A_CALLER_COMPILES_IT
    end

    # A caller that supplies no ceiling pays nothing for the one it could have: `hasproperty` on the
    # parameter type is resolved at compile time, so `caller_αmax` folds to `Inf` and the solve above
    # is the proof. A caller that *does* supply one must not pay either — `params` gains a field, not
    # an allocation — which is what this asserts on the line search directly.
    function ceiling_allocations(m, params)
        ls = Linesearch(make_linesearch_problem(2.0), m; verbosity=0)
        solve_with_status(ls, 1.0, params)
        @allocated solve_with_status(ls, 1.0, params)
    end
    for m in (Static(), Backtracking(), Backtracking(; expand=true), Bisection(), Quadratic(), BierlaireQuadratic())
        @test ceiling_allocations(m, (x=2.0, αmax=10.0)) == 0 skip = !AS_A_CALLER_COMPILES_IT
        @test ceiling_allocations(m, (x=2.0,)) == 0 skip = !AS_A_CALLER_COMPILES_IT
    end

    function wolfe_allocations()
        ls = Linesearch(make_linesearch_problem(2.0), StrongWolfe(); verbosity=0)
        solve_with_status(ls, 1.0)
        @allocated solve_with_status(ls, 1.0)
    end
    @test wolfe_allocations() ≤ 64 skip = !AS_A_CALLER_COMPILES_IT
end
