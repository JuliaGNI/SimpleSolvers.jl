using Random
using SimpleSolvers
using Test

using LinearAlgebra: rmul!, ldiv!
using SimpleSolvers: BierlaireQuadratic, Quadratic, NullParameters
using SimpleSolvers: factorize!, linearsolver, jacobian, jacobian!, cache, linesearch_problem, direction, compute_new_iterate, compute_new_iterate!, direction!, nonlinearproblem, iteration_number
using SimpleSolvers: change_precision, bisection, bracket_root, triple_point_finder
using SimpleSolvers: CurvatureCondition, SufficientDecreaseCondition
using SimpleSolvers: steplength, outcome, trials, armijo_tolerance, backtracking_αmin,
    backtracking_interpolation, with_config, problem, method, config

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

    # with τ = 4 ulps the decision at the degenerate α is taken by the allowance ...
    sdcτ = SufficientDecreaseCondition(1e-4, 1.0, -2.0, α -> nextfloat(1.0); τ=4eps(1.0))
    @test sdcτ(1e-13)
    # ... and never licenses a step where the demanded decrease is meaningful
    @test !sdcτ(1.0)

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

    # the generic fallback makes `solve_with_status` usable for every LinesearchMethod
    prob = LinesearchProblem{Float64}((α, _) -> (α - 0.7)^2, (α, _) -> 2 * (α - 0.7))
    for m in (Static(), Bisection(), Quadratic(), BierlaireQuadratic(), StrongWolfe())
        ls = Linesearch(prob, m; verbosity=0)
        st = solve_with_status(ls, 1.0)
        @test outcome(st) == LINESEARCH_UNKNOWN
        @test steplength(st) == solve(ls, 1.0)
        @test !issufficient(st)
        @test !isfloor(st)
    end
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

@testset "$(rpad("Quadratic returns the tested bracket point (non-descent start)", 80))" begin
    # When the start is not on the descent side, `bracket_minimum_with_fixed_point`
    # flips and the bracket's left endpoint `a` (where the derivative is tested and
    # the early-return fires) differs from the loop's start `α`.  The Quadratic search
    # must return `a`, not `α`.  Here φ(α) = (α + 1)² is increasing at the α = 0 anchor
    # (φ'(0) = 2 > 0) with its minimiser at α = -1.
    prob = LinesearchProblem{Float64}((a, _) -> (a + 1.0)^2, (a, _) -> 2.0 * (a + 1.0))
    ls = Linesearch(prob, Quadratic(); x_abstol=0.0)
    @test solve(ls, 0.0) ≈ -1.0 atol = ∛(2eps())
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
