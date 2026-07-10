using Random
using SimpleSolvers
using Test

using LinearAlgebra: rmul!, ldiv!
using SimpleSolvers: BierlaireQuadratic, Quadratic, NullParameters
using SimpleSolvers: factorize!, linearsolver, jacobian, jacobian!, cache, linesearch_problem, direction, compute_new_iterate, compute_new_iterate!, direction!, nonlinearproblem, iteration_number
using SimpleSolvers: change_precision, bisection, bracket_root, triple_point_finder
using SimpleSolvers: CurvatureCondition

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
    _f(α, _) = f(compute_new_iterate(x₀, α, δx(x₀)))
    _d(α, _) = g(compute_new_iterate(x₀, α, δx(x₀)))
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

    # §2.3 / 2.5: bracket_minimum must return an interval that actually contains
    # the minimum (the early-exit path must not bracket a maximum).
    lo, hi = bracket_minimum(x -> (x - 1)^2, 0.0)
    @test lo < 1.0 < hi
end

@testset "$(rpad("triple_point_finder (§2.3 / 2.5)", 80))" begin
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

@testset "$(rpad("Backtracking stall (§1.3 / 2.1)", 80))" begin
    # f(α) = (α - 100)² starting at α = 1: shrinking α makes the curvature
    # condition impossible to satisfy, so the old shrink-only loop (which
    # required both Wolfe conditions) ran all iterations and silently returned a
    # denormal α ≈ 9.3e-302.  Now the loop terminates on sufficient decrease
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

@testset "$(rpad("Quadratic Linesearch (Bierlaire)", 80))" begin

    test_linesearch(BierlaireQuadratic(), 1)

end

@testset "$(rpad("Quadratic Linesearch (Derivative-Based)", 80))" begin

    test_linesearch(BierlaireQuadratic(), 1)

end


@testset "$(rpad("Quadratic defaults (§2.4 / 2.6)", 80))" begin
    # Quadratic(T, ::SolverMethod) used to square ε, s and s_reduction (an
    # accidental `^2`), disagreeing with the keyword constructor and pushing ε
    # below machine epsilon.  It now matches the keyword constructor defaults and
    # dispatches on ::SolverMethod like its siblings.
    for T in (Float32, Float64)
        q = Quadratic(T, NewtonMethod())
        @test q ≈ Quadratic(T)
        @test q.ε == SimpleSolvers.default_precision(T)
        @test q.s == T(SimpleSolvers.DEFAULT_BRACKETING_s)
        @test q.s_reduction == T(SimpleSolvers.DEFAULT_s_REDUCTION)
    end

    # §5: `default_precision` used to error for any float type other than
    # Float32/Float64 although `8eps(T)` is generic; it is now defined for all
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

    # §1.5 regression: the former `Base.convert(::Type, ::LinesearchMethod)`
    # catch-all was ambiguous with Base and violated the `convert` contract.
    # `convert` on a linesearch method now falls back to Base's default behaviour
    # and no longer throws an ambiguity error.
    @test convert(Any, Static()) === Static()
    @test convert(Static, Static()) === Static()
    # precision changes now go through `change_precision`, which returns the
    # correct element type (not a differently-typed object from `convert`).
    @test change_precision(Float32, Static()) isa Static{Float32}
    @test eltype(change_precision(Float32, Static())) == Float32

end


@testset "$(rpad("Broken convenience entry points (§1.8)", 80))" begin
    x₀ = -3.0
    ls_problem = make_linesearch_problem(x₀)

    # linesearch.jl:69 — the 5-arg `solve(prob, method, α, params, config)` used to
    # call a nonexistent 3-positional `Linesearch` constructor and always threw.
    @test Linesearch(ls_problem, Static(1.0), Options()) isa Linesearch
    @test solve(ls_problem, Static(1.0), 0.0) == 1.0
    @test solve(ls_problem, Static(0.8), 0.0, NullParameters(), Options()) == 0.8

    # bisection.jl:81 — single-`x` `bisection` used to pass a 2-arg callback to
    # `bracket_minimum` (a `MethodError`) and brackets a minimum where a sign
    # change is needed.  It now uses `bracket_root` on a one-argument closure.
    fb(α, _) = α - 1.0
    @test bisection(fb, 0.5) ≈ 1.0 atol = 1e-6

    # bracket_minimum.jl:250 — `bracket_root(prob, params, x)` used to forward to a
    # nonexistent 3-positional `bracket_root(f, df, x)`.
    root_problem = LinesearchProblem{Float64}((α, _) -> α - 1.0, (α, _) -> 1.0)
    lo, hi = bracket_root(root_problem, NullParameters(), 0.5)
    @test lo ≤ 1.0 ≤ hi
end


@testset "$(rpad("Mixed-precision compute_new_iterate! (§2.4)", 80))" begin
    # backtracking_condition.jl:42 — the mixed-precision 3-arg `compute_new_iterate!`
    # used to call the non-mutating `compute_new_iterate` and discard the result,
    # so the array argument was never updated.  It now mutates in place.
    x = Float32[1.0, 2.0]
    p = Float32[1.0, 1.0]
    compute_new_iterate!(x, 1.0, p)   # α is Float64 → mixed precision path
    @test x ≈ Float32[2.0, 3.0]
end


@testset "$(rpad("Phase 3 type-stability fixes", 80))" begin
    # 3.3 — `Bisection(::Type{T}=Float64)` is now inferable (the old
    # `Bisection(T::DataType=Float64)` returned `Bisection{<:Any}`).
    @test (@inferred Bisection()) === Bisection{Float64}()
    @test (@inferred Bisection(Float32)) === Bisection{Float32}()

    # 3.3 — `bisection` promotes integer endpoints to floating point on entry
    # (previously `α` switched type mid-loop and `Options(Int)` was undefined).
    fint(α, _) = α - 2.0
    r = bisection(fint, 0, 4)
    @test r ≈ 2.0 atol = 1e-6
    @test r isa AbstractFloat

    # 3.3 — `CurvatureCondition` encodes the mode in the type (via `Val`) so it is
    # inference-stable, validates `c ∈ (0, 1)`, and the strong condition uses `≤`.
    @test CurvatureCondition(0.9, -1.0, sin, Val(:Strong)) isa CurvatureCondition{Float64,typeof(sin),:Strong}
    @test CurvatureCondition(0.9, -1.0, sin) isa CurvatureCondition{Float64,typeof(sin),:Standard}
    @test_throws AssertionError CurvatureCondition(1.5, -1.0, sin)   # c ∉ (0, 1)
    @test_throws AssertionError CurvatureCondition(0.0, -1.0, sin)   # c ∉ (0, 1)
    # strong-Wolfe boundary: |D(α)| == |c·d| must now pass (was strict `<`)
    ccs = CurvatureCondition(0.9, -1.0, α -> 0.9, Val(:Strong))
    @test ccs(0.0)                                                   # |0.9| ≤ |0.9·(-1)|
    # standard curvature: D(α) ≥ c·d
    ccn = CurvatureCondition(0.9, -1.0, α -> -0.5, Val(:Standard))
    @test ccn(0.0)                                                   # -0.5 ≥ -0.9

    # 3.4 — `Options` tolerance keywords accept any `::Real` (integers, rationals),
    # not just `AbstractFloat`.
    @test Options(Float64; x_abstol=0) isa Options
    @test Options(Float64; f_abstol=1 // 100) isa Options
    @test Options().f_abstol == 0.0
end


@testset "$(rpad("Phase 4.3 bisection hardening", 80))" begin
    # A genuine sign-changing bracket still bisects to the root.
    froot(α, _) = α - 1.0
    @test bisection(froot, 0.0, 2.0) ≈ 1.0 atol = 1e-6

    # No sign change over the bracket: rather than silently collapsing onto α₁
    # (the old bug) or erroring (which would break the line search once the
    # derivative has flattened at a minimum), `bisection` returns the endpoint
    # closest to a root (smallest |f|).
    fpos(α, _) = α + 1.0            # strictly positive on [0, 1] → no sign change
    @test bisection(fpos, 0.0, 1.0) == 0.0    # |f(0)| = 1 < |f(1)| = 2
    @test bisection(fpos, 1.0, 0.0) == 0.0    # endpoints get flipped internally

    # The debug `println` and hard `error("Max iteration number exceeded")` were
    # removed: exhausting the iteration budget returns the best estimate instead
    # of throwing.  A tight tolerance forces exhaustion here.
    fslow(α, _) = α - 1 / 3
    cfg = Options(Float64; max_iterations=2, x_suctol=0.0, f_abstol=0.0, verbosity=0)
    local αbest
    @test (αbest = bisection(fslow, 0.0, 1.0, NullParameters(), cfg)) isa Float64
    @test 0.0 ≤ αbest ≤ 1.0
end

@testset "$(rpad("Phase 6: bisection interval/config disambiguation", 80))" begin
    # `bisection(f, αmin, αmax, config::Options)` used to be ambiguous with the
    # single-`α` convenience form (both matched `(f, ::T, ::T, ::Options)`); Aqua
    # flagged it. A disambiguating method now routes it to the interval form with
    # default params. It must behave exactly like the explicit 5-arg call.
    froot(α, _) = α - 1.0
    cfg = Options(Float64)
    @test bisection(froot, 0.0, 2.0, cfg) ≈ 1.0 atol = 1e-6
    @test bisection(froot, 0.0, 2.0, cfg) == bisection(froot, 0.0, 2.0, NullParameters(), cfg)
end

@testset "$(rpad("Phase 5: StrongWolfe line search (bracket + zoom)", 80))" begin
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

@testset "$(rpad("Phase 5: bracketing line searches are α₀-robust (§5 TODO resolved)", 80))" begin
    # The three former TODO sites asked whether the bracketing line searches
    # (Bisection, Quadratic, BierlaireQuadratic) should start from the caller's α₀
    # instead of α = 0.  Phase 5 resolves this as a *design decision*: they keep the
    # α = 0 anchor (the only point where a descent direction is guaranteed
    # decreasing, which one-sided rightward bracketing requires) with the method's
    # tuned step scale.  The α₀ argument is accepted but does not change the anchor
    # or scale, so the result is independent of α₀.  For f(x) = x² − 1 with the
    # Newton direction δx = −g/2 the line minimiser is at α = 1 (x₀ + 1·δx = 0):
    # every α₀ must converge there.
    prob = make_linesearch_problem(-3.0)
    for method in (Bisection(), Quadratic(), BierlaireQuadratic())
        ls = Linesearch(prob, method; x_abstol=0.0)
        results = [solve(ls, α₀) for α₀ in (0.25, 0.5, 1.0, 2.0, 4.0)]
        for α in results
            @test compute_new_iterate(-3.0, α, δx(-3.0)) ≈ 0.0 atol = ∛(2eps())
        end
        # α₀-independent by design
        @test all(r -> r ≈ first(results), results)
    end
end

# Interface-consistency fix (verification 2026-07-10): Quadratic and
# BierlaireQuadratic now validate their constructor parameters, like
# Backtracking and StrongWolfe always did.
@testset "$(rpad("Quadratic/BierlaireQuadratic constructor validation", 80))" begin
    @test_throws AssertionError Quadratic(Float64; ε=0.0)
    @test_throws AssertionError Quadratic(Float64; s=-1.0)
    @test_throws AssertionError Quadratic(Float64; s_reduction=1.5)
    @test_throws AssertionError BierlaireQuadratic(Float64; ε=0.0)
    @test_throws AssertionError BierlaireQuadratic(Float64; ξ=-1.0)
    @test Quadratic() isa Quadratic                      # defaults are valid
    @test BierlaireQuadratic() isa BierlaireQuadratic    # defaults are valid
end

# §5 leftovers (2026-07-10): `bracket_minimum_with_fixed_point` now returns the
# merit values at the bracket endpoints alongside the bracket — they are computed
# during bracketing anyway, so the Quadratic line search no longer re-evaluates
# them.  Both quadratic line searches iterate instead of recursing, which for
# BierlaireQuadratic also stops fa/fb/fc from being recomputed at every level.
@testset "$(rpad("bracket_minimum_with_fixed_point returns endpoint values (§5)", 80))" begin
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

@testset "$(rpad("Quadratic searches: merit-evaluation canary (§5)", 80))" begin
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
