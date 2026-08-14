"""
    const DEFAULT_WOLFE_αmax

Default upper bound on the step length for the bracketing phase of
[`StrongWolfe`](@ref). This is [`DEFAULT_LINESEARCH_αmax`](@ref), which every method's ceiling now
defaults to: the field was `StrongWolfe`'s alone until the bracketing searches were found to
extrapolate without one, and defining this as that constant is what keeps the two from drifting
apart.
"""
const DEFAULT_WOLFE_αmax = DEFAULT_LINESEARCH_αmax

@doc raw"""
    StrongWolfe{T} <: LinesearchMethod

A line search that finds a step ``\alpha`` satisfying the **strong Wolfe
conditions**

```math
\begin{aligned}
f(\alpha) &\leq f(0) + c_1\,\alpha\,f'(0), &\text{(sufficient decrease / Armijo)}\\
|f'(\alpha)| &\leq c_2\,|f'(0)|, &\text{(strong curvature)}
\end{aligned}
```

with ``0 < c_1 < c_2 < 1``.  It implements the bracketing line search of
[nocedal2006numerical; Alg. 3.5 and 3.6 (`zoom`)](@cite): a *bracketing* phase
grows the step until an interval containing an acceptable point is found, then a
*zoom* phase shrinks that interval (by bisection) until the strong Wolfe
conditions hold.

Unlike [`Backtracking`](@ref) — which enforces only sufficient decrease, since
the curvature condition cannot be honoured by shrinking alone — `StrongWolfe`
actually enforces the curvature condition, at the cost of evaluating the
derivative at each trial step.  Use it when curvature control is genuinely
required; [`Backtracking`](@ref) is cheaper otherwise.

# Keys

- `c₁` (default [`DEFAULT_WOLFE_c₁`](@ref)): the Armijo constant ``c_1``.
- `c₂` (default [`DEFAULT_WOLFE_c₂`](@ref)): the curvature constant ``c_2``. We require ``c_1 < c_2 < 1``.
- `αmax` (default [`DEFAULT_WOLFE_αmax`](@ref)): the largest step the bracketing phase will try.

!!! info
    The strong Wolfe conditions require a *descent direction* (``f'(0) < 0``).  If
    the line search problem is not decreasing at ``\alpha = 0`` the method cannot
    make progress; it then returns the caller's initial step and (at
    `verbosity ≥ 1`) warns.
"""
struct StrongWolfe{T} <: LinesearchMethod{T}
    c₁::T
    c₂::T
    αmax::T

    function StrongWolfe{T}(c₁::T, c₂::T, αmax::T) where {T}
        @assert 0 < c₁ < c₂ < 1 "The Wolfe constants need to satisfy 0 < c₁ < c₂ < 1, they are c₁ = $(c₁), c₂ = $(c₂)."
        @assert αmax > 0 "The maximum step length must be positive, it is $(αmax)."
        new{T}(c₁, c₂, αmax)
    end
end

function StrongWolfe(::Type{T}=Float64;
    c₁=T(DEFAULT_WOLFE_c₁),
    c₂=T(DEFAULT_WOLFE_c₂),
    αmax=T(DEFAULT_WOLFE_αmax)
) where {T}
    StrongWolfe{T}(c₁, c₂, αmax)
end

StrongWolfe(::Type{T}, ::SolverMethod) where {T} = StrongWolfe(T)

method_αmax(m::StrongWolfe) = m.αmax

Base.show(io::IO, ls::StrongWolfe) = print(io, "StrongWolfe with c₁ = $(ls.c₁), c₂ = $(ls.c₂) and αmax = $(ls.αmax).")

function change_precision(::Type{T}, method::StrongWolfe) where {T}
    T ≠ eltype(method) || return method
    StrongWolfe{T}(T(method.c₁), T(method.c₂), T(method.αmax))
end

function Base.isapprox(w₁::StrongWolfe{T}, w₂::StrongWolfe{T}; kwargs...) where {T}
    isapprox(w₁.c₁, w₂.c₁; kwargs...) && isapprox(w₁.c₂, w₂.c₂; kwargs...) && isapprox(w₁.αmax, w₂.αmax; kwargs...)
end

# The one-slot memo of `solve_with_status` below: the `α` at which the merit φ and its derivative φ′
# were last evaluated, and the value each returned there. See the memoisation comment there.
mutable struct WolfeMemo{T}
    φα::T
    φv::T
    dα::T
    dv::T
end

# The `zoom` subroutine (Nocedal & Wright, Alg. 3.6).  On entry [αlo, αhi] brackets
# an acceptable step, with αlo the lower-φ endpoint that already satisfies the
# sufficient-decrease condition.  We interpolate by bisection (robust, no
# curvature assumptions) and maintain the bracket invariant.  On iteration
# exhaustion we return αlo, which by construction satisfies sufficient decrease —
# the method never returns a step worse than Armijo.  The two strong Wolfe
# conditions are checked through the shared `sdc`/`cc` condition objects.
function _wolfe_zoom(ls::Linesearch{T}, φ, dφ, sdc::SufficientDecreaseCondition{T}, cc::CurvatureCondition{T}, αlo::T, αhi::T, φlo::T) where {T}
    αj = αlo
    n = 0
    for _ in 1:config(ls).linesearch_max_iterations
        αj = (αlo + αhi) / 2
        φj = φ(αj)
        n += 1
        # `sdc(αj, φj)` rather than `sdc(αj)`: the one-argument form would evaluate the merit a
        # second time at the very point that was just evaluated, doubling the cost of the zoom.
        if !sdc(αj, φj) || φj ≥ φlo
            αhi = αj
        else
            cc(αj) && return (αj, φj, n)
            dj = dφ(αj)
            if dj * (αhi - αlo) ≥ zero(T)
                αhi = αlo
            end
            αlo = αj
            φlo = φj
        end
        isapprox(αhi, αlo; atol=eps(T)) && break
    end
    # `φlo` tracks `αlo` through every update, so the merit at the returned step is known and the
    # caller does not have to re-evaluate it.
    (αlo, φlo, n)
end

"""
    solve_with_status(ls::Linesearch{T,<:StrongWolfe}, α, params)

Run the strong-Wolfe bracketing line search from the trial step `α` and return the
[`LinesearchStatus`](@ref), emitting no messages. [`solve`](@ref) is this plus the report; see
[`StrongWolfe`](@ref).
"""
function solve_with_status(ls::Linesearch{T,<:StrongWolfe}, α::T, params=NullParameters()) where {T}
    m = method(ls)
    c₁ = m.c₁
    c₂ = m.c₂
    # `StrongWolfe` has had this ceiling since before the other methods did; all that is new is
    # that a caller can ask for a smaller one. Everything downstream — the clamp of the initial
    # trial and the stop of the expansion loop — already respects it.
    αmax = linesearch_αmax(m, params)
    prob = problem(ls)

    # One-slot memoisation of the merit φ and its derivative φ′: the bracketing /
    # zoom logic and the composed Wolfe conditions (`sdc`/`cc`) query the same
    # trial `α`, so caching the last evaluation avoids recomputing the (expensive)
    # merit and derivative.
    # `NaN` never equals a real query, so the first call at any `α` always evaluates.
    # One holder rather than four `Ref`s: the closures below are handed to `_wolfe_zoom` and to the
    # condition objects, so the memory they capture cannot stay on the stack, and one allocation per
    # line search is the whole cost of the memo.
    memo = WolfeMemo{T}(T(NaN), T(NaN), T(NaN), T(NaN))
    function φ(a)
        a == memo.φα || (memo.φα = a; memo.φv = value(prob, a, params))
        memo.φv
    end
    function dφ(a)
        a == memo.dα || (memo.dα = a; memo.dv = derivative(prob, a, params))
        memo.dv
    end

    φ0 = φ(zero(T))
    d0 = dφ(zero(T))

    # The strong Wolfe conditions are only meaningful for a finite, decreasing anchor. The
    # former test was `d0 ≥ zero(T)`, which a `NaN` derivative slips past (`NaN ≥ 0` is false)
    # only to trip `SufficientDecreaseCondition`'s `@assert !isnan(d₀)` below and abort the
    # enclosing solve; `check_anchor` covers the non-finite case too.
    anchor = check_anchor(φ0, d0, α, αmax)
    isnothing(anchor) || return anchor
    # The three tail returns hand back the caller's trial step, so it is bounded here rather than
    # at each of them; the searching returns are bounded by `αi ≤ αmax` already.
    α = min(α, αmax)

    # The strong Wolfe conditions are the Armijo (sufficient decrease) condition
    # plus the strong curvature condition. `StrongWolfe` keeps the *exact* Armijo test (τ = 0
    # inside `sdc`); τ is used only to classify the outcome, i.e. to tell a step that genuinely
    # decreased the merit from one accepted where nothing can decrease it.
    τ = armijo_tolerance(φ0, armijo_ulps(T))
    sdc = SufficientDecreaseCondition(c₁, φ0, d0, φ)
    cc = CurvatureCondition(c₂, d0, dφ, Val(:Strong))

    αprev = zero(T)
    φprev = φ0
    αi = clamp(α > zero(T) ? α : one(T), eps(T), αmax)
    αvalid = αi   # last trial step that satisfied sufficient decrease, and its merit
    # Seeded with the anchor merit so that a zero `linesearch_max_iterations` — the only way to
    # reach the tail without having recorded a pair — reports the floor rather than a stale value.
    φvalid = φ0

    # `n` counts the trial steps α > 0 at which the merit was evaluated, for the status.
    n = 0
    # `_wolfe_zoom` returns its `αlo`, which is seeded with `αprev = 0` on the first expansion
    # round, so a merit that fails sufficient decrease immediately can drive the zoom to return
    # the α = 0 anchor. Returning that would freeze the outer iterate (`x .+= 0 .* d`), so the
    # contract's α > 0 guarantee is enforced here: a non-positive result means no positive step
    # improved the merit, which is the round-off floor, and the caller's trial step is returned.
    # `φres` is always a merit the caller has already computed, never a fresh evaluation: for a
    # `NonlinearSolver` that would be a full residual evaluation per solver step.
    # `n` is passed rather than captured: a counter that this closure captures and the loop below
    # mutates is boxed, which makes the `trials` of every status it builds inferred-`Any`.
    function wolfe_status(n, αres, φres, extra=0)
        αres > zero(T) || return LinesearchStatus{T}(α, LINESEARCH_FLOOR, n + extra, φ0, d0, φ0, τ, zero(T))
        LinesearchStatus{T}(αres, φres ≤ φ0 - τ ? LINESEARCH_DECREASED : LINESEARCH_FLOOR,
            n + extra, φ0, d0, φres, τ, zero(T))
    end

    for i in 1:config(ls).linesearch_max_iterations
        φi = φ(αi)
        n += 1
        # The two-argument `sdc` reuses `φi`; the one-argument form would evaluate the merit
        # again at the same point.
        if !sdc(αi, φi) || (i > 1 && φi ≥ φprev)
            return wolfe_status(n, _wolfe_zoom(ls, φ, dφ, sdc, cc, αprev, αi, φprev)...)
        end
        # αi satisfies sufficient decrease from here on
        cc(αi) && return wolfe_status(n, αi, φi)
        di = dφ(αi)
        if di ≥ zero(T)
            return wolfe_status(n, _wolfe_zoom(ls, φ, dφ, sdc, cc, αi, αprev, φi)...)
        end
        αvalid, φvalid = αi, φi
        # ascend toward αmax; stop expanding once the cap is reached
        αi == αmax && break
        αprev = αi
        φprev = φi
        αi = min(T(2) * αi, αmax)
    end

    # Return the last trial step that satisfied sufficient decrease — never the
    # freshly-doubled, unchecked `αi` from the expansion, and never the zero step. The strong
    # curvature condition was never met, so this is reported as an exhausted search even though
    # the step itself is Armijo-acceptable.
    αvalid > zero(T) || return LinesearchStatus{T}(α, LINESEARCH_FLOOR, n, φ0, d0, φ0, τ, zero(T))
    # `φvalid` was recorded alongside `αvalid`, so reporting it costs no extra evaluation.
    LinesearchStatus{T}(αvalid, LINESEARCH_EXHAUSTED, n, φ0, d0, φvalid, τ, zero(T))
end
