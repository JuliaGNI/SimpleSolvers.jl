using Printf
@doc raw"""
    const DEFAULT_ARMIJO_α₀

The default starting value for ``\alpha`` used in [`Backtracking`](@ref).
Its value is """ * """$(DEFAULT_ARMIJO_α₀).
"""
const DEFAULT_ARMIJO_α₀ = 1.0

"""
    const DEFAULT_ARMIJO_p

Constant used in [`Backtracking`](@ref).
Its value is $(DEFAULT_ARMIJO_p)

This is the default for the constant ``p`` by which `α` is decreased if the [`SufficientDecreaseCondition`](@ref) and the [`CurvatureCondition`](@ref) are not satisfied.
"""
const DEFAULT_ARMIJO_p = 0.5

@doc raw"""
    const DEFAULT_WOLFE_c₁

A constant ``c_1`` that is used in the [`SufficientDecreaseCondition`](@ref) (the
Armijo condition):

```math
f(\alpha) \leq f(\alpha_0) + c_1 \alpha f'(\alpha_0).
```
"""
const DEFAULT_WOLFE_c₁ = 1E-4

@doc raw"""
    const DEFAULT_WOLFE_c₂

The constant used in the second Wolfe condition (the [`CurvatureCondition`](@ref)). According to [nocedal2006numerical,kochenderfer2019algorithms](@cite) we should have
```math
c_2 \in (c_1, 1),
```
where ``c_1`` is the constant specified by [`DEFAULT_WOLFE_c₁`](@ref).

Furthermore [nocedal2006numerical](@cite) recommend ``c_2 = 0.9``; in [kochenderfer2019algorithms](@cite) the authors write: "it is common to set ``c_2=0.1`` when approximate line search is used with the conjugate gradient method and to 0.9 when used with Newton's method."
We use ``c_2 = 0.9`` as default.
"""
const DEFAULT_WOLFE_c₂ = 0.9

@doc raw"""
    const DEFAULT_ARMIJO_τ_ULPS

The number of units in the last place (ulps) of ``\varphi(0)`` by which the
[`SufficientDecreaseCondition`](@ref) is slackened inside [`Backtracking`](@ref), i.e.
``\tau = \mathrm{DEFAULT\_ARMIJO\_τ\_ULPS}\cdot\mathrm{ulp}(\varphi(0))`` in

```math
\varphi(\alpha) \leq \varphi(0) + c_1\alpha\varphi'(0) + \tau .
```

Its value is """ * """$(DEFAULT_ARMIJO_τ_ULPS)""" * raw""". Since the decrease demanded at
``\alpha \approx 1`` is ``2c_1\varphi(0)``, i.e. some ``10^{-4}`` relative, ``\tau`` is
irrelevant except in the region where the condition was previously decided by rounding
rather than by the merit. Set `τ_ulps = 0` in [`Backtracking`](@ref) to recover the exact
condition.

See [`armijo_tolerance`](@ref) and [`backtracking_αmin`](@ref).
"""
const DEFAULT_ARMIJO_τ_ULPS = 4

@doc raw"""
    const BACKTRACKING_SHRINK_MIN

Lower bound on the factor by which a rejected step is shrunk by the interpolation in
[`Backtracking`](@ref): the new trial step is confined to
``[\mathrm{BACKTRACKING\_SHRINK\_MIN}\cdot\alpha, p\alpha]`` (see
[nocedal2006numerical; §3.5](@cite), [dennis1996numerical; Alg. A6.3.1](@cite)).
Its value is """ * """$(BACKTRACKING_SHRINK_MIN).
"""
const BACKTRACKING_SHRINK_MIN = 0.1

@doc raw"""
    Backtracking <: LinesearchMethod

# Keys

The keys are:
- `α₀`=""" * string(DEFAULT_ARMIJO_α₀) * raw""": the initial step size ``\alpha``. This is decreased iteratively by a factor ``p`` until the [`SufficientDecreaseCondition`](@ref) is satisfied.
- `c₁`=""" * string(DEFAULT_WOLFE_c₁) * raw""": the constant ``c_1`` in the [`SufficientDecreaseCondition`](@ref) (Armijo condition). Also see [`DEFAULT_WOLFE_c₁`](@ref).
- `c₂`=""" * string(DEFAULT_WOLFE_c₂) * raw""": the constant on whose basis the [`CurvatureCondition`](@ref) is tested. We should have ``c_2\in(c_1, 1).`` The closer this constant is to 1, the easier it is to satisfy the [`CurvatureCondition`](@ref).
- `p`=""" * string(DEFAULT_ARMIJO_p) * raw""": an *upper bound* on the factor by which ``\alpha`` is decreased in every step until the stopping criterion is satisfied. The actual factor is chosen by interpolation and confined to ``[`` [`BACKTRACKING_SHRINK_MIN`](@ref) ``\cdot\alpha, p\alpha]``, so the trial sequence is never longer than the plain ``\alpha \gets p\alpha`` ladder.
- `τ_ulps`=""" * string(DEFAULT_ARMIJO_τ_ULPS) * raw""": the round-off allowance of the [`SufficientDecreaseCondition`](@ref), in units in the last place of ``\varphi(0)``. See [`DEFAULT_ARMIJO_τ_ULPS`](@ref).

# Implementation

The algorithm starts by setting
```math
\begin{aligned}
\varphi_0 &\gets \varphi(0),\\
d_0 &\gets \varphi'(0),
\end{aligned}
```
where ``\varphi`` is of type [`LinesearchProblem`](@ref). Unless ``\varphi_0`` and ``d_0`` are
finite with ``d_0 < 0`` the search is abandoned at once — no ``\alpha`` can satisfy the
[`SufficientDecreaseCondition`](@ref) along a direction that is not decreasing, so shrinking
``\alpha`` would only waste merit evaluations to find that out.

Otherwise it sets the round-off allowance ``\tau`` ([`armijo_tolerance`](@ref)) and the
smallest informative step ``\alpha_\mathrm{min}`` ([`backtracking_αmin`](@ref)), and shrinks
the trial step by [`backtracking_interpolation`](@ref) until one of the following happens:
1. the [`SufficientDecreaseCondition`](@ref) is satisfied — the step is accepted, and reported
   as a genuine decrease only if ``\varphi(\alpha) \leq \varphi_0 - \tau``;
2. two consecutive trials return ``\varphi(\alpha) = \varphi_0`` bit-exactly — the trial point
   no longer differs from the base point in floating point, so no smaller step can either;
3. ``\alpha \leq \alpha_\mathrm{min}`` — a smaller step could only be judged by rounding;
4. the `linesearch_max_iterations` budget of [`Options`](@ref) is spent.

Cases 2–4 are distinguished in the returned [`LinesearchStatus`](@ref): a merit that does not
vary by more than ``\tau`` has reached its round-off floor (`LINESEARCH_FLOOR`, benign and not
improvable by any line search), whereas one that does vary contradicts ``d_0 < 0``
(`LINESEARCH_EXHAUSTED`, a genuine inconsistency). See [`LinesearchOutcome`](@ref).

The [`CurvatureCondition`](@ref) is not used to terminate the iteration — it cannot be
honoured by shrinking alone — it is only checked afterwards to emit a warning (see
[`curvature_diagnostic`](@ref)).

# Extended help

[Sometimes](https://en.wikipedia.org/wiki/Backtracking_line_search) the parameters ``p`` and ``c_1`` have different names such as ``\tau`` and ``c``. Note that our ``\tau`` is something else entirely (the round-off allowance above).
"""
struct Backtracking{T} <: LinesearchMethod{T}
    α₀::T
    c₁::T
    c₂::T
    p::T
    τ_ulps::T

    function Backtracking{T}(α₀::T, c₁::T, c₂::T, p::T, τ_ulps::T=T(DEFAULT_ARMIJO_τ_ULPS)) where {T}
        @assert 0 < p < 1 "The shrinking parameter needs to satisfy 0 < p < 1, it is $(p)."
        @assert 0 < c₁ < c₂ < 1 "The Wolfe constants need to satisfy 0 < c₁ < c₂ < 1, they are c₁ = $(c₁), c₂ = $(c₂)."
        @assert τ_ulps ≥ 0 "The round-off allowance needs to be nonnegative, it is $(τ_ulps) ulps."
        new{T}(α₀, c₁, c₂, p, τ_ulps)
    end
end

function Backtracking(::Type{T}=Float64;
    α₀=T(DEFAULT_ARMIJO_α₀),
    c₁=T(DEFAULT_WOLFE_c₁),
    c₂=T(DEFAULT_WOLFE_c₂),
    p=T(DEFAULT_ARMIJO_p),
    τ_ulps=T(DEFAULT_ARMIJO_τ_ULPS)
) where {T}
    Backtracking{T}(α₀, c₁, c₂, p, τ_ulps)
end

Backtracking(::Type{T}, ::SolverMethod) where {T} = Backtracking(T)


@doc raw"""
    armijo_tolerance(φ₀, n)

The absolute round-off allowance ``\tau = n\cdot\mathrm{ulp}(\varphi_0)`` of the
[`SufficientDecreaseCondition`](@ref), where `n` is a number of units in the last place.
See [`DEFAULT_ARMIJO_τ_ULPS`](@ref) and [`backtracking_αmin`](@ref).
"""
armijo_tolerance(φ₀::T, n::T) where {T} = n * eps(φ₀)

@doc raw"""
    backtracking_αmin(c₁, d₀, τ)

The smallest step length for which the [`SufficientDecreaseCondition`](@ref) can still be
decided by the merit rather than by rounding:

```math
\alpha_\mathrm{min} = \frac{\tau}{c_1|\varphi'(0)|} .
```

Below ``\alpha_\mathrm{min}`` the demanded decrease ``c_1\alpha|\varphi'(0)|`` is smaller than
the round-off allowance ``\tau``, so a trial step carries no information. Writing
``\tau = n\cdot\mathrm{ulp}(\varphi(0))`` (see [`armijo_tolerance`](@ref)) gives

```math
\alpha_\mathrm{min} = 2n\,\alpha^*, \qquad
\alpha^* = \frac{\mathrm{ulp}(\varphi(0))}{2c_1|\varphi'(0)|},
```

where ``\alpha^*`` is the step below which ``\mathrm{fl}(\varphi(0) + c_1\alpha\varphi'(0))``
rounds back up to ``\varphi(0)`` and the condition degenerates to
``\varphi(\alpha) \leq \varphi(0)`` — a test that a merit sitting at its round-off floor
passes or fails at random. Since ``n \geq 1`` the search therefore stops a factor ``2n``
*before* entering that region, rather than being decided by it.

The result is clamped to ``[\mathrm{eps}(T), \sqrt{\mathrm{eps}(T)}]``: the lower bound is the
historical negligible-step floor, and the upper bound makes sure that a nearly flat but
genuine merit (very small ``|\varphi'(0)|``) is still searched.
"""
function backtracking_αmin(c₁::T, d₀::T, τ::T) where {T}
    αmin = τ / (c₁ * abs(d₀))
    isfinite(αmin) || (αmin = sqrt(eps(T)))
    clamp(αmin, eps(T), sqrt(eps(T)))
end

@doc raw"""
    backtracking_interpolation(φ₀, d₀, α, φα, αp, φp, p)

The next trial step of the safeguarded polynomial backtracking used by [`Backtracking`](@ref)
(see [nocedal2006numerical; §3.5](@cite), [dennis1996numerical; Alg. A6.3.1](@cite)).

`α`/`φα` is the trial step that was just rejected and `αp`/`φp` the one rejected before it
(`αp` is `NaN` on the first backtrack). The model interpolates ``\varphi(0)``,
``\varphi'(0)`` and the rejected value(s) — a quadratic on the first backtrack, a cubic
afterwards — and its minimiser is clamped to
``[`` [`BACKTRACKING_SHRINK_MIN`](@ref) ``\cdot\alpha, p\alpha]``.

The clamp is what makes this safe: an unclamped interpolant can return ``\alpha`` itself (no
progress at all), collapse to numerically zero, or be meaningless because the merit values it
is built from are rounding noise. Because the upper bound is ``p``, the trial sequence is
pointwise never longer than the plain ``\alpha \gets p\alpha`` ladder.
"""
function backtracking_interpolation(φ₀::T, d₀::T, α::T, φα::T, αp::T, φp::T, p::T) where {T}
    αₙ = T(NaN)
    if isfinite(φα)
        if isnan(αp)
            # quadratic model through (0, φ₀), φ'(0) = d₀ and (α, φα)
            den = 2 * (φα - φ₀ - d₀ * α)
            den > zero(T) && (αₙ = -d₀ * α^2 / den)
        elseif isfinite(φp) && α ≠ αp
            # cubic model through (0, φ₀), φ'(0) = d₀, (αp, φp) and (α, φα)
            r₁ = φα - φ₀ - d₀ * α
            r₂ = φp - φ₀ - d₀ * αp
            den = α^2 * αp^2 * (α - αp)
            a = (αp^2 * r₁ - α^2 * r₂) / den
            b = (-αp^3 * r₁ + α^3 * r₂) / den
            disc = b^2 - 3 * a * d₀
            if disc ≥ zero(T)
                αₙ = iszero(a) ? -d₀ / (2 * b) : (-b + sqrt(disc)) / (3 * a)
            end
        end
    end
    (isfinite(αₙ) && αₙ > zero(T)) || (αₙ = p * α)
    clamp(αₙ, T(BACKTRACKING_SHRINK_MIN) * α, p * α)
end

"""
    solve(ls::Linesearch{T,<:Backtracking}, α, params)

Run the backtracking line search from the trial step `α`, report the outcome through
[`linesearch_warnings`](@ref) and return the accepted step length.

Use [`solve_with_status`](@ref) to obtain the [`LinesearchStatus`](@ref) instead: a caller
that has to tell "I found a decreasing step" from "the merit is at its round-off floor and
nothing can decrease it" cannot do so from the step length alone.
"""
function solve(ls::Linesearch{T,<:Backtracking}, α::T, params=NullParameters()) where {T}
    status = solve_with_status(ls, α, params)
    linesearch_warnings(status, ls, params)
    steplength(status)
end

function solve_with_status(ls::Linesearch{T,<:Backtracking}, α::T, params=NullParameters()) where {T}
    m = method(ls)
    f(a) = value(problem(ls), a, params)
    d(a) = derivative(problem(ls), a, params)

    # note that we anchor at α = 0 here as this is the base point of the linesearch problem.
    φ₀ = f(zero(T))
    d₀ = d(zero(T))

    # No α can satisfy the SufficientDecreaseCondition unless the merit is finite and
    # actually decreasing at the anchor, so report that instead of shrinking α fifty times to
    # find out. Like `StrongWolfe`, hand the caller's trial step back rather than a step that
    # was never accepted (and never the α = 0 anchor, which would freeze the outer iterate).
    if !isfinite(φ₀) || !isfinite(d₀) || d₀ > zero(T)
        return LinesearchStatus{T}(α, LINESEARCH_NO_DESCENT, 0, φ₀, d₀, φ₀, zero(T), zero(T))
    elseif iszero(d₀)
        # A stationary anchor. For the ‖F‖² merit of a nonlinear solver this is the exact
        # root (F = 0 ⇒ φ'(0) = 0 and the direction vanishes), so every α is equivalent.
        return LinesearchStatus{T}(α, LINESEARCH_STATIONARY, 0, φ₀, d₀, φ₀, zero(T), zero(T))
    end

    τ = armijo_tolerance(φ₀, m.τ_ulps)
    αmin = backtracking_αmin(m.c₁, d₀, τ)
    sdc = SufficientDecreaseCondition(m.c₁, φ₀, d₀, f; τ=τ)

    αₐ = α       # last trial step that was actually evaluated
    φₐ = φ₀      # the merit there
    αp = T(NaN)  # previous trial step and merit, for the cubic model
    φp = T(NaN)
    frozen = 0   # consecutive trials whose merit is bit-identical to φ(0)
    n = 0        # trial steps at which the merit was evaluated

    for _ in 1:config(ls).linesearch_max_iterations
        αₐ = α
        φₐ = f(α)
        n += 1

        # Accept as soon as the (round-off tolerant) sufficient decrease condition holds.
        # Whether that is a *genuine* decrease or merely a tie at the merit's round-off floor
        # is decided by φₐ: only a decrease exceeding the allowance τ counts. The returned
        # step is the same either way, the reported outcome is not — which is precisely what
        # the former exact test could not express, since its accepts in the region below
        # α* = ulp(φ₀)/(2c₁|d₀|) were ties produced by rounding yet reported as successes.
        if sdc(α, φₐ)
            oc = φₐ ≤ φ₀ - τ ? LINESEARCH_DECREASED : LINESEARCH_FLOOR
            return LinesearchStatus{T}(α, oc, n, φ₀, d₀, φₐ, τ, αmin)
        end

        # A merit that is bit-identical to φ(0) means the trial point no longer differs from
        # the base point in floating point (α below the x-scale eps‖x‖/‖δx‖), or that the
        # merit is flat to the last bit: every smaller α is frozen too, so shrinking cannot
        # help. Two consecutive frozen trials guard against an accidental tie. This is the
        # scalar surrogate for the x-scale minimum step — a line search only ever sees φ.
        φₐ == φ₀ ? (frozen += 1) : (frozen = 0)
        frozen ≥ 2 && return LinesearchStatus{T}(α, LINESEARCH_FLOOR, n, φ₀, d₀, φₐ, τ, αmin)

        α ≤ αmin && break

        αₙ = backtracking_interpolation(φ₀, d₀, α, φₐ, αp, φp, m.p)
        αp, φp = α, φₐ
        α = max(αₙ, αmin)  # a trial exactly at αmin is always taken before giving up
    end

    # Nothing satisfied the sufficient decrease condition. Distinguish the two very different
    # reasons: if the merit at the smallest informative step differs from φ(0) by no more than
    # the round-off allowance, the merit has reached its round-off floor and *no* step can
    # decrease it — the outer iteration is stuck no matter what the line search returns.
    # Otherwise the merit genuinely fails to decrease even at the smallest informative step,
    # which contradicts φ'(0) < 0.
    capped = α > αmin  # the loop ended on the iteration budget, not on the αmin floor
    oc = (!capped && abs(φₐ - φ₀) ≤ τ) ? LINESEARCH_FLOOR : LINESEARCH_EXHAUSTED
    LinesearchStatus{T}(αₐ, oc, n, φ₀, d₀, φₐ, τ, αmin)
end

# The curvature condition cannot be enforced by shrinking alone, so `Backtracking` only
# reports it — and only for a step that was genuinely accepted, because `derivative` costs a
# full Jacobian for the line search problem of a nonlinear solver.
function curvature_diagnostic(status::LinesearchStatus{T}, ls::Linesearch{T,<:Backtracking}, params) where {T}
    issufficient(status) || return nothing
    d(a) = derivative(problem(ls), a, params)
    cc = CurvatureCondition(method(ls).c₂, status.d₀, d, Val(:Standard))
    cc(steplength(status)) || @warn "Backtracking line search: accepted step α = $(steplength(status)) satisfies the sufficient decrease but not the curvature condition."
    nothing
end

Base.show(io::IO, ls::Backtracking) = print(io, "Backtracking with α₀ = $(ls.α₀) c₁ = $(ls.c₁), c₂ = $(ls.c₂), p = $(ls.p) and τ_ulps = $(ls.τ_ulps).")

function change_precision(::Type{T}, method::Backtracking) where {T}
    T ≠ eltype(method) || return method
    Backtracking{T}(T(method.α₀), T(method.c₁), T(method.c₂), T(method.p), T(method.τ_ulps))
end

function Base.isapprox(bt₁::Backtracking{T}, bt₂::Backtracking{T}; kwargs...) where {T}
    isapprox(bt₁.α₀, bt₂.α₀; kwargs...) && isapprox(bt₁.c₁, bt₂.c₁; kwargs...) && isapprox(bt₁.c₂, bt₂.c₂; kwargs...) && isapprox(bt₁.p, bt₂.p; kwargs...) && isapprox(bt₁.τ_ulps, bt₂.τ_ulps; kwargs...)
end
