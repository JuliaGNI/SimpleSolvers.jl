@doc raw"""
    bisection(f, xmin, xmax; config)

Perform bisection of `f` in the interval [`xmin`, `xmax`] with [`Options`](@ref) `config`.

The algorithm is repeated until a root is found (up to tolerance `config.f_abstol` which is determined by [`default_tolerance`](@ref) by default).

!!! info
    When calling `bisection` it first checks if ``x_\mathrm{min} < x_\mathrm{max}`` and else flips the two entries.

!!! info
    You can also call `bisection` with only one `x` as input argument. It then uses [`bracket_minimum`](@ref) to find a suitable interval.

# Extended help

The bisection algorithm divides an interval into equal halves until a root is found (up to a desired accuracy).

We first initialize:
```math
\begin{aligned}
\alpha_0 \gets & \alpha_\mathrm{min}, \\
\alpha_1 \gets & \alpha_\mathrm{max},
\end{aligned}
```
and then repeat:
```math
\begin{aligned}
& \alpha \gets \frac{\alpha_0 + \alpha_1}{2}, \\
& \text{if $f(\alpha_0)f(\alpha) > 0$} \\
& \qquad \alpha_0 \gets \alpha, \\
& \text{else} \\
& \qquad \alpha_1 \gets \alpha, \\
& \text{end}
\end{aligned}
```
So the algorithm checks in each step where the sign change occurred and moves the ``\alpha_0`` or ``\alpha_1`` accordingly. The loop is terminated (and errors) if `config.max_iterations` is reached (by default """ * """$(MAX_ITERATIONS) in [`Options`](@ref) struct).

!!! warning
    The obvious danger with using bisections is that the supplied interval can have multiple roots (or no roots). One should be careful to avoid this when fixing the interval.

!!! info
    Bisection can only locate a root if the endpoints straddle a sign change. If the
    endpoints have the same sign there is no (odd-multiplicity) root in the interval;
    this arises benignly in the line search once the derivative has flattened at a
    minimum (both endpoint values ≈ 0 with the same sign). Rather than erroring,
    `bisection` then returns the endpoint closest to a root (smallest `|f|`) and warns
    only at high verbosity.
"""
function bisection(f::Callable, αmin::T, αmax::T, params=NullParameters(), config::Options=Options(float(T))) where {T<:Number}
    R = float(T)
    α₀ = R(αmin)
    α₁ = R(αmax)
    α = zero(R)

    # flip α₀ and α₁ if the former is bigger than the latter
    α₀ < α₁ || begin
        α₀, α₁ = α₁, α₀
    end

    y₀ = f(α₀, params)
    y₁ = f(α₁, params)
    y = zero(y₀)

    if y₀ * y₁ > zero(y₀)
        config.verbosity ≥ 2 && @warn "Bisection bracket [$(α₀), $(α₁)] shows no sign change (f = $(y₀), $(y₁)); returning the endpoint with the smallest |f|."
        return abs(y₀) ≤ abs(y₁) ? α₀ : α₁
    end

    converged = false
    for _ in 1:config.max_iterations
        α = (α₀ + α₁) / 2
        y = f(α, params)

        # break if y is close to zero.
        if ≈(y, zero(y); atol=config.f_abstol)
            converged = true
            break
        end

        if y₀ * y > 0
            α₀ = α  # Root is in the right half of [α₀,α₁].
            y₀ = y
        else
            α₁ = α  # Root is in the left half of [α₀,α₁].
            y₁ = y
        end

        if isapprox(α₁ - α₀, zero(α), atol=config.x_suctol * max(abs(α₀), abs(α₁)))
            converged = true
            break
        end
    end

    converged || (config.verbosity ≥ 1 && @warn "Bisection did not converge within $(config.max_iterations) iterations; returning best estimate α = $(α).")

    α
end

function bisection(f::Callable, α::T, params=NullParameters(), config::Options=Options(float(T))) where {T<:Number}
    R = float(T)
    lo, hi = bracket_root(β -> f(β, params), R(α))
    bisection(f, lo, hi, params, config)
end

# Disambiguates `(f, ::T, ::T, ::Options)` in favor of the interval form with default `params`.
bisection(f::Callable, αmin::T, αmax::T, config::Options) where {T<:Number} = bisection(f, αmin, αmax, NullParameters(), config)

"""
    Bisection <: Linesearch

See [`bisection`](@ref) for the implementation of the algorithm.

# Extended help

When invoked with a single trial step `α` (i.e. `solve(ls, α)`), the bracket is
always *lower-anchored* at ``\\alpha = 0`` — the only point where a genuine
descent direction is guaranteed to have a decreasing merit (``\\varphi'(0) < 0``),
which one-sided rightward bracketing requires. The caller's `α` is then folded in
via one extra derivative evaluation (see issue #164):

- if ``\\varphi'(\\alpha) \\geq 0`` then `α` overshot the minimum and ``[0, \\alpha]``
  already brackets a stationary point, so it is handed straight to [`bisection`](@ref)
  with no bracketing loop;
- otherwise ``\\alpha`` still lies on the descent side, so the bracket is grown
  outward from ``0`` with the initial step seeded from ``|\\alpha|`` — clamped
  between [`DEFAULT_BRACKETING_s`](@ref) and `1` so a large `α` does not
  over-coarsen the search and a tiny `α` does not crawl — rather than the fixed
  default step.

This keeps the safe ``\\alpha = 0`` anchor while letting the caller's `α` set the
search scale and, when it overshoots, serve directly as the upper bracket bound.
"""
struct Bisection{T} <: LinesearchMethod{T} end

Bisection(::Type{T}=Float64) where {T} = Bisection{T}()
Bisection(::Type{T}, ::SolverMethod) where {T} = Bisection(T)


function solve(ls::Linesearch{T,<:Bisection}, α₀::T, α₁::T, params=NullParameters()) where {T}
    bisection(problem(ls).D, α₀, α₁, params, config(ls))
end

function solve(ls::Linesearch{T,<:Bisection}, α::T, params=NullParameters()) where {T}
    prob = problem(ls)

    # Lower-anchor the bracket at α = 0, where a genuine descent direction has a
    # decreasing merit (φ′(0) < 0). Probe the caller's trial step α (one extra
    # derivative evaluation) to decide how to fold it in; see the docstring and #164.
    if α > zero(T) && derivative(prob, zero(T), params) < zero(T) && derivative(prob, α, params) ≥ zero(T)
        # α overshot the minimum: [0, α] already brackets the stationary point.
        return solve(ls, zero(T), α, params)
    end

    # α is on the descent side (or not a descent step / α ≤ 0): grow the bracket from
    # 0, seeding the step scale from |α| (clamped) instead of the fixed default.
    s = clamp(abs(α), T(DEFAULT_BRACKETING_s), one(T))
    solve(ls, bracket_minimum(prob, params, zero(T), s)..., params)
end

Base.show(io::IO, ::Bisection) = print(io, "Bisection")

function change_precision(::Type{T}, method::Bisection) where {T}
    T ≠ eltype(method) || return method
    Bisection(T)
end

Base.isapprox(::Bisection{T}, ::Bisection{T}; kwargs...) where {T} = true
