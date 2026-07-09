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
"""
function bisection(f::Callable, αmin::T, αmax::T, params=NullParameters(), config::Options=Options(float(T))) where {T<:Number}
    # Promote to a floating point type on entry: with an integer `T` the midpoint
    # `(α₀ + α₁) / 2` would silently switch `α`'s type mid-loop, and `Options(T)`
    # is undefined for integer `T`.
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

    # Bisection can only locate a root if the endpoints straddle a sign change.
    # Same-sign endpoints mean there is no (odd-multiplicity) root in the interval;
    # the loop below would otherwise silently collapse onto α₁ and return a
    # non-root.  This case arises benignly in the line search once the derivative
    # has flattened at a minimum (both endpoint values ≈ 0 with the same sign), so
    # rather than erroring we return the endpoint closest to a root (smallest |f|)
    # and warn only at high verbosity.
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

        # break once the bracket has shrunk below the successive-step tolerance.
        if isapprox(α₁ - α₀, zero(α), atol=config.x_suctol * max(abs(α₀), abs(α₁)))
            converged = true
            break
        end
    end

    # Return the best estimate rather than erroring on exhaustion; warn so the
    # caller knows the tolerance was not met.
    converged || (config.verbosity ≥ 1 && @warn "Bisection did not converge within $(config.max_iterations) iterations; returning best estimate α = $(α).")

    α
end

bisection(f::Callable, α::T, params=NullParameters(), config::Options=Options(float(T))) where {T<:Number} = bisection(f, bracket_root(β -> f(β, params), α)..., params, config)

"""
    Bisection <: Linesearch

See [`bisection`](@ref) for the implementation of the algorithm.
"""
struct Bisection{T} <: LinesearchMethod{T} end

Bisection(::Type{T}=Float64) where {T} = Bisection{T}()
Bisection(::Type{T}, ::SolverMethod) where {T} = Bisection(T)


function solve(ls::Linesearch{T,<:Bisection}, α₀::T, α₁::T, params=NullParameters()) where {T}
    bisection(problem(ls).D, α₀, α₁, params, config(ls))
end

function solve(ls::Linesearch{T,<:Bisection}, α::T, params=NullParameters()) where {T}
    # TODO: The following line should use α instead of zero(T) but that requires a rework of the bracketing algorithm
    # solve(problem, ls, bracket_minimum(problem.F, α)..., params)
    solve(ls, bracket_minimum(problem(ls), params, zero(T))..., params)
end

Base.show(io::IO, ::Bisection) = print(io, "Bisection")

function change_precision(::Type{T}, method::Bisection) where {T}
    T ≠ eltype(method) || return method
    Bisection(T)
end

Base.isapprox(::Bisection{T}, ::Bisection{T}; kwargs...) where {T} = true
