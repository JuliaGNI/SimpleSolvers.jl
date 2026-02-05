@doc raw"""
    bisection(f, xmin, xmax; config)

Perform bisection of `f` in the interval [`xmin`, `xmax`] with [`Options`](@ref) `config`.

The algorithm is repeated until a root is found (up to tolerance `config.f_abstol` which is determined by [`default_tolerance`](@ref) by default).

# implementation

When calling `bisection` it first checks if ``x_\mathrm{min} < x_\mathrm{max}`` and else flips the two entries.

# Extended help

The bisection algorithm divides an interval into equal halves until a root is found (up to a desired accuracy).

We first initialize:
```math
\begin{aligned}
\alpha_0 \gets & \alpha_\mathrm{min},
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
So the algorithm checks in each step where the sign change occurred and moves the ``\alpha_0`` or ``\alpha_1`` accordingly. The loop is terminated (and errors) if `config.max_iterations` is reached (by default""" * """$(MAX_ITERATIONS) and the [`Options`](@ref) struct).

!!! warning
    The obvious danger with using bisections is that the supplied interval can have multiple roots (or no roots). One should be careful to avoid this when fixing the interval.
"""
function bisection(f::Callable, αmin::T, αmax::T, params=NullParameters(), config::Options=Options(T)) where {T<:Number}
    α₀ = αmin
    α₁ = αmax
    α = zero(T)

    # flip α₀ and α₁ if the former is bigger than the latter
    α₀ < α₁ || begin
        α₀, α₁ = α₁, α₀
    end

    y₀ = f(α₀)
    y₁ = f(α₁)
    y = zero(y₀)

    # @assert y₀ * y₁ ≤ 0 "Either no or multiple real roots in [xmin,xmax]."

    for j in 1:config.max_iterations
        α = (α₀ + α₁) / 2
        y = f(α)

        # break if y is close to zero.
        !≈(y, zero(y); atol=config.f_abstol) || break

        if y₀ * y > 0
            α₀ = α  # Root is in the right half of [α₀,α₁].
            y₀ = y
        else
            α₁ = α  # Root is in the left half of [α₀,α₁].
            y₁ = y
        end

        !isapprox(α₁ - α₀, zero(α), atol=config.x_suctol * max(abs(α₀), abs(α₁))) || break

        j != config.max_iterations || (println(α₀, " ", α₁, " ", α₁ - α₀);
        error("Max iteration number exceeded"))
    end

    α
end

"""
    bisection(f, α)

Use [`bracket_minimum`](@ref) to find a starting interval and then do bisections.
"""
bisection(f::Callable, α::T, params=NullParameters(), config::Options=Options(T)) where {T<:Number} = bisection(f, bracket_minimum(f, α)..., params, config)

"""
    Bisection <: Linesearch

See [`bisection`](@ref) for the implementation of the algorithm.
"""
struct Bisection{T} <: LinesearchMethod{T} end

Bisection(T::DataType=Float64) = Bisection{T}()
Bisection(::Type{T}, ::SolverMethod) where {T} = Bisection(T)


function solve(problem::LinesearchProblem{T}, ls::Linesearch{T,<:Bisection}, α₀::T, α₁::T, params=NullParameters()) where {T}
    bisection(problem.D, α₀, α₁, params, config(ls))
end

function solve(problem::LinesearchProblem{T}, ls::Linesearch{T,<:Bisection}, α::T, params=NullParameters()) where {T}
    # TODO: The following line should use α instead of zero(T) but that requires a rework of the bracketing algorithm
    # solve(problem, ls, bracket_minimum(problem.F, α)..., params)
    solve(problem, ls, bracket_minimum(problem.F, zero(T))..., params)
end

Base.show(io::IO, ::Bisection) = print(io, "Bisection")

function Base.convert(::Type{T}, method::Bisection) where {T}
    T ≠ eltype(method) || return method
    Bisection(T)
end

Base.isapprox(::Bisection{T}, ::Bisection{T}; kwargs...) where {T} = true
