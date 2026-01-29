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
x_0 \gets & x_\mathrm{min},
x_1 \gets & x_\mathrm{max},
\end{aligned}
```
and then repeat:
```math
\begin{aligned}
& x \gets \frac{x_0 + x_1}{2}, \\
& \text{if $f(x_0)f(x) > 0$} \\
& \qquad x_0 \gets x, \\
& \text{else} \\
& \qquad x_1 \gets x, \\
& \text{end}
\end{aligned}
```
So the algorithm checks in each step where the sign change occurred and moves the ``x_0`` or ``x_1`` accordingly. The loop is terminated (and errors) if `config.max_iterations` is reached (by default""" * """$(MAX_ITERATIONS) and the [`Options`](@ref) struct).

!!! warning
    The obvious danger with using bisections is that the supplied interval can have multiple roots (or no roots). One should be careful to avoid this when fixing the interval.
"""
function bisection(f::Callable, xmin::T, xmax::T; config::Options) where {T <: Number}
    x₀ = xmin
    x₁ = xmax
    x  = zero(T)

    # flip x₀ and x₁ if the former is bigger than the latter
    x₀ < x₁ || begin x₀, x₁ = x₁, x₀ end

    y₀ = f(x₀)
    y₁ = f(x₁)
    y  = zero(y₀)

    # @assert y₀ * y₁ ≤ 0 "Either no or multiple real roots in [xmin,xmax]."

    for j in 1:config.max_iterations
        x = (x₀ + x₁) / 2
        y = f(x)

        # break if y is close to zero.
        !≈(y, zero(y); atol=config.f_abstol) || break

        if y₀ * y > 0
            x₀ = x  # Root is in the right half of [x₀,x₁].
            y₀ = y
        else
            x₁ = x  # Root is in the left half of [x₀,x₁].
            y₁ = y
        end

        !isapprox(x₁ - x₀, zero(x), atol=config.x_abstol) || break

        j != config.max_iterations || (println(x₀, " ", x₁, " ", x₁ - x₀); error("Max iteration number exceeded"))
    end

    x
end

bisection(problem::AbstractOptimizerProblem, xmin::T, xmax::T; config::Options) where {T <: Number} = bisection(problem.D, xmin, xmax; config = config)
# bisection(problem::AbstractOptimizerProblem, x::T; kwargs...) = bisection(problem.D, x; kwargs...)

"""
    bisection(f, x)

Use [`bracket_minimum`](@ref) to find a starting interval and then do bisections.
"""
bisection(f, x::Number; kwargs...) = bisection(f, bracket_minimum(f, x)...; kwargs...)

"""
    Bisection <: Linesearch

See [`bisection`](@ref) for the implementation of the algorithm.
"""
struct Bisection{T} <: LinesearchMethod{T} end

Bisection(T::DataType=Float64) = Bisection{T}()

function solve(problem::LinesearchProblem{T}, ls::Linesearch{T, LST}, x₀::T, x₁::T) where {T, LST <: Bisection}
    bisection(problem, x₀, x₁; config = ls.config)
end

function solve(problem::LinesearchProblem{T}, ls::Linesearch{T, LST}, x::T=zero(T)) where {T, LST <: Bisection}
    solve(problem, ls, bracket_minimum(problem.F, x)...)
end

Base.show(io::IO, ::Bisection) = print(io, "Bisection")

function Base.convert(::Type{T}, algorithm::Bisection) where {T}
    T ≠ eltype(algorithm) || return algorithm
    Bisection(T)
end

Base.isapprox(::Bisection{T}, ::Bisection{T}; kwargs...) where {T} = true