"""
    bisection(f, xmin, xmax; config)

Perform bisection of `f` in the interval [`xmin`, `xmax`] with [`Options`](@ref) `config`.

The algorithm is repeated until a root is found (up to tolerance `config.f_abstol` which is [`F_ABSTOL`](@ref) by default).
"""
function bisection(f::Callable, xmin::T, xmax::T; config = Options()) where {T <: Number}
    local x₀ = xmin
    local x₁ = xmax
    local x  = zero(T)

    x₀ < x₁ || begin x₀, x₁ = x₁, x₀ end

    local y₀ = f(x₀)
    local y₁ = f(x₁)
    local y  = zero(y₀)

    # y₀ * y₁ ≤ 0 || error("Either no or multiple real roots in [xmin,xmax]")

    for j in 1:config.max_iterations
        x = (x₀ + x₁) / 2
        y = f(x)

        # println("j = ", j, " , x₀ = ", x₀, " , x₁ = ", x₁)

        !isapprox(y, zero(y), atol=config.f_abstol) || break

        if y₀ * y > 0
            x₀ = x  # Root is in the right half of [x₀,x₁].
            y₀ = y
        else
            x₁ = x  # Root is in the left half of [x₀,x₁].
            y₁ = y
        end

        !isapprox(x₁ - x₀, zero(x), atol=config.x_abstol) || break
    end

    # println("α=", x, ", f(α)=", y, ", ftol=", config.f_abstol, ", abs(x₁-x₀)=", abs(x₁-x₀), ", xtol=", config.x_abstol)

    # i != ls.nmax || error("Max iteration number exceeded")

    x
end

bisection(obj::AbstractObjective, xmin::T, xmax::T; config = Options()) where {T <: Number} = bisection(obj.F, xmin, xmax; config = config)

bisection(f, x::Number; kwargs...) = bisection(f, bracket_minimum(f, x)...; kwargs...)

"""
    BisectionState <: LinesearchState

Corresponding [`LinesearchState`](@ref) to [`Bisection`](@ref).

See [`bisection`](@ref) for the implementation of the algorithm.

# Constructors

```julia
BisectionState(options)
BisectionState(; options)
```
"""
mutable struct BisectionState{OPT} <: LinesearchState where {OPT <: Options}
    config::OPT
end

function BisectionState(; config = Options())
    BisectionState(config)
end

Base.show(io::IO, ls::BisectionState) = print(io, "Bisection")

LinesearchState(algorithm::Bisection; T::DataType=Float64, kwargs...) = BisectionState(; kwargs...)

function (ls::BisectionState)(obj::AbstractUnivariateObjective)
    bisection(obj, 0., 1.; config = ls.config)
end
