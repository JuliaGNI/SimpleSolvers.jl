

function bisection(f, xmin::T, xmax::T; config = Options()) where {T}
    local x₀ = xmin
    local x₁ = xmax
    local x  = zero(T)

    x₀ < x₁ || begin x₀, x₁ = x₁, x₀ end

    local y₀ = f(x₀)
    local y₁ = f(x₁)
    local y  = zero(y₀)

    # y₀ * y₁ ≤ 0 || error("Either no or multiple real roots in [xmin,xmax]")

    for _ in 1:config.max_iterations
        x = (x₀ + x₁) / 2
        y = f(x)

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

    # println(j, " bisection iterations, λ=", λ, ", f(λ)=", fλ, ", ftol=", ftol, ", abs(b-a)=", abs(b-a), ", xtol=", xtol)

    # i != ls.nmax || error("Max iteration number exceeded")

    return x
end

bisection(f, x; kwargs...) = BisectionState(f; kwargs...)(x)


"""
simple bisection line search
"""
mutable struct BisectionState{OBJ,OPT} <: LinesearchState where {OBJ, OPT}
    objective::OBJ
    config::OPT

    function BisectionState(objective; config = Options())
        new{typeof(objective), typeof(config)}(objective, config)
    end
end

Base.show(io::IO, ls::BisectionState) = print(io, "Bisection")

LinesearchState(algorithm::Bisection, objective::UnivariateObjective, x; kwargs...) = BisectionState(objective; kwargs...)

(ls::BisectionState)(objective, xmin, xmax) = bisection(objective, xmin, xmax; config = ls.config)
(ls::BisectionState)(xmin, xmax) = bisection(ls.objective, xmin, xmax; config = ls.config)
(ls::BisectionState)(x) = ls(bracket_minimum(ls.objective, x)...)
