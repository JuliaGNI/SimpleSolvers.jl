

function bisection(f, xmin::T, xmax::T; config = Options()) where {T <: Number}
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

    return x
end

bisection(f, x::Number; kwargs...) = bisection(f, bracket_minimum(f, x)...; kwargs...)


"""
simple bisection line search
"""
mutable struct BisectionState{OBJ,OPT} <: LinesearchState where {OBJ <: UnivariateObjective, OPT <: Options}
    objective::OBJ
    config::OPT

    function BisectionState(objective, config)
        new{typeof(objective), typeof(config)}(objective, config)
    end
end

function BisectionState(objective::UnivariateObjective; config = Options())
    BisectionState(objective, config)
end

# function BisectionState(objective::MultivariateObjective; config = Options())
#     cache = LinesearchCache(objective.x_f)
#     ls_objective = linesearch_objective(objective, cache)
#     BisectionState(ls_objective, config)
# end

Base.show(io::IO, ls::BisectionState) = print(io, "Bisection")

LinesearchState(algorithm::Bisection, objective; kwargs...) = BisectionState(objective; kwargs...)

function (ls::BisectionState)()
    bisection(ls.objective, 0., 1.; config = ls.config)
end
