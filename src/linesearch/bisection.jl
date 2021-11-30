

const DEFAULT_BISECTION_xtol = 2eps()
const DEFAULT_BISECTION_ftol = 2eps()

"""
simple bisection line search
"""
mutable struct Bisection{XT,YT,FT} <: LineSearch where {XT <: Number, YT <: Number, FT <: Callable}

    nmax::Int
    xtol::XT
    ftol::YT

    f::FT

    function Bisection(f; nmax=DEFAULT_LINESEARCH_nmax, xtol::XT=DEFAULT_BISECTION_xtol, ftol::YT=DEFAULT_BISECTION_ftol) where {XT,YT}
        new{XT, YT, typeof(f)}(nmax, xtol, ftol, f)
    end
end


function (ls::Bisection)(xmin::T, xmax::T) where {T}
    local x₀ = xmin
    local x₁ = xmax
    local x  = zero(T)

    x₀ < x₁ || begin x₀, x₁ = x₁, x₀ end

    local y₀ = ls.f(x₀)
    local y₁ = ls.f(x₁)

    # y₀ * y₁ ≤ 0 || error("Either no or multiple real roots in [xmin,xmax]")

    for _ in 1:ls.nmax
        x = (x₀ + x₁) / 2
        y = ls.f(x)

        !isapprox(y, zero(y), atol=ls.ftol) || break

        if y₀ * y > 0
            x₀ = x  # Root is in the right half of [x₀,x₁].
            y₀ = y
        else
            x₁ = x  # Root is in the left half of [x₀,x₁].
            y₁ = y
        end

        !isapprox(x₁ - x₀, zero(x), atol=ls.xtol) || break
    end

    # println(j, " bisection iterations, λ=", λ, ", f(λ)=", fλ, ", ftol=", ftol, ", abs(b-a)=", abs(b-a), ", xtol=", xtol)

    # i != ls.nmax || error("Max iteration number exceeded")

    return x
end

(ls::Bisection)(x) = ls(bracket_minimum(ls.f, x)...)

solve!(x, f, g, x₀, x₁, ls::Bisection) = ls(x₀, x₁)
solve!(x₀, x₁, ls::Bisection) = ls(x₀, x₁)
solve!(x, f, g, ls::Bisection) = ls(x)
solve!(x, ls::Bisection) = ls(x)

bisection(f, x₀, x₁; kwargs...) = Bisection(f; kwargs...)(x₀, x₁)
bisection(f, x; kwargs...) = Bisection(f; kwargs...)(x)
