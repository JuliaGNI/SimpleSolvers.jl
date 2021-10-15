

const DEFAULT_BISECTION_xtol = 2eps()
const DEFAULT_BISECTION_ftol = 2eps()

"""
simple bisection line search
"""
struct Bisection{T,DT,AT,FT} <: LineSearch where {T <: Number, DT <: Number, AT <: AbstractArray{DT}, FT <: Callable}

    nmax::Int
    xtol::T
    ftol::T

    F!::FT

    x₀::AT
    x₁::AT
    y₀::AT
    y₁::AT

    y::AT
    
    function Bisection(F!, x, y; nmax=DEFAULT_LINESEARCH_nmax, 
                    xtol::T=DEFAULT_BISECTION_xtol, ftol::T=DEFAULT_BISECTION_ftol) where {T}
        new{T, eltype(y), typeof(y), typeof(F!)}(nmax, xtol, ftol, F!, zero(x), zero(x), zero(y), zero(y), zero(y))
    end
end


function (ls::Bisection)(x::T, x₀::T, x₁::T) where {T}
    ls.x₀ .= x₀
    ls.x₁ .= x₁

    ls.F!(ls.x₀, ls.y₀)
    ls.F!(ls.x₁, ls.y₁)

    # y₀ .* y₁ ≤ 0 || error("Either no or multiple real roots in [λmin,λmax]")

    for _ in 1:ls.nmax
        x .= (ls.x₀ .+ ls.x₁) ./ 2
        ls.F!(x, ls.y)

        !isapprox(ls.y, zero(ls.y), atol=ls.ftol) || break

        if ls.y₀ .* ls.y > 0
            ls.x₀ .= x  # Root is in the right half of [x₀,x₁].
            ls.y₀ .= ls.y
        else
            ls.x₁ .= x  # Root is in the left half of [x₀,x₁].
            ls.y₁ .= ls.y
        end

        !isapprox(ls.x₁ .- ls.x₀, zero(x), atol=ls.xtol) || break
    end

    # println(j, " bisection iterations, λ=", λ, ", f(λ)=", fλ, ", ftol=", ftol, ", abs(b-a)=", abs(b-a), ", xtol=", xtol)

    # i != ls.nmax || error("Max iteration number exceeded")

    return x
end

(ls::Bisection)(x, f, g, x₀, x₁) = ls(x, x₀, x₁)

solve!(x, f, g, x₀, x₁, ls::Bisection) = ls(x, x₀, x₁)
solve!(x, x₀, x₁, ls::Bisection) = ls(x, x₀, x₁)


function bisection(F, x, f, g, x₀, x₁; kwargs...)
    ls = Bisection(F, x, f; kwargs...)
    ls(x, f, g, x₀, x₁)
end
