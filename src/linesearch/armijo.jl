
import Base: Callable

const DEFAULT_LINESEARCH_nmax=100
const DEFAULT_LINESEARCH_rmax=100
const DEFAULT_ARMIJO_λ₀ = 1.0
const DEFAULT_ARMIJO_σ₀ = 0.1
const DEFAULT_ARMIJO_σ₁ = 0.5
const DEFAULT_ARMIJO_ϵ  = 1E-4

"""
simple Armijo line search
"""
struct Armijo{T,DT,AT,FT} <: LineSearch where {T <: Number, DT <: Number, AT <: AbstractArray{DT}, FT <: Callable}

    nmax::Int
    rmax::Int

    λ₀::T
    σ₀::T
    σ₁::T
    ϵ::T

    F!::FT

    y::AT

    function Armijo(F!, y::AbstractArray{T}; nmax=DEFAULT_LINESEARCH_nmax, rmax=DEFAULT_LINESEARCH_rmax,
                    λ₀=DEFAULT_ARMIJO_λ₀, σ₀=DEFAULT_ARMIJO_σ₀, σ₁=DEFAULT_ARMIJO_σ₁, ϵ=DEFAULT_ARMIJO_ϵ) where {T}
        new{T, eltype(y), typeof(y), typeof(F!)}(nmax, rmax, λ₀, σ₀, σ₁, ϵ, F!, zero(y))
    end
end


function (ls::Armijo)(x::AbstractArray{T}, δx::AbstractArray{T}, x₀::AbstractArray{T}, y₀::AbstractArray{T}, g₀::AbstractArray{T}) where {T}
    local λ::T
    local λₜ::T
    local y₀norm::T
    local y₁norm::T
    local p₀::T
    local p₁::T
    local p₂::T

    # set initial λ
    λ = ls.λ₀

    # δy = Jδx
    # mul!(ls.δy, g₀, δx)

    # compute norms of initial solution
    y₀norm = l2norm(y₀)
    
    for _ in 1:ls.nmax
        # x₁ = x₀ + λ δx
        x .= x₀ .+ λ .* δx

        for _ in 1:ls.rmax
            try
                # y = f(x)
                ls.F!(x, ls.y)

                break
            catch DomainError
                # in case the new function value results in some DomainError
                # (e.g., for functions f(x) containing sqrt's or log's),
                # decrease λ and retry

                @warn("Armijo line search encountered Domain Error (lsiter=$lsiter, λ=$λ). Decreasing λ and trying again...")

                λ *= ls.σ₁
            end
        end

        # compute norms of solution
        y₁norm = l2norm(ls.y)

        # if y₁norm ≥ ls.atol + ls.rtol * y₀norm
        if y₁norm ≥ (one(T)-ls.ϵ*λ)*y₀norm
            # determine coefficients of polynomial p(λ) = p₀ + p₁λ + p₂λ²
            p₀ = y₀norm^2
            # p₁ = 2(⋅(y₀, ls.δy))
            p₁ = 2(⋅(y₀, g₀, δx))
            p₂ = (y₁norm^2 - p₀ - p₁*λ) / λ^2

            # compute minimum λₜ of p(λ)
            λₜ = - p₁ / (2p₂)

            if λₜ < ls.σ₀ * λ
                λ = ls.σ₀ * λ
            elseif λₜ > ls.σ₁ * λ
                λ = ls.σ₁ * λ
            else
                λ = λₜ
            end
        else
            break
        end
    end
end


function solve!(x, δx, x₀, y₀, g₀, ls::Armijo)
    ls(x, δx, x₀, y₀, g₀)
end

function armijo(f, x, δx, x₀, y₀, g₀; kwargs...)
    ls = Armijo(f, y₀; kwargs...)
    ls(x, δx, x₀, y₀, g₀)
end
