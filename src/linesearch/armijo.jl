
const DEFAULT_ARMIJO_λ₀ = 1.0
const DEFAULT_ARMIJO_σ₀ = 0.1
const DEFAULT_ARMIJO_σ₁ = 0.5
const DEFAULT_ARMIJO_ϵ  = 1E-4

struct Armijo{T,DT,AT,FT} <: LineSearch where {T <: Number, DT <: Number, AT <: AbstractArray{DT}, FT}

    nmax::Int
    rmax::Int

    λ₀::T
    ϵ::T

    F!::FT

    δx::AT
    y₀::AT
    y::AT

    function Armijo(F!, x, y; nmax=DEFAULT_LINESEARCH_nmax, rmax=DEFAULT_LINESEARCH_rmax,
                    λ₀::T=DEFAULT_ARMIJO_λ₀, ϵ::T=DEFAULT_ARMIJO_ϵ) where {T}
        new{T, eltype(x), typeof(x), typeof(F!)}(nmax, rmax, λ₀, ϵ, F!, zero(x), zero(y), zero(y))
    end
end


function (ls::Armijo)(x::AbstractArray{T}, f::AbstractArray{T}, g::AbstractArray{T}, x₀::AbstractArray{T}, x₁::AbstractArray{T}) where {T}
    local λ::T
    local y₀norm::T
    local y₁norm::T

    # set initial λ
    λ = ls.λ₀

    # δx = x₁ - x₀
    ls.δx .= x₁ .- x₀

    # compute norms of initial solution
    y₀norm = l2norm(f)

    for lsiter in 1:ls.nmax
        # x₁ = x₀ + λ δx
        x .= x₀ .+ λ .* ls.δx

        for _ in 1:ls.rmax
            try
                # y = f(x)
                ls.F!(ls.y, x)

                break
            catch DomainError
                # in case the new function value results in some DomainError
                # (e.g., for functions f(x) containing sqrt's or log's),
                # decrease λ and retry

                @warn("Armijo line search encountered Domain Error (lsiter=$lsiter, λ=$λ). Decreasing λ and trying again...")

                λ /= 2
            end
        end

        # compute norms of solution
        y₁norm = l2norm(ls.y)

        if y₁norm ≥ (one(T) - ls.ϵ * λ) * y₀norm
            λ /= 2
        else
            break
        end
    end

    return x
end


solve!(x, f, g, x₀, x₁, ls::Armijo) = ls(x, f, g, x₀, x₁)


function armijo(F, x, f, g, x₀, x₁; kwargs...)
    ls = Armijo(F, x, f; kwargs...)
    ls(x, f, g, x₀, x₁)
end
