"""
Quadratic Armijo line search
"""
struct ArmijoQuadraticState{T,DT,AT,FT} <: LinesearchState where {T <: Number, DT <: Number, AT <: AbstractArray{DT}, FT}

    nmax::Int
    rmax::Int

    λ₀::T
    σ₀::T
    σ₁::T
    ϵ::T

    F!::FT

    δx::AT
    δy::AT
    y::AT

    function ArmijoQuadraticState(F!, x, y; nmax=DEFAULT_LINESEARCH_nmax, rmax=DEFAULT_LINESEARCH_rmax,
                    λ₀::T=DEFAULT_ARMIJO_λ₀, σ₀::T=DEFAULT_ARMIJO_σ₀, σ₁::T=DEFAULT_ARMIJO_σ₁, ϵ::T=DEFAULT_ARMIJO_ϵ) where {T}
        new{T, eltype(y), typeof(y), typeof(F!)}(nmax, rmax, λ₀, σ₀, σ₁, ϵ, F!, zero(x), zero(y), zero(y))
    end
end


LinesearchState(algorithm::ArmijoQuadratic, objective, x, y; kwargs...) = ArmijoQuadraticState(objective.F, x, y; kwargs...)


function (ls::ArmijoQuadraticState)(x::AbstractArray{T}, f::AbstractArray{T}, g::AbstractArray{T}, x₀::AbstractArray{T}, x₁::AbstractArray{T}) where {T}
    local λ::T
    local λₜ::T
    local y₀norm::T
    local y₁norm::T
    local p₀::T
    local p₁::T
    local p₂::T

    # set initial λ
    λ = ls.λ₀

    # δx = x₁ - x₀
    ls.δx .= x₁ .- x₀

    # δy = Jδx
    mul!(ls.δy, g, ls.δx)

    # compute norms of initial solution
    y₀norm = l2norm(f)

    # determine constant coefficients of polynomial p(λ) = p₀ + p₁λ + p₂λ²
    p₀ = y₀norm^2
    p₁ = 2(⋅(f, ls.δy))
    
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

                λ *= ls.σ₁
            end
        end

        # compute norms of solution
        y₁norm = l2norm(ls.y)

        if y₁norm ≥ (one(T) - ls.ϵ * λ) * y₀norm
            # determine nonconstant coefficient of polynomial p(λ) = p₀ + p₁λ + p₂λ²
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

    return x
end


solve!(x, f, g, x₀, x₁, ls::ArmijoQuadraticState) = ls(x, f, g, x₀, x₁)


function armijo_quadratic(F, x, f, g, x₀, x₁; kwargs...)
    ls = ArmijoQuadraticState(F, x, f; kwargs...)
    ls(x, f, g, x₀, x₁)
end
