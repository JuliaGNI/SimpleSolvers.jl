
abstract type GradientParameters{T} end


struct GradientParametersUser{T, ∇T <: Callable} <: GradientParameters{T}
    ∇F!::∇T
end

struct GradientParametersAD{T, FT <: Callable, ∇T <: ForwardDiff.GradientConfig, VT <: AbstractVector{T}} <: GradientParameters{T}
    F::FT
    ∇config::∇T
    tx::VT

    function GradientParametersAD(F::FT, ∇config::∇T, tx::VT) where {T, FT, ∇T, VT <: AbstractVector{T}}
        new{T, FT, ∇T, VT}(F, ∇config, tx)
    end
end

function GradientParametersAD(F::FT, T, nx::Int) where {FT <: Callable}
    tx = zeros(T, nx)
    ∇config = ForwardDiff.GradientConfig(F, tx)
    GradientParametersAD(F, ∇config, tx)
end

struct GradientParametersFD{T, FT} <: GradientParameters{T}
    ϵ::T
    F::FT
    e::Vector{T}
    tx::Vector{T}
end

function GradientParametersFD(F::FT, ϵ, T, nx::Int) where {FT <: Callable}
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    GradientParametersFD{T, typeof(F)}(ϵ, F, e, tx)
end

function computeGradient(x::AbstractVector{T}, g::AbstractVector{T}, params::GradientParametersUser{T}) where {T}
    params.∇F!(g, x)
end

function computeGradient(x::AbstractVector{T}, g::AbstractVector{T}, params::GradientParametersFD{T}) where {T}
    local ϵⱼ::T

    for j in eachindex(x,g)
        ϵⱼ = params.ϵ * x[j] + params.ϵ
        fill!(params.e, 0)
        params.e[j] = 1
        params.tx .= x .- ϵⱼ .* params.e
        f1 = params.F(params.tx)
        params.tx .= x .+ ϵⱼ .* params.e
        f2 = params.F(params.tx)
        g[j] = (f2 - f1)/(2ϵⱼ)
    end
end

function computeGradient(x::AbstractVector{T}, g::AbstractVector{T}, ∇params::GradientParametersAD{T}) where {T}
    ForwardDiff.gradient!(g, ∇params.F, x, ∇params.∇config)
end

function computeGradientFD(x::AbstractVector{T}, g::AbstractVector{T}, F::FT, ϵ::T) where {T, FT <: Callable}
    params = GradientParametersFD(F, ϵ, T, length(x))
    computeGradient(x, g, params)
end

function computeGradientAD(x::AbstractVector{T}, g::AbstractVector{T}, F::FT) where {T, FT <: Callable}
    params = GradientParametersAD(F, T, length(x))
    computeGradient(x, g, params)
end


function getGradientParameters(g, F, T, nx)
    if g === nothing
        if get_config(:gradient_autodiff)
            ∇params = GradientParametersAD(F, T, nx)
        else
            ∇params = GradientParametersFD(F, get_config(:gradient_fd_ϵ), T, nx)
        end
    else
        ∇params = GradientParametersUser{T, typeof(g)}(g)
    end
    return ∇params
end


function check_gradient(g::AbstractVector)
    println("norm(Gradient):               ", norm(g))
    println("minimum(|Gradient|):          ", minimum(abs.(g)))
    println("maximum(|Gradient|):          ", maximum(abs.(g)))
    println()
end

function print_gradient(g::AbstractVector)
    display(g)
    println()
end
