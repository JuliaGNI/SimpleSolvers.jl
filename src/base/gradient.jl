
const DEFAULT_GRADIENT_ϵ = 8sqrt(eps())


abstract type GradientParameters{T} end

compute_gradient!(g, x, grad::GradientParameters) = grad(g,x)

function compute_gradient(x, grad::GradientParameters)
    g = alloc_g(x)
    grad(g,x)
    return g
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


struct GradientParametersUser{T, ∇T <: Callable} <: GradientParameters{T}
    ∇F!::∇T
end

function (grad::GradientParametersUser{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    grad.∇F!(g, x)
end


struct GradientParametersAD{T, FT <: Callable, ∇T <: ForwardDiff.GradientConfig} <: GradientParameters{T}
    F::FT
    ∇config::∇T

    function GradientParametersAD(F::FT, x::VT) where {T, FT, VT <: AbstractVector{T}}
        ∇config = ForwardDiff.GradientConfig(F, x)
        new{T, FT, typeof(∇config)}(F, ∇config)
    end
end

function GradientParametersAD{T}(F::FT, nx::Int) where {T, FT <: Callable}
    GradientParametersAD(F, zeros(T, nx))
end

function (grad::GradientParametersAD{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    ForwardDiff.gradient!(g, grad.F, x, grad.∇config)
end

function compute_gradient_ad!(g::AbstractVector{T}, x::AbstractVector{T}, F::Callable) where {T}
    grad = GradientParametersAD{T}(F, length(x))
    grad(g,x)
end


struct GradientParametersFD{T, FT} <: GradientParameters{T}
    ϵ::T
    F::FT
    e::Vector{T}
    tx::Vector{T}
end

function GradientParametersFD{T}(F::FT, nx::Int; ϵ=DEFAULT_GRADIENT_ϵ) where {T, FT <: Callable}
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    GradientParametersFD{T,FT}(ϵ, F, e, tx)
end

function (grad::GradientParametersFD{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    local ϵⱼ::T

    for j in eachindex(x,g)
        ϵⱼ = grad.ϵ * x[j] + grad.ϵ
        fill!(grad.e, 0)
        grad.e[j] = 1
        grad.tx .= x .- ϵⱼ .* grad.e
        f1 = grad.F(grad.tx)
        grad.tx .= x .+ ϵⱼ .* grad.e
        f2 = grad.F(grad.tx)
        g[j] = (f2 - f1)/(2ϵⱼ)
    end
end

function compute_gradient_fd!(g::AbstractVector{T}, x::AbstractVector{T}, F::FT; kwargs...) where {T, FT <: Callable}
    grad = GradientParametersFD{T}(F, length(x); kwargs...)
    grad(g,x)
end


function GradientParameters{T}(ForG::Callable, nx::Int; mode = :autodiff, diff_type = :forward, kwargs...) where {T}
    if mode == :autodiff
        if diff_type == :forward
            Gparams = GradientParametersAD{T}(ForG, nx)
        else
            Gparams = GradientParametersFD{T}(ForG, nx; kwargs...)
        end
    else
        Gparams = GradientParametersUser{T, typeof(ForG)}(ForG)
    end
    return Gparams
end

GradientParameters{T}(∇F!::Callable, F, nx::Int; kwargs...) where {T} = GradientParameters{T}(∇F!, nx; mode = :user, kwargs...)

GradientParameters{T}(∇F!::Nothing, F, nx::Int; kwargs...) where {T} = GradientParameters{T}(F, nx;  mode = :autodiff, kwargs...)

GradientParameters(∇F!, F, x::Vector{T}; kwargs...) where {T} = GradientParameters{T}(∇F!, F, length(x); kwargs...)

GradientParameters(F, x::Vector{T}; kwargs...) where {T} = GradientParameters{T}(F, length(x); kwargs...)

function compute_gradient!(g::Vector{T}, x::Vector{T}, ForG::Callable; kwargs...) where {T}
    grad = GradientParameters{T}(ForG, length(x); kwargs...)
    grad(g,x)
end
