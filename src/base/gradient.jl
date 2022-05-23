
const DEFAULT_GRADIENT_ϵ = 8sqrt(eps())


abstract type Gradient{T} end

compute_gradient!(g::AbstractVector, x::AbstractVector, grad::Gradient) = grad(g,x)

function compute_gradient(x, grad::Gradient)
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


struct GradientFunction{T, ∇T} <: Gradient{T}
    ∇F!::∇T
end

function (grad::GradientFunction{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    grad.∇F!(g, x)
end


struct GradientAutodiff{T, FT, ∇T <: ForwardDiff.GradientConfig} <: Gradient{T}
    F::FT
    ∇config::∇T

    function GradientAutodiff(F::FT, x::VT) where {T, FT, VT <: AbstractVector{T}}
        ∇config = ForwardDiff.GradientConfig(F, x)
        new{T, FT, typeof(∇config)}(F, ∇config)
    end
end

function GradientAutodiff{T}(F::FT, nx::Int) where {T, FT}
    GradientAutodiff(F, zeros(T, nx))
end

function (grad::GradientAutodiff{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    ForwardDiff.gradient!(g, grad.F, x, grad.∇config)
end

function compute_gradient_ad!(g::AbstractVector{T}, x::AbstractVector{T}, F) where {T}
    grad = GradientAutodiff{T}(F, length(x))
    grad(g,x)
end


struct GradientFiniteDifferences{T, FT} <: Gradient{T}
    ϵ::T
    F::FT
    e::Vector{T}
    tx::Vector{T}
end

function GradientFiniteDifferences{T}(F::FT, nx::Int; ϵ=DEFAULT_GRADIENT_ϵ) where {T, FT}
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    GradientFiniteDifferences{T,FT}(ϵ, F, e, tx)
end

function (grad::GradientFiniteDifferences{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
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

function compute_gradient_fd!(g::AbstractVector{T}, x::AbstractVector{T}, F::FT; kwargs...) where {T, FT}
    grad = GradientFiniteDifferences{T}(F, length(x); kwargs...)
    grad(g,x)
end


function Gradient{T}(ForG, nx::Int; mode = :autodiff, diff_type = :forward, kwargs...) where {T}
    if mode == :autodiff
        if diff_type == :forward
            Gparams = GradientAutodiff{T}(ForG, nx)
        else
            Gparams = GradientFiniteDifferences{T}(ForG, nx; kwargs...)
        end
    else
        Gparams = GradientFunction{T, typeof(ForG)}(ForG)
    end
    return Gparams
end

Gradient{T}(∇F!, F, nx::Int; kwargs...) where {T} = Gradient{T}(∇F!, nx; mode = :user, kwargs...)

Gradient{T}(∇F!::Nothing, F, nx::Int; kwargs...) where {T} = Gradient{T}(F, nx;  mode = :autodiff, kwargs...)

Gradient(∇F!, F, x::AbstractVector{T}; kwargs...) where {T} = Gradient{T}(∇F!, F, length(x); kwargs...)

Gradient(F, x::AbstractVector{T}; kwargs...) where {T} = Gradient{T}(F, length(x); kwargs...)

function compute_gradient!(g::AbstractVector{T}, x::AbstractVector{T}, ForG; kwargs...) where {T}
    grad = Gradient{T}(ForG, length(x); kwargs...)
    grad(g,x)
end
