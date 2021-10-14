
using Base: Callable

struct NoLineSearch <: LineSearch end

NoLineSearch(::Callable, ::AbstractArray) = NoLineSearch()

function (ls::NoLineSearch)(x::AbstractArray{T}, δx::AbstractArray{T}, x₀::AbstractArray{T}, y₀::AbstractArray{T}, g₀::AbstractArray{T}) where {T}
    x .= x₀ .+ δx
end

function solve!(x, δx, x₀, y₀, g₀, ls::NoLineSearch)
    ls(x, δx, x₀, y₀, g₀)
end
