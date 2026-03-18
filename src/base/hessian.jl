"""
    Hessian

Abstract type. `struct`s derived from this need an associated functor that computes the Hessian of a function (in-place).

Also see [`Gradient`](@ref).

# Implementation

When a custom `Hessian` is implemented, a functor is needed:

```julia
function (hessian::Hessian)(h::AbstractMatrix, x::AbstractVector) end
```

# Examples

Examples include:
- [`HessianFunction`](@ref)
- [`HessianAutodiff`](@ref)
- [`HessianBFGS`](@ref)
- [`HessianDFP`](@ref)

!!! info
    This also includes approximate Hessians such as [`HessianBFGS`](@ref) and [`HessianDFP`](@ref).
"""
abstract type Hessian{T} end

"""
    check_hessian(H)

Check the condition number, determinant, max and min value of the Hessian `H`.

!!! info
    Here the Hessian `H` is a matrix and not of type [`Hessian`](@ref).

```jldoctest; setup = :(using SimpleSolvers)
julia> H = [1. √2.; √2. 3.];

julia> SimpleSolvers.check_hessian(H)
Condition Number of Hessian: 13.9282
Determinant of Hessian:      1.0
minimum(|Hessian|):          1.0
maximum(|Hessian|):          3.0
```
"""
function check_hessian(H::AbstractMatrix; digits::Integer=5)
    println("Condition Number of Hessian: ", round(cond(H); digits=digits))
    println("Determinant of Hessian:      ", round(det(H); digits=digits))
    println("minimum(|Hessian|):          ", round(minimum(abs.(H)); digits=digits))
    println("maximum(|Hessian|):          ", round(maximum(abs.(H)); digits=digits))
    println()
end

"""
    HessianFunction <: Hessian

A `struct` that realizes a [`Hessian`](@ref) by explicitly supplying a function.

# Keys

The `struct` stores:
- `H!`: a function that can be applied in place.

# Functor

The functor does:

```julia
hes(H, x) = hes.H!(H, x)
```
"""
struct HessianFunction{T,HT<:Callable} <: Hessian{T}
    H!::HT
end

HessianFunction(H!::HT, ::AbstractVector{T}) where {T,HT} = HessianFunction{T,HT}(H!)

HessianFunction{T}(H!, n::Integer) where {T} = HessianFunction(H!, zeros(T, n))

function (hes::HessianFunction{T})(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    hes.H!(H, x)
end

"""
    HessianAutodiff <: Hessian

A `struct` that realizes [`Hessian`](@ref) by using `ForwardDiff`.

# Keys

The `struct` stores:
- `F`: a function that has to be differentiated.
- `Hconfig`: result of applying `ForwardDiff.HessianConfig`.

# Constructors

```julia
HessianAutodiff{T}(F, Hconfig)
HessianAutodiff(F, x::AbstractVector)
HessianAutodiff{T}(F, nx::Integer)
```

# Functor

The functor does:

```julia
hes(g, x) = ForwardDiff.hessian!(hes.H, hes.F, x, grad.Hconfig)
```
"""
struct HessianAutodiff{T,FT<:Callable,CT<:ForwardDiff.HessianConfig} <: Hessian{T}
    F::FT
    Hconfig::CT

    function HessianAutodiff{T}(F::FT, Hconfig::CT) where {T,FT,CT}
        new{T,FT,CT}(F, Hconfig)
    end
end

function HessianAutodiff(F::Callable, x::AbstractVector{T}) where {T}
    Hconfig = ForwardDiff.HessianConfig(F, x)
    HessianAutodiff{T}(F, Hconfig)
end

HessianAutodiff(F::OptimizerProblem, x) = HessianAutodiff(F.F, x)

Hessian(::Newton, ForOBJ::Union{Callable,OptimizerProblem}, x::AbstractVector) = HessianAutodiff(ForOBJ, x)

HessianAutodiff{T}(F, nx::Int) where {T} = HessianAutodiff(F, zeros(T, nx))

function (hes::HessianAutodiff{T})(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    ForwardDiff.hessian!(H, hes.F, x, hes.Hconfig)
end

function (hes::HessianAutodiff{T})(x::AbstractVector{T}) where {T}
    H = alloc_h(x)
    ForwardDiff.hessian!(H, hes.F, x, hes.Hconfig)
    H
end
