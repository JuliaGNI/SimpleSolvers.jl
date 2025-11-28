"""
    Hessian

Abstract type. `struct`s derived from this need an associated functor that computes the Hessian of a function (in-place).

Also see [`Gradient`](@ref).

# Implementation

When a custom `Hessian` is implemented, a functor is needed:

```julia
function (hessian::Hessian)(h::AbstractMatrix, x::AbstractVector) end
```
This functor can also be called with [`compute_hessian!`](@ref).

# Examples

Examples include:
- [`HessianFunction`](@ref)
- [`HessianAutodiff`](@ref)
- [`HessianBFGS`](@ref)
- [`HessianDFP`](@ref)
"""
abstract type Hessian{T} end

"""
    initialize!(hessian, x)

See e.g. [`initialize!(::HessianAutodiff, ::AbstractVector)`](@ref).
"""
initialize!(hes::Hessian, ::AbstractVector) = error("initialize! not defined for Hessian of type $(typeof(hes)).")

"""
    compute_hessian!(h, x, hessian)

Compute the Hessian and store it in `h`.
"""
compute_hessian!(h::AbstractMatrix{T}, x::AbstractVector{T}, hessian::Hessian{T}) where {T <: Number} = hessian(h,x)

"""
    compute_hessian(x, hessian)

Compute the Hessian at point `x` and return the result.

Internally this calls [`compute_hessian!`](@ref).
"""
function compute_hessian(x::AbstractVector{T}, hessian::Hessian{T}) where {T <: Number}
    h = alloc_h(x)
    compute_hessian!(h, x, hessian)
    h
end

"""
    check_hessian(H)

Check the condition number, determinant, max and min value of the [`Hessian`](@ref) `H`.

```jldoctest
using SimpleSolvers

H = [1. √2.; √2. 3.]
SimpleSolvers.check_hessian(H)

# output

Condition Number of Hessian: 13.9282
Determinant of Hessian:      1.0
minimum(|Hessian|):          1.0
maximum(|Hessian|):          3.0
```
"""
function check_hessian(H::AbstractMatrix; digits::Integer = 5)
    println("Condition Number of Hessian: ", round(cond(H); digits=digits))
    println("Determinant of Hessian:      ", round(det(H); digits=digits))
    println("minimum(|Hessian|):          ", round(minimum(abs.(H)); digits=digits))
    println("maximum(|Hessian|):          ", round(maximum(abs.(H)); digits=digits))
    println()
end

"""
    update!(hessian, x)

Update the [`Hessian`](@ref) based on the vector `x`. For an explicit example see e.g. [`update!(::HessianAutodiff)`](@ref).
"""
update!(::HT, ::AbstractVector) where {HT <: Hessian} = error("update! not defined for $(HT).")

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
struct HessianFunction{T, HT <: Callable} <: Hessian{T}
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
- `H`: a matrix in which the (updated) [`Hessian`](@ref) is stored. 
- `Hconfig`: result of applying `ForwardDiff.HessianConfig`.

# Constructors

```julia
HessianAutodiff(F, x::AbstractVector)
HessianAutodiff(F, nx::Integer)
```

# Functor

The functor does:

```julia
hes(g, x) = ForwardDiff.hessian!(hes.H, hes.F, x, grad.Hconfig)
```
"""
struct HessianAutodiff{T, FT <: Callable, HT <: AbstractMatrix, CT <: ForwardDiff.HessianConfig} <: Hessian{T}
    F::FT
    H::HT
    Hconfig::CT

    function HessianAutodiff{T}(F::FT, H::HT, Hconfig::CT) where {T, FT, HT, CT}
        new{T, FT, HT, CT}(F, H, Hconfig)
    end
end

function HessianAutodiff(F::Callable, x::AbstractVector{T}) where {T}
    Hconfig = ForwardDiff.HessianConfig(F, x)
    HessianAutodiff{T}(F, alloc_h(x), Hconfig)
end

HessianAutodiff(F::OptimizerProblem, x) = HessianAutodiff(F.F, x)

Hessian(::Newton, ForOBJ::Union{Callable, OptimizerProblem}, x::AbstractVector) = HessianAutodiff(ForOBJ, x)

HessianAutodiff{T}(F, nx::Int) where {T} = HessianAutodiff(F, zeros(T, nx))

function (hes::HessianAutodiff{T})(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    ForwardDiff.hessian!(H, hes.F, x, hes.Hconfig)
end

function (hes::HessianAutodiff{T})(x::AbstractVector{T}) where {T}
    ForwardDiff.hessian!(hes.H, hes.F, x, hes.Hconfig)
end

"""
    initialize!(H, x)

Initialize a [`HessianAutodiff`](@ref) object `H`.

# Implementation

Internally this is calling the [`HessianAutodiff`](@ref) functor and therefore also `ForwardDiff.hessian!`.
"""
function initialize!(H::HessianAutodiff, x::AbstractVector)
    H.H .= alloc_h(x)
    H
end

"""
    update!(H, x)

Update a [`HessianAutodiff`](@ref) object `H`.

This is identical to [`initialize!`](@ref).

# Implementation

Internally this is calling the [`HessianAutodiff`](@ref) functor and therefore also `ForwardDiff.hessian!`.
"""
function update!(H::HessianAutodiff, x::AbstractVector)
    H(x)
    H
end

Base.inv(H::HessianAutodiff) = inv(H.H)

Base.:\(H::HessianAutodiff, b) = solve(LU(), H.H, b)

# TODO: replace the "\" with something that has better performance (and doesn't produce as many allocations)
LinearAlgebra.ldiv!(x, H::HessianAutodiff, b) = x .= H \ b

"""
    compute_hessian!(h, x, ForH)

Compute the hessian of function `ForH` at `x` and store it in `h`.

# Implementation

Internally this allocates a [`Hessian`](@ref) object.
"""
function compute_hessian!(H::AbstractMatrix{T}, x::AbstractVector{T}, ForH; mode = :autodiff) where {T<:Number}
    hessian = if mode == :autodiff
        HessianAutodiff(ForH, x)
    elseif mode == :finite
        HessianFiniteDifferences(ForH, x)
    else
        HessianFunction(ForH, x)
    end
    hessian(H, x)
end
