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
- [`GradientFunction`](@ref)
- [`GradientAutodiff`](@ref)
- [`GradientFiniteDifferences`](@ref)
"""
abstract type Hessian{T} end

initialize!(::Hessian) = nothing

"""
    compute_hessian!(h, x, hessian)

Compute the Hessian and store it in `h`.
"""
compute_hessian!(h::AbstractMatrix, x::AbstractVector, hessian::Hessian) = hessian(h,x)

"""
    compute_hessian(x, hessian)

Compute the Hessian at point `x` and return the result.

Internally this calls [`compute_hessian!`](@ref).
"""
function compute_hessian(x, hessian::Hessian)
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

# function print_hessian(H::AbstractMatrix)
#     display(H)
#     println()
# end

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
struct HessianFunction{T, HT} <: Hessian{T}
    H!::HT
end

HessianFunction(H!::HT, ::AbstractVector{T}) where {T,HT} = HessianFunction{T,HT}(H!)

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
struct HessianAutodiff{T, FT, HT <: AbstractMatrix, CT <: ForwardDiff.HessianConfig} <: Hessian{T}
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

HessianAutodiff(F::MultivariateObjective, x) = HessianAutodiff(F.F, x)

HessianAutodiff{T}(F, nx::Int) where {T} = HessianAutodiff{T}(F, zeros(T, nx))

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
initialize!(H::HessianAutodiff, x) = H(x)

"""
    update!(H, x)

Update a [`HessianAutodiff`](@ref) object `H`.

This is identical to [`initialize!`](@ref).

# Implementation

Internally this is calling the [`HessianAutodiff`](@ref) functor and therefore also `ForwardDiff.hessian!`.
"""
update!(H::HessianAutodiff, x::AbstractVector) = H(x)

Base.inv(H::HessianAutodiff) = inv(H.H)

Base.:\(H::HessianAutodiff, b) = H.H \ b

LinearAlgebra.ldiv!(x, H::HessianAutodiff, b) = x .= H \ b
# LinearAlgebra.ldiv!(x, H::HessianAD, b) = LinearAlgebra.ldiv!(x, H.H, b)
# TODO: Make this work!

function Hessian(ForH, x::AbstractVector{T}; mode = :autodiff, kwargs...) where {T}
    if mode == :autodiff
        HessianAutodiff(ForH, x)
    elseif mode == :function
        HessianFunction(ForH, x)
    elseif mode == :BFGS
        HessianBFGS(ForH, x)
    else
        error("Hessian for mode $(mode) not defined!")
    end
end

Hessian(H!, F, x::AbstractVector; kwargs...) = Hessian(H!, nx; mode = :user, kwargs...)

Hessian(H!::Nothing, F, x::AbstractVector; kwargs...) = Hessian(F, nx;  mode = :autodiff, kwargs...)

Hessian{T}(ForH, nx::Int; kwargs...) where {T} = Hessian(ForH, zeros(T, nx); kwargs...)

Hessian{T}(H!, F, nx::Int; kwargs...) where {T} = Hessian(H!, nx; kwargs...)

Hessian{T}(H!::Nothing, F, nx::Int; kwargs...) where {T} = Hessian(F, nx; kwargs...)

"""
    compute_hessian!(h, x, ForH)

Compute the hessian of function `ForH` at `x` and store it in `h`.

# Implementation

Internally this allocates a [`Hessian`](@ref) object.
"""
function compute_hessian!(H::AbstractMatrix, x::AbstractVector, ForH; kwargs...)
    hessian = Hessian(ForH, x; kwargs...)
    hessian(H, x)
end

"""
    compute_hessian_ad!(g, x, F)

Build a [`HessianAutodiff`](@ref) object based on `F` and apply it to `x`. The result is stored in `H`.

Also see [`gradient_ad!`](@ref) for the [`Gradient`](@ref) version.

# Implementation

This is using [`compute_hessian!`](@ref) with the keyword `mode` set to `autodiff`.
"""
function compute_hessian_ad!(H::AbstractMatrix{T}, x::AbstractVector{T}, F::FT; kwargs...) where {T, FT}
    compute_hessian!(H, x, F; mode = :autodiff, kwargs...)
end