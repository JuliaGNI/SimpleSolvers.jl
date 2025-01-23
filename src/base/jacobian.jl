"""
    DEFAULT_JACOBIAN_ϵ

A constant used for computing the finite difference Jacobian.
"""
const DEFAULT_JACOBIAN_ϵ = 8sqrt(eps())

"""
    Jacobian

Abstract type. `strcut`s that are derived from this need an assoicated functor that computes the Jacobian of a function (in-place).

# Implementation

When a custom `Jacobian` is implemented, a functor is needed:

```julia
function (j::Jacobian)(g::AbstractMatrix, x::AbstractVector) end
```
This functor can also be called with [`compute_jacobian!`](@ref).

# Examples

Examples include:
- [`JacobianFunction`](@ref)
- [`JacobianAutodiff`](@ref)
- [`JacobianFiniteDifferences`](@ref)
"""
abstract type Jacobian{T} end

"""
    compute_jacobian!(j, x, jacobian::Jacobian)

Apply the [`Jacobian`](@ref) and store the result in `j`.
"""
compute_jacobian!(j::AbstractMatrix, x::AbstractVector, jacobian::Jacobian) = jacobian(j,x)

"""
    check_jacobian(J)

Check the condition number, determinant, max and min value of the [`Jacobian`](@ref) `J`.

```jldoctest
using SimpleSolvers

J = [1. √2.; √2. 3.]
SimpleSolvers.check_jacobian(J)

# output

Condition Number of Jacobian: 13.9282
Determinant of Jacobian:      1.0
minimum(|Jacobian|):          1.0
maximum(|Jacobian|):          3.0
```
"""
function check_jacobian(J::AbstractMatrix; digits = 5)
    println("Condition Number of Jacobian: ", round(cond(J); digits=digits))
    println("Determinant of Jacobian:      ", round(det(J); digits=digits))
    println("minimum(|Jacobian|):          ", round(minimum(abs.(J)); digits=digits))
    println("maximum(|Jacobian|):          ", round(maximum(abs.(J)); digits=digits))
    println()
end

# function print_jacobian(J::AbstractMatrix)
#     display(J)
#     println()
# end

"""
    JacobianFunction <: Jacobian

A `struct` that realizes a [`Jacobian`](@ref) by explicitly supplying a function.

# Keys

The `struct` stores:
- `DF!`: a function that can be applied in place.

# Functor

The functor does:

```julia
jac(g, x) = jac.DF!(g, x)
```
"""
struct JacobianFunction{T, JT <: Callable} <: Jacobian{T}
    DF!::JT
end

JacobianFunction(DF!::Callable, ::AbstractArray{T}) where {T} = JacobianFunction{T, typeof(DF!)}(DF!)

function (jac::JacobianFunction{T})(j::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    jac.DF!(j, x)
end

"""
    JacobianAutodiff <: Jacobian

A `struct` that realizes [`Jacobian`](@ref) by using `ForwardDiff`.

# Keys

The `struct` stores:
- `F`: a function that has to be differentiated.
- `Jconfig`: result of applying `ForwardDiff.JacobianConfig`.

# Constructors

```julia
JacobianAutodiff(F, x::AbstractVector)
JacobianAutodiff(F, nx::Integer)
```

# Functor

The functor does:

```julia
jac(J, x) = ForwardDiff.jacobian!(J, jac.ty, x, grad.Jconfig)
```
"""
struct JacobianAutodiff{T, F <: Callable, JT <: ForwardDiff.JacobianConfig, YT <: AbstractVector{T}} <: Jacobian{T}
    F::FT
    Jconfig::JT
    ty::YT

    function JacobianAutodiff{T}(Jconfig::JT, y::YT) where {T, JT, YT}
        new{T, JT, YT}(Jconfig, y)
    end

end

function JacobianAutodiff(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    Jconfig = ForwardDiff.JacobianConfig(nothing, y, x)
    JacobianAutodiff{T}(Jconfig, zero(y))
end

function JacobianAutodiff{T}(nx::Int, ny::Int) where {T}
    tx = zeros(T, nx)
    ty = zeros(T, ny)
    JacobianAutodiff(tx, ty)
end

JacobianAutodiff{T}(n) where {T} = JacobianAutodiff{T}(n, n)

function (jac::JacobianAutodiff{T})(J::AbstractMatrix{T}, x::AbstractVector{T}, f::Callable) where {T}
    ForwardDiff.jacobian!(J, f, jac.ty, x, jac.Jconfig)
end


struct JacobianFiniteDifferences{T} <: Jacobian{T}
    ϵ::T
    f1::Vector{T}
    f2::Vector{T}
    e::Vector{T}
    tx::Vector{T}
end

function JacobianFiniteDifferences{T}(nx::Int, ny::Int; ϵ=DEFAULT_JACOBIAN_ϵ) where {T}
    f1 = zeros(T, ny)
    f2 = zeros(T, ny)
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    JacobianFiniteDifferences{T}(ϵ, f1, f2, e, tx)
end

JacobianFiniteDifferences{T}(n; kwargs...) where {T} = JacobianFiniteDifferences{T}(n, n; kwargs...)

function (jac::JacobianFiniteDifferences{T})(J::AbstractMatrix{T}, x::AbstractVector{T}, f::Callable) where {T}
    local ϵⱼ::T

    for j in eachindex(x)
        ϵⱼ = jac.ϵ * x[j] + jac.ϵ
        fill!(jac.e, 0)
        jac.e[j] = 1
        jac.tx .= x .- ϵⱼ .* jac.e
        f(jac.f1, jac.tx)
        jac.tx .= x .+ ϵⱼ .* jac.e
        f(jac.f2, jac.tx)
        for i in eachindex(x)
            J[i,j] = (jac.f2[i] - jac.f1[i]) / (2ϵⱼ)
        end
    end
end


function Jacobian{T}(nx::Int, ny::Int; mode = :autodiff, diff_type = :forward, kwargs...) where {T}
    if mode == :autodiff
        if diff_type == :forward
            Jparams = JacobianAutodiff{T}(nx, ny)
        else
            Jparams = JacobianFiniteDifferences{T}(nx, ny; kwargs...)
        end
    else
        Jparams = JacobianFunction{T}()
    end
    return Jparams
end

Jacobian{T}(n::Int; kwargs...) where {T} = Jacobian{T}(n, n; kwargs...)

Jacobian{T}(J!::Callable, nx, ny; kwargs...) where {T} = Jacobian{T}(nx, ny; mode = :user, kwargs...)

Jacobian{T}(J!::Union{Nothing,Missing}, nx, ny; kwargs...) where {T} = Jacobian{T}(nx, ny; mode = :autodiff, kwargs...)

Jacobian{T}(J!, n; kwargs...) where {T} = Jacobian{T}(J!, n, n; kwargs...)

function compute_jacobian!(j::AbstractMatrix{T}, x::AbstractVector{T}, ForJ::Callable; kwargs...) where {T}
    jacobian = Jacobian{T}(size(j,1), size(j,2); kwargs...)
    jacobian(j,x,ForJ)
end

function compute_jacobian!(j::AbstractMatrix, x::AbstractVector, ForJ::Callable, jacobian::Jacobian)
    jacobian(j,x,ForJ)
end
