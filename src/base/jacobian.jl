"""
    DEFAULT_JACOBIAN_ϵ

A constant used for computing the finite difference Jacobian.
"""
const DEFAULT_JACOBIAN_ϵ = 8sqrt(eps())

"""
    Jacobian

Abstract type. `struct`s that are derived from this need an associated functor that computes the Jacobian of a function (in-place).

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
    compute_jacobian!(j, x, jacobian::Jacobian, params)

Apply the [`Jacobian`](@ref) and store the result in `j`.
"""
compute_jacobian!(j::AbstractMatrix{T}, x::AbstractVector{T}, jacobian::Jacobian{T}, params) where {T} = jacobian(j,x, params)

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

    function JacobianFunction{T}(DF!::Callable) where T
        j = zeros(T, 3, 3)
        x = zeros(T, 3)
        applicable(DF!, j, x, NullParameters()) || error("The function needs to have the following signature: DF!(j, x, params).")
        new{T, typeof(DF!)}(DF!)
    end
end

function JacobianFunction(DF!::Callable, ::AbstractArray{T}) where {T}
    JacobianFunction{T}(DF!)
end

function (jac::JacobianFunction{T})(j::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
    jac.DF!(j, x, params)
end

"""
    JacobianAutodiff <: Jacobian

A `struct` that realizes [`Jacobian`](@ref) by using `ForwardDiff`.

# Keys

The `struct` stores:
- `F`: a function that has to be differentiated.
- `Jconfig`: result of applying `ForwardDiff.JacobianConfig`.
- `ty`: vector that is used for evaluating `ForwardDiff.jacobian!`

# Constructors

```julia
JacobianAutodiff(F, y::AbstractVector)
JacobianAutodiff(F, nx::Integer)
```

# Functor

The functor does:

```julia
jac(J, x) = ForwardDiff.jacobian!(J, jac.ty, x, grad.Jconfig)
```
"""
struct JacobianAutodiff{T, FT <: Callable, JT <: ForwardDiff.JacobianConfig, YT <: AbstractVector{T}} <: Jacobian{T}
    F::FT
    Jconfig::JT
    ty::YT

    function JacobianAutodiff(F::CT, x::YT, y::YT) where {T, YT <: AbstractArray{T}, CT <: Callable}
        applicable(F, y, x, nothing) || error("The function needs to have the following signature: F(y, x, params).")

        Jconfig = ForwardDiff.JacobianConfig(nothing, y, x)
        new{T, typeof(F), typeof(Jconfig), YT}(F, Jconfig, y)
    end
end

function JacobianAutodiff{T}(F::Callable, nx::Integer, ny::Integer) where {T}
    tx = zeros(T, nx)
    ty = zeros(T, ny)
    JacobianAutodiff(F, tx, ty)
end

JacobianAutodiff{T}(F, n::Integer) where {T} = JacobianAutodiff{T}(F, n, n)

function (jac::JacobianAutodiff{T})(J::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
    F!(j, x) = jac.F(j, x, params)
    F_closure(y, x) = jac.F(y, x, params)
    ForwardDiff.jacobian!(J, F_closure, jac.ty, x, jac.Jconfig)
end

@doc raw"""
    JacobianFiniteDifferences <: Jacobian

A `struct` that realizes [`Jacobian`](@ref) by using finite differences.

# Keys
    
The `struct` stores:
- `F`: a function that has to be differentiated.
- `ϵ`: small constant on whose basis the finite differences are computed.
- `f1`:
- `f2`:
- `e1`: auxiliary vector used for computing finite differences. It's of the form ``e_1 = \begin{bmatrix} 1 & 0 & \cdots & 0 \end{bmatrix}``.
- `e2`:
- `tx`: auxiliary vector used for computing finite differences. It stores the offset in the `x` vector.

# Constructor(s)
    
```julia
JacobianFiniteDifferences{T}(F, nx::Integer, ny::Integer; ϵ)
```

By default for `ϵ` is [`DEFAULT_JACOBIAN_ϵ`](@ref).

# Functor
    
The functor does:
    
```julia
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
```
"""
struct JacobianFiniteDifferences{T, FT <: Callable} <: Jacobian{T}
    F::FT
    ϵ::T
    f1::Vector{T}
    f2::Vector{T}
    e::Vector{T}
    tx::Vector{T}
end

function JacobianFiniteDifferences{T}(F::Callable, nx::Integer, ny::Integer; ϵ=DEFAULT_JACOBIAN_ϵ) where {T}
    f1 = zeros(T, ny)
    f2 = zeros(T, ny)
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    JacobianFiniteDifferences{T, typeof(F)}(F, ϵ, f1, f2, e, tx)
end

JacobianFiniteDifferences{T}(n; kwargs...) where {T} = JacobianFiniteDifferences{T}(n, n; kwargs...)

function (jac::JacobianFiniteDifferences{T})(J::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
    local ϵⱼ::T

    for j in eachindex(x)
        ϵⱼ = jac.ϵ * x[j] + jac.ϵ
        fill!(jac.e, 0)
        jac.e[j] = 1
        jac.tx .= x .- ϵⱼ .* jac.e
        jac.F(jac.f1, jac.tx, params)
        jac.tx .= x .+ ϵⱼ .* jac.e
        jac.F(jac.f2, jac.tx, params)
        for i in eachindex(x)
            J[i,j] = (jac.f2[i] - jac.f1[i]) / (2ϵⱼ)
        end
    end
    J
end

function Jacobian{T}(F::Callable, nx::Integer, ny::Integer; mode = :autodiff, kwargs...) where {T}
    if mode == :autodiff
        return JacobianAutodiff{T}(F, nx, ny)
    elseif mode == :finite
        return JacobianFiniteDifferences{T}(F, nx, ny; kwargs...)
    else
        return JacobianFunction{T}(F)
    end
end

Jacobian{T}(F::Callable, n::Integer; kwargs...) where {T} = Jacobian{T}(F, n, n; kwargs...)

Jacobian(F::Callable, x::AbstractVector{T}; kwargs...) where {T} = Jacobian{T}(F, length(x); kwargs...)
Jacobian(F::Callable, x::AbstractVector{T}, y::AbstractVector{T}; kwargs...) where {T} = Jacobian{T}(F, length(x), length(y); kwargs...)

"""
    compute_jacobian!(j, x, ForJ, params)

Allocate a [`Jacobian`](@ref) object, apply it to `x`, and store the result in `j`.
"""
function compute_jacobian!(j::AbstractMatrix{T}, x::AbstractVector{T}, ForJ::Callable, params; mode = :autodiff, kwargs...) where {T}
    jacobian = Jacobian{T}(ForJ, size(j,1), size(j,2); mode = mode, kwargs...)
    compute_jacobian!(j, x, jacobian, params)
end