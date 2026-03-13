"""
    DEFAULT_GRADIENT_ϵ

A constant on whose basis finite differences are computed. See [`GradientFiniteDifferences`](@ref).

# Extended help

For the [`JacobianFiniteDifferences`](@ref) this is called [`DEFAULT_JACOBIAN_ϵ`](@ref).
"""
const DEFAULT_GRADIENT_ϵ = 8sqrt(eps())

"""
    Gradient

Abstract type. `struct`s that are derived from this need an associated functor that computes the gradient of a function (in-place).

# Implementation

When a custom `Gradient` is implemented, a functor is needed:

```julia
(grad::Gradient)(g::AbstractVector, x::AbstractVector)
```

There is also an out-of place version for convenience:

```julia
(grad::Gradient)(x::AbstractVector)
```

This is using [`alloc_g`](@ref) to allocate the array `g` for the gradient.

# Examples

Examples include:
- [`GradientFunction`](@ref)
- [`GradientAutodiff`](@ref)
- [`GradientFiniteDifferences`](@ref)
"""
abstract type Gradient{T} end

function (::Gradient{T₁})(::AbstractVector{T₂}, ::AbstractVector{T₃}) where {T₁, T₂, T₃}
    (T₁ == T₂ == T₃) ? error("Functor not implemented.") : error("Types $(T₁), $(T₂), $(T₃) in Gradient functor must be the same.")
end

function (grad::Gradient{T})(x::AbstractVector{T}) where {T}
    g = alloc_g(x)
    grad(g, x)
    g
end

"""
    check_gradient(g)

Check norm, maximum value and minimum value of a vector.

# Examples

```jldoctest
using SimpleSolvers

g = [1., 1., 1., 2., 0.9, 3.]
SimpleSolvers.check_gradient(g; digits=3)

# output

norm(Gradient):               4.1
minimum(|Gradient|):          0.9
maximum(|Gradient|):          3.0
```
"""
function check_gradient(g::AbstractVector; digits::Integer = 5)
    println("norm(Gradient):               ", round(norm(g); digits=digits))
    println("minimum(|Gradient|):          ", round(minimum(abs.(g)); digits=digits))
    println("maximum(|Gradient|):          ", round(maximum(abs.(g)); digits=digits))
    println()
end

# do we need this?
# function print_gradient(g::AbstractVector)
#     display(g)
#     println()
# end

"""
    GradientFunction <: Gradient

A `struct` that realizes a [`Gradient`](@ref) by explicitly supplying a function.

# Keys

The `struct` stores:
- `∇F!`: a function that can be applied in place.

# Functor

The functor does:

```julia
grad(g, x) = grad.∇F!(g, x)
```
"""
struct GradientFunction{T, FT<:Callable, GT<:Callable} <: Gradient{T}
    F::FT
    ∇F!::GT
end

function GradientFunction(::Callable, ::AbstractArray)
    error("`GradientFunction` can only be called by providing two `Callable`s or an `OptimizerProblem`.")
end

function GradientFunction{T}(F::TF, ∇F!::TG, ::Integer) where {T, TF <: Callable, TG <: Callable}
    GradientFunction{T, TF, TG}(F, ∇F!)
end

function GradientFunction(F::Callable, ∇F!::Callable, x::AbstractVector{T}) where {T}
    GradientFunction{T}(F, ∇F!, length(x))
end

(grad::GradientFunction{T})(g::VT, x::VT) where {T, VT <: AbstractVector{T}} = grad.∇F!(g, x)

"""
    GradientAutodiff <: Gradient

A `struct` that realizes [`Gradient`](@ref) by using `ForwardDiff`.

# Keys

The `struct` stores:
- `F`: a function that has to be differentiated.
- `∇config`: result of applying `ForwardDiff.GradientConfig`.

# Constructors

```julia
GradientAutodiff(F, x::AbstractVector)
GradientAutodiff{T}(F, nx::Integer)
```

# Functor

The functor does:

```julia
grad(g, x) = ForwardDiff.gradient!(g, grad.F, x, grad.∇config)
```
"""
struct GradientAutodiff{T, FT, ∇T <: ForwardDiff.GradientConfig} <: Gradient{T}
    F::FT
    ∇config::∇T

    function GradientAutodiff(F::FT, x::VT) where {T <: Number, FT <: Callable, VT <: AbstractVector{T}}
        ∇config = ForwardDiff.GradientConfig(F, x)
        new{T, FT, typeof(∇config)}(F, ∇config)
    end
end

function GradientAutodiff{T}(F::Callable, nx::Integer) where {T <: Number}
    GradientAutodiff(F, zeros(T, nx))
end

function (grad::GradientAutodiff{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    ForwardDiff.gradient!(g, grad.F, x, grad.∇config)
end

@doc raw"""
    GradientFiniteDifferences <: Gradient

A `struct` that realizes [`Gradient`](@ref) by using finite differences.

# Keys

The `struct` stores:
- `F`: a function that has to be differentiated.
- `ϵ`: small constant on whose basis the finite differences are computed.
- `e`: auxiliary vector used for computing finite differences. It's of the form ``e_1 = \begin{bmatrix} 1 & 0 & \cdots & 0 \end{bmatrix}``.
- `tx`: auxiliary vector used for computing finite differences. It stores the offset in the `x` vector.

# Constructor(s)

```julia
GradientFiniteDifferences{T}(F, nx::Integer; ϵ)
```

By default for `ϵ` is [`DEFAULT_GRADIENT_ϵ`](@ref).

# Functor

The functor does:

```julia
for j in eachindex(x,g)
    ϵⱼ = grad.ϵ * x[j] + grad.ϵ
    fill!(grad.e, 0)
    grad.e[j] = 1
    grad.tx .= x .- ϵⱼ .* grad.e
    f1 = grad.F(grad.tx)
    grad.tx .= x .+ ϵⱼ .* grad.e
    f2 = grad.F(grad.tx)
    g[j] = (f2 - f1) / (2ϵⱼ)
end
```
"""
struct GradientFiniteDifferences{T, FT <: Callable} <: Gradient{T}
    F::FT
    ϵ::T
    e::Vector{T}
    tx::Vector{T}
end

function GradientFiniteDifferences{T}(F::FT, nx::Int; ϵ=DEFAULT_GRADIENT_ϵ) where {T, FT}
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    GradientFiniteDifferences{T,FT}(F, ϵ, e, tx)
end

function (grad::GradientFiniteDifferences{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    local ϵⱼ::T

    for j in eachindex(x,g)
        ϵⱼ = grad.ϵ * x[j] + grad.ϵ
        fill!(grad.e, zero(T))
        grad.e[j] = one(T)
        grad.tx .= x .- ϵⱼ .* grad.e
        f1 = grad.F(grad.tx)
        grad.tx .= x .+ ϵⱼ .* grad.e
        f2 = grad.F(grad.tx)
        g[j] = (f2 - f1) / (2ϵⱼ)
    end
end
