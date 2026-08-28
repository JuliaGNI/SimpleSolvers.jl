@doc raw"""
    default_ϵ(::Type{T})

The default step size on whose basis finite differences are computed, for the
working precision `T`.  Used by [`GradientFiniteDifferences`](@ref) and
[`JacobianFiniteDifferences`](@ref).

Its value is ``8\sqrt{\varepsilon_T}``, where ``\varepsilon_T`` is the machine
epsilon of `T`.  Being precision-aware (`eps(T)`, not a baked-in `Float64`
epsilon) is essential for `Float32` finite differences to be accurate.

# Examples

```jldoctest; setup = :(using SimpleSolvers: default_ϵ)
julia> default_ϵ(Float64)
1.1920928955078125e-7
```

```jldoctest; setup = :(using SimpleSolvers: default_ϵ)
julia> default_ϵ(Float32)
0.0027621358f0
```
"""
default_ϵ(::Type{T}) where {T<:Number} = 8sqrt(eps(T))

"""
    Gradient

Abstract type. `struct`s that are derived from this need an associated functor that computes the gradient of a function (in-place).

# Examples

Examples include:
- [`GradientFunction`](@ref)
- [`GradientAutodiff`](@ref)
- [`GradientFiniteDifferences`](@ref)
"""
abstract type Gradient{T} end

function (grad::Gradient{T})(x::AbstractVector{T}) where {T}
    g = alloc_g(x)
    grad(g, x)
    g
end

"""
    check_gradient([io], g)

Check norm, maximum value and minimum value of a vector.

Output is written to `io` (defaulting to `stdout`).

# Examples

```jldoctest; setup = :(using SimpleSolvers)
julia> g = [1., 1., 1., 2., 0.9, 3.];

julia> SimpleSolvers.check_gradient(g; digits=3)
norm(Gradient):               4.1
minimum(|Gradient|):          0.9
maximum(|Gradient|):          3.0
```
"""
function check_gradient(io::IO, g::AbstractVector; digits::Integer=5)
    println(io, "norm(Gradient):               ", round(norm(g); digits=digits))
    println(io, "minimum(|Gradient|):          ", round(minimum(abs.(g)); digits=digits))
    println(io, "maximum(|Gradient|):          ", round(maximum(abs.(g)); digits=digits))
    println(io)
end

check_gradient(g::AbstractVector; kwargs...) = check_gradient(stdout, g; kwargs...)

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
- `F`: a function that has to be differentiated.
- `∇F!`: a function that can be applied in place.

# Functor

The functor does:

```julia
grad(g, x) = grad.∇F!(g, x)
```
"""
struct GradientFunction{T,FT<:Callable,GT<:Callable} <: Gradient{T}
    F::FT
    ∇F!::GT
end

function GradientFunction(::Callable, ::AbstractArray)
    error("`GradientFunction` can only be called by providing a `Callable` and an `AbstractArray`.")
end

function GradientFunction{T}(F::TF, ∇F!::TG, ::Integer) where {T,TF<:Callable,TG<:Callable}
    GradientFunction{T,TF,TG}(F, ∇F!)
end

function GradientFunction(F::Callable, ∇F!::Callable, x::AbstractVector{T}) where {T}
    GradientFunction{T}(F, ∇F!, length(x))
end

(grad::GradientFunction{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T} = grad.∇F!(g, x)

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
struct GradientAutodiff{T,FT,∇T<:ForwardDiff.GradientConfig} <: Gradient{T}
    F::FT
    ∇config::∇T

    function GradientAutodiff(F::FT, x::VT) where {T<:Number,FT<:Callable,VT<:AbstractVector{T}}
        ∇config = ForwardDiff.GradientConfig(F, x)
        new{T,FT,typeof(∇config)}(F, ∇config)
    end
end

function GradientAutodiff{T}(F::Callable, nx::Integer) where {T<:Number}
    GradientAutodiff(F, zeros(T, nx))
end

"""
    GradientAutodiff(F, x::AbstractMatrix)

The gradient of `F` at a matrix-shaped argument.

`ForwardDiff` differentiates with respect to a vector, so `x` is flattened here and `F` is composed
with the `reshape` that puts a candidate back into the shape it was written for. Sizing the gradient
with `length(x)` alone is not enough: `F` is called on the flat vector during differentiation and has
to see a matrix.

`AbstractMatrix` and not `Matrix`, so that a package with its own matrix-like iterate can reach this
by falling through -- and can take precedence over it with a method on that type, which is what
`GeometricOptimizers` does for a `Manifold`, where the reconstruction is the manifold's constructor
rather than a `reshape`. This method carried the `Matrix` case there until 0.6.1, where it was type
piracy: neither this function nor `Matrix` was that package's.
"""
GradientAutodiff(F, x::AbstractMatrix) = GradientAutodiff(_x -> F(reshape(_x, size(x)...)), vec(x))

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
- `e`: auxiliary vector used for computing finite differences. It's of the form ``e_1 = \begin{bmatrix} 1 & 0 & \cdots & 0 \end{bmatrix}^T``.
- `tx`: auxiliary vector used for computing finite differences. It stores the offset in the `x` vector.

# Constructor(s)

```julia
GradientFiniteDifferences{T}(F, nx::Integer; ϵ)
```

By default for `ϵ` is [`default_ϵ`](@ref)`(T)`.

# Functor

The functor does (for `grad(g, x)`):

```julia
for j in eachindex(x,g)
    ϵⱼ = grad.ϵ * abs(x[j]) + grad.ϵ
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
struct GradientFiniteDifferences{T,FT<:Callable} <: Gradient{T}
    F::FT
    ϵ::T
    e::Vector{T}
    tx::Vector{T}
end

function GradientFiniteDifferences{T}(F::FT, nx::Integer; ϵ=default_ϵ(T)) where {T,FT}
    e = zeros(T, nx)
    tx = zeros(T, nx)
    GradientFiniteDifferences{T,FT}(F, ϵ, e, tx)
end

function (grad::GradientFiniteDifferences{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    local ϵⱼ::T

    for j in eachindex(x, g)
        ϵⱼ = grad.ϵ * abs(x[j]) + grad.ϵ
        fill!(grad.e, zero(T))
        grad.e[j] = one(T)
        grad.tx .= x .- ϵⱼ .* grad.e
        f1 = grad.F(grad.tx)
        grad.tx .= x .+ ϵⱼ .* grad.e
        f2 = grad.F(grad.tx)
        g[j] = (f2 - f1) / (2ϵⱼ)
    end
end
