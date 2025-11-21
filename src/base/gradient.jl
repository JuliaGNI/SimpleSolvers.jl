"""
    DEFAULT_GRADIENT_ϵ

A constant on whose basis finite differences are computed.
"""
const DEFAULT_GRADIENT_ϵ = 8sqrt(eps())

"""
    Gradient

Abstract type. `strcut`s that are derived from this need an assoicated functor that computes the gradient of a function (in-place).

# Implementation

When a custom `Gradient` is implemented, a functor is needed:

```julia
function (grad::Gradient)(g::AbstractVector, x::AbstractVector) end
```
This functor can also be called with [`gradient!`](@ref).

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

"""
    gradient!(g, grad, x)

Apply the [`Gradient`](@ref) `grad` to `x` and store the result in `g`.

# Implementation

This is equivalent to doing
```jldoctest; setup=:(using SimpleSolvers)
g₁ = zeros(3)
g₂ = zeros(3)
x = [1., 2., 3.]
F(x) = sum(x .^ 2.)
grad = GradientAutodiff(F, x)

grad(g₁, x); gradient!(g₂, grad, x);

g₁ ≈ g₂

# output

true
```
"""
gradient!(g::AbstractVector, grad::Gradient, x::AbstractVector) = grad(g,x)

"""
    compute_gradient!

Alias for [`gradient!`](@ref). Will probably be deprecated.
"""
const compute_gradient! = gradient!

"""
    gradient(x, grad)

Apply `grad` to `x` and return the result. 

# Implementation

Internally this is using [`gradient!`](@ref).
"""
function gradient(x, grad::Gradient)
    g = alloc_g(x)
    gradient!(g, grad, x)
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
struct GradientFunction{T} <: Gradient{T} end

function GradientFunction(::Callable, ::AbstractArray{T}) where {T}
    GradientFunction{T}()
end


function GradientFunction{T}(::Callable, ::Integer) where T
    GradientFunction{T}()
end

GradientFunction(::AbstractArray{T}) where {T} = GradientFunction{T}()

gradient!(::AbstractVector, ::GradientFunction, ::AbstractVector) = error("You have to provide an `OptimizerProblem` when using `GradientFunction`!")

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
GradientAutodiff(F, nx::Integer)
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

"""
    gradient_fd!(g, x, F)

Build a [`GradientFiniteDifferences`](@ref) object based on `F` and apply it to `x`. The result is stored in `g`.

Also see [`gradient_ad!`](@ref) for the autodiff version.
"""
function gradient_fd!(g::AbstractVector{T}, x::AbstractVector{T}, F::FT; kwargs...) where {T, FT}
    grad = GradientFiniteDifferences{T}(F, length(x); kwargs...)
    grad(g,x)
end

# TODO: remove this! (it's here for the moment to keep the tests as close to what they were before as possible)
function gradient!(g::AbstractVector{T}, x::AbstractVector{T}, ForG::Union{Callable, Missing}; mode = :autodiff) where {T}
    grad = if mode == :autodiff
        GradientAutodiff{T}(ForG, length(x))
    elseif mode == :finite
        GradientFiniteDifferences{T}(ForG, length(x))
    else
        GradientFunction{T}(ForG, length(x))
    end

    if typeof(grad) <: GradientFunction
        # use a dummy function to allocate an OptimizerProblem
        function dummyfunction(y, x) end
        obj = OptimizerProblem(dummyfunction, x; gradient = ForG)
        gradient!(obj, grad, x)
        gradient(obj)
    else
        gradient!(g, grad, x)
    end
end