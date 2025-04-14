
macro define(name, definition)
    quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end


"""
    alloc_x(x)

Allocate `NaN`s of the size of `x`.
"""
alloc_x

"""
    alloc_f(x)

Allocate `NaN`s of the size the size of `f` (evaluated at `x`).
"""
alloc_f

"""
    alloc_d(x)

Allocate `NaN`s of the size of the derivative of `f` (with respect to `x`).

This is used in combination with a [`AbstractUnivariateObjective`](@ref).
"""
alloc_d

"""
    alloc_g(x)

Allocate `NaN`s of the size of the gradient of `f` (with respect to `x`).

This is used in combination with a [`MultivariateObjective`](@ref).
"""
alloc_g

"""
    alloc_h(x)

Allocate `NaN`s of the size of the Hessian of `f` (with respect to `x`).

This is used in combination with a [`MultivariateObjective`](@ref).
"""
alloc_h

"""
    alloc_j(x, f)

Allocate `NaN`s of the size of the Jacobian of `f` (with respect to `x`).

This is used in combination with a [`MultivariateObjective`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers: alloc_j)
x = rand(3)
fₓ = rand(2)
alloc_j(x, fₓ)

# output

2×3 Matrix{Float64}:
 NaN  NaN  NaN
 NaN  NaN  NaN
```
"""
alloc_j

alloc_x(x::Number) = typeof(x)(NaN)
alloc_f(x::Number) = real(typeof(x))(NaN)
alloc_d(x::Number) = typeof(x)(NaN)

alloc_x(x::AbstractArray{T}) where {T <: Number} = T(NaN) .* x
alloc_f(::AbstractArray{T}) where {T <: Number} = real(T)(NaN)
alloc_g(x::AbstractArray{T}) where {T <: Number} = T(NaN) .* x
alloc_h(x::AbstractArray{T}) where {T <: Number} = T(NaN) .* x*x'
alloc_j(x::AbstractArray{T}, f::AbstractArray{T}) where {T <: Number} = T(NaN) .* vec(f) .* vec(x)'

function L2norm(x)
    local l2::eltype(x) = 0
    for xᵢ in x
        l2 += xᵢ^2
    end
    l2
end

function l2norm(x)
    sqrt(L2norm(x))
end

function maxnorm(x)
    local r² = zero(eltype(x))
    @inbounds for xᵢ in x
        r² = max(r², xᵢ^2)
    end
    sqrt(r²)
end


function outer!(O, x, y)
    @assert axes(O,1) == axes(x,1)
    @assert axes(O,2) == axes(y,1)
    @inbounds @simd for i in axes(O, 1)
        for j in axes(O, 2)
            O[i,j] = x[i] * y[j]
        end
    end

end
