"""
    alloc_x(x)

Allocate `NaN`s of the size of `x`.
"""
alloc_x

"""
    alloc_g(x)

Allocate `NaN`s of the size of the gradient of `f` (with respect to `x`).
"""
alloc_g

"""
    alloc_h(x)

Allocate `NaN`s of the size of the Hessian of `f` (with respect to `x`).
"""
alloc_h

# The `alloc_*` helpers initialize their storage with `NaN`, which only exists for
# floating-point (and complex-of-floating-point) element types.  Passing an
# integer array would otherwise fail deep inside the broadcast with a cryptic
# `InexactError`; raise a clear error instead.
_nan(::Type{T}) where {T<:AbstractFloat} = T(NaN)
_nan(::Type{Complex{T}}) where {T<:AbstractFloat} = Complex{T}(T(NaN))
_nan(::Type{T}) where {T<:Number} = error("Cannot allocate NaN-initialized storage for element type $(T): only floating-point element types support NaN. Provide a floating-point input.")

alloc_x(x::Number) = _nan(typeof(x))

alloc_x(x::AbstractArray{T}) where {T<:Number} = _nan(T) .* x

alloc_g(x::AbstractArray{T}) where {T<:Number} = _nan(T) .* x
alloc_h(x::AbstractArray{T}) where {T<:Number} = _nan(T) .* x * x'
alloc_j(x::AbstractVector{T}, y::AbstractVector) where {T<:Number} = _nan(T) .* y * x'


function outer!(O, x, y)
    @assert axes(O, 1) == axes(x, 1)
    @assert axes(O, 2) == axes(y, 1)
    @inbounds @simd for i in axes(O, 1)
        for j in axes(O, 2)
            O[i, j] = x[i] * y[j]
        end
    end

end
