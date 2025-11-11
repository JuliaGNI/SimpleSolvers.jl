
"""
    OptimizerResult

Stores `x`, `f` and `g` (as keys).
"""
mutable struct OptimizerResult{T, YT, VT <: AbstractArray{T}}
    x::VT    # current solution
    f::YT    # current function
    g::VT    # current gradient
end

function OptimizerResult(x::VT, y::YT) where {XT, YT, VT <: AbstractVector{XT}}
    result = OptimizerResult{XT,YT,VT}(zero(x), zero(y), zero(x))
    clear!(result)
end

OptimizerResult(x::AbstractVector, obj::AbstractObjective) = OptimizerResult(x, obj(x))

solution(result::OptimizerResult) = result.x
minimizer(result::OptimizerResult) = result.x
Base.minimum(result::OptimizerResult) = result.f

"""
    clear!(result)

Clear all the information contained in `result::`[`OptimizerResult`](@ref).
This also calls [`clear!(::OptimizerStatus)`](@ref).

!!! info
   Calling `initialize!` on an `OptimizerResult` calls `clear!` internally.
"""
function clear!(result::OptimizerResult{XT,YT}) where {XT,YT}
    result.x .= XT(NaN)
    result.f  = YT(NaN)
    result.g .= XT(NaN)

    result
end

function initialize!(result::OptimizerResult{T}, ::AbstractVector{T}) where {T}
    clear!(result)
    result
end

"""
    update!(result, x, f, g)

Update the [`OptimizerResult`](@ref) based on `x`, `f` and `g` (all vectors).
This involves updating the [`OptimizerStatus`](@ref) stored in `result` (by calling [`residual!`](@ref)).

This also calls [`increase_iteration_number!(::OptimizerResult)`](@ref)
"""
function update!(result::OptimizerResult, x::AbstractVector, f::Number, g::AbstractVector)
    result.x .= x
    result.f  = f
    result.g .= g

    result
end

update!(result::OptimizerResult, x::AbstractVector, f::Number, grad::Gradient) = update!(result, x, f, gradient(x, grad))
update!(result::OptimizerResult, x::AbstractVector, obj::AbstractObjective, g) = update!(result, x, obj(x), g)
