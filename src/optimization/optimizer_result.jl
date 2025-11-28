
"""
    OptimizerResult

Stores `x`, `f` and `g` (as keys).
"""
mutable struct OptimizerResult{T, YT, VT <: AbstractArray{T}, OST <: OptimizerStatus{T,YT}}
    status::OST

    x::VT    # current solution
    f::YT    # current function
end

status(result::OptimizerResult) = result.status

solution(result::OptimizerResult) = result.x
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

    result
end
