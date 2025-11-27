
"""
    OptimizerResult

Stores an [`OptimizerStatus`](@ref) as well as `x` and `f` (as keys).
[`OptimizerStatus`](@ref) stores all other information (apart form `x` and `f`; i.e. residuals etc.
"""
mutable struct OptimizerResult{T, YT, VT <: AbstractArray{T}, OST <: OptimizerStatus{T,YT}}
    status::OST   # iteration number, residuals and convergence info

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
    clear!(status(result))

    result.x .= XT(NaN)
    result.f  = YT(NaN)

    result
end
