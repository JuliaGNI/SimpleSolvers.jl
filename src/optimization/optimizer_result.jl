
"""
    OptimizerResult

Stores `x`, `f` and `g` (as keys).
"""
mutable struct OptimizerResult{T,YT,VT<:AbstractArray{T},OST<:OptimizerStatus{T,YT}}
    status::OST

    x::VT    # current solution
    f::YT    # current function
end

status(result::OptimizerResult) = result.status

solution(result::OptimizerResult) = result.x
Base.minimum(result::OptimizerResult) = result.f
