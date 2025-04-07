
"""
    OptimizerResult

Stores an [`OptimizerStatus`](@ref) as well as `x`, `f` and `g` (as keys).
[`OptimizerStatus`](@ref) stores all other information (apart form `x` ,`f` and `g`); i.e. residuals etc.
"""
mutable struct OptimizerResult{T, YT, VT <: AbstractArray{T}, OST <: OptimizerStatus{T,YT}}
    status::OST   # iteration number, residuals and convergence info

    x::VT    # current solution
    f::YT    # current function
    g::VT    # current gradient
end

function OptimizerResult(x::VT, y::YT) where {XT, YT, VT <: AbstractVector{XT}}
    status = OptimizerStatus{XT,YT}()
    result = OptimizerResult{XT,YT,VT,typeof(status)}(status, zero(x), zero(y), zero(x))
    clear!(result)
end

status(result::OptimizerResult) = result.status

solution(result::OptimizerResult) = result.x
minimizer(result::OptimizerResult) = result.x
Base.minimum(result::OptimizerResult) = result.f

"""
    residual!(result, x, f, g)
"""
function residual!(result::OR, x::VT, f::YT, g::VT)::OR where {XT, VT <: AbstractArray{XT}, YT, OST <: OptimizerStatus{XT, YT}, OR <: OptimizerResult{XT, YT, VT, OST}}
    residual!(result.status, x, result.x, f, result.f, g, result.g)
    
    result
end

"""
    clear!(obj)

Similar to [`initialize!`](@ref).
"""
function clear!(result::OptimizerResult{XT,YT}) where {XT,YT}
    clear!(status(result))

    result.x .= XT(NaN)
    result.f  = YT(NaN)
    result.g .= XT(NaN)

    result
end

"""
    update!(result, x, f, g)
"""
function update!(result::OptimizerResult, x::AbstractVector, f::Number, g::AbstractVector)
    increase_iteration_number!(result)
    residual!(result, x, f, g)

    result.x .= x
    result.f  = f
    result.g .= g

    result
end

function increase_iteration_number!(result::OptimizerResult)
    increase_iteration_number!(status(result))
end
