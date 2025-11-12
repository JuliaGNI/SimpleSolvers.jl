
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

OptimizerResult(x::AbstractVector, obj::AbstractOptimizerProblem) = OptimizerResult(x, obj(x))

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
    increase_iteration_number!(result)
    residual!(result, x, f, g)

    result.x .= x
    result.f  = f
    result.g .= g

    result
end

update!(result::OptimizerResult, x::AbstractVector, f::Number, grad::Gradient) = update!(result, x, f, gradient(x, grad))
update!(result::OptimizerResult, x::AbstractVector, obj::AbstractOptimizerProblem, g) = update!(result, x, obj(x), g)

"""
    increase_iteration_number!(result)

Increase the iteration number of `result`[`OptimizerResult`](@ref). This calls [`increase_iteration_number!(::OptimizerStatus)`](@ref).
"""
increase_iteration_number!(result::OptimizerResult) = increase_iteration_number!(status(result))
