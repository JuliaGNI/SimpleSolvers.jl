"""
    OptimizerResult

Stores an [`OptimizerStatus`](@ref) as well as `x`, `f` and `g` (as keys).
[`OptimizerStatus`](@ref) stores all other information (apart form `x` ,`f` and `g`); i.e. residuals etc.
"""
mutable struct OptimizerResult{XT, YT, VT <: AbstractArray{XT}, OST <: OptimizerStatus{XT,YT}}
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


function clear!(result::OptimizerResult{XT,YT}) where {XT,YT}
    clear!(result.status)

    result.x .= XT(NaN)
    result.f  = YT(NaN)
    result.g .= XT(NaN)

    result
end

function residual!(result::OptimizerResult, x, f, g)
    status = result.status

    status.rxₐ = sqeuclidean(x, result.x)
    status.rxᵣ = status.rxₐ / norm(x)

    status.rfₐ = norm(f - result.f)
    status.rfᵣ = status.rfₐ / norm(f)

    status.rgₐ = norm(g - result.g)
    status.rg  = norm(g)

    status.Δf  = f - result.f
    status.Δf̃ = result.g ⋅ x - result.g ⋅ result.x
    
    status.f_increased = abs(f) > abs(result.f)

    status.x_isnan = any(isnan, x)
    status.f_isnan = any(isnan, f)
    status.g_isnan = any(isnan, g)

    status
end

function update!(result::OptimizerResult, x, f, g)
    residual!(result, x, f, g)

    result.x .= x
    result.f  = f
    result.g .= g

    return result
end

function initialize!(result::OptimizerResult, x, f, g)
    clear!(result)
    update!(result, x, f, g)
end

function next_iteration!(result::OptimizerResult)
    increase_iteration_number!(status(result))
end
