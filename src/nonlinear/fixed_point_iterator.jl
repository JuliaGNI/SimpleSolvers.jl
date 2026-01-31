
const FixedPointIterator{T} = NonlinearSolver{T,PicardMethod}

function FixedPointIterator(x::AT, nlp::NLST, y::AT, linesearch::LiSeT, cache::CT; jacobian, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LiSeT,CT}
    cache = NonlinearSolverCache(x, y)
    config = Options(T; options_kwargs...)
    NonlinearSolver(x, nlp, NoLinearProblem(), NoLinearSolver(), linesearch, cache, config; jacobian=jacobian, method=PicardMethod())
end

"""
    FixedPointIterator(x, F)

# Keywords
- `options_kwargs`: see [`Options`](@ref)
"""
function FixedPointIterator(x::AT, F::Callable, y::AT; (DF!)=missing, linesearch=Backtracking(), jacobian=JacobianAutodiff(F, x, y), kwargs...) where {T,AT<:AbstractVector{T}}
    nlp = NonlinearProblem(F, missing, x, x)
    jacobian = ismissing(DF!) ? jacobian : JacobianFunction{T}(F, DF!)
    cache = NonlinearSolverCache(x, y)
    ls = Linesearch(linesearch; T=T)
    FixedPointIterator(x, nlp, y, ls, cache; jacobian=jacobian, kwargs...)
end

function FixedPointIterator(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    FixedPointIterator(x, F, y; kwargs...)
end

NonlinearSolver(::PicardMethod, x...; kwargs...) = FixedPointIterator(x...; kwargs...)

function direction!(d::AbstractVector{T}, x::AbstractVector{T}, it::FixedPointIterator{T}, params) where {T}
    value!(d, nonlinearproblem(it), x, params)
    d .*= -1
end

function direction!(it::FixedPointIterator, x::AbstractVector, params)
    direction!(direction(cache(it)), x, it, params)
end

direction!(it::FixedPointIterator, x::AbstractVector, params, iteration) = direction!(it, x, params)

"""
    update!(iterator, x, params)

Update the `solver::`[`FixedPointIterator`](@ref) based on `x`.
This updates the cache (instance of type [`NonlinearSolverCache`](@ref)) and the status (instance of type [`NonlinearSolverStatus`](@ref)). In course of updating the latter, we also update the `nonlinear` stored in `iterator` (and `status(iterator)`).

!!! info
    At the moment this is neither used in `solver_step!` nor `solve!`.
"""
function update!(it::FixedPointIterator, x₀::AbstractArray, params)
    update!(status(it), x₀, nonlinearproblem(it), params)
    update!(nonlinearproblem(it), x₀, params)
    update!(cache(it), x₀)

    it
end
