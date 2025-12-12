
const FixedPointIterator{T} = NonlinearSolver{T,PicardMethod}

function FixedPointIterator(x::AT, nlp::NLST, y::AT, linesearch::LiSeT, cache::CT; kwargs...) where {T,AT<:AbstractVector{T},NLST,LiSeT,CT}
    cache = NonlinearSolverCache(x, y)
    NonlinearSolver(x, nlp, NoLinearProblem(), NoLinearSolver(), linesearch, cache; method=PicardMethod(), kwargs...)
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
    ls = LinesearchState(linesearch; T=T)
    FixedPointIterator(x, nlp, y, ls, cache; jacobian=jacobian, kwargs...)
end

function FixedPointIterator(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    FixedPointIterator(x, F, y; kwargs...)
end

function compute_new_direction(x, it::FixedPointIterator, params)
    value!(direction(cache(it)), nonlinearproblem(it), x, params)
    direction(cache(it)) .*= -1
end

"""
    update!(iterator, x, params)

Update the `solver::`[`FixedPointIterator`](@ref) based on `x`.
This updates the cache (instance of type [`FixedPointIteratorCache`](@ref)) and the status (instance of type [`NonlinearSolverStatus`](@ref)). In course of updating the latter, we also update the `nonlinear` stored in `iterator` (and `status(iterator)`).

!!! info
    At the moment this is neither used in `solver_step!` nor `solve!`.
"""
function update!(it::FixedPointIterator, x₀::AbstractArray, params)
    update!(status(it), x₀, nonlinearproblem(it), params)
    update!(nonlinearproblem(it), x₀, params)
    update!(cache(it), x₀)

    it
end
