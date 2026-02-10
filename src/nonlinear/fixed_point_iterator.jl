
const PicardSolver{T} = NonlinearSolver{T,PicardMethod}

function PicardSolver(x::AT, nlp::NLST, linesearch::LiSeT, cache::CT; jacobian, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LiSeT,CT}
    config = Options(T; options_kwargs...)
    NonlinearSolver(x, nlp, NoLinearProblem(), NoLinearSolver(), linesearch, cache, config; jacobian=jacobian, method=PicardMethod())
end

"""
    FixedPointIterator(x, F)

# Keywords
- `options_kwargs`: see [`Options`](@ref)
"""
function PicardSolver(x::AT, F::Callable, y::AT; (DF!)=missing, linesearch=Backtracking(T), jacobian=JacobianAutodiff(F, x, y), kwargs...) where {T,AT<:AbstractVector{T}}
    nlp = NonlinearProblem(F, DF!, x, y)
    jacobian = ismissing(DF!) ? jacobian : JacobianFunction{T}(F, DF!)
    cache = NonlinearSolverCache(x, y)
    ls = Linesearch(linesearch_problem(nlp, jacobian, cache), linesearch)
    PicardSolver(x, nlp, ls, cache; jacobian=jacobian, kwargs...)
end

function PicardSolver(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    PicardSolver(x, F, y; kwargs...)
end

NonlinearSolver(::PicardMethod, x...; kwargs...) = PicardSolver(x...; kwargs...)

function direction!(d::AbstractVector{T}, x::AbstractVector{T}, it::PicardSolver{T}, params) where {T}
    value!(d, nonlinearproblem(it), x, params)
    d .*= -1
end

function direction!(it::PicardSolver, x::AbstractVector, params)
    direction!(direction(cache(it)), x, it, params)
end

direction!(it::PicardSolver, x::AbstractVector, params, iteration) = direction!(it, x, params)

"""
    update!(iterator, x, params)

Update the `solver::`[`FixedPointIterator`](@ref) based on `x`.
This updates the cache (instance of type [`NonlinearSolverCache`](@ref)) and the status (instance of type [`NonlinearSolverStatus`](@ref)). In course of updating the latter, we also update the `nonlinear` stored in `iterator` (and `status(iterator)`).

!!! info
    At the moment this is neither used in `solver_step!` nor `solve!`.
"""
function update!(it::PicardSolver, x₀::AbstractArray, params)
    update!(status(it), x₀, nonlinearproblem(it), params)
    update!(nonlinearproblem(it), x₀, params)
    update!(cache(it), x₀)

    it
end
