"""
    FixedPointIteratorCache

Stores the last solution `xₖ`.
"""
struct FixedPointIteratorCache{T,VT<:AbstractVector{T}} <: NonlinearSolverCache{T}
    xₖ::VT
    FixedPointIteratorCache(x::VT) where {T,VT<:AbstractVector{T}} = new{T,VT}(copy(x))
end

solution(cache::FixedPointIteratorCache) = cache.xₖ

function update!(cache::FixedPointIteratorCache{T,VT}, x::VT) where {T,VT<:AbstractVector{T}}
    solution(cache) .= x
    cache
end

const FixedPointIterator{T} = NonlinearSolver{T, PicardMethod}

function FixedPointIterator(x::AT, nls::NLST, cache::CT; options_kwargs...) where {T,AT<:AbstractVector{T},NLST,CT}
    cache = FixedPointIteratorCache(x)
    NonlinearSolver(x, nls, DummyLinearSystem(), DummyLinearSolver(), DummyLinesearchState(T), cache; method = PicardMethod(), options_kwargs...)
end

"""
    FixedPointIterator(x, F)

# Keywords
- `options_kwargs`: see [`Options`](@ref)
"""
function FixedPointIterator(x::AT, F::Callable; kwargs...) where {T,AT<:AbstractVector{T}}
    nls = NonlinearSystem(F, missing, x, x; fixed_point=true)
    cache = FixedPointIteratorCache(x)
    FixedPointIterator(x, nls, cache; kwargs...)
end

function FixedPointIterator(x::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    FixedPointIterator(x, F; kwargs...)
end

"""
    solver_step!(it, x, params)

Solve the problem stored in an instance `it` of [`FixedPointIterator`](@ref).
"""
function solver_step!(it::FixedPointIterator{T}, x::AbstractVector{T}, params) where {T}
    update!(cache(it), x)
    value!(nonlinearsystem(it), x, params)
    x .= value(nonlinearsystem(it))
    isFixedPointFormat(nonlinearsystem(it)) || (x .-= solution(cache(it)))
    x
end

cache(solver::FixedPointIterator)::FixedPointIteratorCache = solver.cache
config(solver::FixedPointIterator)::Options = solver.config
status(solver::FixedPointIterator)::NonlinearSolverStatus = solver.status

"""
    nonlinearsystem(it)

Return the [`NonlinearSystem`](@ref) contained in the [`FixedPointIterator`](@ref). Compare this to [`linearsolver`](@ref).
"""
nonlinearsystem(it::FixedPointIterator)::NonlinearSystem = it.nonlinearsystem

value(it::FixedPointIterator) = value(nonlinearsystem(it))

iteration_number(it::FixedPointIterator)::Integer = iteration_number(status(it))

initialize!(it::FixedPointIterator, x₀::AbstractArray) = initialize!(status(it), x₀)

"""
    update!(iterator, x, params)

Update the `solver::`[`FixedPointIterator`](@ref) based on `x`.
This updates the cache (instance of type [`FixedPointIteratorCache`](@ref)) and the status (instance of type [`NonlinearSolverStatus`](@ref)). In course of updating the latter, we also update the `nonlinear` stored in `iterator` (and `status(iterator)`).

!!! info
    At the moment this is neither used in `solver_step!` nor `solve!`.
"""
function update!(it::FixedPointIterator, x₀::AbstractArray, params)
    update!(status(it), x₀, nonlinearsystem(it), params)
    update!(nonlinearsystem(it), x₀, params)
    update!(cache(it), x₀)

    it
end

"""
    solve!(it, x)

# Extended help

!!! info
    The function `update!` calls [`increase_iteration_number!`](@ref).
"""
function solve!(it::FixedPointIterator, x::AbstractArray, params=NullParameters())
    initialize!(it, x)
    update!(status(it), x, nonlinearsystem(it), params)

    while !meets_stopping_criteria(status(it), config(it))
        increase_iteration_number!(status(it))
        solver_step!(it, x, params)
        update!(status(it), x, nonlinearsystem(it), params)
        residual!(status(it))
    end

    print_status(status(it), config(it))
    warn_iteration_number(status(it), config(it))

    x
end

Base.show(io::IO, it::FixedPointIterator) = show(io, status(it))
