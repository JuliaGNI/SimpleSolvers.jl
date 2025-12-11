"""
    FixedPointIteratorCache

Stores the last solution `xₖ`.
"""
struct FixedPointIteratorCache{T,VT<:AbstractVector{T},JT<:AbstractMatrix{T}} <: NonlinearSolverCache{T}
    δx::VT
    x̄::VT
    x::VT
    y::VT
    j::JT
    function FixedPointIteratorCache(x::VT, y::VT) where {T,VT<:AbstractVector{T}}
        j = alloc_j(x, y)
        new{T,VT,typeof(j)}(copy(x), copy(x), copy(x), copy(x), j)
    end
end

direction(cache::FixedPointIteratorCache) = cache.δx
jacobian(cache::FixedPointIteratorCache) = cache.j

function update!(cache::FixedPointIteratorCache{T,VT}, x::VT) where {T,VT<:AbstractVector{T}}
    copy!(cache.x̄, x)
    cache
end

const FixedPointIterator{T} = NonlinearSolver{T,PicardMethod}

function FixedPointIterator(x::AT, nlp::NLST, y::AT, linesearch::LiSeT, cache::CT; kwargs...) where {T,AT<:AbstractVector{T},NLST,LiSeT,CT}
    cache = FixedPointIteratorCache(x, y)
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
    cache = FixedPointIteratorCache(x, y)
    ls = LinesearchState(linesearch; T=T)
    FixedPointIterator(x, nlp, y, ls, cache; jacobian=jacobian, kwargs...)
end

function FixedPointIterator(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    FixedPointIterator(x, F, y; kwargs...)
end

"""
    solver_step!(x, it, params)

Solve the problem stored in an instance `it` of [`FixedPointIterator`](@ref).
"""
function solver_step!(x::AbstractVector{T}, it::FixedPointIterator{T}, params) where {T}
    update!(cache(it), x)
    for _ in 1:LINESEARCH_NAN_MAX_ITERATIONS
        value!(direction(cache(it)), nonlinearproblem(it), x, params)
        direction(cache(it)) .*= -1
        if any(isnan, direction(cache(it)))
            (it.config.verbosity >= 2 && @warn "NaN detected in Picard solver. Reducing length of direction vector.")
            direction(cache(it)) ./= 2
        else
            break
        end
    end
    α = linesearch(it)(linesearch_problem(it, params))
    compute_new_iterate!(x, α, direction(cache(it)))
    x
end

cache(solver::FixedPointIterator)::FixedPointIteratorCache = solver.cache
config(solver::FixedPointIterator)::Options = solver.config
status(solver::FixedPointIterator)::NonlinearSolverStatus = solver.status

"""
    nonlinearproblem(it)

Return the [`NonlinearProblem`](@ref) contained in the [`FixedPointIterator`](@ref). Compare this to [`linearsolver`](@ref).
"""
nonlinearproblem(it::FixedPointIterator)::NonlinearProblem = it.nonlinearproblem

value(it::FixedPointIterator) = value(nonlinearproblem(it))

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
    update!(status(it), x₀, nonlinearproblem(it), params)
    update!(nonlinearproblem(it), x₀, params)
    update!(cache(it), x₀)

    it
end

"""
    solve!(x, it)

# Extended help

!!! info
    The function `update!` calls [`increase_iteration_number!`](@ref).
"""
function solve!(x::AbstractArray, it::FixedPointIterator, params=NullParameters())
    initialize!(it, x)
    update!(status(it), x, nonlinearproblem(it), params)

    while !meets_stopping_criteria(status(it), config(it))
        increase_iteration_number!(status(it))
        solver_step!(x, it, params)
        update!(status(it), x, nonlinearproblem(it), params)
        residual!(status(it))
    end

    print_status(status(it), config(it))
    warn_iteration_number(status(it), config(it))

    x
end

Base.show(io::IO, it::FixedPointIterator) = show(io, status(it))
