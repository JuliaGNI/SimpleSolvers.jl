"""
    NonlinearSolverState(x, y)
    NonlinearSolverState(x)
    NonlinearSolverState{T}(n, m)
    NonlinearSolverState{T}(n)

The `NonlinearSolverState` to be used together with a [`NonlinearSolver`](@ref).

!!! warn
    Note the difference to the [`NonlinearSolverCache`](@ref) and the [`NonlinearSolverStatus`](@ref).
"""
mutable struct NonlinearSolverState{T,XT<:AbstractVector{T},YT<:AbstractVector{T}} <: AbstractSolverState
    x̄::XT
    f̄::YT
    iterations::Int

    function NonlinearSolverState(x::AbstractVector{T}, y::AbstractVector{T}=x) where {T}
        x̄ = zero(x)
        f̄ = zero(y)
        x̄ .= T(NaN)
        f̄ .= T(NaN)
        new{T,typeof(x̄),typeof(f̄)}(x̄, f̄, 0)
    end
end

solution(state::NonlinearSolverState) = state.x̄
value(state::NonlinearSolverState) = state.f̄

iteration_number(state::NonlinearSolverState) = state.iterations
"""
    increase_iteration_number!(s)

To be used together with [`NonlinearSolver`](@ref).
"""
function increase_iteration_number!(state::NonlinearSolverState)
    state.iterations += 1
end

function NonlinearSolverState{T}(n::Integer, m::Integer=n) where {T}
    x = zeros(T, n)
    y = zeros(T, m)
    x .= T(NaN)
    y .= T(NaN)
    NonlinearSolverState(x, y)
end

function initialize!(state::NonlinearSolverState{T}, x::AbstractVector{T}, f::AbstractVector{T}) where {T}
    state.iterations = 0
    solution(state) .= x
    value(state) .= f
end

function update!(state::NonlinearSolverState{T}, x::AbstractVector{T}, f::AbstractVector{T}) where {T}
    solution(state) .= x
    value(state) .= f
end
