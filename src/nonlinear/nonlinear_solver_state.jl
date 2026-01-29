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
    iterations::Int

    x::XT
    x̄::XT
    y::YT
    ȳ::YT

    function NonlinearSolverState(X::AbstractVector{T}, Y::AbstractVector{T}=x) where {T}
        x = zero(X)
        x̄ = zero(X)
        y = zero(Y)
        ȳ = zero(Y)

        x .= T(NaN)
        x̄ .= T(NaN)
        y .= T(NaN)
        ȳ .= T(NaN)

        new{T,typeof(x),typeof(y)}(0, x, x̄, y, ȳ)
    end
end

iteration_number(state::NonlinearSolverState) = state.iterations
solution(state::NonlinearSolverState) = state.x
value(state::NonlinearSolverState) = state.y

previoussolution(state::NonlinearSolverState) = state.x̄
previousvalue(state::NonlinearSolverState) = state.ȳ

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

function initialize!(state::NonlinearSolverState{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    state.iterations = 0
    state.x .= x
    state.y .= y
    state.x̄ .= T(NaN)
    state.ȳ .= T(NaN)
end

function update!(state::NonlinearSolverState{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    state.x̄ .= state.x
    state.ȳ .= state.y
    state.x .= x
    state.y .= y
end
