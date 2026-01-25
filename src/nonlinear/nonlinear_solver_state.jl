"""
    NonlinearSolverState(x, y)
    NonlinearSolverState(x)
    NonlinearSolverState{T}(n, m)
    NonlinearSolverState{T}(n)

The `NonlinearSolverState` to be used together with a [`NonlinearSolver`](@ref).

!!! warn
    Note the difference to the [`NonlinearSolverCache`](@ref) and the [`NonlinearSolverStatus`](@ref).
"""
struct NonlinearSolverState{T, XT <: AbstractVector{T}, YT <: AbstractVector{T}}
    x̄::XT
    f̄::YT
    f₀::YT

    function NonlinearSolverState(x::AbstractVector{T}, y::AbstractVector{T}=x) where {T}
        x̄ = zero(x)
        f̄ = zero(y)
        f₀ = zero(y)
        x̄ .= T(NaN)
        f̄ .= T(NaN)
        f₀ .= T(NaN)
        new{T, typeof(x), typeof(y)}(x̄, f̄, f₀)
    end
end

solution(state::NonlinearSolverState) = state.x̄
value(state::NonlinearSolverState) = state.f̄

function NonlinearSolverState{T}(n::Integer, m::Integer=n) where {T}
    x = zeros(T, n)
    y = zeros(T, m)
    x .= T(NaN)
    y .= T(NaN)
    NonlinearSolverState(x, y)
end

function update!(state::NonlinearSolverState{T}, x::AbstractVector{T}, f::AbstractVector{T}, iteration_number::Integer) where {T}
    iteration_number == 0 && (state.f₀ .= f)
    solution(state) .= x
    value(state) .= f
end