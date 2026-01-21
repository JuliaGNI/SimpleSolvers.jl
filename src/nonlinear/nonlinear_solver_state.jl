struct NonlinearSolverState{T, XT <: AbstractVector{T}, YT <: AbstractVector{T}}
    x̄::XT
    f̄::YT
    f₀::YT

    function NonlinearSolverState(x̄::XT, f̄::YT, f₀::YT) where {T, XT <: AbstractVector{T}, YT <: AbstractVector{T}}
        new{T, XT, YT}(x̄, f̄, f₀)
    end
end

solution(state::NonlinearSolverState) = state.x̄
value(state::NonlinearSolverState) = state.f̄

function NonlinearSolverState(x::AbstractVector{T}, y::AbstractVector{T}=x) where {T}
    f̄ = zero(y)
    f₀ = zero(y)
    f̄ .= T(NaN)
    f₀ .= T(NaN)
    NonlinearSolverState(x, f̄, f₀)
end

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