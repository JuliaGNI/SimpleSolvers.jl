"""
    NonlinearSolverState <: AbstractSolverState

The `NonlinearSolverState` to be used together with a [`NonlinearSolver`](@ref).

!!! info
    Note the difference to the [`NonlinearSolverCache`](@ref) and the [`NonlinearSolverStatus`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers)
julia> state = NonlinearSolverState(zeros(3))
NonlinearSolverState{Float64, Vector{Float64}, Vector{Float64}}(0, [NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN])
```
"""
mutable struct NonlinearSolverState{T,XT<:AbstractVector{T},YT<:AbstractVector{T}} <: AbstractSolverState
    iterations::Int

    x::XT
    x̄::XT
    y::YT
    ȳ::YT

    function NonlinearSolverState(X::AbstractVector{T}, Y::AbstractVector{T}=X) where {T}
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
    increase_iteration_number!(state)

To be used together with [`NonlinearSolverState`](@ref).
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

"""
    update!(state, x, y)

Update `x̄`, `ȳ`, `x` and `y`.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NullParameters)
julia> f(y, x, params) = y .= sin.(x .- .5) .^ 2
f (generic function with 1 method)

julia> x = ones(1) / 4
1-element Vector{Float64}:
 0.25

julia> y = zero(x); f(y, x, NullParameters())
1-element Vector{Float64}:
 0.06120871905481365

julia> state = NonlinearSolverState(x)
NonlinearSolverState{Float64, Vector{Float64}, Vector{Float64}}(0, [NaN], [NaN], [NaN], [NaN])

julia> update!(state, x, y)
NonlinearSolverState{Float64, Vector{Float64}, Vector{Float64}}(0, [0.25], [NaN], [0.06120871905481365], [NaN])

julia> x = ones(1) / 2
1-element Vector{Float64}:
 0.5

julia> f(y, x, NullParameters())
1-element Vector{Float64}:
 0.0

julia> update!(state, x, y)
NonlinearSolverState{Float64, Vector{Float64}, Vector{Float64}}(0, [0.5], [0.25], [0.0], [0.06120871905481365])
```

The [`NonlinearSolverState`](@ref) stores the previous solution, the previous value, the current solution and the current value.

All of these are updated during one [`update!`](@ref) step (and initialized with `NaN`s).
"""
function update!(state::NonlinearSolverState{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    previoussolution(state) .= solution(state)
    previousvalue(state) .= value(state)
    solution(state) .= x
    value(state) .= y

    state
end
