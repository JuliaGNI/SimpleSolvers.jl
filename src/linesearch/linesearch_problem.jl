"""
    LinesearchProblem <: AbstractProblem

In practice `LinesearchProblem`s are allocated by calling [`linesearch_problem`](@ref).

# Constructors

!!! warning "Calling line search problems"
    Below we show a constructor that can be used to allocate a `LinesearchProblem`. Note however that in practice one should call `linesearch_problem` and not use the constructor directly.

```jldoctest; setup = :(using SimpleSolvers: LinesearchProblem, compute_new_iterate)
f(x) = x^2 - 1
g(x) = 2x
δx(x) = - g(x) / 2
x₀ = 3.
_f(α,_) = f(compute_new_iterate(x₀, α, δx(x₀)))
_d(α,_) = g(compute_new_iterate(x₀, α, δx(x₀)))
ls_obj = LinesearchProblem{typeof(x₀)}(_f, _d)

# output

LinesearchProblem{Float64, typeof(_f), typeof(_d)}(_f, _d)
```
"""
struct LinesearchProblem{T,TF,TD} <: AbstractProblem
    F::TF
    D::TD

    LinesearchProblem{T}(f, d) where {T<:Number} = new{T,typeof(f),typeof(d)}(f, d)
end


value(problem::LinesearchProblem, x::Number, params) = problem.F(x, params)
derivative(problem::LinesearchProblem, x::Number, params) = problem.D(x, params)
