"""
    LinesearchProblem <: AbstractOptimizerProblem

Doesn't store `f`, `d`, `x_f` and `x_d`.

In practice `LinesearchProblem`s are allocated by calling [`linesearch_problem`](@ref).

# Constructors

!!! warning "Calling line search problems"
    Below we show a few constructors that can be used to allocate `LinesearchProblem`s. Note however that in practice one probably should not do that and instead call `linesearch_problem`.

```jldoctest; setup = :(using SimpleSolvers: LinesearchProblem, compute_new_iterate)
f(x) = x^2 - 1
g(x) = 2x
δx(x) = - g(x) / 2
x₀ = 3.
_f(α) = f(compute_new_iterate(x₀, α, δx(x₀)))
_d(α) = g(compute_new_iterate(x₀, α, δx(x₀)))
ls_obj = LinesearchProblem{typeof(x₀)}(_f, _d)

# output

LinesearchProblem{Float64, typeof(_f), typeof(_d)}(_f, _d)
```

Alternatively one can also do:

```jldoctest; setup = :(using SimpleSolvers: LinesearchProblem, compute_new_iterate; f(x) = x^2 - 1; g(x) = 2x; δx(x) = - g(x) / 2; x₀ = 3.; _f(α) = f(compute_new_iterate(x₀, α, δx(x₀))); _d(α) = g(compute_new_iterate(x₀, α, δx(x₀))))
ls_obj = LinesearchProblem(_f, _d, x₀)

# output

LinesearchProblem{Float64, typeof(_f), typeof(_d)}(_f, _d)
```

Here we wrote `ls_obj` to mean *line search problem*.
"""
struct LinesearchProblem{Tx,TF,TD} <: AbstractProblem
    F::TF
    D::TD
end

LinesearchProblem{Tx}(f, d) where {Tx<:Number} = LinesearchProblem{Tx,typeof(f),typeof(d)}(f, d)

LinesearchProblem(f, d, ::Tx=zero(Float64)) where {Tx<:Number} = LinesearchProblem{Tx}(f, d)

value(obj::LinesearchProblem, x::Number) = obj.F(x)
derivative(obj::LinesearchProblem, x::Number) = obj.D(x)
