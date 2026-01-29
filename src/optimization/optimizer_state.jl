"""
An `OptimizerState` is a data structure that is used to dispatch on different algorithms.

It needs to implement three methods,
```
initialize!(alg::OptimizerState, ::AbstractVector)
update!(alg::OptimizerState, ::AbstractVector)
solver_step!(::AbstractVector, alg::OptimizerState)
```
that initialize and update the state of the algorithm and perform an actual optimization step.

Further the following convenience methods should be implemented,
```
problem(alg::OptimizerState)
gradient(alg::OptimizerState)
hessian(alg::OptimizerState)
linesearch(alg::OptimizerState)
```
which return the problem to optimize, its gradient and (approximate) Hessian as well as the
linesearch algorithm used in conjunction with the optimization algorithm if any.

See [`NewtonOptimizerState`](@ref) for a `struct` that was derived from `OptimizerState`.

!!! info
    Note that a `OptimizerState` is not necessarily a `NewtonOptimizerState` as we can also have other optimizers, *Adam* for example.
"""
abstract type OptimizerState{T} <: AbstractSolverState end

OptimizerState(alg::OptimizerMethod, args...; kwargs...) = error("OptimizerState not implemented for $(typeof(alg))")

"""
    isaOptimizerState(alg)

Verify if an object implements the [`OptimizerState`](@ref) interface.
"""
function isaOptimizerState(alg)
    x = rand(3)

    applicable(gradient, alg) &&
    applicable(hessian, alg) &&
    applicable(linesearch, alg) &&
    applicable(problem, alg) &&
    applicable(initialize!, alg, x) &&
    applicable(update!, alg, x) &&
    applicable(solver_step!, x, alg)
end

iteration_number(state::OptimizerState) = state.iterations

function increase_iteration_number!(state::OptimizerState)
    state.iterations = iteration_number(state) + 1
end