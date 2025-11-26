"""
An `OptimizationAlgorithm` is a data structure that is used to dispatch on different algorithms.

It needs to implement three methods,
```
initialize!(alg::OptimizationAlgorithm, ::AbstractVector)
update!(alg::OptimizationAlgorithm, ::AbstractVector)
solver_step!(::AbstractVector, alg::OptimizationAlgorithm)
```
that initialize and update the state of the algorithm and perform an actual optimization step.

Further the following convenience methods should be implemented,
```
problem(alg::OptimizationAlgorithm)
gradient(alg::OptimizationAlgorithm)
hessian(alg::OptimizationAlgorithm)
linesearch(alg::OptimizationAlgorithm)
```
which return the problem to optimize, its gradient and (approximate) Hessian as well as the
linesearch algorithm used in conjunction with the optimization algorithm if any.

See [`NewtonOptimizerState`](@ref) for a `struct` that was derived from `OptimizationAlgorithm`.

!!! info
    Note that a `OptimizationAlgorithm` is not necessarily a `NewtonOptimizerState` as we can also have other optimizers, *Adam* for example.
"""
abstract type OptimizationAlgorithm end

OptimizerState(alg::OptimizationAlgorithm, args...; kwargs...) = error("OptimizerState not implemented for $(typeof(alg))")

"""
    isaOptimizationAlgorithm(alg)

Verify if an object implements the [`OptimizationAlgorithm`](@ref) interface.
"""
function isaOptimizationAlgorithm(alg)
    x = rand(3)

    applicable(gradient, alg) &&
    applicable(hessian, alg) &&
    applicable(linesearch, alg) &&
    applicable(problem, alg) &&
    applicable(initialize!, alg, x) &&
    applicable(update!, alg, x) &&
    applicable(solver_step!, x, alg)
end