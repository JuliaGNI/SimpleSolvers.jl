# Optimizer Problems

A central object in `SimpleSolvers` are *optimizer problems* (see [`SimpleSolvers.AbstractOptimizerProblem`](@ref)). They are either [`SimpleSolvers.LinesearchProblem`](@ref)s or [`OptimizerProblem`](@ref)s. The goal of a *solver* (both [`LinearSolver`](@ref)s and [`NonlinearSolver`](@ref)s) is to make the optimizer problem have value zero. The goal of an [`Optimizer`](@ref) is to minimize a [`OptimizerProblem`](@ref).

## Examples

### Multivariate Optimizer Problems

[`OptimizerProblem`](@ref)s are used in a way similar to [`LinesearchProblem`](@ref)s.

```@example optimizer_problem
using SimpleSolvers # hide
using Random # hide
Random.seed!(123) # hide
f(x::AbstractArray) = sum(x .^ 2)
x = rand(3)

obj = OptimizerProblem(f, x)
```