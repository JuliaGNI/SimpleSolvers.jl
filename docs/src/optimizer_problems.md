# Optimizer Problems

A central object in `SimpleSolvers` are *optimizer problems* (see [`SimpleSolvers.AbstractOptimizerProblem`](@ref)). They are either [`SimpleSolvers.LinesearchProblem`](@ref)s or [`MultivariateOptimizerProblem`](@ref)s. The goal of a *solver* (both [`LinearSolver`](@ref)s and [`NonlinearSolver`](@ref)s) is to make the optimizer problem have value zero. The goal of an [`Optimizer`](@ref) is to minimize a [`MultivariateOptimizerProblem`](@ref).

## Examples

### Multivariate Optimizer Problems

[`MultivariateOptimizerProblem`](@ref)s are used in a way similar to [`LinesearchProblem`](@ref)s, the difference is that the *derivative functions* are replaced by *gradient functions*, i.e.:
- [`derivative`](@ref) ``\implies`` [`gradient`](@ref),
- [`derivative!`](@ref) ``\implies`` [`gradient!`](@ref),
- [`derivative!!`](@ref) ``\implies`` [`gradient!!`](@ref).

```@example optimizer_problem
Random.seed!(123) # hide
f(x::AbstractArray) = sum(x .^ 2)
x = rand(3)

obj = MultivariateOptimizerProblem(f, x)
```

Every instance of [`MultivariateOptimizerProblem`](@ref) stores an instance of [`Gradient`](@ref) to which we [similarly can apply the functions](@ref "Gradients") [`gradient`](@ref) or [`gradient!`](@ref):

```@example optimizer_problem
gradient(obj, x)
```

The difference to [`Gradient`](@ref) is that we also store the value for the evaluated gradient, which can be accessed by calling:

```@example optimizer_problem
gradient(obj)
```