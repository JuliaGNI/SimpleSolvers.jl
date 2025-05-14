# Solver Status

In `SimpleSolvers` we can use the [`SimpleSolvers.NonlinearSolverStatus`](@ref) to provide a diagnostic tool for a [`NonlinearSolver`](@ref). We first make an instance of [`MultivariateObjective`](@ref):

```@example status
using SimpleSolvers # hide
using SimpleSolvers: SufficientDecreaseCondition, NewtonOptimizerCache, update!, gradient!, linesearch_objective, ldiv! # hide

x = [3., 1.3]
f = x -> tanh.(x)
obj = MultivariateObjective(f, x)
```

We now create an instance of [`NewtonSolver`](@ref) which also allocates a [`SimpleSolvers.NonlinearSolverStatus`](@ref):

```@example status
solver = NewtonSolver(obj, x, obj(x))
```

Note that all variables are [initialized with `NaN`s](@ref "Reasoning behind Initialization with `NaN`s").

For the first step we therefore have to call [`SimpleSolvers.update!`](@ref)[^1]:

[^1]: Also see the [page on the `update!` function](@ref "Updates").

```@example status
update!(solver, x)
```