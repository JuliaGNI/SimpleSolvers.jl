# Solver Status

In `SimpleSolvers` we can use the [`SimpleSolvers.NonlinearSolverStatus`](@ref) to provide a diagnostic tool for a [`NonlinearSolver`](@ref). We first make an instance of [`NonlinearSystem`](@ref):

```@example status
using SimpleSolvers # hide
using SimpleSolvers: SufficientDecreaseCondition, NewtonOptimizerCache, update!, gradient!, linesearch_objective, ldiv! # hide

x = [3., 1.3]
f = x -> tanh.(x)
nls = NonlinearSystem(f, x)
```

We now create an instance of [`NewtonSolver`](@ref) which also allocates a [`SimpleSolvers.NonlinearSolverStatus`](@ref):

```@example status
solver = NewtonSolver(x, f)
```

Note that all variables are [initialized with `NaN`s](@ref "Reasoning behind Initialization with `NaN`s").

For the first step we therefore have to call [`update!`](@ref)[^1]:

[^1]: Also see the [page on the `update!` function](@ref "Updates").

```@example status
update!(solver, x)
```

Note that the residuals are still `NaN`s however as we need to perform at least two updates in order to compute them. As a next step we write:

```@example status
x = [2., 1.2]
update!(solver, x)
```