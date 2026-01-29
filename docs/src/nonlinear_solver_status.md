# Solver Status

In `SimpleSolvers` we can use the [`SimpleSolvers.NonlinearSolverStatus`](@ref) to provide a diagnostic tool for a [`NonlinearSolver`](@ref). We first make an instance of [`NonlinearProblem`](@ref):

```@example status
using SimpleSolvers # hide
using SimpleSolvers: NonlinearSolverState, update!, solver_step! # hide

x = [3., 1.3]
f = x -> tanh.(x)
F!(y, x, params) = y .= f(x)
nlp = NonlinearProblem(F!, x, f(x))
```

We now create an instance of [`NewtonSolver`](@ref) and [`NonlinearSolverState`](@ref):

```@example status
solver = NewtonSolver(x, f(x); F = F!)
state = NonlinearSolverState(x)
```

Note that all variables are [initialized with `NaN`s](@ref "Reasoning behind Initialization with `NaN`s").

For the first step we call [`solver_step!`](@ref) (which updates the `state` internally via [`update!`](@ref)[^1]):

[^1]: Also see the [page on the `update!` function](@ref "Updates").

```@example status
using SimpleSolvers: NullParameters, cache # hide
params = NullParameters()
solver_step!(x, solver, state, params)
```

We now compute the [`NonlinearSolverStatus`](@ref):

```@example status
using SimpleSolvers: NonlinearSolverStatus # hide

NonlinearSolverStatus(state, config(solver))
```