# Static Line Search

Static line search is the simplest form of line search in which the *guess for ``\alpha``* is always just a fixed value. In the following we demonstrate how to use this line search.

## [Example](@id static_example)

We show how to use linesearches in `SimpleSolvers` to solve a simple toy problem:

```@example static
using SimpleSolvers # hide

x = [1., 0., 0.]
f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2)
obj = OptimizerProblem(f, x)

α = .1
ls_method = Static(α)
nothing # hide
```

`SimpleSolvers` contains a function [`SimpleSolvers.linesearch_problem`](@ref) that allocates a [`SimpleSolvers.LinesearchProblem`](@ref) that only depends on ``\alpha``:

```@example static
using SimpleSolvers: linesearch_problem, NewtonOptimizerCache, update! # hide
cache = NewtonOptimizerCache(x)
state = NewtonOptimizerState(x)
grad = GradientAutodiff{Float64}(obj.F, length(x))
update!(cache, state, grad, x)
x₂ = [.9, 0., 0.]
update!(cache, state, grad, x₂)
ls_obj = linesearch_problem(obj, grad, cache, state)
nothing # hide
```

We now use this to compute a *static line search*:

```@example static
ls = Linesearch(ls_method)
solve(ls_obj, ls)
```

!!! info
    We note that for the static line search we always just return ``\alpha``.