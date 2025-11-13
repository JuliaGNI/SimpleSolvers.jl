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
sl = Static(α)
nothing # hide
```

`SimpleSolvers` contains a function [`SimpleSolvers.linesearch_problem`](@ref) that allocates a [`SimpleSolvers.LinesearchProblem`](@ref) that only depends on ``\alpha``:

```@example static
using SimpleSolvers: linesearch_problem, NewtonOptimizerCache, LinesearchState, update! # hide
cache = NewtonOptimizerCache(x)

update!(cache, x)
x₂ = [.9, 0., 0.]
update!(cache, x₂)
value!(obj, x₂)
gradient!(obj, x₂)
ls_obj = linesearch_problem(obj, cache)
nothing # hide
```

We now use this to compute a *static line search*[^1]:

[^1]: We also note the use of the [`SimpleSolvers.LinesearchState`](@ref) constructor here, which has to be used together with a [`SimpleSolvers.LinesearchMethod`](@ref).

```@example static
ls = LinesearchState(sl)
ls(ls_obj, α)
```

!!! info
    We note that for the static line search we always just return ``\alpha``.