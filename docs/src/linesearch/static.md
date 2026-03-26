# Static Line Search

Static line search is the simplest form of line search in which the *guess for ``\alpha``* is always just a fixed value.

## [Example](@id static_example)

We show how to use linesearches in `SimpleSolvers` to solve a simple toy problem:

```@example static
using SimpleSolvers # hide
using SimpleSolvers: NullParameters, direction!, linesearch_problem, update!, direction, cache # hide
x = [3., 1.3]
y = similar(x)
f(y, x, params) = y .= 10 .* x .^ 3 ./ 6 .- x .^ 2 ./ 2
_params = NullParameters()
f(y, x, _params)
s = NewtonSolver(x, y; F = f)
c₁ = 1e-4
state = NonlinearSolverState(x)
update!(state, x, y)
direction!(s, x, _params, 0)
p = copy(direction(cache(s))) # hide
params = (x = state.x, parameters = _params)

α = .1
ls_method = Static(α)
nothing # hide
```

`SimpleSolvers` contains a function [`linesearch_problem`](@ref) that allocates a [`LinesearchProblem`](@ref) that only depends on ``\alpha``:

```@example static
ls_obj = linesearch_problem(s)
nothing # hide
```

We now use this to compute a *static line search*:

```@example static
ls = Linesearch(ls_obj, ls_method)
solve(ls, 1.0, params)
```
