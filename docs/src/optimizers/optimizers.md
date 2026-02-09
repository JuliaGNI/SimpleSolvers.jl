# Optimizers

An [`Optimizer`](@ref) stores an [`OptimizerState`](@ref), a [`OptimizerProblem`](@ref), the [`SimpleSolvers.OptimizerResult`](@ref) and a [`SimpleSolvers.NonlinearMethod`](@ref). Its purposes are:

```@example optimizer
using SimpleSolvers
using LinearAlgebra: norm
import Random # hide
Random.seed!(123) # hide

x = rand(3)
obj = OptimizerProblem(x -> sum((x - [0., 0., 1.]) .^ 2), x)
bt = Backtracking()
alg = Newton()
opt = Optimizer(x, obj; algorithm = alg, linesearch = bt)
```

## Optimizer Constructor

Internally the constructor for [`Optimizer`](@ref) calls [`SimpleSolvers.OptimizerResult`](@ref) and [`SimpleSolvers.NewtonOptimizerState`](@ref) and [`Hessian`](@ref). We can also allocate these objects manually and then call a different constructor for [`Optimizer`](@ref):

```@example optimizer
using SimpleSolvers: NewtonOptimizerState, NewtonOptimizerCache, initialize!, linesearch_problem, Linesearch

_cache = NewtonOptimizerCache(x)
hes = Hessian(alg, obj, x)
grad = GradientAutodiff(obj.F, x)
state = NewtonOptimizerState(x)
update!(_cache, state, grad, x)
ls_obj = linesearch_problem(obj, grad, _cache)
ls_method = Static(.1)
ls = Linesearch(ls_obj, ls_method)
opt₂ = Optimizer(alg, obj, hes, _cache, ls_method)
```

If we want to solve the problem, we can call [`solve!`](@ref) on the [`Optimizer`](@ref) instance:

```@example optimizer
x₀ = copy(x)
state = NewtonOptimizerState(x)
solve!(x₀, state, opt)
```

Internally [`SimpleSolvers.solve!`](@ref) repeatedly calls [`SimpleSolvers.solver_step!`](@ref) until [`SimpleSolvers.meets_stopping_criteria`](@ref) is satisfied.

```@example optimizer
using SimpleSolvers: solver_step!

x = rand(3)
solver_step!(x, state, opt)
```

The function [`SimpleSolvers.solver_step!`](@ref) in turn does the following:

```julia
# update problem, hessian, state and result
update!(opt, state, x)

# solve H δx = - ∇f
# rhs is -g
ldiv!(direction(opt), hessian(opt), rhs(opt))

# apply line search
α = linesearch(opt)(linesearch_problem(problem(opt), gradient(opt), cache(opt), state))

# compute new minimizer
x .= compute_new_iterate(x, α, direction(opt))
cache(opt).x .= x
```

### Solving the Line Search Problem with Backtracking

Calling [`solve`](@ref) together with [`Linesearch`](@ref) (in this case [`SimpleSolvers.Backtracking`](@ref)) on an [`SimpleSolvers.LinesearchProblem`](@ref) in turn does:

```julia
α *= ls.p
```

as long as the [`SimpleSolvers.SufficientDecreaseCondition`](@ref) isn't satisfied. This condition checks the following:

```julia
fₖ₊₁ ≤ sdc.fₖ + sdc.c₁ * αₖ * sdc.pₖ' * sdc.gradₖ
```

`sdc` is first allocated as:

```@example optimizer
using SimpleSolvers: SufficientDecreaseCondition, linesearch, linesearch_problem, problem, method, cache, NullParameters # hide
ls = linesearch(opt)
α = method(ls).α₀
x₀ = zero(α)
grad = GradientAutodiff{Float64}(problem(opt).F, length(x))
params = (x = state.x̄, parameters = NullParameters())
sdc = SufficientDecreaseCondition(method(ls).c₁, ls_obj.F(0., params), ls_obj.D(0., params), alpha -> ls_obj.F(alpha, params))
```
