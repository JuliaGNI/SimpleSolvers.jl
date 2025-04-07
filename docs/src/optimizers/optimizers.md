# Optimizers

An [`Optimizer`](@ref) stores an [`OptimizationAlgorithm`](@ref), a [`MultivariateObjective`](@ref), the [`SimpleSolvers.OptimizerResult`](@ref) and a [`SimpleSolvers.NonlinearMethod`](@ref). Its purposes are:

```@example optimizer
using SimpleSolvers
using LinearAlgebra: norm
import Random # hide
Random.seed!(123) # hide

x = rand(3)
obj = MultivariateObjective(x -> norm(x - [0., 0., 1.]) ^ 2, x)
bt = Backtracking()
config = Options()
alg = Newton()
opt = Optimizer(x, obj; algorithm = alg, linesearch = bt, config = config)
```

## Optimization Algorithm

Internally the constructor for [`Optimizer`](@ref) calls [`SimpleSolvers.OptimizationAlgorithm`](@ref) and [`SimpleSolvers.OptimizerResult`](@ref). [`SimpleSolvers.OptimizationAlgorithm`](@ref) in turn calls [`SimpleSolvers.LinesearchState`](@ref) and [`Hessian`](@ref):

```@example optimizer
using SimpleSolvers: OptimizationAlgorithm

state = OptimizationAlgorithm(alg, obj, x; linesearch = bt)
```

The call above is equivalent to 

```@example optimizer
using SimpleSolvers: NewtonOptimizerCache, NewtonOptimizerState, LinesearchState, linesearch_objective, initialize!, update!

cache = NewtonOptimizerCache(x)
hess = Hessian(obj, x; mode = :autodiff)
initialize!(hess, x)
ls = LinesearchState(bt)
lso = linesearch_objective(obj, cache)

NewtonOptimizerState(obj, hess, ls, lso, cache)
```

Note that we use a separate objective here that only depends on ``\alpha`` (i.e. the step length for a single iteration) via [`SimpleSolvers.linesearch_objective`](@ref).

Also note that:

```@example optimizer
NewtonOptimizerState <: OptimizationAlgorithm
```

If we want to use the [`Optimizer`](@ref) we can call:

```@example optimizer
x₀ = copy(x)
update!(opt, rand(3))

solve!(x, opt)
```

Internally [`SimpleSolvers.solve!`](@ref) repeatedly calls [`SimpleSolvers.solver_step!`](@ref) (and [`SimpleSolvers.update!`](@ref)) until [`SimpleSolvers.meets_stopping_criteria`](@ref) is satisfied.

```@example optimizer
using SimpleSolvers: solver_step!

update!(opt, rand(3))
solver_step!(x, state)
```

The function [`SimpleSolvers.solver_step!`](@ref) in turn does the following:

```julia
update!(state, x)
ldiv!(direction(state), hessian(state), rhs(state))
ls = linesearch(state)
α = ls(state.ls_objective)
x .= x .+ α .* direction(state)
```

Calling an instance of [`SimpleSolvers.LinesearchState`](@ref) (in this case [`SimpleSolvers.BacktrackingState`](@ref)) on an [`SimpleSolvers.AbstractUnivariateObjective`](@ref) in turn does:

```julia
α *= ls.p
```

as long as the [`SimpleSolvers.SufficientDecreaseCondition`](@ref) isn't satisfied. This condition checks the following:

```julia
fₖ₊₁ ≤ sdc.fₖ + sdc.c₁ * αₖ * sdc.pₖ' * sdc.gradₖ
```

`sdc` is first allocated as:

```@example optimizer
using SimpleSolvers: SufficientDecreaseCondition # hide
α = ls.α₀
x₀ = zero(α)
y₀ = value!(lso, x₀)
d₀ = derivative!(lso, x₀)

sdc = SufficientDecreaseCondition(ls.ϵ, x₀, y₀, d₀, d₀, obj)
```