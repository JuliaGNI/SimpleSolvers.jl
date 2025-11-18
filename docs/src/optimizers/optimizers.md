# Optimizers

An [`Optimizer`](@ref) stores an [`OptimizationAlgorithm`](@ref), a [`OptimizerProblem`](@ref), the [`SimpleSolvers.OptimizerResult`](@ref) and a [`SimpleSolvers.NonlinearMethod`](@ref). Its purposes are:

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
using SimpleSolvers: NewtonOptimizerState, OptimizerResult, OptimizerStatus, initialize!

result = OptimizerResult(x, value!(obj, x))
status = OptimizerStatus{eltype(x)}()
initialize!(result, x)
state = NewtonOptimizerState(x; linesearch = bt)
hes = Hessian(alg, obj, x)
opt₂ = Optimizer(alg, obj, hes, status, result, state)
```

If we want to solve the problem, we can call [`solve!`](@ref) on the [`Optimizer`](@ref) instance:

```@example optimizer
x₀ = copy(x)

solve!(opt, x₀)
```

Internally [`SimpleSolvers.solve!`](@ref) repeatedly calls [`SimpleSolvers.solver_step!`](@ref) until [`SimpleSolvers.meets_stopping_criteria`](@ref) is satisfied.

```@example optimizer
using SimpleSolvers: solver_step!

x = rand(3)
solver_step!(opt, x)
```

The function [`SimpleSolvers.solver_step!`](@ref) in turn does the following:

```julia
# update problem, hessian, state and result
update!(opt, x)
# solve H δx = - ∇f
ldiv!(direction(opt), hessian(opt), rhs(opt))
# apply line search
α = linesearch(state(opt))(linesearch_problem(problem(opt), cache(opt)))
# compute new minimizer
x .= compute_new_iterate(x, α, direction(opt))
```

### Solving the Line Search Problem with Backtracking

Calling an instance of [`SimpleSolvers.LinesearchState`](@ref) (in this case [`SimpleSolvers.BacktrackingState`](@ref)) on an [`SimpleSolvers.LinesearchProblem`](@ref) in turn does:

```julia
α *= ls.p
```

as long as the [`SimpleSolvers.SufficientDecreaseCondition`](@ref) isn't satisfied. This condition checks the following:

```julia
fₖ₊₁ ≤ sdc.fₖ + sdc.c₁ * αₖ * sdc.pₖ' * sdc.gradₖ
```

`sdc` is first allocated as:

```@example optimizer
using SimpleSolvers: SufficientDecreaseCondition, linesearch, linesearch_problem, problem, cache # hide
ls = linesearch(opt)
α = ls.α₀
x₀ = zero(α)
lso = linesearch_problem(problem(opt), cache(opt))
y₀ = value!(lso, x₀)
d₀ = derivative!(lso, x₀)

sdc = SufficientDecreaseCondition(ls.ϵ, x₀, y₀, d₀, d₀, obj)
```