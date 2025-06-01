# Updates

One of the most central objects in `SimpleSolvers` are [`update!`](@ref) routines. They can be used together with many different `types` and `structs`:
- [`SimpleSolvers.update!(::Hessian, ::AbstractVector)`](@ref): this routine exists for most [`Hessian`](@ref)s, i.e. for [`HessianFunction`](@ref), [`HessianAutodiff`](@ref), [`HessianBFGS`](@ref) and [`HessianDFP`](@ref),
- [`SimpleSolvers.update!(::SimpleSolvers.NewtonSolverCache, ::AbstractVector)`](@ref),
- [`SimpleSolvers.update!(::SimpleSolvers.NonlinearSolverStatus, ::AbstractVector, ::Base.Callable)`](@ref),
- [`SimpleSolvers.update!(::SimpleSolvers.NewtonOptimizerCache, ::AbstractVector, ::AbstractVector, ::Hessian)`](@ref),
- [`SimpleSolvers.update!(::SimpleSolvers.NewtonOptimizerState, ::AbstractVector)`](@ref).
- [`SimpleSolvers.update!(::SimpleSolvers.OptimizerResult, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).

So [`update!`](@ref) always takes an object that has to be updated and a single vector in the simplest case. For some methods more arguments need to be provided. 

## Examples

### `Hessian`

If we look at the case of the [Hessian](@ref "Hessians"), we store a matrix ``H`` that has to be updated in every iteration. We first initialize the matrix[^1]:

[^1]: The constructor uses the function [`SimpleSolvers.initialize!`](@ref).

```@example update
using SimpleSolvers # hide
using LinearAlgebra: norm # hide
f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2)
x = [1., 0., 0.]
hes = Hessian(f, x; mode = :autodiff)
hes.H
```

And then update:

```@example update
update!(hes, x)

hes.H
```

### `NewtonOptimizerCache`

In order to update an instance of [`SimpleSolvers.NewtonOptimizerCache`](@ref) we have to supply a value of the [`Gradient`](@ref) and the [`Hessian`](@ref) in addition to `x`:

```@example update
using SimpleSolvers: initialize!, NewtonOptimizerCache # hide
grad = Gradient(f, x; mode = :autodiff)
cache = NewtonOptimizerCache(x)
update!(cache, x, grad, hes)
```

!!! info
    We note that when calling `update!` on the `NewtonOptimizerCache`, the Hessian `hes` is not automatically updated! This has to be done manually.

!!! info
    Calling `update!` on the `NewtonOptimizerCache` updates everything except `x` as this in general requires another line search!

In order that we do not have to update the [`Hessian`](@ref) and the [`SimpleSolvers.NewtonOptimizerCache`](@ref) separately we can use [`SimpleSolvers.NewtonOptimizerState`](@ref):

```@example update
using SimpleSolvers: NewtonOptimizerState # hide
obj = MultivariateObjective(f, x)
state = NewtonOptimizerState(x)
update!(state, x, Gradient(obj), hes)
```

### `OptimizerResult`

We also show how to update an instance of [`SimpleSolvers.OptimizerResult`](@ref):

```@example update
using SimpleSolvers: OptimizerResult # hide

result = OptimizerResult(x, obj)

update!(result, x, obj, grad)
```

Note that the residuals are still `NaN`s here. In order to get proper values for these we have to *perform two updating steps*:

```@example update
x₂ = [.9, 0., 0.]
update!(result, x₂, obj, grad)
```

!!! warn
    `NewtonOptimizerCache`, `OptimizerResult` and `NewtonOptimizerState` (through `MultivariateObjective`) all store things that are somewhat similar, for example `x`. This may make it somewhat difficult to keep track of all the things that happen during optimization.

An [`Optimizer`](@ref) stores a [`MultivariateObjective`](@ref), an [`SimpleSolvers.OptimizerResult`](@ref) and an [`OptimizationAlgorithm`](@ref) (and therefore the [`MultivariateObjective`](@ref) again). We also give an example:

```@example update
opt = Optimizer(x, obj)

update!(opt, x)
```

Equivalent to calling [`update!`](@ref) on [`SimpleSolvers.OptimizerResult`](@ref), the diagnostics cannot be computed with only one iterations; we have to compute a second one:

```@example update
x₂ = [.9, 0., 0.]
update!(opt, x₂)
```

We note that simply calling [`update!`](@ref) on an instance of [`SimpleSolvers.Optimizer`](@ref) is not enough to perform a complete iteration since the computation of a new ``x`` requires a [line search](@ref "Line Search") procedure in general.

We also note that [`update!`](@ref) always returns the first input argument.