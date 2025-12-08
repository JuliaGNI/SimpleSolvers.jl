# Updates

One of the most central objects in `SimpleSolvers` are [`update!`](@ref) routines. They can be used together with many different `types` and `structs`:
- [`SimpleSolvers.update!(::Hessian, ::AbstractVector)`](@ref): this routine exists for most [`Hessian`](@ref)s, i.e. for [`HessianFunction`](@ref), [`HessianAutodiff`](@ref), [`HessianBFGS`](@ref) and [`HessianDFP`](@ref),
- [`SimpleSolvers.update!(::SimpleSolvers.NewtonSolverCache, ::AbstractVector)`](@ref),
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
hes = HessianAutodiff(f, x)
H = SimpleSolvers.alloc_h(x)
```

And then update:

```@example update
hes(H, x)

H
```

### `NewtonOptimizerCache`

In order to update an instance of [`SimpleSolvers.NewtonOptimizerCache`](@ref) we have to supply a value of the [`Gradient`](@ref) and the [`Hessian`](@ref) in addition to `x`:

```@example update
using SimpleSolvers: initialize!, NewtonOptimizerCache # hide
grad = GradientAutodiff(f, x)
cache = NewtonOptimizerCache(x)
state = NewtonOptimizerState(x)
update!(cache, state, grad, hes, x)
```

!!! info
    We note that when calling `update!` on the `NewtonOptimizerCache`, the Hessian `hes` is not automatically updated! This has to be done manually.

!!! info
    Calling `update!` on the `NewtonOptimizerCache` updates everything except `x` as this in general requires another line search!

!!! info
    When updating the `cache` we also need to supply the `state`. This is needed for the `direction`.


### `OptimizerResult`

!!! warn
    `NewtonOptimizerCache`, `OptimizerResult` and `NewtonOptimizerState` (through `OptimizerProblem`) all store things that are somewhat similar, for example `x`. This may make it somewhat difficult to keep track of all the things that happen during optimization.

An [`Optimizer`](@ref) stores a [`OptimizerProblem`](@ref), an [`SimpleSolvers.OptimizerResult`](@ref) and an [`OptimizerState`](@ref) (and therefore the [`OptimizerProblem`](@ref) again). We also give an example:

```@example update
obj = OptimizerProblem(f, x)
opt = Optimizer(x, obj; algorithm = Newton())

update!(opt, state, x)
```

Equivalent to calling [`update!`](@ref) on [`SimpleSolvers.OptimizerResult`](@ref), the diagnostics cannot be computed with only one iterations; we have to compute a second one:

```@example update
x₂ = [.9, 0., 0.]
update!(opt, state, x₂)
```

We note that simply calling [`update!`](@ref) on an instance of [`SimpleSolvers.Optimizer`](@ref) is not enough to perform a complete iteration since the computation of a new ``x`` requires a [line search](@ref "Line Search") procedure in general.

We also note that [`update!`](@ref) always returns the first input argument.