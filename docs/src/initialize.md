# Initialization

Before we can use [`SimpleSolvers.update!`](@ref) we have to initialize with [`SimpleSolvers.initialize!`](@ref)[^1].

[^1]: The different methods for [`SimpleSolvers.initialize!`](@ref) are however often called with the constructor of a `struct` (e.g. for [`SimpleSolvers.NewtonOptimizerCache`](@ref)).

Similar to [`SimpleSolvers.update!`](@ref), [`SimpleSolvers.initialize!`](@ref) returns the first input argument as output.

One of the most central objects in `SimpleSolvers` are [`SimpleSolvers.initialize!`](@ref) routines. They can be used together with many different `types` and `structs`:
- [`SimpleSolvers.initialize!(::Hessian, ::AbstractVector)`](@ref): this routine exists for most [`Hessian`](@ref)s, i.e. for [`HessianFunction`](@ref), [`HessianAutodiff`](@ref), [`HessianBFGS`](@ref) and [`HessianDFP`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.NewtonSolverCache, ::AbstractVector)`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.NonlinearSolverStatus, ::AbstractVector, ::Base.Callable)`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.NewtonOptimizerCache, ::AbstractVector)`](@ref).

We show a few examples. First for an instance of [`SimpleSolvers.Hessian`](@ref):

```@example initialization
using SimpleSolvers # hide
using SimpleSolvers: initialize! # hide
using LinearAlgebra: norm
import Random # hide
Random.seed!(123) # hide

x = rand(3)
obj = MultivariateObjective(x -> norm(x - vcat(0., 0., 1.)) ^ 2, x)
bt = Backtracking() # hide
config = Options() # hide
alg = Newton() # hide
# opt = Optimizer(x, obj; algorithm = alg, linesearch = bt, config = config) # hide

hes = Hessian(obj, x; mode = :autodiff)
initialize!(hes, x)
hes.H
```

For an instance of [`SimpleSolvers.NewtonOptimizerCache`](@ref)[^2]:

[^2]: Here we remark that [`SimpleSolvers.NewtonOptimizerCache`](@ref) has five keys: `x`, `x̄`, `δ`, `g` and `rhs`. All of them are initialized with `NaN`s except for `x`. We also remark that the constructor, which is called by providing a single vector `x` as input argument, internally also calls [`SimpleSolvers.initialize!`](@ref).

```@example initialization
cache = SimpleSolvers.NewtonOptimizerCache(x)
initialize!(cache, x)
cache.g
```

## Clear Routines

For [`SimpleSolvers.OptimizerResult`](@ref) and [`SimpleSolvers.OptimizerStatus`](@ref) the [`SimpleSolvers.clear!`](@ref) routines are used instead of the [`SimpleSolvers.initialize!`](@ref) routines.

## Reasoning behind Initialization with `NaN`s

We initialize with `NaN`s instead of with zeros (or other values) as this clearly divides the initialization from the numerical operations (which are done with [`SimpleSolvers.update!`](@ref)).