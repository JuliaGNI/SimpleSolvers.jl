# Initialization

Before we can use [`update!`](@ref) we have to initialize with [`SimpleSolvers.initialize!`](@ref)[^1].

[^1]: The different methods for [`SimpleSolvers.initialize!`](@ref) are however often called with the constructor of a `struct` (e.g. for [`SimpleSolvers.NewtonOptimizerCache`](@ref)).

Similar to [`update!`](@ref), [`SimpleSolvers.initialize!`](@ref) returns the first input argument as output. Examples include:
- [`SimpleSolvers.initialize!(::Hessian, ::AbstractVector)`](@ref): this routine exists for most [`Hessian`](@ref)s, i.e. for [`HessianFunction`](@ref), [`HessianAutodiff`](@ref), [`HessianBFGS`](@ref) and [`HessianDFP`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.NewtonSolverCache, ::AbstractVector)`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.NewtonOptimizerCache, ::AbstractVector)`](@ref).

We demonstrate these examples in code. First for an instance of [`SimpleSolvers.Hessian`](@ref):

```@example initialization
using SimpleSolvers # hide
using SimpleSolvers: initialize! # hide
using LinearAlgebra: norm
import Random # hide
Random.seed!(123) # hide

x = rand(3)
obj = OptimizerProblem(x -> norm(x - vcat(0., 0., 1.)) ^ 2, x)
bt = Backtracking() # hide
alg = Newton() # hide
# opt = Optimizer(x, obj; algorithm = alg, linesearch = bt, config = config) # hide

hes = HessianAutodiff(obj, x)
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

## Reasoning behind Initialization with `NaN`s

We initialize with `NaN`s instead of with zeros (or other values) as this clearly divides the initialization from the numerical operations (which are done with [`update!`](@ref)).

## Alloc Functions