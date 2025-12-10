# Initialization

Before we can use [`update!`](@ref) we have to initialize with [`SimpleSolvers.initialize!`](@ref)[^1].

[^1]: The different methods for [`SimpleSolvers.initialize!`](@ref) are however often called with the constructor of a `struct` (e.g. for [`SimpleSolvers.NewtonOptimizerCache`](@ref)).

Similar to [`update!`](@ref), [`SimpleSolvers.initialize!`](@ref) returns the first input argument as output. Examples include:
- [`SimpleSolvers.initialize!(::SimpleSolvers.NewtonSolverCache, ::AbstractVector)`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.NewtonOptimizerCache, ::AbstractVector)`](@ref).

We demonstrate this for an instance of [`SimpleSolvers.NewtonOptimizerCache`](@ref)[^2]:

[^2]: Here we remark that [`SimpleSolvers.NewtonOptimizerCache`](@ref) has five keys: `x`, `x̄`, `δ`, `g` and `rhs`. All of them are initialized with `NaN`s except for `x`. We also remark that the constructor, which is called by providing a single vector `x` as input argument, internally also calls [`SimpleSolvers.initialize!`](@ref).

```@example initialization
using SimpleSolvers # hide
using SimpleSolvers: initialize! # hide
using Random: seed! # hide
seed!(123) # hide
x = rand(3)
cache = SimpleSolvers.NewtonOptimizerCache(x)
initialize!(cache, x)
cache.g
```

## Reasoning behind Initialization with `NaN`s

We initialize with `NaN`s instead of with zeros (or other values) as this clearly divides the initialization from the numerical operations (which are done with [`update!`](@ref)).

## Alloc Functions