# Initialization

Before we can use [`SimpleSolvers.update!`](@ref) we have to initialize with [`SimpleSolvers.initialize!`](@ref)[^1].

[^1]: The different methods for [`SimpleSolvers.initialize!`](@ref) are however often called with the constructor of a `struct` (e.g. for [`NewtonOptimizerCache`](@ref)).

Similar to [`SimpleSolvers.update!`](@ref), [`SimpleSolvers.initialize!`](@ref) returns the first input argument as output.

One of the most central objects in `SimpleSolvers` are [`SimpleSolvers.initialize!`](@ref) routines. They can be used together with many different `types` and `structs`:
- [`SimpleSolvers.initialize!(::Hessian, ::AbstractVector)`](@ref): this routine exists for most [`Hessian`](@ref)s, i.e. for [`HessianFunction`](@ref), [`HessianAutodiff`](@ref), [`HessianBFGS`](@ref) and [`HessianDFP`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.NewtonSolverCache, ::AbstractVector)`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.NonlinearSolverStatus, ::AbstractVector, ::Base.Callable)`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.NewtonOptimizerCache, ::AbstractVector)`](@ref) and [`SimpleSolvers.update!(::SimpleSolvers.NewtonOptimizerCache, ::AbstractVector, ::AbstractVector)`](@ref),
- [`SimpleSolvers.initialize!(::SimpleSolvers.OptimizerResult, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).