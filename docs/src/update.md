# Updates

One of the most central objects in `SimpleSolvers` are [`SimpleSolvers.update!`](@ref) routines. They can be used together with many different `types` and `structs`:
- [`SimpleSolvers.update!(::Hessian, ::AbstractVector)`](@ref): this routine exists for most [`Hessian`](@ref)s, i.e. for [`HessianFunction`](@ref), [`HessianAutodiff`](@ref), [`HessianBFGS`](@ref) and [`HessianDFP`](@ref),
- [`SimpleSolvers.update!(::SimpleSolvers.NewtonSolverCache, ::AbstractVector)`](@ref),
- [`SimpleSolvers.update!(::SimpleSolvers.NonlinearSolverStatus, ::AbstractVector, ::Base.Callable)`](@ref),
- [`SimpleSolvers.update!(::SimpleSolvers.NewtonOptimizerCache, ::AbstractVector)`](@ref) and [`SimpleSolvers.update!(::SimpleSolvers.NewtonOptimizerCache, ::AbstractVector, ::AbstractVector)`](@ref),
- [`SimpleSolvers.update!(::SimpleSolvers.OptimizerResult, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).

So [`SimpleSolvers.update!`](@ref) always takes an object that has to be updated and a single vector in the simplest case. For some methods more arguments need to be provided. If we look at the case of the [Hessian](@ref "Hessians"), we store a matrix ``H`` that has to be updated in every iteration. We first initialize the matrix:

```@example hessian_update
using SimpleSolvers # hide
using SimpleSolvers: update! # hide
using LinearAlgebra: norm # hide
f = x -> norm(x .^ 3)
x = [1., 0., 0.]
hes = Hessian(f, x)
hes.H
```

And then update:

```@example hessian_update
update!(hes, x)

hes.H
```

We also note that [`update!`](@ref) always returns the first input argument.