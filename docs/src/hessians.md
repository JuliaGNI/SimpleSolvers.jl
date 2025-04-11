# Hessians

Hessians are a crucial ingredient in [`NewtonSolver`](@ref)s and [`SimpleSolvers.NewtonOptimizerState`](@ref)s.

```@example hessian
using SimpleSolvers
using LinearAlgebra: norm

x = rand(3)
obj = MultivariateObjective(x -> norm(x - vcat(0., 0., 1.))  ^ 2, x)
hes = HessianAutodiff(obj, x)
```

An instance of [`HessianAutodiff`](@ref) stores a Hessian matrix:

```@example hessian
hes.H
```

The instance of [`HessianAutodiff`](@ref) can be called:

```@example hessian
hes(x)
```

Or equivalently with:

```julia
update!(hes, x)
```

This updates `hes.H`:

```@example hessian
hes.H
```

## BFGS Hessian

```@example hessian
using SimpleSolvers: initialize!
hes = HessianBFGS(obj, x)
initialize!(hes, x)
```

For computational reasons we save the inverse of the Hessian, it can be accessed by calling `inv`:

```@example hessian
inv(hes)
```

Similarly to [`HessianAutodiff`](@ref) we can call [`SimpleSolvers.update!`](@ref):

```@example hessian
using SimpleSolvers: update!

update!(hes, x)
```