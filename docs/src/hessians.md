# Hessians

Hessians are a crucial ingredient in [`NewtonSolver`](@ref)s and [`SimpleSolvers.NewtonOptimizerState`](@ref)s.

```@example hessian
using SimpleSolvers
using LinearAlgebra: norm

x = rand(3)
obj = OptimizerProblem(x -> norm(x - vcat(0., 0., 1.))  ^ 2, x)
hes = HessianAutodiff(obj, x)
```

The instance of [`HessianAutodiff`](@ref) can be called:

```@example hessian
hes(x)
```

Or alternative in-place:

```@example hessian
H = SimpleSolvers.alloc_h(x)
hes(H, x)
H
```