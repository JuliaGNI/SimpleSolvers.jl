# Jacobians

The supertype [`Jacobian`](@ref) comprises different ways of taking Jacobians:
- [`JacobianFunction`](@ref),
- [`JacobianAutodiff`](@ref),
- [`JacobianFiniteDifferences`](@ref).

We first start by showing [`JacobianAutodiff`](@ref):

```@example jacobian
using SimpleSolvers, Random; using SimpleSolvers: JacobianAutodiff, Jacobian, JacobianFunction, JacobianFiniteDifferences; Random.seed!(123) # hide
# the input and output dimensions of this function are the same
F(y::AbstractArray, x::AbstractArray, params) = y .= tanh.(x)
dim = 3
x = rand(dim)
jac = JacobianAutodiff{eltype(x)}(F, dim)
```

When calling an instance of [`Jacobian`](@ref) we can use the function [`compute_jacobian!`]:

```@example jacobian
params = nothing
j = zeros(dim, dim)
compute_jacobian!(j, x, jac, params)
```

This is equivalent to calling:

```@example jacobian
jac(j, x, params)
```