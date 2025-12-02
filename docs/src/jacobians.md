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

And the functor:

```@example jacobian
j = zeros(3, 3)
jac(j, x, nothing)
```