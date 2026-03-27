# Hessians

Hessians come in essentially two forms in `SimpleSolvers`: *exact* Hessians and *iterative* Hessians.

Exact [`Hessian`](@ref)s are used with the `Newton` method. They encompass
- [`HessianFunction`](@ref) and
- [`HessianAutodiff`](@ref).
