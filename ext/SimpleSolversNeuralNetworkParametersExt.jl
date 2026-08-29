"""
    SimpleSolversNeuralNetworkParametersExt

The `Gradient` and `alloc_h` methods for a set of neural network parameters.

A solver here works on a flat `AbstractVector`. A set of neural network parameters is a tree, so
something has to flatten it, hand the solver the vector, and put the answer back in the shape the
objective was written for. `NeuralNetworkParameters` provides exactly that
([`NeuralNetworkParameters.flatten`](@extref) and a [`NeuralNetworkParameters.ParameterLayout`](@extref)
that is a *value*), and these three methods are the whole of the seam.

`GeometricOptimizers` carried all three until 0.6.1. `GradientAutodiff`, `GradientFunction` and
`alloc_h` are this package's functions and a parameter set is `NeuralNetworkParameters`', so those
methods owned neither side of their own signatures; a weak dependency puts them with the functions
and costs nothing to anyone who does not load `NeuralNetworkParameters`.
"""
module SimpleSolversNeuralNetworkParametersExt

using NeuralNetworkParameters: NetworkParameters, flatten, unflatten, flatlength, mapparameters,
    parameter_eltype

using SimpleSolvers: _nan
import SimpleSolvers: GradientAutodiff, GradientFunction, alloc_h

"""
    GradientAutodiff(F, ps::NetworkParameters)

The automatic-differentiation gradient of `F` at a set of neural network parameters.

`ps` is flattened once here and the layout is captured in the closure, so `ForwardDiff` sees the flat
vector it wants while `F` sees the shape it was written for. The element type comes from `ps` itself
rather than defaulting to `Float64`, which is what silently promoted a `Float32` network under
`ParameterHandling`.

A [`NeuralNetworkParameters.ParameterLayout`](@extref) is a *value* and not a chain of closures,
which is the difference that matters here: one closure, type stable, and the same layout on every
call.
"""
function GradientAutodiff(F, ps::NetworkParameters)
    v, layout = flatten(ps)
    GradientAutodiff(_x -> F(unflatten(layout, _x)), v)
end

"""
    GradientFunction(F, ∇F!, ps::NetworkParameters)

The gradient of `F` at a set of neural network parameters, computed by the supplied `∇F!`.

`∇F!` is called on the *flattened* parameters, i.e. on `flatten(ps)[1]`, which is what lets a caller
hand over a `Zygote` gradient it has already written against the flat vector.
"""
function GradientFunction(F, ∇F!, ps::NetworkParameters)
    v, layout = flatten(ps)
    GradientFunction(_x -> F(unflatten(layout, _x)), ∇F!, v)
end

"""
    alloc_h(ps::NetworkParameters)

`NaN`s of the size of the Hessian with respect to a set of neural network parameters.

# Implementation

The fallback is `x * x'`, which for anything but a plain vector is the wrong shape. `Q` is sized by
the *intrinsic* dimension of the parameters instead — the length of the flattening of the space the
direction lives in, which for a manifold leaf is its horizontal lift and not its dense matrix. For a
bare `StiefelManifold` of size `(3, 1)` the fallback gives `3 × 3` where the lift has 2 free
parameters, and the cache and the state then disagree about how big `Q` is.

`flatlength` of a *zero* of `ps` and not of `ps` itself, which is what makes that hold: `zero` of a
leaf is a point of its tangent space, so the walk below reaches the lift's dimension rather than the
dense one. `mapparameters` applies `zero` to the leaves at whatever depth they are and does not
rebuild them, so a manifold leaf comes back as its lift rather than being forced into a manifold
again.
"""
function alloc_h(ps::NetworkParameters)
    z = mapparameters(zero, ps)
    n = flatlength(z)
    fill(_nan(parameter_eltype(z)), n, n)
end

end
