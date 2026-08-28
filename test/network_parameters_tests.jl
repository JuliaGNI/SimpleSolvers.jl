# The `NeuralNetworkParameters` extension: the two `Gradient` constructors and `alloc_h` for a
# parameter set.
#
# These three methods lived in `GeometricOptimizers` until 0.6.1, where they were type piracy --
# the functions are this package's and the parameter set is `NeuralNetworkParameters`', so neither
# side of the signature belonged to the package defining them. They are asserted here and not only
# downstream because a guarantee holds where it is asserted and nowhere else.

using ForwardDiff
using NeuralNetworkParameters
using SimpleSolvers
using SimpleSolvers: GradientAutodiff, GradientFunction, alloc_h
using Test

const nt = (L1 = (W = [1.0 2.0; 3.0 4.0], b = [5.0, 6.0]), L2 = (W = [7.0 8.0], b = [9.0]))
const ps = NetworkParameters(nt)

# A quadratic in every entry, so that the gradient is `2x` at every leaf and the answer can be
# written down rather than compared against a second implementation of the same walk.
F(x) = foldstorage((acc, s) -> acc + sum(abs2, s), 0.0, x)

@testset "the extension loads beside `NeuralNetworkParameters`" begin
    ext = Base.get_extension(SimpleSolvers, :SimpleSolversNeuralNetworkParametersExt)
    @test ext isa Module
    # Every one of the three methods has to come from the extension and not from a downstream
    # package: that is the property, and a count of sites goes stale where this does not.
    for m in (which(GradientAutodiff, Tuple{typeof(F),typeof(ps)}),
              which(GradientFunction, Tuple{typeof(F),Function,typeof(ps)}),
              which(alloc_h, Tuple{typeof(ps)}))
        @test m.module === ext
    end
end

@testset "`GradientAutodiff` on a $(name)" for (name, p) in (("NamedTuple", nt), ("NetworkParameters", ps))
    v, layout = flatten(p)
    grad = GradientAutodiff(F, p)

    # The gradient is taken of the flat form, so the functor is the ordinary vector one ...
    g = similar(v)
    grad(g, v)
    @test g ≈ 2v
    # ... and `unflatten` puts it back in the shape `F` was written for.
    @test flatten(unflatten(layout, g))[1] ≈ 2v

    # `F` really is called through the layout: it sees the tree, not the vector.
    @test grad(v) ≈ 2v
    @test ForwardDiff.gradient(_x -> F(unflatten(layout, _x)), v) ≈ 2v
end

@testset "`GradientFunction` is called on the flattened parameters" begin
    v, _ = flatten(ps)
    ∇F!(g, x) = (g .= 2 .* x)
    grad = GradientFunction(F, ∇F!, ps)
    g = similar(v)
    grad(g, v)
    @test g ≈ 2v
end

@testset "element type comes from the parameters and is not promoted" begin
    ps32 = NetworkParameters((L1 = (W = Float32[1 2; 3 4], b = Float32[5, 6]),))
    v, _ = flatten(ps32)
    @test eltype(v) === Float32
    grad = GradientAutodiff(F, ps32)
    @test grad isa GradientAutodiff{Float32}
    g = similar(v)
    grad(g, v)
    @test eltype(g) === Float32
end

@testset "`alloc_h` is sized by the flattening and not by the entry count" begin
    for p in (nt, ps)
        n = flatlength(p)
        H = alloc_h(p)
        @test size(H) == (n, n)
        @test eltype(H) === Float64
        @test all(isnan, H)
        # The failure this replaces: `length(p)` is the number of layers, i.e. 2 rather than 11.
        @test n != length(p)
    end
end

@testset "`GradientAutodiff` on a matrix argument" begin
    x = [1.0 2.0; 3.0 4.0]
    G(A) = sum(abs2, A)
    grad = GradientAutodiff(G, x)
    g = similar(vec(x))
    grad(g, vec(x))
    @test g ≈ 2vec(x)
    # A `reshape` and not a `vec` of the candidate: `G` has to see a matrix while `ForwardDiff` sees
    # a vector of `Dual`s.
    @test which(GradientAutodiff, Tuple{typeof(G),Matrix{Float64}}).module === SimpleSolvers
end
