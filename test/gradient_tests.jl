using SimpleSolvers
using Test
import Random
Random.seed!(123)

const n = 2
const x = rand(n)
const g = 2x
const T = eltype(x)

function F(x::Vector)
    1 + sum(x .^ 2)
end

function ∇F!(g::Vector, x::Vector)
    g .= 0
    for i in eachindex(x, g)
        g[i] = 2x[i]
    end
    g
end

const ∇PAD = GradientAutodiff{T}(F, n)
const ∇PFD = GradientFiniteDifferences{T}(F, n)
const ∇PUS = GradientFunction{T}(F, ∇F!, n)

@test typeof(∇PAD) <: GradientAutodiff
@test typeof(∇PFD) <: GradientFiniteDifferences
@test typeof(∇PUS) <: GradientFunction


function test_grad(g1, g2, atol)
    for i in eachindex(g1, g2)
        @test g1[i] ≈ g2[i] atol = atol
    end
end


gad = zero(g)
gfd = zero(g)
gus = zero(g)

∇PAD(gad, x)
∇PFD(gfd, x)
∇PUS(gus, x)

test_grad(gad, g, eps())
test_grad(gfd, g, 1E-7)
test_grad(gus, g, zero(eltype(g)))

gad1 = zero(g)
gfd1 = zero(g)
gus1 = zero(g)

∇PAD(gad1, x)
∇PFD(gfd1, x)
∇PUS(gus1, x)

test_grad(gad, gad1, 0)
test_grad(gfd, gfd1, 0)
test_grad(gus, gus1, 0)

# §2.2 regression: the default finite-difference step used to bake in the
# Float64 machine epsilon (`8sqrt(eps())`) regardless of the working precision,
# so a Float32 finite-difference gradient used a ~1 ulp step and produced
# garbage.  `default_ϵ(T) = 8sqrt(eps(T))` is now precision-aware.
@testset "Float32 finite-difference gradient accuracy (§2.2)" begin
    F32(x) = 1 + sum(x .^ 2)
    x32 = Float32[0.3, 0.7]
    g32 = 2x32
    ∇fd32 = GradientFiniteDifferences{Float32}(F32, 2)
    gfd32 = zero(g32)
    ∇fd32(gfd32, x32)
    @test eltype(gfd32) == Float32
    # with a precision-aware step this is accurate to ~sqrt(eps(Float32)); a
    # Float64-epsilon step would be off by orders of magnitude.
    for i in eachindex(gfd32, g32)
        @test gfd32[i] ≈ g32[i] atol = 1e-3
    end
end
