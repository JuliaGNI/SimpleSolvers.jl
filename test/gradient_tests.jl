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

# Regression: the default finite-difference step used to bake in the
# Float64 machine epsilon (`8sqrt(eps())`) regardless of the working precision,
# so a Float32 finite-difference gradient used a ~1 ulp step and produced
# garbage.  `default_ϵ(T) = 8sqrt(eps(T))` is now precision-aware.
@testset "Float32 finite-difference gradient accuracy" begin
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

# Regression: the `GradientFunction` functor used to require both
# arguments to have the *identical* concrete type (`g::VT, x::VT`), so a
# `Vector`/`SubArray` pair (same eltype, different container) hit the misleading
# "Functor not implemented." fallback.  The arguments are now two independent
# `AbstractVector{T}`.
@testset "GradientFunction accepts independent container types" begin
    # a derivative closure that accepts any AbstractVector (the fix is that the
    # SimpleSolvers functor no longer forces g and x to the *same* concrete type)
    ∇g!(g::AbstractVector, x::AbstractVector) = (g .= 2 .* x; g)
    ∇lenient = GradientFunction{T}(F, ∇g!, n)
    M = [0.3 0.0; 0.7 0.0]
    xsub = @view M[:, 1]        # SubArray{Float64}
    gv = zeros(2)              # Vector{Float64}
    @test typeof(gv) != typeof(xsub)
    ∇lenient(gv, xsub)          # would previously hit the "Functor not implemented" fallback
    @test gv ≈ 2 .* collect(xsub)
end

# Regression: `GradientFiniteDifferences{T}` used to restrict `nx::Int`,
# while its siblings accept any `::Integer`.
@testset "GradientFiniteDifferences{T} accepts any Integer nx" begin
    ∇int = GradientFiniteDifferences{Float64}(F, Int32(2))
    @test ∇int isa GradientFiniteDifferences
    gg = zeros(2)
    ∇int(gg, x)
    @test gg ≈ 2x atol = 1e-6
end

# The generic `Gradient` functor fallback (which raised a
# home-grown "Functor not implemented." error and masked `MethodError`s) was
# removed.  A `Gradient` subtype without a functor now yields a proper
# `MethodError`, so `hasmethod`/`applicable` report the truth.
@testset "Gradient functor fallback removed" begin
    struct UnimplementedGradient{S} <: SimpleSolvers.Gradient{S} end
    ug = UnimplementedGradient{Float64}()
    @test !hasmethod(ug, Tuple{Vector{Float64},Vector{Float64}})
    @test_throws MethodError ug(zeros(2), zeros(2))
end
