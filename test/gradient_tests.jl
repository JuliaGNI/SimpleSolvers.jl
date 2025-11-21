using SimpleSolvers
using Test
import Random
Random.seed!(123)

const n = 2
const x = rand(n)
const g = 2x
const T = eltype(x)

function F(x::Vector)
    1 + sum(x.^2)
end

function ∇F!(g::Vector, x::Vector)
    g .= 0
    for i in eachindex(x,g)
        g[i] = 2x[i]
    end
end

# this is needed for the analytic gradient (called with `GradientFunction`)
const obj = OptimizerProblem(F, x; gradient = ∇F!)

const ∇PAD = GradientAutodiff{T}(F, n)
const ∇PFD = GradientFiniteDifferences{T}(F, n)
const ∇PUS = GradientFunction{T}(∇F!, n)

@test typeof(∇PAD) <: GradientAutodiff
@test typeof(∇PFD) <: GradientFiniteDifferences
@test typeof(∇PUS) <: GradientFunction


function test_grad(g1, g2, atol)
    for i in eachindex(g1,g2)
        @test g1[i] ≈ g2[i] atol=atol
    end
end


gad = zero(g)
gfd = zero(g)
gus = zero(g)

SimpleSolvers.compute_gradient!(gad, ∇PAD, x)
SimpleSolvers.compute_gradient!(gfd, ∇PFD, x)
@test_throws "You have to provide an `OptimizerProblem` when using `GradientFunction`!" SimpleSolvers.compute_gradient!(gus, ∇PUS, x)

# call GradientFunction
SimpleSolvers.compute_gradient!(obj, ∇PFD, x)

test_grad(gad, g, eps())
test_grad(gfd, g, 1E-7)
test_grad(gradient(obj), g, 0)

gad1 = zero(g)
gfd1 = zero(g)
gus1 = zero(g)

SimpleSolvers.compute_gradient!(gad1, x, F; mode = :autodiff)
SimpleSolvers.compute_gradient!(gfd1, x, F; mode = :finite)
SimpleSolvers.compute_gradient!(gus1, x, ∇F!; mode = :user)

test_grad(gad, gad1, 0)
test_grad(gfd, gfd1, 0)
test_grad(gus, gus1, 0)


gad2 = zero(g)
gfd2 = zero(g)

SimpleSolvers.gradient_ad!(gad2, x, F)
SimpleSolvers.gradient_fd!(gfd2, x, F)

test_grad(gad, gad2, 0)
test_grad(gfd, gfd2, 0)