
using SimpleSolvers
using Test

import SimpleSolvers: get_config, set_config


n = 1
x = [2.]
g = 2x
T = eltype(x)


function F(x::Vector)
    1 + sum(x.^2)
end

function ∇F!(g::Vector, x::Vector)
    g .= 0
    for i in eachindex(x,g)
        g[i] = 2x[i]
    end
end


∇PAD = GradientParameters{T}(F, n; mode = :autodiff, diff_type = :forward)
∇PFD = GradientParameters{T}(F, n; mode = :autodiff, diff_type = :finite)
∇PUS = GradientParameters{T}(∇F!, n; mode = :user)

@test typeof(∇PAD) <: GradientParametersAD
@test typeof(∇PFD) <: GradientParametersFD
@test typeof(∇PUS) <: GradientParametersUser


function test_grad(g1, g2, atol)
    for i in eachindex(g1,g2)
        @test g1[i] ≈ g2[i] atol=atol
    end
end


gad = zero(g)
gfd = zero(g)
gus = zero(g)

compute_gradient!(gad, x, ∇PAD)
compute_gradient!(gfd, x, ∇PFD)
compute_gradient!(gus, x, ∇PUS)

test_grad(gad, g, eps())
test_grad(gfd, g, 1E-7)
test_grad(gus, g, 0)


gad1 = zero(g)
gfd1 = zero(g)
gus1 = zero(g)

compute_gradient!(gad1, x, F; mode = :autodiff, diff_type = :forward)
compute_gradient!(gfd1, x, F; mode = :autodiff, diff_type = :finite)
compute_gradient!(gus1, x, ∇F!; mode = :user)

test_grad(gad, gad1, 0)
test_grad(gfd, gfd1, 0)
test_grad(gus, gus1, 0)


gad2 = zero(g)
gfd2 = zero(g)

compute_gradient_ad!(gad2, x, F)
compute_gradient_fd!(gfd2, x, F)

test_grad(gad, gad2, 0)
test_grad(gfd, gfd2, 0)
