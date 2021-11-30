
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


set_config(:gradient_autodiff, true)
∇PAD = getGradientParameters(nothing, F, T, n)

set_config(:gradient_autodiff, false)
∇PFD = getGradientParameters(nothing, F, T, n)

set_config(:gradient_autodiff, true)
∇PUS = getGradientParameters(∇F!, F, T, n)

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

computeGradient(x, gad, ∇PAD)
computeGradient(x, gfd, ∇PFD)
computeGradient(x, gus, ∇PUS)

test_grad(gad, g, eps())
test_grad(gfd, g, 1E-7)
test_grad(gus, g, 0)

gad2 = zero(g)
gfd2 = zero(g)

computeGradientAD(x, gad2, F)
computeGradientFD(x, gfd2, F, get_config(:gradient_fd_ϵ))

test_grad(gad, gad2, 0)
test_grad(gfd, gfd2, 0)
