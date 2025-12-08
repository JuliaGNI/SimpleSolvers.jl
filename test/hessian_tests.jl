using SimpleSolvers
using SimpleSolvers: inverse_hessian, OptimizerCache
using Test
using Random: seed!

seed!(123)

n = 2
x = rand(n)
h = zeros(n,n)
T = eltype(x)

function F(x::Vector)
    1 + sum(x.^2)
end

function H!(h::Matrix, x::Vector)
    h .= 0
    for i in eachindex(x)
        h[i,i] = 2
    end
end

H!(h,x)

HPAD = HessianAutodiff{T}(F, n)
HPUS = HessianFunction{T}(H!, n)
H_BFGS = HessianBFGS{T}(F, n)
H_DFP = HessianDFP{T}(F, n)

function test_hessian(h1, h2, atol)
    for i in eachindex(h1,h2)
        @test h1[i] ≈ h2[i] atol=atol
    end
end

had = zero(h)
hus = zero(h)

HPAD(had, x)
HPUS(hus, x)

test_hessian(had, h, eps())
test_hessian(hus, h, zero(eltype(hus)))

BFGS_cache = OptimizerCache(BFGS(), x)
BFGS_state = OptimizerState(BFGS(), x)
BFGS_state.x̄ .= x
BFGS_state.ḡ .= GradientAutodiff(F, x)(x)

for α ∈ .9:-.01:1.0
    update!(BFGS_cache, BFGS_state, GradientAutodiff(F, x), α * x)
    update!(H_DFP, α * x)
end

@test inverse_hessian(BFGS_state) ≈ H_DFP.Q