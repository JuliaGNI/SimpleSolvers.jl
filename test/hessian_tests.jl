using SimpleSolvers
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

for α ∈ .1:-.01:0.0
    update!(H_BFGS, α * x)
    update!(H_DFP, α * x)
end

@test H_BFGS.Q ≈ H_DFP.Q