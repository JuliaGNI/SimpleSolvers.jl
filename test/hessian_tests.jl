using SimpleSolvers
using Test

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

@test typeof(HPAD) <: HessianAutodiff
@test typeof(HPUS) <: HessianFunction

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