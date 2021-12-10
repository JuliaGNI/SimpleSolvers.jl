
using SimpleSolvers
using Test

import SimpleSolvers: get_config, set_config


n = 2
T = Float64
x = rand(T, n)
h = zeros(T, n, n)


function F(x::Vector)
    sum(x.^2)
end

function H!(h::Matrix, x::Vector)
    h .= 0
    for i in eachindex(x)
        h[i,i] = 2
    end
end


H!(h,x)


HPAD = HessianParameters{T}(F, n; mode = :autodiff)
HPUS = HessianParameters{T}(H!, n; mode = :user)

@test typeof(HPAD) <: HessianParametersAD
@test typeof(HPUS) <: HessianParametersUser


function test_hessian(h1, h2, atol)
    for i in eachindex(h1,h2)
        @test h1[i] ≈ h2[i] atol=atol
    end
end


had = zero(h)
hus = zero(h)

compute_hessian!(had, x, HPAD)
compute_hessian!(hus, x, HPUS)

test_hessian(had, h, eps())
test_hessian(hus, h, 0)


had1 = zero(h)
hus1 = zero(h)

compute_hessian!(had1, x, F; mode = :autodiff)
compute_hessian!(hus1, x, H!; mode = :user)

test_hessian(had, had1, 0)
test_hessian(hus, hus1, 0)


had2 = zero(h)

compute_hessian_ad!(had2, x, F)

test_hessian(had, had2, 0)