using SimpleSolvers
using SimpleSolvers: initialize!, solver_step!
using Test
using Random
using ForwardDiff
Random.seed!(123)

f(x::T) where {T<:Number} = abs(tanh(x - 0.1)) # exp(x) * (x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
f2(x) = x - f(x)
F(x) = f2.(x)
F!(y, x, params) = y .= F(x)

n = 1
x₀ = rand(n)
root₁ = 0.1

for T ∈ (Float64, Float32)
    # x = T.(copy(x₀))
    x = ones(T, n)
    # println(x)
    y = F(x)
    # println(x)
    it = FixedPointIterator(x; F=F!)
    # println(x)

    @test config(it) == it.config
    @test status(it) == it.status

    solve!(it, x)
    # println(x)
    for _x in x
        @test _x ≈ T(root₁) atol = ∛(2eps(T))
    end
end
