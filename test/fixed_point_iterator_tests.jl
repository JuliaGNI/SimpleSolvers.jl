using ForwardDiff
using Random
using SimpleSolvers
using Test

using NaNMath: log
using SimpleSolvers: initialize!, solver_step!

Random.seed!(123)

F(x::T) where {T<:Number} = abs(tanh(x - T(0.1))) # exp(x) * (x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
F!(y, x, params) = y .= F.(x)

n = 1
x₀ = rand(n)
root₁ = 0.1

for T ∈ (Float64, Float32)
    # x = T.(copy(x₀))
    x = ones(T, n)
    # println(x)
    y = F.(x)
    # println(x)
    it = FixedPointIterator(x, y; F=F!)
    # println(x)

    @test config(it) == it.config

    solve!(x, it)
    # println(x)
    for _x in x
        @test _x ≈ T(root₁) atol = ∛(2eps(T))
    end
end

# test new constructor
for T ∈ (Float64, Float32)
    x = ones(T, n)
    y = F.(x)
    it = NonlinearSolver(PicardMethod(), x, y; F=F!)
    # println(x)

    @test config(it) == it.config

    solve!(x, it)
    for _x in x
        @test _x ≈ T(root₁) atol = ∛(2eps(T))
    end
end
