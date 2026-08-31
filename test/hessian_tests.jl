using SimpleSolvers
using Test
using Random: seed!

seed!(123)

n = 2
x = rand(n)
h = zeros(n, n)
T = eltype(x)

function F(x::Vector)
    1 + sum(x .^ 2)
end

function H!(h::Matrix, x::Vector)
    h .= 0
    for i in eachindex(x)
        h[i, i] = 2
    end
end

H!(h, x)

HPAD = HessianAutodiff{T}(F, n)
HPUS = HessianFunction{T}(H!, n)

function test_hessian(h1, h2, atol)
    for i in eachindex(h1, h2)
        @test h1[i] ≈ h2[i] atol = atol
    end
end

had = zero(h)
hus = zero(h)

HPAD(had, x)
HPUS(hus, x)

test_hessian(had, h, eps())
test_hessian(hus, h, zero(eltype(hus)))

# The exported `check_hessian` diagnostic writes to an `io` argument, so its
# output is captured with `sprint` (no stdout redirection) and checked directly.
@testset "check_hessian writes the correct diagnostics" begin
    H = [1.0 √2.0; √2.0 3.0]                      # cond = 7+4√3 ≈ 13.9282, det = 1
    out = replace(sprint(check_hessian, H), r"[ \t]+" => " ")
    @test occursin("Condition Number of Hessian: 13.9282", out)
    @test occursin("Determinant of Hessian: 1.0", out)
    @test occursin("minimum(|Hessian|): 1.0", out)
    @test occursin("maximum(|Hessian|): 3.0", out)

    # the convenience form without `io` writes to stdout and forwards keywords
    # (called silently; content is covered by the `sprint` checks above)
    @test redirect_stdout(() -> check_hessian(H), devnull) === nothing
    @test redirect_stdout(() -> check_hessian(H; digits = 3), devnull) === nothing
end
