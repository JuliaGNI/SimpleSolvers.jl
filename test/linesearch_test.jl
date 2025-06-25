using SimpleSolvers
using SimpleSolvers: LinesearchState, StaticState, compute_new_iterate, AbstractObjective, BierlaireQuadratic, BierlaireQuadraticState, QuadraticState2
using Test

f(x) = x^2 - 1
g(x) = 2x
δx(x) = - g(x) / 2

function make_linesearch_objective(x₀::Number)
    _f(α) = f(compute_new_iterate(x₀, α, δx(x₀)))
    _d(α) = g(compute_new_iterate(x₀, α, δx(x₀)))
    TemporaryUnivariateObjective(_f, _d)
end

function compute_next_iterate(ls::LinesearchState, x₀::T) where {T}
    ls_obj = make_linesearch_objective(x₀)
    α = ls(ls_obj)
    compute_new_iterate(x₀, α, δx(x₀))
end

function compute_next_iterate(ls::LinesearchState, x₀::T, n::Integer) where {T}
    x = x₀
    for _ in 1:n
        x = compute_next_iterate(ls, x)
    end
    x
end

function test_linesearch(algorithm::LinesearchMethod, n::Integer = 1)

    x₀ = -3.
    x₁ = +3.0
    xₛ =  0.0
    
    ls = LinesearchState(algorithm; x_abstol = zero(x₀))

    @test compute_next_iterate(ls, x₀, n) ≈ xₛ  atol=∛(2eps())
    @test compute_next_iterate(ls, x₀, n) ≈ xₛ  atol=∛(2eps())
end

@testset "$(rpad("Bracketing",80))" begin
    @test bracket_minimum(x -> x^2) == (-SimpleSolvers.DEFAULT_BRACKETING_s, +SimpleSolvers.DEFAULT_BRACKETING_s)
    @test bracket_minimum(x -> (x-1)^2) == (0.64, 2.56)
end

@testset "$(rpad("Static",80))" begin
    x₀ = -3.
    x₁ = +3.
    δx = x₁ - x₀
    x  = copy(x₀)

    ls = StaticState()

    @test ls == LinesearchState(Static())
    @test ls == LinesearchState(Static(1.0))

    @test ls() == 1.

    o1  = UnivariateObjective(f, x)
    o2  = UnivariateObjective(f, g, x)
    
    ls1 = Linesearch(algorithm = Static())
    ls2 = Linesearch(algorithm = Static(1.0))
    ls3 = Linesearch(algorithm = Static(0.8))

    @test ls1(o1) == ls1(o2) == ls1(f,g) == 1
    @test ls2(o1) == ls2(o2) == ls2(f,g) == 1
    @test ls3(o1) == ls3(o2) == ls3(f,g) == 0.8

end

@testset "$(rpad("Bisection", 80))" begin

    test_linesearch(Bisection(), 1)

end

@testset "$(rpad("Backtracking", 80))" begin

    test_linesearch(Backtracking(), 20)

end

@testset "$(rpad("Quadratic Linesearch (Bierlaire)", 80))" begin

    test_linesearch(BierlaireQuadratic(), 1)

end

@testset "$(rpad("Quadratic Linesearch (Derivative-Based)", 80))" begin

    test_linesearch(BierlaireQuadratic(), 1)

end