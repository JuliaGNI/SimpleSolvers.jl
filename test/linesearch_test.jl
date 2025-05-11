using SimpleSolvers
using SimpleSolvers: LinesearchState, StaticState, compute_new_iterate, AbstractObjective, BierlaireQuadratic, BierlaireQuadraticState
using Test

include("optimizers_problems.jl")

f(x) = x^2 - 1
g(x) = 2x
δx(x) = - g(x) / 2

function compute_next_iterate(ls::BierlaireQuadraticState, x₀::T; α₀ = one(T), mode = :autodiff) where {T}
    _f(α) = f(compute_new_iterate(x₀, α, δx(x₀)))
    _d(α) = g(compute_new_iterate(x₀, α, δx(x₀)))

    obj = mode == :autodiff ? UnivariateObjective(_f, zero(T); mode = :autodiff) : UnivariateObjective(_f, _d, α₀; mode = :function)
    α = ls(obj, zero(T))
    compute_new_iterate(x₀, α, δx(x₀))
end

function compute_next_iterate(ls::LinesearchState, x₀::T; α₀ = one(T), mode = :autodiff) where {T}
    _f(α) = f(compute_new_iterate(x₀, α, δx(x₀)))
    _d(α) = g(compute_new_iterate(x₀, α, δx(x₀)))

    obj = mode == :autodiff ? UnivariateObjective(_f, α₀; mode = :autodiff) : UnivariateObjective(_f, _d, α₀; mode = :function)
    α = ls(obj, α₀)
    compute_new_iterate(x₀, α, δx(x₀))
end

function compute_next_iterate(ls::LinesearchState, x₀::T, n::Integer; kwargs...) where {T}
    x = x₀
    for _ in 1:n
        x = compute_next_iterate(ls, x; kwargs...)
    end
    x
end

function test_linesearch(algorithm, n = 1; kwargs...)

    x₀ = -3.0
    x₁ = +3.0
    xₛ =  0.0

    options = Options(x_abstol = zero(x₀))
    
    ls = LinesearchState(algorithm; config = options, kwargs...)

    @test compute_next_iterate(ls, x₀, n) ≈ xₛ  atol=∛(2eps())
    @test compute_next_iterate(ls, x₀, n) ≈ xₛ  atol=∛(2eps())

    # start with a different initial guess
    @test compute_next_iterate(ls, x₀, n; α₀ = 2one(x₀)) ≈ xₛ  atol=∛(2eps())
    @test compute_next_iterate(ls, x₀, n; α₀ = 2one(x₀)) ≈ xₛ  atol=∛(2eps())
end

@testset "$(rpad("Bracketing",80))" begin
    @test bracket_minimum(x -> x^2) == (-SimpleSolvers.DEFAULT_BRACKETING_s, +SimpleSolvers.DEFAULT_BRACKETING_s)
    @test bracket_minimum(x -> (x-1)^2) == (0.64, 2.56)
end

@testset "$(rpad("Static",80))" begin
    x₀ = -3.
    x₁ = +1.
    δx = x₁ - x₀
    x  = copy(x₀)


    o  = UnivariateObjective(f, x)
    ls = StaticState()

    @test ls == LinesearchState(Static())
    @test ls == LinesearchState(Static(1.0))

    # x1 = copy(x₀); x2 = copy(x₀); @test solve!(x1, δx, ls) == ls(x2, δx) == x₁
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

@testset "$(rpad("Polynomial Linesearch", 80))" begin

    test_linesearch(BierlaireQuadratic(), 10)

end