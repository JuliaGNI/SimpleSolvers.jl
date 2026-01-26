using Random
using SimpleSolvers
using Test

using LinearAlgebra: rmul!, ldiv!
using SimpleSolvers: LinesearchState, BierlaireQuadraticState, StaticState, QuadraticState
using SimpleSolvers: AbstractOptimizerProblem, BierlaireQuadratic, Quadratic, NullParameters
using SimpleSolvers: factorize!, linearsolver, jacobian, jacobian!, cache, linesearch_problem, direction, compute_new_iterate, direction!, nonlinearproblem

f(x) = x^2 - 1
g(x) = 2x
δx(x) = - g(x) / 2

function make_linesearch_problem(x₀::Number)
    _f(α) = f(compute_new_iterate(x₀, α, δx(x₀)))
    _d(α) = g(compute_new_iterate(x₀, α, δx(x₀)))
    LinesearchProblem(_f, _d)
end

function compute_next_iterate(ls::LinesearchState, x₀::T) where {T}
    ls_obj = make_linesearch_problem(x₀)
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

    ls1 = Linesearch(algorithm = Static())
    ls2 = Linesearch(algorithm = Static(1.0))
    ls3 = Linesearch(algorithm = Static(0.8))

    @test ls1(f,g) == 1
    @test ls2(f,g) == 1
    @test ls3(f,g) == 0.8

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


@testset "$(rpad("Additional Linesearch Tests", 80))" begin

    Random.seed!(1234)

    x = -10 * rand(1)

    function make_linesearch_problem2(x::AbstractVector{T}, params=NullParameters()) where {T}
        f(x::T) where {T<:Number} = exp(x) * (T(.5) * x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
        f(x::AbstractArray{T}) where {T<:Number} = @. exp(x) * (T(.5) * x^3 - 5 * x^2 + 2x) + 2one(T)
        f!(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T} = y .= f.(x)

        function j!(j::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
            f_closure!(y, x) = f!(y, x, params)
            SimpleSolvers.ForwardDiff.jacobian!(j, f_closure!, similar(x), x)
        end

        jacobian_instance = JacobianFunction{T}(f!, j!)
        solver = NewtonSolver(x, f.(x); F = f!, DF! = j!, jacobian = jacobian_instance)
        state = NonlinearSolverState(x, value(cache(solver)))
        direction!(solver, x, params)

        update!(state, x, value(cache(solver)), 0)
        linesearch_problem(nonlinearproblem(solver), jacobian_instance, cache(solver), state, params)
    end

    function check_linesearch(ls::LinesearchState, ls_obj::LinesearchProblem)
        α = ls(ls_obj)
        T = eltype(α)
        @test ≈(ls_obj.D(α), zero(T); atol = atol=∛(2eps(T)))
    end

    for T ∈ (Float32, Float64)
        for ls_method ∈ (Bisection(), Quadratic(), BierlaireQuadratic())
            ls = LinesearchState(ls_method; T = T)
            ls_obj = make_linesearch_problem2(T.(x))
            check_linesearch(ls, ls_obj)
        end
    end

end
