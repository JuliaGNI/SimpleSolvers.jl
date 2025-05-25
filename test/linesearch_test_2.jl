using SimpleSolvers
using SimpleSolvers: update!, compute_jacobian!, factorize!, linearsolver, jacobian, cache, linesearch_objective, direction, LinesearchState
using LinearAlgebra: rmul!, ldiv!
using Test
using Random 
Random.seed!(1234) 

f(x::T) where {T<:Number} = exp(x) * (x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
f(x::AbstractArray{T}) where {T<:Number} = exp.(x) .* (.5 * (x .^ 3) - 5 * (x .^ 2) + 2x) .+ 2one(T)
f!(y::AbstractVector{T}, x::AbstractVector{T}) where {T} = y .= f.(x)
j!(j::AbstractMatrix{T}, x::AbstractVector{T}) where {T} = SimpleSolvers.ForwardDiff.jacobian!(j, f!, similar(x), x)
x = -10 * rand(1)

function make_linesearch_objective(x::AbstractVector)
    solver = NewtonSolver(x, f.(x); F = f)
    update!(solver, x)
    compute_jacobian!(solver, x, j!; mode = :function)

    # compute rhs
    f!(cache(solver).rhs, x)
    rmul!(cache(solver).rhs, -1)

    # multiply rhs with jacobian
    factorize!(linearsolver(solver), jacobian(solver))
    ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)

    nls = NonlinearSystem(f, x)
    linesearch_objective(nls, cache(solver))
end

function check_linesearch(ls::LinesearchState, ls_obj::TemporaryUnivariateObjective)
    α = ls(ls_obj)
    @test ls_obj.D(α) ≈ zero(α)
end

for T ∈ (Float32, Float64)
    for ls_method ∈ (Bisection(), Quadratic(), BierlaireQuadratic())
        ls = LinesearchState(ls_method; T = T)
        ls_obj = make_linesearch_objective(T.(x))
        ls(ls, ls_obj)
    end
end