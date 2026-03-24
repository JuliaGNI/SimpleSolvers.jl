const PicardSolver{T} = NonlinearSolver{T,PicardMethod}

function PicardSolver(x::AT, nlp::NLST, linesearch::LiSeT, cache::CT; jacobian, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LiSeT,CT}
    config = Options(T; options_kwargs...)
    NonlinearSolver(x, nlp, NoLinearProblem(), NoLinearSolver(), linesearch, cache, config; jacobian=jacobian, method=PicardMethod())
end

"""
    PicardSolver(x, F)

# Arguments
- `x`: the initial guess for the solution.
- `F`: the nonlinear function to solve.
- `y`

# Keywords
- `DF!`: the Jacobian of `F`,
- `linesearch`: the linesearch algorithm to use, defaults to [`Backtracking`](@ref),
- `jacobian`: the Jacobian of `F`, defaults to [`JacobianAutodiff`](@ref),
- `options_kwargs`: see [`Options`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers)
F(y, x, params) = y .= sin.(x) .^ 2
x = zeros(2)
y = similar(x)

s = PicardSolver(x, F, y)
state = SolverState(s)

solve!(x, s, state)

# output

2-element Vector{Float64}:
 0.0
 0.0
```
"""
function PicardSolver(x::AT, F::Callable, y::AT; (DF!)=missing, linesearch=Backtracking(T), jacobian=JacobianAutodiff(F, x, y), kwargs...) where {T,AT<:AbstractVector{T}}
    nlp = NonlinearProblem(F, DF!, x, y)
    jacobian = ismissing(DF!) ? jacobian : JacobianFunction{T}(F, DF!)
    cache = NonlinearSolverCache(x, y)
    ls = Linesearch(linesearch_problem(nlp, jacobian, cache), linesearch)
    PicardSolver(x, nlp, ls, cache; jacobian=jacobian, kwargs...)
end

function PicardSolver(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    PicardSolver(x, F, y; kwargs...)
end

NonlinearSolver(::PicardMethod, x...; kwargs...) = PicardSolver(x...; kwargs...)

function direction!(d::AbstractVector{T}, x::AbstractVector{T}, it::PicardSolver{T}, params) where {T}
    value!(d, nonlinearproblem(it), x, params)
    d .*= -1
end

function direction!(it::PicardSolver, x::AbstractVector, params)
    direction!(direction(cache(it)), x, it, params)
end

direction!(it::PicardSolver, x::AbstractVector, params, iteration) = direction!(it, x, params)
