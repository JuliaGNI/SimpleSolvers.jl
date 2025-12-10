using Printf

"""
    NonlinearSolverCache

An abstract type that comprises e.g. the [`NewtonSolverCache`](@ref).
"""
abstract type NonlinearSolverCache{T} end

"""
    NonlinearSolver

A `struct` that comprises *Newton solvers* (see [`NewtonMethod`](@ref)) and the *fixed point iterator* (see [`PicardMethod`](@ref)).

# Constructors

```julia
NonlinearSolver(x, nlp, ls, linearsolver, linesearch, cache; method)
```

The `NonlinearSolver` can be called with an [`NonlinearProblem`](@ref) or with a `Callable`. Note however that the latter will probably be deprecated in the future. See [`NewtonSolver`](@ref) for examples (as well as [`NonlinearSolverStatus`](@ref)).

It's arguments are:
- `nlp::`[`NonlinearProblem`](@ref): the system that has to be solved. This can be accessed by calling [`nonlinearproblem`](@ref),
- `ls::`[`LinearProblem`](@ref),
- `linearsolver::`[`LinearSolver`](@ref): the linear solver is used to compute the [`direction`](@ref) of the solver step (see [`solver_step!`](@ref)). This can be accessed by calling [`linearsolver`](@ref),
- `linesearch::`[`LinesearchState`](@ref)
- `cache::`[`NonlinearSolverCache`](@ref)
- `config::`[`Options`](@ref)
- `status::`[`NonlinearSolverStatus`](@ref):
"""
struct NonlinearSolver{T,MT<:NonlinearSolverMethod,AT,NLST<:NonlinearProblem{T},LST<:AbstractLinearProblem,JT<:Jacobian{T},LSoT<:AbstractLinearSolver,LiSeT<:LinesearchState{T},CT<:NonlinearSolverCache{T},NSST<:NonlinearSolverStatus{T}} <: AbstractSolver
    nonlinearproblem::NLST
    linearproblem::LST
    jacobian::JT
    linearsolver::LSoT
    linesearch::LiSeT
    method::MT

    cache::CT
    config::Options{T}
    status::NSST

    function NonlinearSolver(x::AT, nlp::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT; method::MT = NewtonMethod(), jacobian::JT=JacobianAutodiff(nlp.F, x), options_kwargs...) where {T,AT<:AbstractVector{T},MT<:NonlinearSolverMethod,JT<:Jacobian,NLST,LST,LSoT,LiSeT,CT}
        status = NonlinearSolverStatus(x)
        config = Options(T; options_kwargs...)
        new{T,MT,AT,NLST,LST,JT,LSoT,LiSeT,CT,typeof(status)}(nlp, ls, jacobian, linearsolver, linesearch, method, cache, config, status)
    end
end

config(s::NonlinearSolver) = s.config
status(s::NonlinearSolver) = s.status
initialize!(s::NonlinearSolver, ::AbstractArray) = error("initialize! not implemented for $(typeof(s))")
solver_step!(s::NonlinearSolver) = error("solver_step! not implemented for $(typeof(s))")
method(s::NonlinearSolver) = s.method
nonlinearproblem(s::NonlinearSolver) = s.nonlinearproblem
Jacobian(s::NonlinearSolver)::Jacobian = s.jacobian

jacobian!(s::NonlinearSolver{T}, x::AbstractVector{T}, params) where {T} = Jacobian(s)(jacobian(cache(s)), x, params)

struct NonlinearSolverException <: Exception
    msg::String
end

Base.showerror(io::IO, e::NonlinearSolverException) = print(io, "Nonlinear Solver Exception: ", e.msg, "!")