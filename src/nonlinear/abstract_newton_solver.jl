

struct NewtonSolverCache{T, AT <: AbstractArray{T}}
    x₀::Vector{T}
    x₁::Vector{T}
    δx::Vector{T}
    
    y₀::Vector{T}
    y₁::Vector{T}
    δy::Vector{T}

    function NewtonSolverCache(x::AT, y::AT) where {T, AT <: AbstractArray{T}}
        new{T,AT}(zero(x), zero(x), zero(x),
                  zero(y), zero(y), zero(y))
    end
end


abstract type AbstractNewtonSolver{T,AT} <: NonlinearSolver{T} end


@define newton_solver_variables begin
    x::AT
    y::AT
    J::Matrix{T}

    F!::FT
    Jparams::TJ

    linear::TL
    linesearch::TS

    cache::NewtonSolverCache{T,AT}
    config::Options{T}
    status::NonlinearSolverStatus{T}
end

config(solver::AbstractNewtonSolver) = solver.config
status(solver::AbstractNewtonSolver) = solver.status

compute_jacobian!(s::AbstractNewtonSolver) = compute_jacobian!(s.J, s.x, s.Jparams)
check_jacobian(s::AbstractNewtonSolver) = check_jacobian(s.J)
print_jacobian(s::AbstractNewtonSolver) = print_jacobian(s.J)

function initialize!(s::AbstractNewtonSolver{T}, x₀::Vector{T}) where {T}
    copyto!(s.x, x₀)
    s.F!(s.y, s.x)
    initialize!(status(s), s.x, s.y)
end
