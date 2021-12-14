
abstract type AbstractNewtonSolver{T} <: NonlinearSolver{T} end

@define newton_solver_variables begin
    x::Vector{T}
    y::Vector{T}
    J::Matrix{T}

    x₀::Vector{T}
    x₁::Vector{T}
    y₀::Vector{T}
    y₁::Vector{T}
    δx::Vector{T}
    δy::Vector{T}

    F!::FT
    Jparams::TJ

    linear::TL
    ls::TS

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
