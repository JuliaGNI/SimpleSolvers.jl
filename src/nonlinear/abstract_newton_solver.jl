
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
    params::NonlinearSolverParameters{T}
    status::NonlinearSolverStatus{T}
end

function initialize!(s::AbstractNewtonSolver{T}, x₀::Vector{T}) where {T}
    s.x .= x₀
    s.F!(s.y, s.x)
end

status(solver::AbstractNewtonSolver) = solver.status
params(solver::AbstractNewtonSolver) = solver.params

compute_jacobian!(s::AbstractNewtonSolver) = compute_jacobian!(s.J, s.x, s.Jparams)
check_jacobian(s::AbstractNewtonSolver) = check_jacobian(s.J)
print_jacobian(s::AbstractNewtonSolver) = print_jacobian(s.J)
