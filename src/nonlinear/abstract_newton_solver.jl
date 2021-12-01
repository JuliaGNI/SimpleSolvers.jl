
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

    params::NonlinearSolverParameters{T}
    status::NonlinearSolverStatus{T}
end

function setInitialConditions!(s::AbstractNewtonSolver{T}, x₀::Vector{T}) where {T}
    s.x .= x₀
    s.F!(s.y, s.x)
end

status(solver::AbstractNewtonSolver) = solver.status
params(solver::AbstractNewtonSolver) = solver.params

computeJacobian(s::AbstractNewtonSolver) = computeJacobian(s.x, s.J, s.Jparams)
check_jacobian(s::AbstractNewtonSolver) = check_jacobian(s.J)
print_jacobian(s::AbstractNewtonSolver) = print_jacobian(s.J)
