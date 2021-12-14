
abstract type Optimizer{T} <: NonlinearSolver{T} end

config(s::Optimizer) = error("config not implemented for $(typeof(s))")
status(s::Optimizer) = error("status not implemented for $(typeof(s))")
objective(s::Optimizer) = error("objective not implemented for $(typeof(s))")
initialize!(s::Optimizer, args...) = error("initialize! not implemented for $(typeof(s))")
solver_step!(s::Optimizer) = error("solver_step! not implemented for $(typeof(s))")


function solve!(x, s::Optimizer)
    initialize!(s, x)

    while !meets_stopping_criteria(status(s), config(s))
        next_iteration!(status(s))
        solver_step!(s)
        residual!(status(s))
    end

    warn_iteration_number(status(s), config(s))

    copyto!(x, solution(status(s)))

    return x
end
