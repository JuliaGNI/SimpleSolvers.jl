
import NLsolve: OnceDifferentiable, NewtonCache, newton_
import LineSearches


struct NLsolveNewton{T, AT, JT, FT, DT, CT, LST, LT, TST} <: AbstractNewtonSolver{T,AT}
    x::AT
    f::AT
    J::JT

    F!::FT
    DF::DT

    line_search::LST
    linear_solver::LT

    cache::CT
    config::Options{T}
    status::TST

    function NLsolveNewton(x::AT, f::AT, J::JT, F!::FT, DF::DT, cache::CT,
            line_search::ST, linear_solver::LT; options_kwargs...) where {
                T, AT <: AbstractVector{T}, JT <: AbstractMatrix{T}, FT, DT, CT, ST, LT
            }

        status = NonlinearSolverStatus{T}(length(x))
        options = Options(T; options_kwargs...)

        new{T,AT,JT,FT,DT,CT,ST,LT, typeof(status)}(x, f, J, F!, DF, line_search, linear_solver, cache, options, status)
    end
end


function NLsolveNewton(x::AbstractVector{T}, f::AbstractVector{T}, F!::Function; J!::Union{Callable,Nothing,Missing} = nothing, mode = :autodiff, diff_type = :forward) where {T}
    linear_solver = LinearSolver(x)

    df = linear_solver.A

    if J! === nothing || ismissing(J!) || mode == :autodiff
        DF = OnceDifferentiable(F!, x, f, df; autodiff=diff_type, inplace=true)
    else
        DF = OnceDifferentiable(F!, J!, x, f, df; inplace=true)
    end

    NLsolveNewton(x, f, df, F!, DF, NewtonCache(DF), LineSearches.Static(), linear_solver)
end


function linsolve!(s::NLsolveNewton, x, A, b)
    factorize!(s.linear_solver, A)
    ldiv!(x, s.linear_solver, b)
end

function solve!(x, f, forj, s::NLsolveNewton)
    res = newton_(s.DF, x, s.config.x_abstol, s.config.f_abstol, s.config.max_iterations, false, false, false,
                  s.line_search, (x, A, b) -> linsolve!(s, x, A, b), s.cache)

    copyto!(x, res.zero)

    s.status.i = res.iterations
    s.status.rfₐ = res.residual_norm

    return x
end