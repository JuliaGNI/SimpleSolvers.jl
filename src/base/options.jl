"absolute tolerance for `x` (the function argument)."
const X_ABSTOL::Real = -Inf
"relative tolerance for `x` (the function argument)."
const X_RELTOL::Real = 2eps()
"succesive tolerance for `x"
const X_SUCTOL::Real = 2eps()
"Absolute tolerance for how close the function value should be to zero. Used in e.g. [`bisection`](@ref)."
const F_ABSTOL::Real = 1e-50
"relative tolerance for the function value."
const F_RELTOL::Real = 2eps()
"succesive tolerance for the function value"
const F_SUCTOL::Real = 2eps()
"minimum value by which the function has to decrease."
const F_MINDEC::Real = 1e-4
"tolerance for the residual (?) of the gradient."
const G_RESTOL::Real = sqrt(eps())
const X_ABSTOL_BREAK::Real = Inf
const X_RELTOL_BREAK::Real = Inf
const F_ABSTOL_BREAK::Real = Inf
const F_RELTOL_BREAK::Real = Inf
const G_ABSTOL_BREAK::Real = Inf
const G_RESTOL_BREAK::Real = Inf
const F_CALLS_LIMIT::Int = 0
const G_CALLS_LIMIT::Int = 0
const H_CALLS_LIMIT::Int = 0
const ALLOW_F_INCREASES::Bool = true
const MIN_ITERATIONS::Int = 0
"The maximum number of iterations used in an alorithm, e.g. [`bisection`](@ref) and the functor for [`BacktrackingState`](@ref)."
const MAX_ITERATIONS::Int = 1_000
const WARN_ITERATIONS::Int = 1_000
const SHOW_TRACE::Bool = false
const STORE_TRACE::Bool = false
const EXTENDED_TRACE::Bool = false
const SHOW_EVERY::Int = 1
const VERBOSITY::Int = 1

"""
Configurable options with defaults (values 0 and NaN indicate unlimited):
- `x_abstol = $(X_ABSTOL)`,
- `x_reltol = $(X_RELTOL)`,
- `x_suctol = $(X_SUCTOL)`
- `f_abstol = $(F_ABSTOL)`,
- `f_reltol = $(F_RELTOL)`,
- `f_suctol = $(F_SUCTOL)`,
- `f_mindec = $(F_MINDEC)`,
- `g_restol = $(G_RESTOL)`,
- `x_abstol_break = $(X_ABSTOL_BREAK)`,
- `x_reltol_break = $(X_RELTOL_BREAK)`,
- `f_abstol_break = $(F_ABSTOL_BREAK)`,
- `f_reltol_break = $(F_RELTOL_BREAK)`,
- `g_restol_break = $(G_RESTOL_BREAK)`,
- `f_calls_limit = $(F_CALLS_LIMIT)`,
- `g_calls_limit = $(G_CALLS_LIMIT)`,
- `h_calls_limit = $(H_CALLS_LIMIT)`,
- `allow_f_increases = $(ALLOW_F_INCREASES)`,
- `min_iterations = $(MIN_ITERATIONS)`,
- `max_iterations = $(MAX_ITERATIONS)`,
- `warn_iterations = $(WARN_ITERATIONS)`,
- `show_trace = $(SHOW_TRACE)`,
- `store_trace = $(STORE_TRACE)`,
- `extended_trace = $(EXTENDED_TRACE)`,
- `show_every = $(SHOW_EVERY)`,
- `verbosity = $(VERBOSITY)`
"""
struct Options{T}
    x_abstol::T
    x_reltol::T
    x_suctol::T
    f_abstol::T
    f_reltol::T
    f_suctol::T
    f_mindec::T
    g_restol::T
    x_abstol_break::T
    x_reltol_break::T
    f_abstol_break::T
    f_reltol_break::T
    g_restol_break::T
    f_calls_limit::Int
    g_calls_limit::Int
    h_calls_limit::Int
    allow_f_increases::Bool
    min_iterations::Int
    max_iterations::Int
    warn_iterations::Int
    show_trace::Bool
    store_trace::Bool
    extended_trace::Bool
    show_every::Int
    verbosity::Int
end

function Options(;
        x_tol = nothing,
        f_tol = nothing,
        g_tol = nothing,
        x_abstol::Real = X_ABSTOL,
        x_reltol::Real = X_RELTOL,
        x_suctol::Real = X_SUCTOL,
        f_abstol::Real = F_ABSTOL,
        f_reltol::Real = F_RELTOL,
        f_suctol::Real = F_SUCTOL,
        f_mindec::Real = F_MINDEC,
        g_restol::Real = G_RESTOL,
        x_abstol_break::Real = X_ABSTOL_BREAK,
        x_reltol_break::Real = X_RELTOL_BREAK,
        f_abstol_break::Real = G_ABSTOL_BREAK,
        f_reltol_break::Real = F_RELTOL_BREAK,
        g_restol_break::Real = G_RESTOL_BREAK,
        f_calls_limit::Integer = F_CALLS_LIMIT,
        g_calls_limit::Integer = G_CALLS_LIMIT,
        h_calls_limit::Integer = H_CALLS_LIMIT,
        allow_f_increases::Bool = ALLOW_F_INCREASES,
        min_iterations::Integer = MIN_ITERATIONS,
        max_iterations::Integer = MAX_ITERATIONS,
        warn_iterations::Integer = WARN_ITERATIONS,
        show_trace::Bool = SHOW_TRACE,
        store_trace::Bool = STORE_TRACE,
        extended_trace::Bool = EXTENDED_TRACE,
        show_every::Integer = SHOW_EVERY,
        verbosity::Integer = VERBOSITY)

    show_every = show_every > 0 ? show_every : 1
    
    if extended_trace
       show_trace = true
    end
    if !isnothing(x_tol)
        x_abstol = x_tol
    end
    if !isnothing(g_tol)
        g_restol = g_tol
    end
    if !isnothing(f_tol)
        f_reltol = f_tol
    end

    Options(promote(    x_abstol, 
                        x_reltol,  
                        x_suctol, 
                        f_abstol, 
                        f_reltol, 
                        f_suctol, 
                        f_mindec, 
                        g_restol,
                        x_abstol_break,
                        x_reltol_break,    
                        f_abstol_break, 
                        f_reltol_break, 
                        g_restol_break)...,
                        f_calls_limit, 
                        g_calls_limit, 
                        h_calls_limit, 
                        allow_f_increases, 
                        min_iterations, 
                        max_iterations, 
                        warn_iterations,
                        show_trace, 
                        store_trace, 
                        extended_trace, 
                        show_every, 
                        verbosity)
end

function Options(T::DataType, options::Options)

    floatopts = (
        options.x_abstol,
        options.x_reltol,
        options.x_suctol,
        options.f_abstol,
        options.f_reltol,
        options.f_suctol,
        options.f_mindec,
        options.g_restol,
        options.x_abstol_break,
        options.x_reltol_break,
        options.f_abstol_break,
        options.f_reltol_break,
        options.g_restol_break,
    )

    nonfloats = (
        options.f_calls_limit,
        options.g_calls_limit,
        options.h_calls_limit,
        options.allow_f_increases,
        options.min_iterations,
        options.max_iterations,
        options.warn_iterations,
        options.show_trace,
        options.store_trace,
        options.extended_trace,
        options.show_every,
        options.verbosity,
    )

    Options(map(x -> convert(T, x), floatopts)..., nonfloats...)
end

function Base.show(io::IO, o::SimpleSolvers.Options)
    for k in fieldnames(typeof(o))
        v = getfield(o, k)
        if v isa Nothing
            @printf io "%24s = %s\n" k "nothing"
        else
            @printf io "%24s = %s\n" k v
        end
    end
end

x_abstol(o::Options) = o.x_abstol
x_reltol(o::Options) = o.x_reltol
x_suctol(o::Options) = o.x_suctol
f_abstol(o::Options) = o.f_abstol
f_reltol(o::Options) = o.f_reltol
f_suctol(o::Options) = o.f_suctol
f_mindec(o::Options) = o.f_mindec
g_restol(o::Options) = o.g_restol

verbosity(o::Options) = o.verbosity