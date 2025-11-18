"""
    default_tolerance(T)

Determine the default tolerance for a specific data type. This is used in the constructor of [`Options`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers: default_tolerance)
default_tolerance(Float64)

# output

4.440892098500626e-16
```

```jldoctest; setup = :(using SimpleSolvers: default_tolerance)
default_tolerance(Float32)

# output

2.3841858f-7
```

```jldoctest; setup = :(using SimpleSolvers: default_tolerance)
default_tolerance(Float16)

# output

Float16(0.001953)
```
"""
function default_tolerance(::Type{T}) where {T <: AbstractFloat}
    2eps(T)
end

"""
    absolute_tolerance(T)

Determine the absolute tolerance for a specific data type. This is used in the constructor of [`Options`](@ref).

In comparison to [`default_tolerance`](@ref), this should return a very small number, close to zero (i.e. not just machine precision).

# Examples

```jldoctest; setup = :(using SimpleSolvers: absolute_tolerance)
absolute_tolerance(Float64)

# output

0.0
```

```jldoctest; setup = :(using SimpleSolvers: absolute_tolerance)
absolute_tolerance(Float32)

# output

0.0f0
```
"""
function absolute_tolerance(::Type{T}) where {T <: AbstractFloat}
    zero(T)
end

"""
    minimum_decrease_threshold(T)

The minimum value by which a function ``f`` should decrease during an iteration.

The default value of ``10^-4`` is often used in the literature [bierlaire2015optimization], nocedal2006numerical(@cite).

# Examples

```jldoctest; setup = :(using SimpleSolvers: minimum_decrease_threshold)
minimum_decrease_threshold(Float64)

# output

0.0001
```

```jldoctest; setup = :(using SimpleSolvers: minimum_decrease_threshold)
minimum_decrease_threshold(Float32)

# output

0.0001f0
```
"""
function minimum_decrease_threshold(::Type{T}) where {T <: AbstractFloat}
    T(10)^-4
end

const F_CALLS_LIMIT::Int = 0
const G_CALLS_LIMIT::Int = 0
const H_CALLS_LIMIT::Int = 0
const ALLOW_F_INCREASES::Bool = true
const MIN_ITERATIONS::Int = 0
const MAX_ITERATIONS::Int = 1_000
const WARN_ITERATIONS::Int = 1_000
const SHOW_TRACE::Bool = false
const STORE_TRACE::Bool = false
const EXTENDED_TRACE::Bool = false
const SHOW_EVERY::Int = 1
const VERBOSITY::Int = 1

"""
    Options

# Keys

Configurable options with defaults (values 0 and NaN indicate unlimited):
- `x_abstol = 2eps(T)`: absolute tolerance for `x` (the function argument). Used in e.g. [`assess_convergence!`](@ref) and [`bisection`](@ref),
- `x_reltol = 2eps(T)`: relative tolerance for `x` (the function argument). Used in e.g. [`assess_convergence!`](@ref),
- `x_suctol = 2eps(T)`: succesive tolerance for `x`. Used in e.g. [`assess_convergence!`](@ref),
- `f_abstol = zero(T)`: absolute tolerance for how close the function value should be to zero. See [`absolute_tolerance`](@ref). Used in e.g. [`bisection`](@ref) and [`assess_convergence!`](@ref),
- `f_reltol = 2eps(T)`: relative tolerance for the function value. Used in e.g. [`assess_convergence!`](@ref),
- `f_suctol = 2eps(T)`: succesive tolerance for the function value. Used in e.g. [`assess_convergence!`](@ref),
- `f_mindec = T(10)^-4`: minimum value by which the function has to decrease (also see [`minimum_decrease_threshold`](@ref)),
- `g_restol = 2eps(T)`: tolerance for the residual (?) of the gradient,
- `x_abstol_break = -Inf`: see [`meets_stopping_criteria`](@ref),
- `x_reltol_break = Inf`: see [`meets_stopping_criteria`](@ref),
- `f_abstol_break = Inf`: see [`meets_stopping_criteria`](@ref),
- `f_reltol_break = Inf`: see [`meets_stopping_criteria`](@ref).,
- `g_restol_break = Inf`,
- `f_calls_limit = $(F_CALLS_LIMIT)`,
- `h_calls_limit = $(H_CALLS_LIMIT)`,
- `allow_f_increases = $(ALLOW_F_INCREASES)`,
- `min_iterations = $(MIN_ITERATIONS)`,
- `max_iterations = $(MAX_ITERATIONS)`: the maximum number of iterations used in an alorithm, e.g. [`bisection`](@ref) and the functor for [`BacktrackingState`](@ref),
- `warn_iterations = $(WARN_ITERATIONS)`,
- `show_trace = $(SHOW_TRACE)`,
- `store_trace = $(STORE_TRACE)`,
- `extended_trace = $(EXTENDED_TRACE)`,
- `show_every = $(SHOW_EVERY)`,
- `verbosity = $(VERBOSITY)`

Some of the constants are defined by the functions [`default_tolerance`](@ref) and [`absolute_tolerance`](@ref).
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

function Options(T = Float64;
        x_abstol::AbstractFloat = default_tolerance(T),
        x_reltol::AbstractFloat = default_tolerance(T),
        x_suctol::AbstractFloat = default_tolerance(T),
        f_abstol::AbstractFloat = absolute_tolerance(T),
        f_reltol::AbstractFloat = default_tolerance(T),
        f_suctol::AbstractFloat = default_tolerance(T),
        f_mindec::AbstractFloat = minimum_decrease_threshold(T),
        g_restol::AbstractFloat = √(default_tolerance(T) / 2),
        x_abstol_break::AbstractFloat = T(Inf),
        x_reltol_break::AbstractFloat = T(Inf),
        f_abstol_break::AbstractFloat = T(Inf),
        f_reltol_break::AbstractFloat = T(Inf),
        g_restol_break::AbstractFloat = T(Inf),
        f_calls_limit::Integer = F_CALLS_LIMIT,
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

    Options{T}(promote( x_abstol, 
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