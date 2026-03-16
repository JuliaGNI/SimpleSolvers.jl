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
function default_tolerance(::Type{T}) where {T<:AbstractFloat}
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
function absolute_tolerance(::Type{T}) where {T<:AbstractFloat}
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
function minimum_decrease_threshold(::Type{T}) where {T<:AbstractFloat}
    T(10)^-4
end

const ALLOW_F_INCREASES::Bool = true
const MIN_ITERATIONS::Int = 0
const MAX_ITERATIONS::Int = 1_000
const WARN_ITERATIONS::Int = 1_000
const SHOW_TRACE::Bool = false
const STORE_TRACE::Bool = false
const EXTENDED_TRACE::Bool = false
const SHOW_EVERY::Int = 1
const VERBOSITY::Int = 1

const NAN_MAX_ITERATIONS = 10
const NAN_FACTOR = 0.5
const REGULARIZATION_FACTOR = 0

"""
    Options

# Examples

```jldoctest; setup = :(using SimpleSolvers)
Options()

# output

                x_abstol = 4.440892098500626e-16
                x_reltol = 4.440892098500626e-16
                x_suctol = 4.440892098500626e-16
                f_abstol = 0.0
                f_reltol = 4.440892098500626e-16
                f_suctol = 4.440892098500626e-16
                f_mindec = 0.0001
                g_restol = 1.4901161193847656e-8
          x_abstol_break = Inf
          x_reltol_break = Inf
          f_abstol_break = Inf
          f_reltol_break = Inf
          g_restol_break = Inf
       allow_f_increases = true
          min_iterations = 0
          max_iterations = 1000
         warn_iterations = 1000
              show_trace = false
             store_trace = false
          extended_trace = false
              show_every = 1
               verbosity = 1
      nan_max_iterations = 10
              nan_factor = 0.5
   regularization_factor = 0.0

```

!!! info
    For the first few constants (`x_abstol` to `g_restol`) the default constructor uses the functions [`default_tolerance`](@ref) and [`absolute_tolerance`](@ref).

!!! info
    Also see [`meets_stopping_criteria`](@ref).
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
    allow_f_increases::Bool
    min_iterations::Int
    max_iterations::Int
    warn_iterations::Int
    show_trace::Bool
    store_trace::Bool
    extended_trace::Bool
    show_every::Int
    verbosity::Int
    nan_max_iterations::Int
    nan_factor::T
    regularization_factor::T
end

function Options(T=Float64;
    x_abstol::AbstractFloat=default_tolerance(T),
    x_reltol::AbstractFloat=default_tolerance(T),
    x_suctol::AbstractFloat=default_tolerance(T),
    f_abstol::AbstractFloat=4absolute_tolerance(T),
    f_reltol::AbstractFloat=default_tolerance(T),
    f_suctol::AbstractFloat=default_tolerance(T),
    f_mindec::AbstractFloat=minimum_decrease_threshold(T),
    g_restol::AbstractFloat=(√(default_tolerance(T) / 2)),
    x_abstol_break::AbstractFloat=T(Inf),
    x_reltol_break::AbstractFloat=T(Inf),
    f_abstol_break::AbstractFloat=T(Inf),
    f_reltol_break::AbstractFloat=T(Inf),
    g_restol_break::AbstractFloat=T(Inf),
    allow_f_increases::Bool=ALLOW_F_INCREASES,
    min_iterations::Integer=MIN_ITERATIONS,
    max_iterations::Integer=MAX_ITERATIONS,
    warn_iterations::Integer=WARN_ITERATIONS,
    show_trace::Bool=SHOW_TRACE,
    store_trace::Bool=STORE_TRACE,
    extended_trace::Bool=EXTENDED_TRACE,
    show_every::Integer=SHOW_EVERY,
    verbosity::Integer=VERBOSITY,
    nan_max_iterations::Integer=NAN_MAX_ITERATIONS,
    nan_factor::AbstractFloat=NAN_FACTOR,
    regularization_factor::AbstractFloat=T(REGULARIZATION_FACTOR),
)

    show_every = show_every > 0 ? show_every : 1

    Options{T}(promote(x_abstol,
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
        allow_f_increases,
        min_iterations,
        max_iterations,
        warn_iterations,
        show_trace,
        store_trace,
        extended_trace,
        show_every,
        verbosity,
        nan_max_iterations,
        nan_factor,
        regularization_factor,
    )
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
