using Printf
@doc raw"""
    const DEFAULT_ARMIJO_α₀

The default starting value for ``\alpha`` used in [`Backtracking`](@ref).
Its value is """ * """$(DEFAULT_ARMIJO_α₀).
"""
const DEFAULT_ARMIJO_α₀ = 1.0

"""
    const DEFAULT_ARMIJO_p

Constant used in [`Backtracking`](@ref).
Its value is $(DEFAULT_ARMIJO_p)

This is the default for the constant ``p`` by which `α` is decreased if the [`SufficientDecreaseCondition`](@ref) and the [`CurvatureCondition`](@ref) are not satisfied.
"""
const DEFAULT_ARMIJO_p = 0.5

@doc raw"""
    const DEFAULT_WOLFE_c₁

A constant ``c_1`` that is used in the [`SufficientDecreaseCondition`](@ref) (the
Armijo condition):

```math
f(\alpha) \leq f(\alpha_0) + c_1 \alpha f'(\alpha_0).
```
"""
const DEFAULT_WOLFE_c₁ = 1E-4

@doc raw"""
    const DEFAULT_WOLFE_c₂

The constant used in the second Wolfe condition (the [`CurvatureCondition`](@ref)). According to [nocedal2006numerical,kochenderfer2019algorithms](@cite) we should have
```math
c_2 \in (c_1, 1),
```
where ``c_1`` is the constant specified by [`DEFAULT_WOLFE_c₁`](@ref).

Furthermore [nocedal2006numerical](@cite) recommend ``c_2 = 0.9``; in [kochenderfer2019algorithms](@cite) the authors write: "it is common to set ``c_2=0.1`` when approximate line search is used with the conjugate gradient method and to 0.9 when used with Newton's method."
We use ``c_2 = 0.9`` as default.
"""
const DEFAULT_WOLFE_c₂ = 0.9

@doc raw"""
    Backtracking <: LinesearchMethod

# Keys

The keys are:
- `α₀`=""" * string(DEFAULT_ARMIJO_α₀) * raw""": the initial step size ``\alpha``. This is decreased iteratively by a factor ``p`` until the Wolfe conditions (the [`SufficientDecreaseCondition`](@ref) and the [`CurvatureCondition`](@ref)) are satisfied.
- `c₁`=""" * string(DEFAULT_WOLFE_c₁) * raw""": the constant ``c_1`` in the [`SufficientDecreaseCondition`](@ref) (Armijo condition). Also see [`DEFAULT_WOLFE_c₁`](@ref).
- `c₂`=""" * string(DEFAULT_WOLFE_c₂) * raw""": the constant on whose basis the [`CurvatureCondition`](@ref) is tested. We should have ``c_2\in(c_1, 1).`` The closer this constant is to 1, the easier it is to satisfy the [`CurvatureCondition`](@ref).
- `p`=""" * string(DEFAULT_ARMIJO_p) * raw""": a parameter with which ``\alpha`` is decreased in every step until the stopping criterion is satisfied.

# Implementation

The algorithm starts by setting:
```math
\begin{aligned}
x_0 &\gets 0,\\
y_0 &\gets f(x_0),\\
d_0 &\gets f'(x_0),\\
\alpha &\gets \alpha_0,
\end{aligned}
```
where ``f`` is of type [`LinesearchProblem`](@ref) and ``\alpha_0`` is stored in `ls`. It then repeatedly does ``\alpha \gets \alpha\cdot{}p`` until either (i) the maximum number of iterations is reached (the `max_iterations` keyword in [`Options`](@ref)) or (ii) the [`SufficientDecreaseCondition`](@ref) and the [`CurvatureCondition`](@ref) are satisfied.

# Extended help

[Sometimes](https://en.wikipedia.org/wiki/Backtracking_line_search) the parameters ``p`` and ``\epsilon`` have different names such as ``\tau`` and ``c``.
"""
struct Backtracking{T} <: LinesearchMethod{T}
    α₀::T
    c₁::T
    c₂::T
    p::T

    function Backtracking{T}(α₀::T, c₁::T, c₂::T, p::T) where {T}
        @assert 0 < p < 1 "The shrinking parameter needs to satisfy 0 < p < 1, it is $(p)."
        @assert 0 < c₁ < c₂ < 1 "The Wolfe constants need to satisfy 0 < c₁ < c₂ < 1, they are c₁ = $(c₁), c₂ = $(c₂)."
        new{T}(α₀, c₁, c₂, p)
    end
end

function Backtracking(::Type{T}=Float64;
    α₀=T(DEFAULT_ARMIJO_α₀),
    c₁=T(DEFAULT_WOLFE_c₁),
    c₂=T(DEFAULT_WOLFE_c₂),
    p=T(DEFAULT_ARMIJO_p)
) where {T}
    Backtracking{T}(α₀, c₁, c₂, p)
end

Backtracking(::Type{T}, ::SolverMethod) where {T} = Backtracking(T)


# function solve(ls::Linesearch{T,<:Backtracking}, α::T=method(ls).α₀) where {T,LST}
function solve(ls::Linesearch{T,<:Backtracking}, α::T, params=NullParameters()) where {T}
    f(α) = value(problem(ls), α, params)
    d(α) = derivative(problem(ls), α, params)

    α₀ = zero(α)
    y₀ = f(α₀)
    d₀ = d(α₀)

    # note that we set pₖ ← 0 here as this is the descent direction for the linesearch problem.
    sdc = SufficientDecreaseCondition(method(ls).c₁, y₀, d₀, f)

    # Textbook backtracking (Nocedal & Wright, Alg. 3.1) terminates on the
    # sufficient-decrease (Armijo) condition alone.  Enforcing the curvature
    # condition inside the shrink-only loop below can never terminate, because
    # shrinking α makes the curvature condition harder to satisfy (α → 0 ⇒
    # D(α) → d₀ < c₂·d₀).  The curvature condition is therefore checked only
    # post-hoc (see below), not as a loop termination criterion.
    αₐ = α₀  # last α satisfying sufficient decrease (α₀ = 0 trivially does)
    satisfied = false
    for i in 1:config(ls).max_iterations
        if sdc(α)
            αₐ = α
            satisfied = true
            break
        else
            α *= method(ls).p
        end
    end

    if !satisfied
        # Never silently return a denormal α: fall back to the last α that
        # satisfied sufficient decrease (α₀ = 0 if none did).
        config(ls).verbosity ≥ 1 && @warn "Backtracking line search did not satisfy the sufficient decrease condition within $(config(ls).max_iterations) iterations. Returning α = $(αₐ)."
        return αₐ
    end

    # Opt-in, post-hoc curvature (second Wolfe) check.  If the accepted step
    # fails it we warn rather than shrinking further (which would break the
    # sufficient decrease guarantee); enforcing curvature requires a proper
    # bracketing/zoom line search.
    cc = CurvatureCondition(method(ls).c₂, d₀, d, Val(:Standard))
    (config(ls).verbosity ≥ 2 && !cc(α)) && @warn "Backtracking line search: accepted step α = $(α) satisfies the sufficient decrease but not the curvature condition."

    α
end

Base.show(io::IO, ls::Backtracking) = print(io, "Backtracking with α₀ = $(ls.α₀) c₁ = $(ls.c₁), c₂ = $(ls.c₂) and p = $(ls.p).")

function change_precision(::Type{T}, method::Backtracking) where {T}
    T ≠ eltype(method) || return method
    Backtracking{T}(T(method.α₀), T(method.c₁), T(method.c₂), T(method.p))
end

function Base.isapprox(bt₁::Backtracking{T}, bt₂::Backtracking{T}; kwargs...) where {T}
    isapprox(bt₁.α₀, bt₂.α₀; kwargs...) && isapprox(bt₁.c₁, bt₂.c₁; kwargs...) && isapprox(bt₁.c₂, bt₂.c₂; kwargs...) && isapprox(bt₁.p, bt₂.p; kwargs...)
end
