"""
    NonlinearProblem

A `NonlinearProblem` describes ``F(x) = y``, where we want to solve for ``x`` and ``F`` is in nonlinear in general (also compare this to [`LinearProblem`](@ref)).

# Keys
- `F`
- `J::Union{Callable, Missing}`: accessed by calling [`jacobian`](@ref).

# Constructors

We show an example for one particular constructor:
```jldoctest; setup = :(using SimpleSolvers)
F(y, x, params) = y .= sin.(x) .^ 2
NonlinearProblem(F, zeros(3))

# output

NonlinearProblem{typeof(F), Missing}(F, missing)
```
"""
struct NonlinearProblem{TF <: Callable, TJ <: Union{Callable, Missing}} <: AbstractProblem
    F::TF
    J::TJ

    # `x` and `f` are only used to size/type-check the problem on construction; the
    # struct stores neither, so they may be independent array types (e.g. a
    # `Vector` and a `SubArray` with the same eltype).
    function NonlinearProblem(F::Callable, J::Union{Callable, Missing}, x::AbstractArray, f::AbstractArray = x)
        @assert eltype(x) == eltype(f) "x and f must have the same element type."
        new{typeof(F), typeof(J)}(F, J)
    end
end

function NonlinearProblem(F::Callable, x::AbstractArray, f::AbstractArray = x)
    NonlinearProblem(F, missing, x, f)
end

"""
    value!(y, nlp, x, params)

Evaluate the [`NonlinearProblem`](@ref) at `x`.
"""
function value!(y::AbstractArray{T}, nlp::NonlinearProblem, x::AbstractArray{T}, params) where {T}
    nlp.F(y, x, params)
    y
end

"""
    jacobian(nlp)

Return the *Jacobian function* stored in the [`NonlinearProblem`](@ref) `nlp`.
"""
jacobian(nlp::NonlinearProblem) = nlp.J

function jacobian!(j::AbstractMatrix{T}, nlp::NonlinearProblem, x::AbstractArray{T}, params) where {T}
    nlp.J(j, x, params)
end

function jacobian!(::AbstractMatrix{T}, ::NonlinearProblem{FT, Missing},
        ::AbstractArray{T}, params) where {T, FT <: Callable}
    error("NonlinearProblem does not contain Jacobian.")
end
